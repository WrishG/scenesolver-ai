import os
import sys
import time
import traceback
import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from functools import wraps
from datetime import timedelta
import torch
import torchvision.transforms as T
from transformers import pipeline
from collections import Counter

# --- Setup Project Path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import Custom Modules ---
from backend.load_models import load_all 
from video_crime_analyzer import process_video, aggregate_labels, summarize_captions
from scripts.constants import CLIP_MEAN, CLIP_STD

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'a_default_fallback_key_for_dev')
app.permanent_session_lifetime = timedelta(hours=1)
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Database Connection ---
try:
    MONGO_URI = os.environ.get("MONGO_URI")
    if not MONGO_URI:
        print("CRITICAL ERROR: MONGO_URI environment variable not set.", file=sys.stderr)
        sys.exit(1)
    client = MongoClient(MONGO_URI)
    db = client['scene_solver_db']
    users = db['users']
    analysis_history = db['analysis_history']
    print("✅ MongoDB Atlas connected successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not connect to MongoDB Atlas: {e}", file=sys.stderr)
    sys.exit(1)

# --- Global Model Variables ---
GLOBAL_MODELS = {}
SUMMARIZER_MODEL_NAME = "facebook/bart-large-cnn"
summarizer_pipeline_global = None
clip_transform_global = None

def load_summarizer_if_needed():
    global summarizer_pipeline_global
    if summarizer_pipeline_global is None:
        print("INFO: Loading summarizer model (lazy load)...")
        device_obj = GLOBAL_MODELS.get("device", torch.device("cpu"))
        device_id = 0 if device_obj.type == "cuda" else -1
        
        summarizer_pipeline_global = pipeline("summarization", model=SUMMARIZER_MODEL_NAME, device=device_id)
        
        if device_obj.type == 'cpu':
            print("INFO: Applying dynamic quantization to Summarizer model...")
            summarizer_pipeline_global.model = torch.quantization.quantize_dynamic(
                summarizer_pipeline_global.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("✅ Summarizer model dynamically quantized.")
        print("✅ Summarizer loaded.")
    return summarizer_pipeline_global

# --- All Flask Routes ---
# (Your routes like /signin, /register, /index, etc. go here, unchanged from before)
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('signin'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return redirect(url_for('signin'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    # This function remains unchanged
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            flash('Username and password are required.', 'error')
            return redirect(url_for('register'))
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return redirect(url_for('register'))
        if users.find_one({'username': username}):
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        users.insert_one({'username': username, 'password': hashed_password, 'created_at': datetime.datetime.now()})
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('signin'))
    return render_template('register.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    # This function remains unchanged
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        user = users.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            session['user_id'] = str(user['_id'])
            session.permanent = True
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('frontpage'))
        else:
            flash('Invalid username or password.', 'error')
            return redirect(url_for('signin'))
    return render_template('signin.html')

@app.route('/logout')
def logout():
    # This function remains unchanged
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('signin'))

@app.route('/frontpage')
@login_required
def frontpage():
    # This function remains unchanged
    return render_template('frontpage.html', username=session.get('username'))

@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    # This function remains unchanged
    if request.method == 'POST':
        if 'video_file' not in request.files:
            flash('No file part in the request.', 'error')
            return redirect(request.url)
        
        video_file = request.files['video_file']
        if video_file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        if video_file:
            filename = os.path.basename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_path)

            try:
                start_time = time.time()
                print(f"\n--- Starting Analysis for: {filename} ---")

                analysis_result = process_video(
                    video_path=video_path, 
                    classifier_model=GLOBAL_MODELS["classifier_model"], 
                    binary_classifier_model=GLOBAL_MODELS["binary_model"],
                    blip_processor=GLOBAL_MODELS["blip_processor"], 
                    blip_model=GLOBAL_MODELS["blip_model"], 
                    yolo_model=GLOBAL_MODELS["yolo_model"], 
                    clip_transform=clip_transform_global,      
                    device=GLOBAL_MODELS["device"]             
                )

                if not analysis_result or not analysis_result.get("captions"):
                    flash("Video analysis failed. The video might be too short or corrupted.", 'error')
                    if os.path.exists(video_path): os.remove(video_path)
                    return redirect(url_for('index'))

                video_crime_class, crime_dominance = aggregate_labels(
                    analysis_result["frame_labels"], 
                    analysis_result["frame_confs"]
                )
                
                summarizer = load_summarizer_if_needed()
                video_summary = summarize_captions(
                    analysis_result["captions"], 
                    summarizer,
                    video_crime_class
                )

                object_counts = Counter(analysis_result["detected_objects"])
                top_objects_display = [f"{label} (seen in {count} frames)" for label, count in object_counts.most_common(5)]
                if not top_objects_display:
                    top_objects_display.append("No notable objects detected.")

                total_duration = time.time() - start_time
                print(f"--- Analysis Complete for {filename} in {total_duration:.2f}s ---")

                session['analysis_results'] = {
                    'video_file_name': filename,
                    'overall_crime': video_crime_class,
                    'confidence_score': crime_dominance,
                    'detected_objects': top_objects_display,
                    'summary': video_summary,
                    'analysis_duration': f"{total_duration:.2f}",
                    'video_url': url_for('uploaded_file', filename=filename)
                }
                
                analysis_history.insert_one({
                    'username': session.get('username'),
                    'filename': filename,
                    'upload_time': datetime.datetime.now(),
                    'crime_type': video_crime_class,
                    'confidence_score': crime_dominance,
                    'detected_objects': list(object_counts.keys()),
                    'summary': video_summary,
                    'video_url': url_for('uploaded_file', filename=filename)
                })
                flash('Analysis complete and saved to history.', 'success')
                return redirect(url_for('result'))
            
            except Exception as e:
                flash(f"An error occurred during analysis: {e}", 'error')
                print(f"ERROR during video analysis: {e}", file=sys.stderr)
                traceback.print_exc()
                if os.path.exists(video_path): os.remove(video_path)
                return redirect(url_for('index'))

    return render_template('index.html', username=session.get('username'))

@app.route('/result')
@login_required
def result():
    # This function remains unchanged
    analysis_data = session.pop('analysis_results', None)
    if not analysis_data:
        flash('No analysis results found. Please upload a video first.', 'error')
        return redirect(url_for('index'))
    return render_template('result.html', username=session.get('username'), **analysis_data)

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    # This function remains unchanged
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
@login_required
def history():
    # This function remains unchanged
    username = session.get('username')
    try:
        user_history = list(analysis_history.find({'username': username}).sort('upload_time', -1))
        return render_template('history.html', username=username, history=user_history)
    except Exception as e:
        print(f"ERROR retrieving history: {e}", file=sys.stderr)
        flash('Could not retrieve analysis history.', 'error')
        return render_template('history.html', username=username, history=[])

@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    # This function remains unchanged
    if request.method == 'POST':
        feedback_text = request.form.get('feedback')
        if not feedback_text:
            flash('Feedback cannot be empty.', 'error')
            return redirect(url_for('feedback'))
        db['feedback'].insert_one({
            'username': session.get('username'),
            'name': request.form.get('name'),
            'email': request.form.get('email'),
            'feedback_text': feedback_text,
            'submitted_at': datetime.datetime.now()
        })
        flash('Thank you for your feedback!', 'success')
        return redirect(url_for('feedback'))
    return render_template('feedback.html', username=session.get('username'))

# --- Main Execution Block ---
# In app.py

# ... (all your routes and functions) ...

# --- Main Execution Block ---
if __name__ == '__main__':
    print("--- Initializing SceneSolver ---")
    
    preferred_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    GLOBAL_MODELS = load_all(device=preferred_device)
    
    clip_transform_global = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])

    print("--- All models loaded. ---")
    print("--- To run the application, use a WSGI server like Gunicorn: ---")
    print("--- Example: gunicorn --workers 3 --bind 0.0.0.0:5000 app:app ---")

# REMOVE the app.run(...) line from here