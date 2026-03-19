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
app.config['CLIPS_FOLDER'] = os.path.join(PROJECT_ROOT, 'static', 'uploads', 'clips')
app.config['DEMO_FOLDER'] = os.path.join(PROJECT_ROOT, 'static', 'demo')
os.makedirs(os.path.join(PROJECT_ROOT, 'static', 'uploads', 'clips'), exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DEMO_FOLDER'], exist_ok=True)

# --- Database Connection ---
try:
    MONGO_URI = os.environ.get("MONGO_URI")
    if not MONGO_URI:
        print("CRITICAL ERROR: MONGO_URI environment variable not set.", file=sys.stderr)
        sys.exit(1)
    client = MongoClient(
        MONGO_URI,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=20000,
        socketTimeoutMS=20000
    )
    db = client['scene_solver_db']
    users = db['users']
    analysis_history = db['analysis_history']
    print("✅ MongoDB Atlas connected successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not connect to MongoDB Atlas: {e}", file=sys.stderr)
    sys.exit(1)

# --- Global Model Variables & Immediate Loading ---
print("--- Initializing SceneSolver Models ---")
preferred_device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    GLOBAL_MODELS = load_all(device=preferred_device)
    clip_transform_global = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])
    print("✅ All Base Models Loaded Successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load models: {e}", file=sys.stderr)
    GLOBAL_MODELS = {}
    clip_transform_global = None

SUMMARIZER_MODEL_NAME = "facebook/bart-large-cnn"
summarizer_pipeline_global = None

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

# --- Auth Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('signin'))
        return f(*args, **kwargs)
    return decorated_function

# --- Pre-computed Demo Data (no models needed) ---
DEMO_DATA = {
    'video_file_name': 'demo_fighting_clip.mp4',
    'overall_crime': 'Fighting',
    'confidence_score': 1.0,
    'detected_objects': [
        'person (seen in 6 frames)',
        'person (seen in 4 frames)',
        'person (seen in 3 frames)',
    ],
    'summary': (
        'The footage captures a violent altercation between multiple individuals outside a '
        'commercial premises. One suspect delivers a powerful blow directly at an opponent '
        'near the storefront entrance. Five individuals are tracked across multiple frames '
        'with consistent IDs, indicating sustained confrontation. Law enforcement response '
        'is advised. Incident classified as Fighting with high confidence.'
    ),
    'analysis_duration': '79.82',
    'video_url': '/static/demo/demo_fighting.mp4',
    'is_stream': False,
    'crime_clips': [{
        'filename': 'demo_clip_Fighting.mp4',
        'crime_label': 'Fighting',
        'trigger_frame': 0
    }],
}

# --- Routes ---
@app.route('/')
def home():
    return redirect(url_for('signin'))

@app.route('/demo')
def demo():
    """Public demo — no login required. Uses pre-computed results so no GPU/models needed."""
    return render_template('result.html', username='Guest', **DEMO_DATA)

@app.route('/static/demo/<filename>')
def serve_demo_file(filename):
    return send_from_directory(app.config['DEMO_FOLDER'], filename)

@app.route('/register', methods=['GET', 'POST'])
def register():
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
        users.insert_one({
            'username': username,
            'password': hashed_password,
            'created_at': datetime.datetime.now()
        })
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('signin'))
    return render_template('register.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
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
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('signin'))

@app.route('/frontpage')
@login_required
def frontpage():
    return render_template('frontpage.html', username=session.get('username'))

@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        stream_url = request.form.get('stream_url', '').strip()

        video_file = request.files.get('video_file')

        video_source = None
        filename = None
        max_frames_to_process = None
        is_stream = False

        # --- Determine source: stream takes priority ---
        if stream_url:
            video_source = 0 if stream_url == '0' else stream_url
            filename = f"stream_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            max_frames_to_process = 900  # ~30 seconds at 30fps
            is_stream = True

        elif video_file and video_file.filename != '':
            filename = os.path.basename(video_file.filename)
            video_source = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_source)
            is_stream = False

        else:
            flash('Please upload a video file or enter a stream URL.', 'error')
            return redirect(request.url)

        try:
            start_time = time.time()
            print(f"\n--- Starting Analysis for: {filename} ---")

            if "classifier_model" not in GLOBAL_MODELS:
                raise KeyError("Models were not loaded properly. Check terminal for startup errors.")

            analysis_result = process_video(
                video_path=video_source,
                classifier_model=GLOBAL_MODELS["classifier_model"],
                binary_classifier_model=GLOBAL_MODELS["binary_model"],
                blip_processor=GLOBAL_MODELS["blip_processor"],
                blip_model=GLOBAL_MODELS["blip_model"],
                yolo_model=GLOBAL_MODELS["yolo_model"],
                clip_transform=clip_transform_global,
                device=GLOBAL_MODELS["device"],
                max_frames=max_frames_to_process,
                clips_output_dir=app.config['CLIPS_FOLDER']
            )


            if not analysis_result or not analysis_result.get("captions"):
                flash("Video analysis failed. The video might be too short or corrupted.", 'error')
                if not is_stream and isinstance(video_source, str) and os.path.exists(video_source):
                    try:
                        os.remove(video_source)
                    except PermissionError:
                        pass
                return redirect(url_for('index'))

            video_crime_class, crime_dominance = aggregate_labels(
                analysis_result["frame_labels"],
                analysis_result["frame_confs"]
            )

            summarizer = load_summarizer_if_needed()

            # --- FIXED: correct signature includes detected_objects ---
            video_summary = summarize_captions(
                analysis_result["captions"],
                analysis_result["detected_objects"],
                summarizer,
                video_crime_class
            )

            # detected_objects is list of lists — flatten first
            flat_objects = [obj for sublist in analysis_result["detected_objects"] for obj in (sublist if isinstance(sublist, list) else [sublist])]
            object_counts = Counter(flat_objects)
            top_objects_display = [
                f"{label} (seen in {count} frames)"
                for label, count in object_counts.most_common(5)
            ]
            if not top_objects_display:
                top_objects_display.append("No notable objects detected.")

            total_duration = time.time() - start_time
            print(f"--- Analysis Complete for {filename} in {total_duration:.2f}s ---")

            # --- Safe video URL: no file saved for streams ---
            safe_video_url = "#" if is_stream else url_for('uploaded_file', filename=filename)

            session['pdf_export_data'] = {
                'video_file_name': filename,
                'overall_crime': video_crime_class,
                'confidence_score': crime_dominance,
                'detected_objects': top_objects_display,
                'summary': video_summary,
                'analysis_duration': f"{total_duration:.2f}",
                'crime_clips': analysis_result.get('crime_clips', [])[:3],
            }

            session['analysis_results'] = {
                'video_file_name': filename,
                'overall_crime': video_crime_class,
                'confidence_score': crime_dominance,
                'detected_objects': top_objects_display,
                'summary': video_summary,
                'analysis_duration': f"{total_duration:.2f}",
                'video_url': safe_video_url,
                'is_stream': is_stream,
                'crime_clips': analysis_result.get('crime_clips', [])[:3],  # max 3 clips in session
            }

            analysis_history.insert_one({
                'username': session.get('username'),
                'filename': filename,
                'upload_time': datetime.datetime.now(),
                'crime_type': video_crime_class,
                'confidence_score': crime_dominance,
                'detected_objects': list(object_counts.keys()),
                'summary': video_summary,
                'video_url': safe_video_url
            })
            flash('Analysis complete and saved to history.', 'success')
            return redirect(url_for('result'))

        except Exception as e:
            flash(f"An error occurred during analysis: {e}", 'error')
            print(f"ERROR during video analysis: {e}", file=sys.stderr)
            traceback.print_exc()
            if not is_stream and isinstance(video_source, str) and os.path.exists(video_source):
                try:
                    os.remove(video_source)
                except PermissionError:
                    pass
            return redirect(url_for('index'))

    return render_template('index.html', username=session.get('username'))

@app.route('/result')
@login_required
def result():
    analysis_data = session.pop('analysis_results', None)
    if not analysis_data:
        flash('No analysis results found. Please upload a video first.', 'error')
        return redirect(url_for('index'))
    return render_template('result.html', username=session.get('username'), **analysis_data)

@app.route('/export-pdf')
@login_required
def export_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.units import cm
    from flask import send_file
    import io

    data = session.get('pdf_export_data')
    if not data:
        flash('No analysis data available for export. Please run an analysis first.', 'error')
        return redirect(url_for('index'))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=24, textColor=colors.HexColor('#0f56eb'),
                                  spaceAfter=6)
    elements.append(Paragraph("SceneSolver — Forensic Analysis Report", title_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0f56eb')))
    elements.append(Spacer(1, 0.4*cm))

    # Metadata
    meta_style = ParagraphStyle('Meta', parent=styles['Normal'],
                                 fontSize=10, textColor=colors.HexColor('#666666'))
    elements.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
    elements.append(Paragraph(f"Analyst: {session.get('username', 'Unknown')}", meta_style))
    elements.append(Paragraph(f"File: {data.get('video_file_name', 'N/A')}", meta_style))
    elements.append(Spacer(1, 0.6*cm))

    # Crime verdict
    verdict_color = colors.HexColor('#dc2626') if 'Normal' not in data.get('overall_crime', '') else colors.HexColor('#16a34a')
    verdict_style = ParagraphStyle('Verdict', parent=styles['Normal'],
                                    fontSize=16, textColor=verdict_color,
                                    spaceAfter=4, fontName='Helvetica-Bold')
    elements.append(Paragraph("VERDICT", styles['Heading2']))
    elements.append(Paragraph(data.get('overall_crime', 'N/A'), verdict_style))

    conf = data.get('confidence_score', 0)
    elements.append(Paragraph(f"Confidence Score: {conf * 100:.1f}%", styles['Normal']))
    elements.append(Paragraph(f"Analysis Duration: {data.get('analysis_duration', 'N/A')}s", styles['Normal']))
    elements.append(Spacer(1, 0.6*cm))

    # Detected objects table
    elements.append(Paragraph("KEY OBJECTS DETECTED", styles['Heading2']))
    objects = data.get('detected_objects', [])
    if objects:
        table_data = [['Object', 'Frequency']]
        for obj in objects:
            parts = obj.rsplit('(', 1)
            label = parts[0].strip()
            freq = '(' + parts[1] if len(parts) > 1 else ''
            table_data.append([label, freq])
        t = Table(table_data, colWidths=[10*cm, 6*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f56eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8f9fa'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(t)
    else:
        elements.append(Paragraph("No significant objects detected.", styles['Normal']))
    elements.append(Spacer(1, 0.6*cm))

    # Summary
    elements.append(Paragraph("COMPREHENSIVE INCIDENT SUMMARY", styles['Heading2']))
    summary_style = ParagraphStyle('Summary', parent=styles['Normal'],
                                    fontSize=11, leading=16,
                                    borderPad=8, borderColor=colors.HexColor('#0f56eb'),
                                    borderWidth=1, borderRadius=4)
    elements.append(Paragraph(data.get('summary', 'No summary available.'), summary_style))
    elements.append(Spacer(1, 0.6*cm))

    # Crime clips info
    clips = data.get('crime_clips', [])
    if clips:
        elements.append(Paragraph("EXTRACTED CRIME CLIPS", styles['Heading2']))
        for i, clip in enumerate(clips, 1):
            elements.append(Paragraph(
                f"Clip {i}: {clip.get('crime_label', 'Unknown')} — Triggered at frame {clip.get('trigger_frame', 'N/A')} — File: {clip.get('filename', 'N/A')}",
                styles['Normal']
            ))
        elements.append(Spacer(1, 0.4*cm))

    # Footer
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Spacer(1, 0.2*cm))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'],
                                   fontSize=8, textColor=colors.grey, alignment=1)
    elements.append(Paragraph("Generated by SceneSolver — AI-Powered Forensic Video Analysis", footer_style))

    doc.build(elements)
    buffer.seek(0)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return send_file(buffer, as_attachment=True,
                     download_name=f"SceneSolver_Report_{timestamp}.pdf",
                     mimetype='application/pdf')

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/clips/<filename>')
@login_required
def serve_clip(filename):
    return send_from_directory(app.config['CLIPS_FOLDER'], filename)

@app.route('/history')
@login_required
def history():
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
if __name__ == '__main__':
    app.run(debug=True)