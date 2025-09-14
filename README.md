SceneSolver AI: Automated Video Crime Analysis Pipeline

SceneSolver is an intelligent, full-stack web application designed to automatically analyze video footage for potential criminal activities. By leveraging a sophisticated five-stage pipeline of deep learning models, the system can ingest a video file, classify the activity, generate contextual descriptions, detect relevant objects, and produce a concise, human-readable summary of the events.

The project's primary achievement is not just the multi-model analysis, but the extensive optimization performed to transform a multi-gigabyte, slow, research-grade system into a lightweight, high-performance, and deployable application ready for real-world use on both GPU and cost-effective CPU cloud servers.

The main interface of the SceneSolver application.
Live Demo & Video Walkthrough

    Live Demo: (Placeholder: Add your Hugging Face Spaces link here when it's running)

    Video Walkthrough: (https://youtu.be/Q8G-9GuV2rU?si=HhPYywkP69H14zMN)

Key Features

    Multi-Class Crime Classification: Identifies specific activities such as Theft, Fighting, Explosion, and Shooting.

    AI-Powered Scene Description: Generates human-like captions for frames containing critical activity.

    Relevant Object Detection: Uses YOLOv8 to detect and list objects pertinent to the scene (e.g., person, handbag, backpack).

    Automated Summary Generation: Synthesizes all gathered information into a single, coherent summary paragraph.

    Full-Stack Web Interface: A complete Flask and MongoDB application for user management, video uploads, and viewing analysis history.

The 5-Stage AI Analysis Pipeline

SceneSolver employs an intelligent pipeline where the output of one model informs the next, creating an efficient and accurate analysis workflow that saves computational resources.

    Stage 1: Initial Triage (Binary Classification)

        Model: Fine-tuned CLIP ViT-B/32.

        Task: Acts as a rapid, low-cost filter, classifying frames as "Crime" or "Normal." Normal frames are quickly passed over, saving significant computation time.

    Stage 2: Detailed Analysis (Multi-Class Classification)

        Model: A second fine-tuned CLIP ViT-B/32 with a different classification head.

        Task: Categorizes flagged frames into specific crime types (e.g., "Theft," "Fighting"). This more expensive model is only run on the small subset of frames that pass the initial triage.

    Stage 3: Context Generation (Image Captioning)

        Model: Fine-tuned Salesforce BLIP.

        Task: Generates rich, descriptive sentences for frames confirmed to contain a crime, providing crucial details like "a person running away with a handbag."

    Stage 4: Object Recognition (Object Detection)

        Model: YOLOv8n.

        Task: Detects a predefined list of objects relevant to criminal activity, providing concrete evidence and keywords for the report.

    Stage 5: Final Report (Text Summarization)

        Model: Facebook BART.

        Task: Synthesizes the unique captions and crime labels into a single, concise paragraph, filtering out redundant information to present a clear narrative of the events.

From Research to Reality: The Optimization Journey

A key focus of this project was transforming a massive, slow prototype into a deployable product. This was achieved through three pillars of optimization:
1. Model Shrinking (Deployment Readiness)

The initial models were over 2GB combined, making deployment impractical.

    Classifier Head Extraction: Instead of saving the entire 500MB+ fine-tuned CLIP model, I engineered a method to extract and save only the tiny, newly-trained 16MB classification head. The application now loads the base pre-trained CLIP model at startup and simply attaches these custom-trained head weights.

    Half-Precision Conversion: The fine-tuned BLIP model was converted to .safetensors format in half-precision (FP16), cutting its file size in half without a significant loss in quality.

2. GPU & CPU Performance

The application is dual-optimized to run at high speed on expensive GPU hardware and efficiently on cheap CPU-only cloud hardware.

    For GPU: Automatic Mixed-Precision is enabled for CLIP models, and 8-bit quantization (BitsAndBytes) is used for the large BLIP model to reduce memory footprint and accelerate performance.

    For CPU Cloud Deployment: PyTorch's Dynamic Quantization is automatically applied to the BLIP model, classifier heads, and the BART summarizer when no GPU is detected. This converts their weights to the highly efficient INT8 format, leading to significantly faster inference on standard cloud CPUs.

3. Code and Logic Efficiency

    Batch Processing: The analysis logic was re-architected to process frames in batches, allowing models to analyze multiple frames in a single pass. This is exponentially more efficient and fully leverages modern hardware.

    Lazy Loading: The BART summarizer model is only loaded into memory on the first request that requires a summary, reducing the application's initial memory footprint and startup time.

Technology Stack

    Backend: Flask, Gunicorn

    Database: MongoDB

    Deep Learning: PyTorch, Hugging Face Transformers

    Core Models: OpenAI CLIP, Salesforce BLIP, Ultralytics YOLOv8, Facebook BART

    Key Libraries: OpenCV, Pillow, Werkzeug

    Deployment: Docker

Local Setup and Installation
Prerequisites

    Python 3.9+

    Git

    An environment variable tool (e.g., python-dotenv)

1. Clone the Repository

git clone [https://github.com/WrishG/scenesolver-ai.git](https://github.com/WrishG/scenesolver-ai.git)
cd scenesolver-ai

2. Set Up a Virtual Environment

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Download Pre-Trained Models

The required model weights are hosted on Hugging Face Hub. Create a models directory in the project root and download the files into it.

mkdir models
cd models
# --- Download the following files from the Hugging Face Hub repo ---
# (Placeholder: Add direct wget/curl links to your model files here)
# Example:
# wget [https://huggingface.co/your-username/your-repo/resolve/main/yolov8n.pt](https://huggingface.co/your-username/your-repo/resolve/main/yolov8n.pt)
# wget [https://huggingface.co/your-username/your-repo/resolve/main/auto_head_multi.pth](https://huggingface.co/your-username/your-repo/resolve/main/auto_head_multi.pth)
# ...etc.
cd ..

5. Set Up Environment Variables

Create a file named .env in the root of the project directory and add your secret keys:

# .env file
SECRET_KEY="your_super_strong_random_secret_key"
MONGO_URI="your_mongodb_atlas_connection_string"

6. Run the Application

You can use the built-in Flask development server or a production-grade server like Waitress.

# Using Flask's development server
flask run

# Using Waitress (for Windows)
waitress-serve --host 127.0.0.1 --port 5000 app:app

The application will be available at http://127.0.0.1:5000.
Contact

Wrish - [Wrishg@gmail.com] - [Your LinkedIn Profile URL]

Project Link: https://github.com/WrishG/scenesolver-ai
