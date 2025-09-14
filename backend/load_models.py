import os
from pathlib import Path
import logging
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
from ultralytics import YOLO
from safetensors.torch import load_file as safeload
from collections import OrderedDict

from models import CLIPMultiClassClassifier 
from scripts.constants import NUM_CLASSES, NUM_BINARY_CLASSES

# --- DYNAMIC PATH CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FINETUNED_BLIP_DIR = MODELS_DIR / "blip_finetuned_crime"
BLIP_SAFETENSORS_PATH = MODELS_DIR / "blip_finetuned_fp16.safetensors"
MULTI_HEAD_PATH = MODELS_DIR / "auto_head_multi.pth"
BINARY_HEAD_PATH = MODELS_DIR / "auto_head_binary.pth"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger("model_loader")

def load_blip_model(device: torch.device):
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = None

    if device.type == "cuda":
        log.info("Applying 8-bit BitsAndBytes quantization for GPU.")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model_source = str(FINETUNED_BLIP_DIR) if FINETUNED_BLIP_DIR.exists() else "Salesforce/blip-image-captioning-base"
        blip_model = BlipForConditionalGeneration.from_pretrained(model_source, quantization_config=bnb_config, device_map="auto")
    else:
        log.info("Applying dynamic quantization for CPU.")
        model_source = str(FINETUNED_BLIP_DIR) if FINETUNED_BLIP_DIR.exists() else "Salesforce/blip-image-captioning-base"
        blip_model = BlipForConditionalGeneration.from_pretrained(model_source)
        blip_model = torch.quantization.quantize_dynamic(blip_model, {torch.nn.Linear}, dtype=torch.qint8)
        log.info("✅ BLIP model dynamically quantized for CPU.")

    blip_model.eval()
    log.info("✅ BLIP model loaded successfully.")
    return blip_processor, blip_model

def load_all(device: str = "cpu"):
    if torch.cuda.is_available() and device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}")

    blip_processor, blip_model = load_blip_model(device)

    log.info("Creating classifier wrappers...")
    multi_classifier = CLIPMultiClassClassifier(num_classes=NUM_CLASSES)
    binary_classifier = CLIPMultiClassClassifier(num_classes=NUM_BINARY_CLASSES)
    
    # --- FINAL FIX ---
    # We remove the manual .half() calls. 
    # The .to(device) call is sufficient for the model to work correctly on the GPU.
    multi_classifier.to(device)
    binary_classifier.to(device)
    # --- END OF FIX ---

    if MULTI_HEAD_PATH.exists():
        log.info(f"Loading weights for multi-class head...")
        full_state_dict = torch.load(MULTI_HEAD_PATH, map_location=device)
        head_state_dict = OrderedDict((key[len("classifier."):], value) for key, value in full_state_dict.items() if key.startswith("classifier."))
        multi_classifier.classifier.load_state_dict(head_state_dict)
        log.info(f"✅ Loaded multi-class head weights.")
    else:
        log.warning(f"Multi-class model file not found.")

    if BINARY_HEAD_PATH.exists():
        log.info(f"Loading weights for binary head...")
        full_state_dict = torch.load(BINARY_HEAD_PATH, map_location=device)
        head_state_dict = OrderedDict((key[len("classifier."):], value) for key, value in full_state_dict.items() if key.startswith("classifier."))
        binary_classifier.classifier.load_state_dict(head_state_dict)
        log.info(f"✅ Loaded binary-class head weights.")
    else:
        log.warning(f"Binary model file not found.")

    if device.type == 'cpu':
        log.info("Applying dynamic quantization to classifiers...")
        multi_classifier = torch.quantization.quantize_dynamic(multi_classifier, {torch.nn.Linear}, dtype=torch.qint8)
        binary_classifier = torch.quantization.quantize_dynamic(binary_classifier, {torch.nn.Linear}, dtype=torch.qint8)
        log.info("✅ Classifiers dynamically quantized.")

    multi_classifier.eval()
    binary_classifier.eval()

    yolo_model = None
    if YOLO_MODEL_PATH.exists():
        log.info(f"Loading YOLO model...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        yolo_model.to(device)
        log.info("✅ YOLO model loaded successfully.")
    else:
        log.warning(f"YOLO model not found.")

    return {
        "device": device,
        "blip_processor": blip_processor,
        "blip_model": blip_model,
        "classifier_model": multi_classifier,
        "binary_model": binary_classifier,
        "yolo_model": yolo_model,
    }