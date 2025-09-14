import sys
import os
import cv2
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from collections import Counter
from ultralytics import YOLO
from typing import Tuple, List, Dict, Any

# --- Project Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- THIS IS THE CHANGED LINE ---
# We no longer import the deleted CLIPBinaryClassifier
from models import CLIPMultiClassClassifier
# --- END OF CHANGE ---

from scripts.constants import (
    FRAME_INTERVAL, CAPTION_MAX_LENGTH, CAPTION_NUM_BEAMS,
    CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD,
    IDX_TO_LABEL, NUM_CLASSES, BINARY_IDX_TO_LABEL,
    CRIME_OBJECTS, OBJ_CONF_THRESHOLD, VIDEO_DOM_THRESHOLD,
    BINARY_NORMAL_CONF_THRESHOLD, MULTI_CLASS_CRIME_CONF_THRESHOLD, SINGLE_FRAME_ALERT_THRESHOLD
)

# --- Core Processing Functions ---

def classify_binary_frame_batch(inputs_batch: torch.Tensor, binary_classifier_model: CLIPMultiClassClassifier, device: torch.device) -> Tuple[List[str], List[float]]:
    """Classifies a batch of images as 'Crime' or 'Normal'."""
    with torch.no_grad():
        logits = binary_classifier_model(inputs_batch)
        probs = torch.softmax(logits, dim=1)
        confidences, indices = torch.max(probs, dim=1)
    
    labels = [BINARY_IDX_TO_LABEL[idx.item()] for idx in indices]
    confs = confidences.tolist()
    return labels, confs

def classify_frame_batch(inputs_batch: torch.Tensor, classifier_model: CLIPMultiClassClassifier, device: torch.device) -> Tuple[List[str], List[float]]:
    """
    Classifies a batch of images into specific crime types.
    """
    with torch.no_grad():
        logits = classifier_model(inputs_batch)
        probs = torch.softmax(logits, dim=1)

        k_val = min(2, NUM_CLASSES) 
        top_probs, top_indices = torch.topk(probs, k=k_val, dim=1)
    
    final_labels = []
    final_confs = []

    for i in range(inputs_batch.shape[0]):
        top1_idx = top_indices[i, 0].item()
        top1_prob = top_probs[i, 0].item()
        top1_label = IDX_TO_LABEL[top1_idx]

        current_label = top1_label
        current_conf = top1_prob

        if k_val > 1:
            top2_idx = top_indices[i, 1].item()
            top2_label = IDX_TO_LABEL[top2_idx]

            if top1_label == "Shooting" and top2_label == "Theft":
                current_label = "Theft"
                current_conf = top_probs[i, 1].item()
        
        final_labels.append(current_label)
        final_confs.append(current_conf)

    return final_labels, final_confs

def caption_frame_batch(images_batch: List[Image.Image], blip_processor: BlipProcessor, blip_model, device: torch.device) -> List[str]:
    """Generates captions for a batch of images."""
    if not images_batch:
        return []

    inputs = blip_processor(images=images_batch, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=CAPTION_MAX_LENGTH, num_beams=CAPTION_NUM_BEAMS, early_stopping=True)
    
    captions = blip_processor.batch_decode(out, skip_special_tokens=True)
    return captions

def detect_objects_batch(frames_batch: List[Any], yolo_model: YOLO) -> List[List[str]]:
    """Detects specified objects in a batch of video frames."""
    if not frames_batch:
        return []
        
    results_batch = yolo_model(frames_batch, verbose=False, conf=OBJ_CONF_THRESHOLD)
    
    all_detected_objs = []
    for results in results_batch:
        detected_objs_for_frame = []
        if results.boxes:
            for box in results.boxes:
                label = yolo_model.model.names[int(box.cls)]
                if label in CRIME_OBJECTS:
                    detected_objs_for_frame.append(label)
        all_detected_objs.append(detected_objs_for_frame)
    return all_detected_objs

def _process_frame_batch(
    pil_images_batch: List[Image.Image], cv_frames_batch: List[Any], frame_indices_batch: List[int],
    classifier_model: CLIPMultiClassClassifier, binary_classifier_model: CLIPMultiClassClassifier,
    blip_processor: BlipProcessor, blip_model: Any, yolo_model: YOLO,
    clip_transform: T.Compose, device: torch.device
) -> Dict[str, Any]:
    """Helper function to process one batch of frames."""
    batch_results = {"labels": [], "confs": [], "captions": [], "objects": []}

    if not pil_images_batch:
        return batch_results

    clip_inputs_batch = torch.stack([clip_transform(img) for img in pil_images_batch]).to(device)
    batch_binary_labels, batch_binary_confs = classify_binary_frame_batch(clip_inputs_batch, binary_classifier_model, device)
    
    frames_for_detailed_analysis_indices_in_batch = []
    pil_images_for_detailed_analysis = []
    for i, (binary_label, binary_conf) in enumerate(zip(batch_binary_labels, batch_binary_confs)):
        if binary_label == "Crime" or (binary_label == "Normal" and binary_conf <= BINARY_NORMAL_CONF_THRESHOLD):
            frames_for_detailed_analysis_indices_in_batch.append(i)
            pil_images_for_detailed_analysis.append(pil_images_batch[i])

    batch_crime_labels_detailed, batch_crime_confs_detailed, batch_captions_detailed = [], [], []
    if frames_for_detailed_analysis_indices_in_batch:
        clip_inputs_detailed_batch = torch.stack([clip_transform(pil_images_batch[i]) for i in frames_for_detailed_analysis_indices_in_batch]).to(device)
        batch_crime_labels_detailed, batch_crime_confs_detailed = classify_frame_batch(clip_inputs_detailed_batch, classifier_model, device)
        batch_captions_detailed = caption_frame_batch(pil_images_for_detailed_analysis, blip_processor, blip_model, device)

    batch_detected_objects_lists = detect_objects_batch(cv_frames_batch, yolo_model)

    detailed_analysis_ptr = 0
    for i in range(len(pil_images_batch)):
        current_frame_label = "Normal Activity"
        current_frame_conf = batch_binary_confs[i]
        current_frame_caption = "Normal activity observed."
        
        if i in frames_for_detailed_analysis_indices_in_batch:
            crime_label = batch_crime_labels_detailed[detailed_analysis_ptr]
            crime_conf = batch_crime_confs_detailed[detailed_analysis_ptr]
            caption = batch_captions_detailed[detailed_analysis_ptr]

            if crime_conf > MULTI_CLASS_CRIME_CONF_THRESHOLD:
                current_frame_label = crime_label
                current_frame_conf = crime_conf
                current_frame_caption = caption
                if crime_conf >= SINGLE_FRAME_ALERT_THRESHOLD:
                    print(f"    🚨 HIGH CONFIDENCE ALERT: Detected '{crime_label}' with {crime_conf:.2f} confidence in frame {frame_indices_batch[i]}.")
            
            detailed_analysis_ptr += 1

        batch_results["labels"].append(current_frame_label)
        batch_results["confs"].append(current_frame_conf)
        batch_results["captions"].append(current_frame_caption)
        batch_results["objects"].extend(batch_detected_objects_lists[i])
        
    return batch_results

# --- THIS IS THE OTHER CHANGED LINE ---
# Updated the type hint for binary_classifier_model
def process_video(
    video_path: str, classifier_model: CLIPMultiClassClassifier, binary_classifier_model: CLIPMultiClassClassifier,
    blip_processor: BlipProcessor, blip_model: Any, yolo_model: YOLO,
    clip_transform: T.Compose, device: torch.device
) -> Dict[str, Any]:
# --- END OF CHANGE ---
    """
    Processes a video by analyzing frames at a set interval using batch processing.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    results = {"frame_labels": [], "frame_confs": [], "detected_objects": [], "captions": [], "video_fps": fps}
    frame_idx, BATCH_SIZE = 0, 8
    pil_images_batch, cv_frames_batch, frame_indices_batch = [], [], []

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % FRAME_INTERVAL == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_images_batch.append(pil_img)
            cv_frames_batch.append(frame)
            frame_indices_batch.append(frame_idx)

            if len(pil_images_batch) >= BATCH_SIZE:
                batch_output = _process_frame_batch(
                    pil_images_batch, cv_frames_batch, frame_indices_batch,
                    classifier_model, binary_classifier_model, blip_processor, blip_model, yolo_model,
                    clip_transform, device
                )
                results["frame_labels"].extend(batch_output["labels"])
                results["frame_confs"].extend(batch_output["confs"])
                results["captions"].extend(batch_output["captions"])
                results["detected_objects"].extend(batch_output["objects"])
                pil_images_batch, cv_frames_batch, frame_indices_batch = [], [], []

        frame_idx += 1

    if pil_images_batch:
        batch_output = _process_frame_batch(
            pil_images_batch, cv_frames_batch, frame_indices_batch,
            classifier_model, binary_classifier_model, blip_processor, blip_model, yolo_model,
            clip_transform, device
        )
        results["frame_labels"].extend(batch_output["labels"])
        results["frame_confs"].extend(batch_output["confs"])
        results["captions"].extend(batch_output["captions"])
        results["detected_objects"].extend(batch_output["objects"])

    cap.release()
    return results

def aggregate_labels(labels: list, confs: list) -> Tuple[str, float]:
    """Aggregates frame-level labels into an overall video conclusion."""
    if not labels:
        return "No Activity Detected", 0.0

    all_label_counts = Counter(labels)
    most_common_label, count = all_label_counts.most_common(1)[0]
    dominance = count / len(labels)
    
    if most_common_label == "Normal Activity":
        if dominance >= VIDEO_DOM_THRESHOLD:
            return "Normal Activity Verified", dominance
        else:
            specific_crime_labels = [lbl for lbl in labels if lbl != "Normal Activity"]
            if specific_crime_labels:
                most_common_crime, _ = Counter(specific_crime_labels).most_common(1)[0]
                return f"Mixed Incident (predominantly Normal, with {most_common_crime})", dominance
            else:
                return "Normal Activity Verified", dominance
    else:
        if dominance >= VIDEO_DOM_THRESHOLD:
            return most_common_label, dominance
        else:
            return f"Mixed Incident (featuring {most_common_label})", dominance

def summarize_captions(caps: list, summarizer: pipeline, overall_crime_class: str) -> str:
    """Summarizes a list of captions, with special handling for normal videos."""
    if "Normal Activity" in overall_crime_class:
        return "The video footage was analyzed and determined to show routine activities with no significant crime-related events detected."

    text = " ".join(dict.fromkeys(cap for cap in caps if cap != "Normal activity observed."))
    if not text.strip():
        return "No specific details could be extracted from the scene."
    
    try:
        if summarizer is None:
            return (text[:750] + "...") if len(text) > 750 else text
        
        summary = summarizer(text, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        print(f"⚠️ Summarization failed: {e}. Returning raw text.", file=sys.stderr)
        return (text[:750] + "...") if len(text) > 750 else text