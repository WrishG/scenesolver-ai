import sys
import os
import cv2
import torch
import datetime
from PIL import Image
import torchvision.transforms as T
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from collections import Counter, deque
from ultralytics import YOLO
from typing import Tuple, List, Dict, Any, Union

FFMPEG_PATH = r"C:\Users\wrish\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
# --- Project Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import CLIPMultiClassClassifier

from scripts.constants import (
    CAPTION_MAX_LENGTH, CAPTION_NUM_BEAMS,
    CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD,
    IDX_TO_LABEL, NUM_CLASSES, BINARY_IDX_TO_LABEL,
    CRIME_OBJECTS, OBJ_CONF_THRESHOLD, VIDEO_DOM_THRESHOLD,
    BINARY_NORMAL_CONF_THRESHOLD, MULTI_CLASS_CRIME_CONF_THRESHOLD, SINGLE_FRAME_ALERT_THRESHOLD
)

# --- Core Processing Functions ---

def classify_binary_frame_batch(inputs_batch: torch.Tensor, binary_classifier_model: CLIPMultiClassClassifier, device: torch.device) -> Tuple[List[str], List[float]]:
    with torch.no_grad():
        logits = binary_classifier_model(inputs_batch)
        probs = torch.softmax(logits, dim=1)
        confidences, indices = torch.max(probs, dim=1)
    labels = [BINARY_IDX_TO_LABEL[idx.item()] for idx in indices]
    confs = confidences.tolist()
    return labels, confs

def classify_frame_batch(inputs_batch: torch.Tensor, classifier_model: CLIPMultiClassClassifier, device: torch.device) -> Tuple[List[str], List[float]]:
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
    if not images_batch:
        return []
    inputs = blip_processor(images=images_batch, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=CAPTION_MAX_LENGTH, num_beams=CAPTION_NUM_BEAMS, early_stopping=True)
    captions = blip_processor.batch_decode(out, skip_special_tokens=True)
    return captions

def detect_objects_batch(frames_batch: List[Any], yolo_model: YOLO) -> List[List[str]]:
    if not frames_batch:
        return []
    results_batch = yolo_model.track(frames_batch, persist=True, tracker="bytetrack.yaml", verbose=False, conf=OBJ_CONF_THRESHOLD)
    all_detected_objs = []
    for results in results_batch:
        detected_objs_for_frame = []
        if results.boxes:
            if results.boxes.id is not None:
                track_ids = results.boxes.id.int().cpu().tolist()
                class_ids = results.boxes.cls.int().cpu().tolist()
                names = results.names
                for track_id, cls_id in zip(track_ids, class_ids):
                    label = names[cls_id]
                    if label in CRIME_OBJECTS:
                        detected_objs_for_frame.append(f"{label} ID: {track_id}")
            else:
                for box in results.boxes:
                    label = results.names[int(box.cls)]
                    if label in CRIME_OBJECTS:
                        detected_objs_for_frame.append(label)
        all_detected_objs.append(detected_objs_for_frame)
    return all_detected_objs

def _process_frame_batch(
    pil_images_batch, cv_frames_batch, frame_indices_batch,
    classifier_model, binary_classifier_model,
    blip_processor, blip_model, yolo_model,
    clip_transform, device
) -> Dict[str, Any]:
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
        batch_results["objects"].append(batch_detected_objects_lists[i])

    return batch_results


def save_crime_clip(frame_buffer: list, future_frames: list, fps: float, crime_label: str, output_dir: str, yolo_model=None) -> str:
    """Saves a crime clip with YOLO bounding boxes, re-encoded to H.264 for browser playback."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_label = crime_label.replace(" ", "_")
    temp_path = os.path.join(output_dir, f"temp_{safe_label}_{timestamp}.avi")
    clip_filename = f"clip_{safe_label}_{timestamp}.mp4"
    clip_path = os.path.join(output_dir, clip_filename)

    if not frame_buffer:
        return None

    h, w = frame_buffer[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

    for f in list(frame_buffer) + list(future_frames):
        annotated = f.copy()
        if yolo_model is not None:
            yolo_results = yolo_model(f, verbose=False, conf=OBJ_CONF_THRESHOLD)
            for result in yolo_results:
                if result.boxes:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        label = result.names[int(box.cls)]
                        conf = float(box.conf)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        text = f"{label} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), (0, 0, 255), -1)
                        cv2.putText(annotated, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"DETECTED: {crime_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        writer.write(annotated)

    writer.release()

    # Re-encode to H.264 MP4 for browser playback
    import subprocess
    result = subprocess.run([
        FFMPEG_PATH, "-y", "-i", temp_path,
        "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
        "-movflags", "+faststart", clip_path
    ], capture_output=True)

    os.remove(temp_path)

    if os.path.exists(clip_path):
        print(f"✅ Crime clip saved: {clip_filename}")
        return clip_filename
    else:
        print(f"⚠️ ffmpeg conversion failed: {result.stderr.decode()}")
        return None

def process_video(
    video_path: Union[str, int], classifier_model: Any, binary_classifier_model: Any,
    blip_processor: Any, blip_model: Any, yolo_model: Any,
    clip_transform: Any, device: Any, max_frames: int = None,
    clips_output_dir: str = None
) -> Dict[str, Any]:
    """
    Processes a video or live stream with batch processing, motion pre-filter,
    and event-triggered crime clip extraction.
    """
    if video_path == 0 or (isinstance(video_path, str) and str(video_path).isdigit()):
        cap = cv2.VideoCapture(int(video_path), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Error opening video source: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # FRAME_INTERVAL: 30 = every frame at 30fps (streams), 90 = every 3s (video files)
    FRAME_INTERVAL = 30 if max_frames is not None else 90
    BUFFER_SECONDS = 5
    POST_EVENT_SECONDS = 5
    BUFFER_SIZE = int(fps * BUFFER_SECONDS)
    POST_EVENT_FRAMES = int(fps * POST_EVENT_SECONDS)
    BATCH_SIZE = 8
    # MOTION_THRESHOLD: low for streams (catch everything), high for video files (skip static)
    MOTION_THRESHOLD = 10 if max_frames is not None else 500
    # Cap clips per video to avoid session size issues
    MAX_CLIPS = 3

    results = {
        "frame_labels": [], "frame_confs": [], "detected_objects": [],
        "captions": [], "video_fps": fps, "crime_clips": []
    }

    frame_idx = 0
    pil_images_batch, cv_frames_batch, frame_indices_batch = [], [], []

    rolling_buffer = deque(maxlen=BUFFER_SIZE)
    post_event_capture = False
    post_event_frames_remaining = 0
    post_event_buffer = []
    pre_event_snapshot = []
    pending_crime_label = None
    clipped_frame_indices = set()
    prev_gray_frame = None

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            print(f"Stream reached max frames ({max_frames}). Generating report...")
            break

        ret, frame = cap.read()
        if not ret:
            break

        rolling_buffer.append(frame.copy())

        # --- Post-event frame collection ---
        if post_event_capture:
            post_event_buffer.append(frame.copy())
            post_event_frames_remaining -= 1
            if post_event_frames_remaining <= 0:
                post_event_capture = False
                if clips_output_dir and pending_crime_label and len(results["crime_clips"]) < MAX_CLIPS:
                    clip_filename = save_crime_clip(
                        pre_event_snapshot, post_event_buffer,
                        fps, pending_crime_label, clips_output_dir, yolo_model=yolo_model
                    )
                    if clip_filename:
                        results["crime_clips"].append({
                            "filename": clip_filename,
                            "crime_label": pending_crime_label,
                            "trigger_frame": frame_idx - POST_EVENT_FRAMES
                        })
                post_event_buffer = []
                pending_crime_label = None

        if frame_idx % FRAME_INTERVAL == 0:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            motion_detected = True

            if prev_gray_frame is not None:
                frame_delta = cv2.absdiff(prev_gray_frame, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                movement_score = cv2.countNonZero(thresh)
                if movement_score < MOTION_THRESHOLD:
                    motion_detected = False

            prev_gray_frame = gray

            if not motion_detected:
                results["frame_labels"].append("Normal Activity")
                results["frame_confs"].append(1.0)
                results["captions"].append("No significant movement detected in the scene.")
                results["detected_objects"].append([])
            else:
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

                    # --- Event-triggered clip extraction ---
                    for i, (lbl, conf) in enumerate(zip(batch_output["labels"], batch_output["confs"])):
                        trigger_idx = frame_indices_batch[i]
                        if (conf >= SINGLE_FRAME_ALERT_THRESHOLD
                                and lbl != "Normal Activity"
                                and trigger_idx not in clipped_frame_indices
                                and not post_event_capture
                                and clips_output_dir
                                and len(results["crime_clips"]) < MAX_CLIPS):
                            print(f"🎬 Clip triggered: '{lbl}' at frame {trigger_idx}")
                            pre_event_snapshot = list(rolling_buffer)
                            clipped_frame_indices.add(trigger_idx)
                            if max_frames is not None:
                                # Stream: save immediately, return early
                                clip_filename = save_crime_clip(
                                    pre_event_snapshot, [], fps, lbl, clips_output_dir, yolo_model=yolo_model
                                )
                                if clip_filename:
                                    results["crime_clips"].append({
                                        "filename": clip_filename,
                                        "crime_label": lbl,
                                        "trigger_frame": trigger_idx
                                    })
                                cap.release()
                                return results
                            else:
                                # Video: collect post-event frames then save
                                post_event_capture = True
                                post_event_frames_remaining = POST_EVENT_FRAMES
                                pending_crime_label = lbl

                    pil_images_batch, cv_frames_batch, frame_indices_batch = [], [], []

        frame_idx += 1

    # --- Process remaining frames in batch ---
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

        for i, (lbl, conf) in enumerate(zip(batch_output["labels"], batch_output["confs"])):
            trigger_idx = frame_indices_batch[i]
            if (conf >= SINGLE_FRAME_ALERT_THRESHOLD
                    and lbl != "Normal Activity"
                    and trigger_idx not in clipped_frame_indices
                    and not post_event_capture
                    and clips_output_dir
                    and len(results["crime_clips"]) < MAX_CLIPS):
                print(f"🎬 Clip triggered (final batch): '{lbl}' at frame {trigger_idx}")
                pre_event_snapshot = list(rolling_buffer)
                clip_filename = save_crime_clip(
                    pre_event_snapshot, [], fps, lbl, clips_output_dir, yolo_model=yolo_model
                )
                if clip_filename:
                    results["crime_clips"].append({
                        "filename": clip_filename,
                        "crime_label": lbl,
                        "trigger_frame": trigger_idx
                    })
                clipped_frame_indices.add(trigger_idx)

    # --- Save pending clip if video ended mid-collection ---
    if post_event_capture and pending_crime_label and clips_output_dir and len(results["crime_clips"]) < MAX_CLIPS:
        clip_filename = save_crime_clip(
            pre_event_snapshot, post_event_buffer,
            fps, pending_crime_label, clips_output_dir, yolo_model=yolo_model
        )
        if clip_filename:
            results["crime_clips"].append({
                "filename": clip_filename,
                "crime_label": pending_crime_label,
                "trigger_frame": frame_idx
            })

    cap.release()
    return results


def aggregate_labels(labels: list, confs: list) -> Tuple[str, float]:
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


def summarize_captions(caps: list, detected_objects: list, summarizer: Any, overall_crime_class: str) -> str:
    if "Normal Activity" in overall_crime_class or "Normal" in overall_crime_class:
        return "The video footage was analyzed and determined to show routine activities with no significant crime-related events detected."

    unique_caps = list(set(caps))
    base_text = " ".join(unique_caps)

    # Flatten detected_objects if it's a list of lists
    if detected_objects and isinstance(detected_objects[0], list):
        detected_objects = [obj for sublist in detected_objects for obj in sublist]

    unique_tracked_objects = list(set([obj for obj in detected_objects if "ID:" in obj.upper()]))

    if unique_tracked_objects:
        object_context = f" The scene involves {len(unique_tracked_objects)} tracked individuals identified across multiple frames."
        text = base_text + object_context
    else:
        text = base_text

    if not text or text == "Normal activity observed.":
        return "No specific details could be extracted from the scene."

    word_count = len(text.split())
    if word_count < 25:
        return text

    try:
        dynamic_max = min(120, int(word_count * 1.5))
        dynamic_min = min(30, int(word_count * 0.5))
        dynamic_max = min(dynamic_max, word_count)
        summary = summarizer(text, max_length=dynamic_max, min_length=dynamic_min, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        print(f"⚠️ Summarization failed: {e}. Returning raw text.", file=sys.stderr)
        return (text[:750] + "...") if len(text) > 750 else text