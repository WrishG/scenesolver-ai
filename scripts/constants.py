# --- Configuration Constants ---
FRAME_INTERVAL = 150
CAPTION_MAX_LENGTH = 30 # As per your previous request
CAPTION_NUM_BEAMS = 5
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IDX_TO_LABEL = {0: "Explosion", 1: "Fighting", 2: "Theft", 3: "Shooting"}
NUM_CLASSES = len(IDX_TO_LABEL)

BINARY_IDX_TO_LABEL = {0: "Crime", 1: "Normal"}
NUM_BINARY_CLASSES = len(BINARY_IDX_TO_LABEL)

CRIME_OBJECTS = {"person", "backpack", "handbag", "suitcase", "cell phone", "bottle", "sports ball", "baseball bat"}
OBJ_CONF_THRESHOLD = 0.5
VIDEO_DOM_THRESHOLD = 0.35
BINARY_NORMAL_CONF_THRESHOLD = 0.70
MULTI_CLASS_CRIME_CONF_THRESHOLD = 0.40
SINGLE_FRAME_ALERT_THRESHOLD = 0.85
