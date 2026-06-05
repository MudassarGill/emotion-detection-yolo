"""
EmotionSense AI — Configuration & Constants
"""

# ─── Page Config ────────────────────────────────────────────────
APP_TITLE = "EmotionSense AI"
APP_ICON = "🧠"
APP_LAYOUT = "wide"

# ─── YOLO Face Detection ────────────────────────────────────────
YOLO_MODEL_NAME = "yolov8n-face.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.45
YOLO_IOU_THRESHOLD = 0.5

# ─── Emotion Labels ─────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ─── Emotion Color Palette (BGR for OpenCV) ──────────────────────
EMOTION_COLORS = {
    "angry":    (0, 0, 255),       # Red
    "disgust":  (0, 140, 255),     # Dark Orange
    "fear":     (0, 165, 255),     # Orange
    "happy":    (0, 255, 100),     # Green
    "sad":      (255, 150, 0),     # Blue
    "surprise": (255, 0, 255),     # Magenta
    "neutral":  (200, 200, 200),   # Gray
}

# ─── Emotion Emoji Mapping ──────────────────────────────────────
EMOTION_EMOJIS = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "sad":      "😢",
    "surprise": "😲",
    "neutral":  "😐",
}

# ─── UI Colors (Hex for Streamlit) ──────────────────────────────
EMOTION_HEX_COLORS = {
    "angry":    "#FF4444",
    "disgust":  "#FF8C00",
    "fear":     "#FFA500",
    "happy":    "#00FF64",
    "sad":      "#4A90D9",
    "surprise": "#FF00FF",
    "neutral":  "#C8C8C8",
}

# ─── Camera Settings ────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ─── FPS Settings ───────────────────────────────────────────────
FPS_SMOOTHING_WINDOW = 30
