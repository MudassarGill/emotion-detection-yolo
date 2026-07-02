"""
EmotionSense AI — Face Detection Module
Uses YOLOv8n-face for real-time face detection.
Falls back to OpenCV Haar Cascade if YOLO face model is unavailable.
"""

import cv2
import numpy as np
from config import YOLO_CONFIDENCE_THRESHOLD, YOLO_IOU_THRESHOLD

class FaceDetector:
    """Detects faces in a frame using YOLO or Haar Cascade fallback."""

    def __init__(self):
        self.backend = None
        self.model = None
        self._initialize()

    def _initialize(self):
        """Try YOLO first, fall back to Haar Cascade."""
        # ── Attempt 1: YOLO face model ──────────────────────────
        try:
            from ultralytics import YOLO

            # Try yolov8n-face (community face-detection model)
            self.model = YOLO("yolov8n-face.pt")
            self.backend = "yolo"
            print("Face detector: YOLOv8n-face loaded successfully")
            return
        except Exception as e:
            print(f"  YOLO face model failed: {e}")

        # ── Attempt 2: Haar Cascade ─────────────────────────────
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.model = cv2.CascadeClassifier(cascade_path)
            if self.model.empty():
                raise RuntimeError("Cascade classifier is empty")
            self.backend = "haar"
            print(" Face detector: Haar Cascade loaded as fallback")
        except Exception as e:
            raise RuntimeError(f" No face detector available: {e}")

    def detect(self, frame: np.ndarray) -> list:
        """
        Detect faces in a frame.

        Returns:
            List of dicts: [{"bbox": (x1, y1, x2, y2), "confidence": float}, ...]
        """
        detections = []
        if self.backend in ["yolo", "yolo_general"]:
            detections = self._detect_yolo_face(frame)
            
        # If YOLO fails to find anything, try Haar as a dynamic fallback
        if not detections:
            detections = self._detect_haar(frame)
            
        return detections

    def _detect_yolo_face(self, frame: np.ndarray) -> list:
        """Detect faces using specialized YOLO face model."""
        results = self.model(
            frame,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            verbose=False,
        )
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append({"bbox": (x1, y1, x2, y2), "confidence": conf})
        
        if not detections and self.backend != "haar":
             print(f"DEBUG: YOLO ({self.backend}) found 0 faces at conf {YOLO_CONFIDENCE_THRESHOLD}")
             
        return detections

    def _detect_haar(self, frame: np.ndarray) -> list:
        """Detect faces using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.model.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=7, minSize=(50, 50)
        )
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                "bbox": (x, y, x + w, y + h),
                "confidence": 0.95,  # Haar doesn't provide confidence
            })
        return detections

    @property
    def backend_name(self) -> str:
        """Return a human-readable name for the active backend."""
        names = {
            "yolo": "YOLOv8n-Face",
            "haar": "Haar Cascade",
        }
        return names.get(self.backend, "Unknown")
