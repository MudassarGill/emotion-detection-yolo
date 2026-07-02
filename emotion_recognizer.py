"""
EmotionSense AI — Emotion Recognition Module
Uses DeepFace for pretrained facial emotion classification.
"""

import cv2
import numpy as np


class EmotionRecognizer:
    """Classifies facial emotions using DeepFace (pretrained, no training required)."""

    def __init__(self):
        self._deepface = None
        self._initialize()

    def _initialize(self):
        """Import and warm up DeepFace."""
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            print(" Emotion recognizer: DeepFace loaded successfully")
        except ImportError:
            raise ImportError(
                "DeepFace is required. Install it with: pip install deepface tf-keras"
            )

    def recognize(self, face_crop: np.ndarray) -> dict:
        """
        Classify emotion from a face crop.

        Args:
            face_crop: BGR image of a cropped face region.

        Returns:
            dict with keys:
                - "dominant_emotion": str
                - "confidence": float (0-100)
                - "all_emotions": dict of emotion → score
            Returns None if recognition fails.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        # Ensure minimum face size for reliable recognition
        h, w = face_crop.shape[:2]
        if h < 20 or w < 20:
            return None

        try:
            # Resize very small faces up for better accuracy
            if h < 45 or w < 45:
                face_crop = cv2.resize(face_crop, (48, 48), interpolation=cv2.INTER_CUBIC)

            results = self._deepface.analyze(
                img_path=face_crop,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )

            # DeepFace may return a list of results
            if isinstance(results, list):
                result = results[0]
            else:
                result = results

            dominant = result.get("dominant_emotion", "neutral")
            emotions = result.get("emotion", {})
            confidence = emotions.get(dominant, 0.0)

            return {
                "dominant_emotion": dominant.lower(),
                "confidence": round(confidence, 1),
                "all_emotions": {k.lower(): round(v, 1) for k, v in emotions.items()},
            }

        except Exception:
            return None
