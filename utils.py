"""
EmotionSense AI — Utility Functions
Drawing helpers, FPS counter, and color utilities.
"""

import time
from collections import deque

import cv2
import numpy as np

from config import (
    EMOTION_COLORS,
    EMOTION_EMOJIS,
    FPS_SMOOTHING_WINDOW,
)


class FPSCounter:
    """Rolling average FPS calculator."""

    def __init__(self, window_size: int = FPS_SMOOTHING_WINDOW):
        self._timestamps = deque(maxlen=window_size)
        self._fps = 0.0

    def tick(self):
        """Call once per frame."""
        now = time.time()
        self._timestamps.append(now)
        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self._fps = (len(self._timestamps) - 1) / elapsed

    @property
    def fps(self) -> float:
        return round(self._fps, 1)


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes, emotion labels, and confidence bars on the frame.

    Args:
        frame: BGR image.
        detections: List of dicts with keys "bbox", "confidence", and optionally "emotion".

    Returns:
        Annotated frame.
    """
    overlay = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        emotion_data = det.get("emotion")

        if emotion_data:
            emotion = emotion_data["dominant_emotion"]
            conf = emotion_data["confidence"]
            emoji = EMOTION_EMOJIS.get(emotion, "")
            color = EMOTION_COLORS.get(emotion, (200, 200, 200))
        else:
            emotion = "detecting"
            conf = 0
            emoji = ""
            color = (200, 200, 200)

        # ── Bounding box ────────────────────────────────────────
        # Draw a semi-transparent filled rectangle behind the box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Corner accents for a premium look
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thickness = 3
        # Top-left
        cv2.line(overlay, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(overlay, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Top-right
        cv2.line(overlay, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(overlay, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Bottom-left
        cv2.line(overlay, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(overlay, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Bottom-right
        cv2.line(overlay, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(overlay, (x2, y2), (x2, y2 - corner_len), color, thickness)

        # ── Label background ────────────────────────────────────
        label = f"{emoji} {emotion.capitalize()} {conf:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        label_y1 = max(0, y1 - th - 14)
        label_y2 = y1 - 2

        # Dark background for label
        cv2.rectangle(
            overlay,
            (x1, label_y1),
            (x1 + tw + 10, label_y2 + baseline + 4),
            (30, 30, 30),
            cv2.FILLED,
        )

        # Label text
        cv2.putText(
            overlay,
            label,
            (x1 + 5, label_y2 + baseline),
            font,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )

        # ── Confidence bar ──────────────────────────────────────
        if conf > 0:
            bar_width = x2 - x1
            bar_height = 4
            bar_y = y2 + 6
            filled = int(bar_width * (conf / 100))

            cv2.rectangle(
                overlay, (x1, bar_y), (x2, bar_y + bar_height), (50, 50, 50), cv2.FILLED
            )
            cv2.rectangle(
                overlay, (x1, bar_y), (x1 + filled, bar_y + bar_height), color, cv2.FILLED
            )

    # Blend overlay with original for a subtle transparency
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter on the top-right corner."""
    h, w = frame.shape[:2]
    label = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)

    x = w - tw - 15
    y = 30

    cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 8), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (x, y), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def draw_face_count(frame: np.ndarray, count: int) -> np.ndarray:
    """Draw face count on the top-left corner."""
    label = f"Faces: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)

    cv2.rectangle(frame, (5, 5), (tw + 20, th + 20), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (10, th + 12), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
