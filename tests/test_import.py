"""
EmotionSense AI — Import Tests
Validates that all core modules can be imported successfully.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_import():
    """Test that config module imports correctly."""
    from config import APP_TITLE, EMOTIONS, EMOTION_COLORS
    assert APP_TITLE == "EmotionSense AI"
    assert len(EMOTIONS) == 7
    assert len(EMOTION_COLORS) == 7
    print("✅ config.py imported successfully")


def test_face_detector_import():
    """Test that FaceDetector class can be imported."""
    from face_detector import FaceDetector
    assert FaceDetector is not None
    print("✅ face_detector.py imported successfully")


def test_emotion_recognizer_import():
    """Test that EmotionRecognizer class can be imported."""
    from emotion_recognizer import EmotionRecognizer
    assert EmotionRecognizer is not None
    print("✅ emotion_recognizer.py imported successfully")


def test_utils_import():
    """Test that utility functions can be imported."""
    from utils import FPSCounter, draw_detections, draw_fps, draw_face_count
    assert FPSCounter is not None
    assert callable(draw_detections)
    assert callable(draw_fps)
    assert callable(draw_face_count)
    print("✅ utils.py imported successfully")


if __name__ == "__main__":
    test_config_import()
    test_face_detector_import()
    test_emotion_recognizer_import()
    test_utils_import()
    print("\n🎉 All import tests passed!")
