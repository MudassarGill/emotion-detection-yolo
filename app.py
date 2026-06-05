"""
EmotionSense AI — Real-Time Facial Emotion Recognition
Main Streamlit Application
"""

import cv2
import numpy as np
import streamlit as st

from config import (
    APP_ICON,
    APP_LAYOUT,
    APP_TITLE,
    CAMERA_INDEX,
    EMOTION_EMOJIS,
    EMOTION_HEX_COLORS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)
from emotion_recognizer import EmotionRecognizer
from face_detector import FaceDetector
from utils import FPSCounter, draw_detections, draw_face_count, draw_fps


# ═══════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════
# Custom CSS for Premium UI
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Global ────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ────────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #888;
        font-size: 1rem;
    }

    /* ── Metric Cards ──────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.4rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    .metric-card .label {
        color: #888;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fff;
    }

    /* ── Emotion Badge ─────────────────────────────────────── */
    .emotion-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.2rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* ── Sidebar Styling ───────────────────────────────────── */
    .sidebar-section {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    .sidebar-title {
        color: #667eea;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }

    /* ── Confidence Bar ────────────────────────────────────── */
    .conf-bar-bg {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        height: 8px;
        width: 100%;
        margin-top: 4px;
    }
    .conf-bar-fill {
        height: 8px;
        border-radius: 6px;
        transition: width 0.3s ease;
    }

    /* ── Video container ───────────────────────────────────── */
    .video-container {
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }

    /* ── Status Indicator ──────────────────────────────────── */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Session State Initialization
# ═══════════════════════════════════════════════════════════════════
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "face_detector" not in st.session_state:
    st.session_state.face_detector = None
if "emotion_recognizer" not in st.session_state:
    st.session_state.emotion_recognizer = None


# ═══════════════════════════════════════════════════════════════════
# Model Loading (cached)
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_face_detector():
    return FaceDetector()


@st.cache_resource
def load_emotion_recognizer():
    return EmotionRecognizer()


# ═══════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo / Brand
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size: 3rem;">🧠</div>
        <h2 style="
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0.3rem 0;
        ">EmotionSense AI</h2>
        <p style="color: #888; font-size: 0.8rem;">Real-Time Emotion Detection</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Camera Controls ─────────────────────────────────────────
    st.markdown('<div class="sidebar-title">📹 Camera Controls</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Start", use_container_width=True, type="primary")
    with col2:
        stop_btn = st.button("⏹ Stop", use_container_width=True)

    if start_btn:
        st.session_state.camera_running = True
    if stop_btn:
        st.session_state.camera_running = False

    # Status indicator
    if st.session_state.camera_running:
        st.markdown(
            '<p><span class="status-dot" style="background:#00ff64;"></span> Camera Active</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p><span class="status-dot" style="background:#ff4444;"></span> Camera Stopped</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Model Info ──────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">🤖 Model Info</div>', unsafe_allow_html=True)

    try:
        detector = load_face_detector()
        st.success(f"Face: {detector.backend_name}")
    except Exception as e:
        detector = None
        st.error(f"Face detector error: {e}")

    try:
        recognizer = load_emotion_recognizer()
        st.success("Emotion: DeepFace")
    except Exception as e:
        recognizer = None
        st.error(f"Emotion model error: {e}")

    st.divider()

    # ── Detectable Emotions ─────────────────────────────────────
    st.markdown('<div class="sidebar-title">🎭 Detectable Emotions</div>', unsafe_allow_html=True)

    emotions_display = ""
    for emo, emoji in EMOTION_EMOJIS.items():
        color = EMOTION_HEX_COLORS.get(emo, "#fff")
        emotions_display += f'<span class="emotion-badge" style="border-color:{color}; color:{color};">{emoji} {emo.capitalize()}</span>'
    st.markdown(emotions_display, unsafe_allow_html=True)

    st.divider()

    # ── About ───────────────────────────────────────────────────
    st.markdown("""
    <div style="color: #666; font-size: 0.75rem; text-align: center; padding-top: 0.5rem;">
        Built with ❤️ using Streamlit<br>
        YOLO + DeepFace<br>
        © 2026 EmotionSense AI
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Main Content Area
# ═══════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1> EmotionSense AI</h1>
    <p>Real-time facial emotion recognition powered by YOLO & DeepFace</p>
</div>
""", unsafe_allow_html=True)

# Analytics placeholders
analytics_cols = st.columns(4)
with analytics_cols[0]:
    fps_metric = st.empty()
with analytics_cols[1]:
    face_count_metric = st.empty()
with analytics_cols[2]:
    dominant_emotion_metric = st.empty()
with analytics_cols[3]:
    confidence_metric = st.empty()

# Video feed
video_placeholder = st.empty()

# Detailed analytics section
st.markdown("---")
detail_cols = st.columns([2, 1])

with detail_cols[0]:
    emotion_chart_placeholder = st.empty()
with detail_cols[1]:
    emotion_list_placeholder = st.empty()


# ═══════════════════════════════════════════════════════════════════
# Camera Loop
# ═══════════════════════════════════════════════════════════════════
if st.session_state.camera_running and detector is not None and recognizer is not None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        st.error(" Cannot open webcam. Please check your camera connection.")
        st.session_state.camera_running = False
    else:
        fps_counter = FPSCounter()

        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.warning(" Failed to read frame from webcam.")
                break

            fps_counter.tick()

            # ── Detect faces ────────────────────────────────────
            face_detections = detector.detect(frame)

            # ── Recognize emotions ──────────────────────────────
            results = []
            for det in face_detections:
                x1, y1, x2, y2 = det["bbox"]
                # Clamp to frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_crop = frame[y1:y2, x1:x2]
                emotion_data = recognizer.recognize(face_crop)

                results.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": det["confidence"],
                    "emotion": emotion_data,
                })

            # ── Draw on frame ───────────────────────────────────
            annotated = draw_detections(frame, results)
            annotated = draw_fps(annotated, fps_counter.fps)
            annotated = draw_face_count(annotated, len(results))

            # ── Convert BGR → RGB for Streamlit ─────────────────
            display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # ── Update video feed ───────────────────────────────
            video_placeholder.image(display_frame, channels="RGB", use_container_width=True)

            # ── Update analytics ────────────────────────────────
            fps_metric.markdown(f"""
            <div class="metric-card">
                <div class="label">⚡ FPS</div>
                <div class="value">{fps_counter.fps}</div>
            </div>
            """, unsafe_allow_html=True)

            face_count_metric.markdown(f"""
            <div class="metric-card">
                <div class="label">👤 Faces Detected</div>
                <div class="value">{len(results)}</div>
            </div>
            """, unsafe_allow_html=True)

            # Dominant emotion across all faces
            if results and any(r["emotion"] for r in results):
                emotions_found = [
                    r["emotion"] for r in results if r["emotion"] is not None
                ]
                if emotions_found:
                    # Find the highest-confidence detection
                    best = max(emotions_found, key=lambda e: e["confidence"])
                    best_emoji = EMOTION_EMOJIS.get(best["dominant_emotion"], "")
                    best_color = EMOTION_HEX_COLORS.get(best["dominant_emotion"], "#fff")

                    dominant_emotion_metric.markdown(f"""
                    <div class="metric-card">
                        <div class="label">🎭 Dominant Emotion</div>
                        <div class="value" style="color:{best_color};">
                            {best_emoji} {best["dominant_emotion"].capitalize()}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    confidence_metric.markdown(f"""
                    <div class="metric-card">
                        <div class="label"> Confidence</div>
                        <div class="value" style="color:{best_color};">{best["confidence"]:.1f}%</div>
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:{best['confidence']}%; background:{best_color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Detailed emotion breakdown ──────────────
                    # Bar chart data
                    all_emotions = best.get("all_emotions", {})
                    if all_emotions:
                        chart_html = "<div>"
                        for emo, score in sorted(all_emotions.items(), key=lambda x: -x[1]):
                            emoji = EMOTION_EMOJIS.get(emo, "")
                            color = EMOTION_HEX_COLORS.get(emo, "#888")
                            chart_html += f"""
                            <div style="margin: 6px 0;">
                                <div style="display:flex; justify-content:space-between; color:#ccc; font-size:0.85rem;">
                                    <span>{emoji} {emo.capitalize()}</span>
                                    <span>{score:.1f}%</span>
                                </div>
                                <div class="conf-bar-bg">
                                    <div class="conf-bar-fill" style="width:{score}%; background:{color};"></div>
                                </div>
                            </div>
                            """
                        chart_html += "</div>"
                        emotion_chart_placeholder.markdown(
                            f'<div class="metric-card"><div class="label"> Emotion Distribution</div>{chart_html}</div>',
                            unsafe_allow_html=True,
                        )

                    # Per-face list
                    faces_html = ""
                    for i, r in enumerate(results):
                        if r["emotion"]:
                            e = r["emotion"]
                            e_emoji = EMOTION_EMOJIS.get(e["dominant_emotion"], "")
                            e_color = EMOTION_HEX_COLORS.get(e["dominant_emotion"], "#888")
                            faces_html += f"""
                            <div style="padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                                <span style="color:#888;">Face {i+1}:</span>
                                <span style="color:{e_color}; font-weight:600;">
                                    {e_emoji} {e["dominant_emotion"].capitalize()}
                                </span>
                                <span style="color:#666;"> ({e["confidence"]:.1f}%)</span>
                            </div>
                            """
                    if faces_html:
                        emotion_list_placeholder.markdown(
                            f'<div class="metric-card"><div class="label">👥 Per-Face Results</div>{faces_html}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    dominant_emotion_metric.markdown("""
                    <div class="metric-card">
                        <div class="label"> Dominant Emotion</div>
                        <div class="value">—</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                dominant_emotion_metric.markdown("""
                <div class="metric-card">
                    <div class="label"> Dominant Emotion</div>
                    <div class="value">—</div>
                </div>
                """, unsafe_allow_html=True)

                confidence_metric.markdown("""
                <div class="metric-card">
                    <div class="label"> Confidence</div>
                    <div class="value">—</div>
                </div>
                """, unsafe_allow_html=True)

        cap.release()

elif not st.session_state.camera_running:
    # Show placeholder when camera is off
    video_placeholder.markdown("""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 400px;
        background: linear-gradient(135deg, #0a0a1a, #1a1a2e, #16213e);
        border-radius: 16px;
        border: 2px dashed rgba(102, 126, 234, 0.3);
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📹</div>
        <h3 style="
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        ">Camera Ready</h3>
        <p style="color: #666; margin-top: 0.5rem;">
            Click <strong>▶ Start</strong> in the sidebar to begin
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Empty analytics
    for col, (icon, label) in zip(analytics_cols, [
        ("⚡", "FPS"), ("👤", "Faces"), ("🎭", "Emotion"), ("📊", "Confidence"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{icon} {label}</div>
                <div class="value">—</div>
            </div>
            """, unsafe_allow_html=True)
