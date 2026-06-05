# 🧠 EmotionSense AI — Real-Time Facial Emotion Recognition

A production-ready real-time facial emotion recognition application powered by **YOLO** face detection and **DeepFace** emotion classification, with a premium **Streamlit** dashboard.

---

## 📖 About This Project

**EmotionSense AI** is a real-time computer vision application that detects human faces through your webcam and classifies their emotional state. It uses two pretrained AI models working together:

1. **YOLOv8n-face** detects face locations in each video frame
2. **DeepFace** analyzes each detected face and classifies the emotion

No model training or datasets are needed — both models come pretrained and are automatically downloaded on first run. The application processes your webcam feed frame-by-frame, drawing bounding boxes around detected faces and displaying the recognized emotion with confidence scores.

### How It Works (Pipeline)

```
Webcam Frame → YOLO Face Detection → Crop Each Face → DeepFace Emotion Classification → Draw Results → Display in Streamlit
```

1. **Capture**: OpenCV captures frames from your webcam at high resolution
2. **Detect**: YOLO scans the frame and returns bounding box coordinates for every face
3. **Crop**: Each face region is extracted from the frame
4. **Classify**: DeepFace analyzes each face crop and returns emotion probabilities
5. **Annotate**: Bounding boxes, labels, confidence bars, and corner accents are drawn
6. **Display**: The annotated frame is rendered in the Streamlit dashboard with analytics

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Face-Only Detection** | Strictly optimized for human faces using YOLOv8n-face |
| 🖼️ **Image Analysis** | Upload static photos (JPG, PNG) for detailed emotion scanning |
| 🎭 **7 Emotions** | Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral |
| 👥 **Multi-Face Support** | Detect and classify emotions for multiple faces simultaneously |
| ⚡ **Real-Time FPS** | Live frames-per-second counter for webcam mode |
| 📊 **Analytics Dashboard** | Emotion distribution bars, per-face results, confidence scores |
| 🎨 **Premium UI** | Gradient design, glassmorphism cards, animated status indicators |
| 🛡️ **Error Handling** | Graceful fallbacks for model loading, camera access, and small faces |

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Core programming language |
| **Streamlit** | ≥1.30.0 | Web-based frontend dashboard |
| **OpenCV** | ≥4.8.0 | Webcam capture & image processing |
| **Ultralytics YOLO** | ≥8.0.0 | YOLOv8n-face for face detection |
| **DeepFace** | ≥0.0.89 | Pretrained emotion classification (FER2013) |
| **TF-Keras** | ≥2.15.0 | Backend for DeepFace models |
| **NumPy** | ≥1.24.0 | Array operations |
| **Pillow** | ≥10.0.0 | Image handling |

---

## 🚀 Setup & Installation

### Prerequisites

- **Python 3.8+** installed ([Download Python](https://www.python.org/downloads/))
- **Webcam** (for live mode)
- **Internet connection** (for first-run model download only)
- ~2GB free disk space (for model weights)

### Step 1: Clone the Repository

```bash
git clone https://github.com/MudassarGill/emotion-detection-yolo.git
cd emotion-detection-yolo
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Common Mistake**: Do NOT run `pip install requirements.txt` — you need the `-r` flag!

### Step 4: Run the Application

```bash
streamlit run app.py
```

The app will automatically open in your browser at **http://localhost:8501**.

---

## 🎮 How to Use

1. **Launch** → Run `streamlit run app.py` in your terminal
2. **Select Mode** → Choose between **🎥 Live Webcam** or **📂 Image Upload** in the sidebar
3. **Analyze**:
   - **Webcam**: Click **▶ Start** to begin real-time detection
   - **Image**: Drop a photo (JPG/PNG) into the uploader to scan it
4. **View Detections** → Faces are highlighted with:
   - Color-coded bounding boxes with corner accents
   - Emotion label + emoji above each face
   - Confidence percentage
5. **Check Analytics** → The dashboard shows:
   - ⚡ **FPS** (Webcam) or 📸 **Mode** (Image)
   - 👤 **Faces Detected** — Total face count
   - 🎭 **Dominant Emotion** — Overall impression
   - 📊 **Emotion Distribution** — Detailed score breakdown
   - 👥 **Per-Face Results** — Individual results for every person

---

## 📁 Project Structure

```
emotion-detection-yolo/
│
├── app.py                    # 🖥️  Streamlit main application
│                             #     - Sidebar controls (Start/Stop)
│                             #     - Live webcam feed display
│                             #     - Real-time analytics dashboard
│                             #     - Premium CSS styling
│
├── config.py                 # ⚙️  App-wide constants & settings
│                             #     - YOLO model name & thresholds
│                             #     - Emotion color palettes (BGR + Hex)
│                             #     - Emoji mappings
│                             #     - Camera resolution settings
│
├── face_detector.py          # 🎯  Face detection module
│                             #     - YOLOv8n-face (primary)
│                             #     - YOLOv8n general (fallback)
│                             #     - Haar Cascade (final fallback)
│
├── emotion_recognizer.py     # 🎭  Emotion classification module
│                             #     - DeepFace.analyze() wrapper
│                             #     - Returns dominant emotion + all scores
│                             #     - Handles small/invalid faces gracefully
│
├── utils.py                  # 🔧  Utility functions
│                             #     - FPSCounter (rolling average)
│                             #     - draw_detections() — boxes, labels, bars
│                             #     - draw_fps() & draw_face_count()
│
├── requirements.txt          # 📦  Python dependencies
├── README.md                 # 📖  This file
├── LICENSE                   # 📜  MIT License
└── .gitignore                # 🚫  Git ignore rules
```

---

## 🤖 Models Used

### Face Detection — YOLOv8n-face
- **Architecture**: YOLOv8 Nano (face-specific variant)
- **Source**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Download**: Automatic on first run
- **Speed**: ~50+ FPS on modern hardware
- **Fallback**: If unavailable, falls back to YOLOv8n general → Haar Cascade

### Emotion Classification — DeepFace
- **Architecture**: VGG-Face based CNN (trained on FER2013 dataset)
- **Source**: [DeepFace](https://github.com/serengil/deepface)
- **Download**: Automatic on first run
- **Emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Input**: 48×48 grayscale face crops (auto-resized)

> **Note:** All models are **pretrained**. No training or datasets are required to run this application.

---

## 🎭 Supported Emotions

| Emotion | Emoji | Color |
|---------|-------|-------|
| Happy | 😊 | 🟢 Green |
| Sad | 😢 | 🔵 Blue |
| Angry | 😠 | 🔴 Red |
| Fear | 😨 | 🟠 Orange |
| Surprise | 😲 | 🟣 Magenta |
| Disgust | 🤢 | 🟠 Dark Orange |
| Neutral | 😐 | ⚪ Gray |

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `pip install requirements.txt` error | Use `pip install -r requirements.txt` (with `-r` flag) |
| Camera not opening | Check webcam connection; close other apps using the camera |
| Slow FPS / Laggy | Close other applications; try reducing resolution in `config.py` |
| Model download fails | Ensure you have an internet connection on first run |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| TensorFlow warnings | These are normal; the app still works correctly |
| Small faces not detected | Move closer to the camera or adjust `YOLO_CONFIDENCE_THRESHOLD` in `config.py` |

---

## ⚙️ Configuration

You can customize settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `YOLO_CONFIDENCE_THRESHOLD` | 0.45 | Minimum confidence for face detection |
| `FRAME_WIDTH` | 1280 | Webcam capture width |
| `FRAME_HEIGHT` | 720 | Webcam capture height |
| `CAMERA_INDEX` | 0 | Camera device index (0 = default) |
| `FPS_SMOOTHING_WINDOW` | 30 | Number of frames for FPS averaging |

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <b>Built with ❤️ using Streamlit, YOLO & DeepFace</b><br>
  <i>© 2026 EmotionSense AI</i>
</div>