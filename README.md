# 🧠 EmotionSense AI — Real-Time Facial Emotion Recognition

A production-ready real-time facial emotion recognition application powered by **YOLO** face detection and **DeepFace** emotion classification, with a premium **Streamlit** dashboard.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Face Detection** | YOLOv8n-face with Haar Cascade fallback |
| 🎭 **7 Emotions** | Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral |
| 👥 **Multi-Face** | Detect and classify emotions for multiple faces simultaneously |
| ⚡ **Real-Time** | Live webcam feed with FPS counter |
| 📊 **Analytics** | Emotion distribution bars, per-face results, confidence scores |
| 🎨 **Premium UI** | Gradient design, glassmorphism cards, animated indicators |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MudassarGill/emotion-detection-yolo.git
cd emotion-detection-yolo
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
emotion-detection-yolo/
├── app.py                    # Streamlit main application
├── config.py                 # App-wide constants & settings
├── face_detector.py          # YOLO face detection wrapper
├── emotion_recognizer.py     # DeepFace emotion classification
├── utils.py                  # Drawing utilities & FPS counter
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🎮 How to Use

1. **Start the app** → Run `streamlit run app.py`
2. **Click ▶ Start** in the sidebar to activate your webcam
3. **View detections** → Faces are highlighted with bounding boxes, emotion labels, and confidence bars
4. **Check analytics** → FPS, face count, dominant emotion, and emotion distribution are shown in real-time
5. **Click ⏹ Stop** to pause the camera

---

## 🤖 Models Used

| Model | Purpose | Source |
|-------|---------|--------|
| **YOLOv8n-face** | Face detection | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| **DeepFace** | Emotion classification | [DeepFace](https://github.com/serengil/deepface) |

> **Note:** All models are pretrained and auto-downloaded on first run. No training or datasets required.

---

## ⚙️ Requirements

- Python 3.8+
- Webcam
- ~2GB disk space (for model weights on first download)

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not opening | Check webcam connection and permissions |
| Slow FPS | Close other camera apps; reduce resolution in `config.py` |
| Model download fails | Ensure internet connection on first run |
| Import errors | Run `pip install -r requirements.txt` again |

---

## 📜 License

This project is licensed under the MIT License.