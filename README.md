# CNN Module — Stress & Calmness Detection

Part of the **ALIA AI Avatar** system · VITAL Lab

---

## Overview

This module performs real-time stress and calmness level detection using a live webcam feed. It leverages **DeepFace** to extract facial analysis metrics and **OpenCV** to capture and process video frames. The resulting stress/calmness score is passed to the ALIA agent pipeline as a contextual signal to adapt the avatar's responses accordingly.

---

## How It Works

1. OpenCV captures a continuous video stream from the webcam
2. Each frame is analyzed by DeepFace, which extracts facial metrics (emotion probabilities, dominant emotion, etc.)
3. The module maps those metrics to a **stress/calmness score**
4. The score is returned as structured output for use by other ALIA agents

---

## Metrics Used

DeepFace provides per-frame emotion probabilities. The module derives stress and calmness levels from the following emotional signals:

| Signal | Contribution |
|---|---|
| `angry`, `fear`, `disgust` | Contribute to stress score |
| `happy`, `neutral` | Contribute to calmness score |
| `sad`, `surprise` | Weighted contextually |

The final output is a normalized score indicating the user's current affective state.

---

## Technologies

| Tool | Role |
|---|---|
| OpenCV | Webcam capture and frame preprocessing |
| DeepFace | Facial analysis and emotion metric extraction |

---

## Installation

Make sure the following dependencies are installed in your virtual environment:

```bash
pip install deepface opencv-python
```

---

## Usage

Start the Flask development server:

```bash
python app.py
```

The module will be available with the full project of ALIA as well 

The webcam stream is processed server-side and the stress/calmness scores are served through the Flask API endpoints.

---

## Output Format

The module returns a JSON object per frame:

```json
{
  "stress_level": 0.72,
  "calmness_level": 0.28,
  "dominant_emotion": "angry",
  "raw_emotions": {
    "angry": 0.61,
    "fear": 0.11,
    "neutral": 0.20,
    "happy": 0.08
  }
}
```

---

## Notes

> This module requires a working webcam. Performance may vary depending on lighting conditions and camera quality. DeepFace runs inference on each captured frame, so a GPU is recommended for smoother real-time analysis.
