import base64
import io

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from emotion_analyzer import EmotionAnalyzer


app = Flask(__name__)

# Single analyzer instance for the session
analyzer = EmotionAnalyzer(buffer_size=300)


def _b64_to_bgr_image(b64_data: str):
    """Decode a base64 PNG/JPEG string (data URL) to an OpenCV BGR image."""
    # Strip header like "data:image/png;base64,"
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/frame", methods=["POST"])
def api_frame():
    """Receive one frame from the browser, run emotion analysis, return metrics."""
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get("image")
    if not frame_b64:
        return jsonify({"error": "missing image"}), 400

    frame = _b64_to_bgr_image(frame_b64)
    if frame is None:
        return jsonify({"error": "invalid image"}), 400

    metrics = analyzer.analyze(frame)
    return jsonify(metrics)


@app.route("/api/summary", methods=["GET"])
def api_summary():
    """Return the current smoothed metrics as a simple session summary."""
    metrics = analyzer.compute_metrics()
    return jsonify(metrics)


if __name__ == "__main__":
    # Run in debug mode for development
    app.run(host="127.0.0.1", port=5000, debug=True)
