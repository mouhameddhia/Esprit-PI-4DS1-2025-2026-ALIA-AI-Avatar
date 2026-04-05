from collections import deque
import traceback

import numpy as np
import requests
from deepface import DeepFace


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class EmotionAnalyzer:
    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.emotion_buffer = deque(maxlen=buffer_size)
        self.last_result = None

    def analyze(self, frame):
        """Analyze a single frame, update the temporal buffer, and return smoothed metrics."""
        probabilities = None

        try:
            # Prefer OpenCV detector; fall back if this DeepFace version
            # does not support the detector_backend argument.
            try:
             analysis = DeepFace.analyze(
             frame,
             actions=["emotion"],
             enforce_detection=False,
             detector_backend="opencv",
             )
            except TypeError:
             analysis = DeepFace.analyze(
             frame,
             actions=["emotion"],
             enforce_detection=False,
             )

            # DeepFace may return a list or a dict
            if isinstance(analysis, list):
                analysis = analysis[0]

            emotions = analysis.get("emotion", {}) or {}
            dominant = analysis.get("dominant_emotion")

            # Extract full distribution
            probabilities = {e: float(emotions.get(e, 0.0)) for e in EMOTIONS}
            total = sum(probabilities.values())

            # If distribution is empty or all zeros but DeepFace provided a
            # dominant_emotion label, fall back to a one-hot distribution.
            if total == 0.0 and isinstance(dominant, str) and dominant in EMOTIONS:
                probabilities = {e: (1.0 if e == dominant else 0.0) for e in EMOTIONS}
                total = 1.0

            # Normalize to 0–1 if we have a valid total
            if total > 0:
                probabilities = {k: v / total for k, v in probabilities.items()}
            else:
                probabilities = None

        except Exception as e:
            # In case DeepFace fails, skip updating with this frame.
            # Printed once per error type typically; if it is noisy we
            # can remove this later.
            import traceback
            print(f"DeepFace error ({type(e).__name__}): {e}")
            traceback.print_exc()
            probabilities = None

        if probabilities is not None:
            self.emotion_buffer.append(probabilities)

        metrics = self.compute_metrics()
        self.last_result = metrics
        return metrics

    def compute_metrics(self):
        """Compute stress, calmness, dominant emotion, and emotional stability from the temporal buffer."""
        if not self.emotion_buffer:
            return {
                "stress_score": 0.0,
                "calmness_score": 0.0,
                "dominant_emotion": "neutral",
                "emotional_stability": 0.0,
            }

        # Shape: (T, E) where T = time steps, E = number of emotions
        arr = np.array([[d[e] for e in EMOTIONS] for d in self.emotion_buffer])

        mean_probs = arr.mean(axis=0)
        var_over_time = arr.var(axis=0)

        # Stress: fear + angry + sad
        stress_score = float(
            np.clip(
                mean_probs[EMOTIONS.index("fear")]
                + mean_probs[EMOTIONS.index("angry")]
                + mean_probs[EMOTIONS.index("sad")],
                0.0,
                1.0,
            )
        )

        # Calmness: happy + neutral
        calmness_score = float(
            np.clip(
                mean_probs[EMOTIONS.index("happy")]
                + mean_probs[EMOTIONS.index("neutral")],
                0.0,
                1.0,
            )
        )

        # Dominant emotion from averaged probabilities
        dominant_emotion = EMOTIONS[int(np.argmax(mean_probs))]

        # Emotional stability: inverse of average variance over time.
        # Variance of probabilities is in [0, 0.25]; map to [1, 0].
        mean_variance = float(var_over_time.mean())
        emotional_stability = float(max(0.0, 1.0 - 4.0 * mean_variance))

        return {
            "stress_score": stress_score,
            "calmness_score": calmness_score,
            "dominant_emotion": dominant_emotion,
            "emotional_stability": emotional_stability,
        }


def send_metrics(
    metrics: dict,
    url: str = "http://localhost:8000/analyze",
    timeout: float = 0.5,
):
    """Optional helper to send metrics via HTTP POST."""
    try:
        requests.post(url, json=metrics, timeout=timeout)
    except Exception:
        # Fail silently to avoid interrupting the main loop.
        pass
