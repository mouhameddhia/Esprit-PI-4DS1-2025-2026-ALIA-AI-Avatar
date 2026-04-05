import cv2
import json

from emotion_analyzer import EmotionAnalyzer, send_metrics


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    analyzer = EmotionAnalyzer(buffer_size=30)
    frame_count = 0
    deepface_interval = 5  # Run DeepFace every N frames
    send_api = False       # Set to True to enable HTTP POST

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize for performance
            frame = cv2.resize(frame, (640, 480))

            frame_count += 1

            if frame_count % deepface_interval == 0:
                metrics = analyzer.analyze(frame)
            else:
                # Reuse last metrics / smoothed buffer
                if analyzer.last_result is not None:
                    metrics = analyzer.last_result
                else:
                    metrics = analyzer.compute_metrics()

            # Print JSON metrics to console continuously
            print(json.dumps(metrics), flush=True)

            # Optional: send to API
            if send_api:
                send_metrics(metrics)

            # Overlay metrics on video
            stress_text = f"Stress: {metrics['stress_score']:.2f}"
            emotion_text = f"Emotion: {metrics['dominant_emotion']}"

            cv2.putText(
                frame,
                stress_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                emotion_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Real-Time Facial Emotion Analysis", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
