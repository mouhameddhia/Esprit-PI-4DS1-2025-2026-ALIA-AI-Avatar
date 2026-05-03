from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _face_template(rng: np.random.Generator, n_landmarks: int = 468) -> np.ndarray:
    # Coarse oval-like face template in normalized coordinates.
    theta = np.linspace(0, 2 * np.pi, n_landmarks, endpoint=False)
    x = 0.5 + 0.18 * np.cos(theta) + rng.normal(0, 0.01, size=n_landmarks)
    y = 0.5 + 0.24 * np.sin(theta) + rng.normal(0, 0.01, size=n_landmarks)
    z = rng.normal(0, 0.015, size=n_landmarks)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _pose_template() -> np.ndarray:
    # Minimal human-like normalized skeleton (33 landmarks).
    pts = np.zeros((33, 3), dtype=np.float32)
    pts[:] = np.array([0.5, 0.55, 0.0], dtype=np.float32)

    # Core torso / head anchors (MediaPipe Pose ids).
    pts[0] = [0.5, 0.36, -0.02]   # nose
    pts[11] = [0.43, 0.50, 0.00]  # left shoulder
    pts[12] = [0.57, 0.50, 0.00]  # right shoulder
    pts[13] = [0.39, 0.60, 0.02]  # left elbow
    pts[14] = [0.61, 0.60, 0.02]  # right elbow
    pts[15] = [0.36, 0.70, 0.04]  # left wrist
    pts[16] = [0.64, 0.70, 0.04]  # right wrist
    pts[23] = [0.46, 0.68, 0.01]  # left hip
    pts[24] = [0.54, 0.68, 0.01]  # right hip
    pts[25] = [0.45, 0.86, 0.05]  # left knee
    pts[26] = [0.55, 0.86, 0.05]  # right knee
    pts[27] = [0.44, 1.00, 0.08]  # left ankle
    pts[28] = [0.56, 1.00, 0.08]  # right ankle
    return pts


def _ar1_noise(prev: np.ndarray, rng: np.random.Generator, sigma: float, alpha: float) -> np.ndarray:
    eps = rng.normal(0, sigma, size=prev.shape).astype(np.float32)
    return alpha * prev + (1.0 - alpha) * eps


def main() -> None:
    rng = np.random.default_rng(42)
    n_subjects = 28
    n_frames = 360
    n_facemesh = 468
    n_pose = 33

    # Approximate expressive regions (indices are valid FaceMesh ids).
    brow_ids = np.array([70, 63, 105, 66, 107, 336, 296, 334, 293, 300])
    eye_top_ids = np.array([159, 386])
    mouth_ids = np.array([13, 14, 61, 291, 78, 308])

    rows = []
    for s in range(n_subjects):
        label = int(s % 2)  # 0: no stress, 1: stress

        face_base = _face_template(rng, n_facemesh)
        pose_base = _pose_template().copy()
        pose_base += rng.normal(0, 0.01, size=pose_base.shape).astype(np.float32)

        face_noise_state = np.zeros((n_facemesh, 3), dtype=np.float32)
        pose_noise_state = np.zeros((n_pose, 3), dtype=np.float32)

        # Subject style variation (some people are naturally more expressive).
        subj_expressiveness = float(rng.uniform(0.8, 1.25))

        for f in range(n_frames):
            t = float(f)

            # Shared low-frequency head/body drift.
            drift_x = 0.007 * np.sin(t / 35.0)
            drift_y = 0.006 * np.cos(t / 28.0)

            face_pts = face_base.copy()
            pose_pts = pose_base.copy()

            face_pts[:, 0] += drift_x
            face_pts[:, 1] += drift_y
            pose_pts[:, 0] += 0.8 * drift_x
            pose_pts[:, 1] += 0.8 * drift_y

            # AR(1) noise gives natural temporal continuity.
            base_face_sigma = 0.004 if label == 0 else 0.010
            base_pose_sigma = 0.003 if label == 0 else 0.008
            face_noise_state = _ar1_noise(face_noise_state, rng, sigma=base_face_sigma, alpha=0.85)
            pose_noise_state = _ar1_noise(pose_noise_state, rng, sigma=base_pose_sigma, alpha=0.80)
            face_pts += subj_expressiveness * face_noise_state
            pose_pts += pose_noise_state

            if label == 1:
                # Stress signatures: eyebrow raise + mouth tension + shoulder lift + wrist jitter.
                tension = 0.020 * abs(np.sin(t / 7.0))
                face_pts[brow_ids, 1] -= tension
                face_pts[mouth_ids, 1] += 0.012 * abs(np.cos(t / 9.0))
                face_pts[eye_top_ids, 1] += 0.006 * np.sin(t / 5.5)

                # Shoulder elevation and mild hunching.
                pose_pts[[11, 12], 1] -= 0.018 * abs(np.sin(t / 8.0))
                pose_pts[[11, 12], 2] += 0.010 * abs(np.sin(t / 11.0))

                # Faster hand tremor.
                pose_pts[[15, 16], 0] += 0.010 * np.sin(t / 2.2)
                pose_pts[[15, 16], 1] += 0.008 * np.cos(t / 2.6)

                # Occasional micro-spike events.
                if rng.random() < 0.03:
                    burst = rng.normal(0, 0.018, size=(len(mouth_ids), 3)).astype(np.float32)
                    face_pts[mouth_ids] += burst
            else:
                # Calm signatures: slower, smaller movement.
                face_pts[brow_ids, 1] -= 0.004 * abs(np.sin(t / 18.0))
                pose_pts[[11, 12], 1] -= 0.006 * abs(np.sin(t / 20.0))

            # Keep coordinates in plausible normalized bounds.
            face_pts = np.clip(face_pts, -1.0, 2.0)
            pose_pts = np.clip(pose_pts, -1.0, 2.0)

            face_flat = face_pts.reshape(-1)
            pose_flat = pose_pts.reshape(-1)

            row = {
                "subject_id": f"S{s:03d}",
                "frame": f,
                "label": label,
            }

            for i, val in enumerate(face_flat):
                row[f"feat_{i}"] = float(val)

            for i, val in enumerate(pose_flat):
                row[f"feat_{n_facemesh * 3 + i}"] = float(val)

            rows.append(row)

    df = pd.DataFrame(rows)
    out = Path("data/facemesh_sequences.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(f"  Shape: {df.shape}")
    print(
        f"  Features: {df.shape[1] - 3} "
        "(468 FaceMesh + 33 Pose = 501 landmarks * 3 = 1503 features)"
    )
    print(f"  Labels: {dict(df['label'].value_counts())}")
    print(f"  Subjects: {df['subject_id'].nunique()}")


if __name__ == "__main__":
    main()
