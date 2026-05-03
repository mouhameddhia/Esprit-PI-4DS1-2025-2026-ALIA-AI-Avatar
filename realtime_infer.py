from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import yaml

from src.model import StressLSTM
from src.preprocess import center_and_scale_facemesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime stress inference with OpenCV + MediaPipe FaceMesh")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--sequence-length", type=int, default=None)
    p.add_argument("--predict-every", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--calibration-seconds", type=float, default=8.0)
    p.add_argument("--smooth-alpha", type=float, default=0.25)
    p.add_argument("--json-out", type=str, default="artifacts/realtime_session.json")
    p.add_argument("--session-id", type=str, default=None)
    p.add_argument("--show-mesh", action="store_true")
    return p.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_scaler(artifact_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    mean_path = artifact_dir / "scaler_mean.npy"
    scale_path = artifact_dir / "scaler_scale.npy"
    if mean_path.exists() and scale_path.exists():
        return np.load(mean_path), np.load(scale_path)
    return None, None


def apply_scaler(frame_feat: np.ndarray, mean: np.ndarray | None, scale: np.ndarray | None) -> np.ndarray:
    if mean is None or scale is None:
        return frame_feat
    if frame_feat.shape[0] != mean.shape[0] or frame_feat.shape[0] != scale.shape[0]:
        raise ValueError(
            "Scaler dimension mismatch. Checkpoint/scaler artifacts were not generated from "
            "the same feature pipeline used for realtime FaceMesh inference."
        )
    return (frame_feat - mean) / np.maximum(scale, 1e-8)


def infer_checkpoint_input_dim(state_dict: dict[str, torch.Tensor]) -> int:
    key = "lstm.weight_ih_l0"
    if key not in state_dict:
        raise KeyError(f"Missing key '{key}' in checkpoint state_dict.")
    return int(state_dict[key].shape[1])


def build_session_report(
    session_id: str,
    config_path: str,
    checkpoint_path: str,
    use_pose: bool,
    add_velocity: bool,
    sequence_length: int,
    predict_every: int,
    threshold: float,
    smooth_alpha: float,
    calibration_seconds: float,
    baseline_prob: float | None,
    total_frames: int,
    face_detected_frames: int,
    pose_detected_frames: int,
    feature_valid_frames: int,
    num_predictions: int,
    sum_prob: float,
    sum_prob_sq: float,
    min_prob: float,
    max_prob: float,
    above_threshold_count: int,
    label_switches: int,
    longest_stress_streak: int,
) -> dict:
    if num_predictions == 0:
        stress_mean = 0.0
        stress_std = 0.0
        stress_min = 0.0
        stress_max = 0.0
        stress_ratio = 0.0
    else:
        stress_mean = float(sum_prob / num_predictions)
        variance = max((sum_prob_sq / num_predictions) - (stress_mean * stress_mean), 0.0)
        stress_std = float(np.sqrt(variance))
        stress_min = float(min_prob)
        stress_max = float(max_prob)
        stress_ratio = float(above_threshold_count / num_predictions)

    detector = {
        "face_detect_rate": float(face_detected_frames / max(total_frames, 1)),
        "pose_detect_rate": float(pose_detected_frames / max(total_frames, 1)),
        "valid_feature_rate": float(feature_valid_frames / max(total_frames, 1)),
        "total_frames": int(total_frames),
        "face_detected_frames": int(face_detected_frames),
        "pose_detected_frames": int(pose_detected_frames),
        "feature_valid_frames": int(feature_valid_frames),
    }

    return {
        "schema_version": "1.0",
        "session": {
            "session_id": session_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
        },
        "runtime": {
            "mode": "FaceMesh+Pose" if use_pose else "FaceMesh",
            "use_pose": bool(use_pose),
            "use_velocity": bool(add_velocity),
            "sequence_length": int(sequence_length),
            "predict_every": int(predict_every),
            "threshold": float(threshold),
            "smooth_alpha": float(smooth_alpha),
            "calibration_seconds": float(calibration_seconds),
            "baseline_prob": None if baseline_prob is None else float(baseline_prob),
        },
        "detector": detector,
        "stress_metrics": {
            "num_predictions": int(num_predictions),
            "mean_stress_prob": stress_mean,
            "std_stress_prob": stress_std,
            "min_stress_prob": stress_min,
            "max_stress_prob": stress_max,
            "stress_ratio_above_threshold": stress_ratio,
            "label_switches": int(label_switches),
            "longest_stress_streak_predictions": int(longest_stress_streak),
        },
    }


def extract_frame_features(
    frame_bgr: np.ndarray,
    face_mesh: mp.solutions.face_mesh.FaceMesh,
    pose: mp.solutions.pose.Pose,
    expected_face_landmarks: int,
    use_pose: bool,
    add_velocity: bool,
    prev_base: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, object | None, object | None]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Extract FaceMesh landmarks
    face_result = face_mesh.process(rgb)
    if not face_result.multi_face_landmarks:
        return None, prev_base, None, None
    
    face_lm_list = face_result.multi_face_landmarks[0]
    if len(face_lm_list.landmark) < expected_face_landmarks:
        return None, prev_base, None, None
    
    # Extract Pose landmarks
    pose_result = pose.process(rgb)
    pose_lm_list = None
    pose_xyz = np.zeros(33 * 3, dtype=np.float32)  # 33 pose landmarks * 3 (xyz)

    if use_pose and pose_result.pose_landmarks:
        pose_lm_list = pose_result.pose_landmarks
        pose_xyz = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks.landmark[:33]],
            dtype=np.float32,
        ).reshape(-1)
    
    # Get FaceMesh xyz
    facemesh_xyz = np.array(
        [[lm.x, lm.y, lm.z] for lm in face_lm_list.landmark[:expected_face_landmarks]],
        dtype=np.float32,
    ).reshape(-1)
    
    # Center and scale FaceMesh
    facemesh_base = center_and_scale_facemesh(facemesh_xyz).astype(np.float32)
    
    # Use either FaceMesh-only or FaceMesh+Pose features based on checkpoint.
    if use_pose:
        combined_base = np.concatenate([facemesh_base, pose_xyz], axis=0).astype(np.float32)
    else:
        combined_base = facemesh_base
    
    if add_velocity:
        if prev_base is None:
            vel = np.zeros_like(combined_base)
        else:
            vel = combined_base - prev_base
        feat = np.concatenate([combined_base, vel], axis=0)
    else:
        feat = combined_base
    
    return feat, combined_base, face_lm_list, pose_lm_list


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    seq_len = args.sequence_length or int(cfg["data"]["sequence_length"])
    
    # FaceMesh: 468 landmarks * 3 (xyz)
    facemesh_dim = 468 * 3
    # Pose: 33 landmarks * 3 (xyz)
    pose_dim = 33 * 3
    # Combined dimension
    base_dim = facemesh_dim + pose_dim

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = select_device()
    state_dict = torch.load(ckpt, map_location=device)
    ckpt_input_dim = infer_checkpoint_input_dim(state_dict)

    if ckpt_input_dim == facemesh_dim:
        use_pose = False
        add_velocity = False
    elif ckpt_input_dim == 2 * facemesh_dim:
        use_pose = False
        add_velocity = True
    elif ckpt_input_dim == base_dim:
        use_pose = True
        add_velocity = False
    elif ckpt_input_dim == 2 * base_dim:
        use_pose = True
        add_velocity = True
    else:
        raise ValueError(
            "Checkpoint input_dim is incompatible with realtime features. "
            f"Checkpoint expects {ckpt_input_dim}, while realtime supports: "
            f"{facemesh_dim} (FaceMesh xyz), {2 * facemesh_dim} (FaceMesh xyz+velocity), "
            f"{base_dim} (FaceMesh+Pose xyz), or {2 * base_dim} (FaceMesh+Pose xyz+velocity)."
        )

    model_cfg = dict(cfg["model"])
    model_cfg["input_dim"] = ckpt_input_dim
    model = StressLSTM(**model_cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    expected_face_landmarks = 468

    artifact_dir = ckpt.parent
    scaler_mean, scaler_scale = load_scaler(artifact_dir)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    face_draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)
    pose_draw_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))

    buf: deque[np.ndarray] = deque(maxlen=seq_len)
    prev_base: np.ndarray | None = None
    pred_label = "N/A"
    pred_prob = 0.0
    raw_prob = 0.0
    frame_idx = 0
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1.0 or np.isnan(fps):
        fps = 30.0
    calibration_frames = max(1, int(args.calibration_seconds * fps / max(args.predict_every, 1)))
    calibration_probs: list[float] = []
    baseline_prob: float | None = None
    session_id = args.session_id or datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%SZ")

    total_frames = 0
    face_detected_frames = 0
    pose_detected_frames = 0
    feature_valid_frames = 0
    num_predictions = 0
    sum_prob = 0.0
    sum_prob_sq = 0.0
    min_prob = 1.0
    max_prob = 0.0
    above_threshold_count = 0
    label_switches = 0
    last_pred_class: int | None = None
    current_stress_streak = 0
    longest_stress_streak = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as face_mesh, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total_frames += 1

            feat, prev_base, face_lm_list, pose_lm_list = extract_frame_features(
                frame,
                face_mesh,
                pose,
                expected_face_landmarks=expected_face_landmarks,
                use_pose=use_pose,
                add_velocity=add_velocity,
                prev_base=prev_base,
            )
            if face_lm_list is not None:
                face_detected_frames += 1
            if pose_lm_list is not None:
                pose_detected_frames += 1

            if feat is not None:
                feature_valid_frames += 1
                try:
                    feat = apply_scaler(feat, scaler_mean, scaler_scale)
                except ValueError:
                    # Skip incompatible frames/artifacts instead of killing realtime session.
                    continue
                buf.append(feat)

                if len(buf) == seq_len and (frame_idx % max(args.predict_every, 1) == 0):
                    seq = np.stack(buf, axis=0).astype(np.float32)
                    x = torch.from_numpy(seq).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                    raw_prob = float(probs[1])

                    # Calibrate to user's neutral baseline during first seconds.
                    if baseline_prob is None:
                        calibration_probs.append(raw_prob)
                        if len(calibration_probs) >= calibration_frames:
                            baseline_prob = float(np.mean(calibration_probs))

                    adjusted_prob = raw_prob
                    if baseline_prob is not None:
                        adjusted_prob = float(np.clip(raw_prob - baseline_prob + 0.5, 0.0, 1.0))

                    alpha = float(np.clip(args.smooth_alpha, 0.01, 1.0))
                    pred_prob = alpha * adjusted_prob + (1.0 - alpha) * pred_prob
                    pred_label = "STRESS" if pred_prob >= args.threshold else "NO_STRESS"

                    pred_class = int(pred_prob >= args.threshold)
                    num_predictions += 1
                    sum_prob += float(pred_prob)
                    sum_prob_sq += float(pred_prob * pred_prob)
                    min_prob = float(min(min_prob, pred_prob))
                    max_prob = float(max(max_prob, pred_prob))
                    above_threshold_count += int(pred_class)

                    if last_pred_class is not None and pred_class != last_pred_class:
                        label_switches += 1
                    last_pred_class = pred_class

                    if pred_class == 1:
                        current_stress_streak += 1
                        longest_stress_streak = max(longest_stress_streak, current_stress_streak)
                    else:
                        current_stress_streak = 0

            # Draw FaceMesh
            if args.show_mesh and face_lm_list is not None:
                mp_draw.draw_landmarks(
                    frame,
                    face_lm_list,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=face_draw_spec,
                    connection_drawing_spec=face_draw_spec,
                )
            
            # Draw Pose
            if args.show_mesh and pose_lm_list is not None:
                mp_draw.draw_landmarks(
                    frame,
                    pose_lm_list,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=pose_draw_spec,
                    connection_drawing_spec=pose_draw_spec,
                )

            text = f"{pred_label} p(stress)={pred_prob:.2f}"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            mode = "FaceMesh+Pose" if use_pose else "FaceMesh"
            cv2.putText(frame, f"mode={mode} buf={len(buf)}/{seq_len}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 180), 2)
            if baseline_prob is None:
                cv2.putText(
                    frame,
                    f"calibrating neutral... {len(calibration_probs)}/{calibration_frames}",
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (100, 220, 255),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    f"raw={raw_prob:.2f} neutral_base={baseline_prob:.2f}",
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (170, 230, 170),
                    2,
                )
            cv2.putText(frame, "Press q to quit", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow("Realtime Stress Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    report = build_session_report(
        session_id=session_id,
        config_path=args.config,
        checkpoint_path=str(ckpt),
        use_pose=use_pose,
        add_velocity=add_velocity,
        sequence_length=seq_len,
        predict_every=max(args.predict_every, 1),
        threshold=float(args.threshold),
        smooth_alpha=float(np.clip(args.smooth_alpha, 0.01, 1.0)),
        calibration_seconds=float(args.calibration_seconds),
        baseline_prob=baseline_prob,
        total_frames=total_frames,
        face_detected_frames=face_detected_frames,
        pose_detected_frames=pose_detected_frames,
        feature_valid_frames=feature_valid_frames,
        num_predictions=num_predictions,
        sum_prob=sum_prob,
        sum_prob_sq=sum_prob_sq,
        min_prob=min_prob,
        max_prob=max_prob,
        above_threshold_count=above_threshold_count,
        label_switches=label_switches,
        longest_stress_streak=longest_stress_streak,
    )

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved realtime JSON report to {json_out}")


if __name__ == "__main__":
    main()
