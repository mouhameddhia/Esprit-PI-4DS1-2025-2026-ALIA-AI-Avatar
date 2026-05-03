"""
Improved synthetic data generator for stress detection with MediaPipe FaceMesh + Pose.

Features realistic stress patterns:
- Facial: jaw tension, eye widening/narrowing, mouth tightening
- Body: shoulder elevation, arm tension, posture changes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# FaceMesh key landmark indices for stress patterns
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
JAW_BOTTOM = 175
NOSE_TIP = 1
LEFT_EYE = 33
RIGHT_EYE = 263
LEFT_SHOULDER = 11  # In MediaPipe Pose
RIGHT_SHOULDER = 12  # Adjust if using separate pose array


def generate_stress_pattern(rng: np.random.RandomState, n_frames: int, stress_level: float) -> dict:
    """
    Generate realistic stress pattern with temporal dynamics.
    stress_level: 0.0 (relaxed) to 1.0 (highly stressed)
    Returns dict of modulation factors for different facial/body regions.
    """
    # Base patterns with sinusoidal variation
    t = np.linspace(0, 4 * np.pi, n_frames)
    
    # Jaw tension modulation - increases with stress
    jaw_tension = 0.3 + 0.4 * stress_level + 0.2 * np.sin(t / 4) * stress_level
    
    # Eye opening - decreases slightly with stress
    eye_opening = 1.0 - 0.2 * stress_level + 0.1 * np.sin(t / 5) * stress_level
    
    # Mouth tightness - increases with stress
    mouth_tight = 0.3 + 0.5 * stress_level + 0.15 * np.sin(t / 3) * stress_level
    
    # Shoulder elevation - increases with stress
    shoulder_elev = 0.2 * stress_level + 0.1 * np.sin(t / 6) * stress_level
    
    # Head tilt variation - more rigid with stress
    head_tilt = 0.15 * (1 - stress_level) * np.sin(t / 7)
    
    # Micro-tremors (more pronounced under stress) - array for all frames
    tremor_scale = 0.02 + 0.03 * stress_level + 0.01 * np.sin(t / 8) * stress_level
    
    return {
        "jaw_tension": jaw_tension,
        "eye_opening": eye_opening,
        "mouth_tight": mouth_tight,
        "shoulder_elev": shoulder_elev,
        "head_tilt": head_tilt,
        "tremor_scale": tremor_scale,
    }


def apply_facemesh_stress(
    baseline_pts: np.ndarray,
    stress_factors: dict,
    frame_idx: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Apply stress-induced deformations to FaceMesh landmarks."""
    pts = baseline_pts.copy()
    n_landmarks = pts.shape[0]
    
    # Jaw clenching (move jaw points inward and up)
    if JAW_BOTTOM < n_landmarks:
        jaw_factor = stress_factors["jaw_tension"][frame_idx]
        pts[JAW_BOTTOM] -= np.array([0, jaw_factor * 0.02, 0])
    
    # Eye narrowing (move eyelids closer)
    if LEFT_EYE < n_landmarks and RIGHT_EYE < n_landmarks:
        eye_factor = stress_factors["eye_opening"][frame_idx]
        # Simulate eye narrowing by moving upper/lower lids
        for i in range(n_landmarks):
            # Heuristic: landmarks 159-181 are right eye, 386-408 are left eye
            if (159 <= i <= 181) or (386 <= i <= 408):
                pts[i, 1] *= eye_factor
    
    # Mouth tightness (flatten mouth, compress horizontally)
    if MOUTH_LEFT < n_landmarks and MOUTH_RIGHT < n_landmarks:
        mouth_factor = stress_factors["mouth_tight"][frame_idx]
        mouth_pts = [61, 78, 95, 185, 291, 308, 324, 415]  # Mouth boundary landmarks
        for idx in mouth_pts:
            if idx < n_landmarks:
                pts[idx, 1] -= mouth_factor * 0.01  # Slight upward compression
    
    # Head micro-tremors
    tremor = stress_factors["tremor_scale"][frame_idx] * rng.normal(0, 1, 3)
    pts += tremor
    
    return pts


def apply_pose_stress(
    baseline_pts: np.ndarray,
    stress_factors: dict,
    frame_idx: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Apply stress-induced deformations to Pose landmarks."""
    pts = baseline_pts.copy()
    n_landmarks = pts.shape[0]
    
    # Shoulder elevation (20,21 are shoulders in full Pose, or 11,12 in reduced)
    shoulder_elev = stress_factors["shoulder_elev"][frame_idx]
    if n_landmarks >= 33:
        # MediaPipe Pose: 11=left shoulder, 12=right shoulder
        if 11 < n_landmarks:
            pts[11, 1] -= shoulder_elev * 0.05
        if 12 < n_landmarks:
            pts[12, 1] -= shoulder_elev * 0.05
    
    # Posture stiffness (reduced movement variation)
    if stress_factors["shoulder_elev"][frame_idx] > 0.15:
        # Under stress, reduce natural sway
        for i in range(n_landmarks):
            if i not in [11, 12]:  # Skip shoulders
                pts[i] *= 0.95  # Reduce amplitude slightly
    
    # Tremor on full body
    tremor = stress_factors["tremor_scale"][frame_idx] * rng.normal(0, 1, 3)
    pts += tremor * 0.5
    
    return pts


def generate_subject_data(
    subject_id: str,
    stress_label: int,
    n_frames: int,
    n_facemesh: int,
    n_pose: int,
    rng: np.random.RandomState,
) -> list[dict]:
    """Generate synthetic data for one subject with stress patterns."""
    
    stress_level = float(stress_label)
    stress_factors = generate_stress_pattern(rng, n_frames, stress_level)
    
    rows = []
    
    for frame_idx in range(n_frames):
        # Baseline FaceMesh landmarks
        baseline_facemesh = rng.normal(loc=0.5, scale=0.15, size=(n_facemesh, 3)).astype(np.float32)
        facemesh_pts = apply_facemesh_stress(baseline_facemesh, stress_factors, frame_idx, rng)
        
        # Baseline Pose landmarks (33 landmarks with x,y,z)
        baseline_pose = rng.normal(loc=0.5, scale=0.15, size=(n_pose, 3)).astype(np.float32)
        pose_pts = apply_pose_stress(baseline_pose, stress_factors, frame_idx, rng)
        
        # Combine features
        combined_features = np.concatenate([
            facemesh_pts.reshape(-1),
            pose_pts.reshape(-1),
        ], axis=0).astype(np.float32)
        
        row = {
            "subject_id": subject_id,
            "frame": frame_idx,
            "label": stress_label,
        }
        
        # Add all combined features
        for i, val in enumerate(combined_features):
            row[f"feat_{i}"] = float(val)
        
        rows.append(row)
    
    return rows


def main() -> None:
    rng = np.random.RandomState(42)
    
    # Configuration
    n_subjects = 20  # More subjects for better coverage
    n_frames = 300  # Longer sequences for temporal patterns
    n_facemesh = 468
    n_pose = 33
    
    print(f"Generating synthetic data:")
    print(f"  Subjects: {n_subjects}")
    print(f"  Frames per subject: {n_frames}")
    print(f"  FaceMesh landmarks: {n_facemesh}")
    print(f"  Pose landmarks: {n_pose}")
    print(f"  Total features: {(n_facemesh + n_pose) * 3}")
    
    all_rows = []
    
    # Generate balanced stress/non-stress data
    for s in range(n_subjects):
        stress_label = int(s % 2)  # Alternate stress/non-stress
        print(f"  Subject S{s:03d}: stress={stress_label}")
        
        rows = generate_subject_data(
            subject_id=f"S{s:03d}",
            stress_label=stress_label,
            n_frames=n_frames,
            n_facemesh=n_facemesh,
            n_pose=n_pose,
            rng=rng,
        )
        all_rows.extend(rows)
    
    df = pd.DataFrame(all_rows)
    out = Path("data/facemesh_sequences.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    
    print(f"\nWrote {len(df)} frames to {out}")
    print(f"Data shape: {df.shape}")
    print(f"Samples per class:")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
