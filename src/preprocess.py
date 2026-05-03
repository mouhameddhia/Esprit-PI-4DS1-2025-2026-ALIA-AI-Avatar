from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Stable FaceMesh anchor ids for rough geometric normalization.
NOSE_TIP_ID = 1
LEFT_EYE_OUTER_ID = 33
RIGHT_EYE_OUTER_ID = 263


@dataclass
class PreprocessConfig:
    center_on_nose: bool = True
    scale_by_iod: bool = True
    compute_velocity: bool = True
    smooth_window: int = 3
    normalize: str = "zscore"


@dataclass
class DataConfig:
    subject_col: str
    label_col: str
    frame_col: str
    feature_cols: List[str]
    sequence_length: int
    stride: int


def infer_feature_columns(df: pd.DataFrame, feature_cols: Sequence[str]) -> List[str]:
    if feature_cols:
        return list(feature_cols)
    excluded = {"subject_id", "label", "frame", "landmark_id", "x", "y", "z"}
    return [c for c in df.columns if c not in excluded]


def moving_average(features: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return features
    out = np.copy(features)
    half = window // 2
    for t in range(features.shape[0]):
        lo = max(0, t - half)
        hi = min(features.shape[0], t + half + 1)
        out[t] = features[lo:hi].mean(axis=0)
    return out


def center_and_scale_facemesh(frame_vec: np.ndarray) -> np.ndarray:
    """
    Normalize one frame represented as [x1,y1,z1,x2,y2,z2,...].
    If extra features are appended after FaceMesh (e.g., Pose landmarks),
    only FaceMesh coordinates are normalized and the rest are preserved.
    """
    face_dim = 468 * 3
    has_appended_features = frame_vec.shape[0] > face_dim

    if has_appended_features:
        face_vec = frame_vec[:face_dim]
        rest_vec = frame_vec[face_dim:]
    else:
        face_vec = frame_vec
        rest_vec = None

    pts = face_vec.reshape(-1, 3).copy()
    if pts.shape[0] <= max(NOSE_TIP_ID, LEFT_EYE_OUTER_ID, RIGHT_EYE_OUTER_ID):
        return frame_vec

    nose = pts[NOSE_TIP_ID]
    left_eye = pts[LEFT_EYE_OUTER_ID]
    right_eye = pts[RIGHT_EYE_OUTER_ID]

    pts = pts - nose
    iod = np.linalg.norm(left_eye - right_eye)
    if iod > 1e-8:
        pts = pts / iod

    face_out = pts.reshape(-1)
    if rest_vec is not None:
        return np.concatenate([face_out, rest_vec.astype(np.float32)], axis=0)
    return face_out


def maybe_add_velocity(sequence: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return sequence
    vel = np.diff(sequence, axis=0, prepend=sequence[:1])
    return np.concatenate([sequence, vel], axis=1)


def build_windows(
    frame_features: np.ndarray,
    label: int,
    sequence_length: int,
    stride: int,
) -> Tuple[List[np.ndarray], List[int]]:
    X, y = [], []
    if len(frame_features) < sequence_length:
        return X, y
    for start in range(0, len(frame_features) - sequence_length + 1, stride):
        end = start + sequence_length
        X.append(frame_features[start:end])
        y.append(label)
    return X, y


def build_sequences(
    df: pd.DataFrame,
    data_cfg: DataConfig,
    prep_cfg: PreprocessConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = infer_feature_columns(df, data_cfg.feature_cols)
    X_all: List[np.ndarray] = []
    y_all: List[int] = []
    groups: List[str] = []

    for subject_id, g in df.groupby(data_cfg.subject_col):
        g = g.sort_values(data_cfg.frame_col)
        label = int(g[data_cfg.label_col].iloc[0])
        frame_features = g[feature_cols].to_numpy(dtype=np.float32)

        if prep_cfg.center_on_nose and frame_features.shape[1] % 3 == 0:
            frame_features = np.stack(
                [center_and_scale_facemesh(f) for f in frame_features], axis=0
            )

        frame_features = moving_average(frame_features, prep_cfg.smooth_window)
        frame_features = maybe_add_velocity(frame_features, prep_cfg.compute_velocity)

        Xi, yi = build_windows(
            frame_features,
            label=label,
            sequence_length=data_cfg.sequence_length,
            stride=data_cfg.stride,
        )
        X_all.extend(Xi)
        y_all.extend(yi)
        groups.extend([str(subject_id)] * len(Xi))

    X = np.asarray(X_all, dtype=np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    groups_arr = np.asarray(groups)
    return X, y, groups_arr


def fit_feature_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(flat)
    return scaler


def apply_feature_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    flat = X.reshape(-1, X.shape[-1])
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(X.shape).astype(np.float32)
