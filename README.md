# LSTM Stress Detection From MediaPipe FaceMesh + Pose Sequences

This project proposes and implements an end-to-end LSTM baseline for stress detection from temporal FaceMesh + body Pose landmark features.

It also includes realtime webcam inference using OpenCV + MediaPipe FaceMesh/Pose in `realtime_infer.py`.

## 1) Proposed Architecture

Input sequence shape:
- `T x F` where:
- `T`: number of frames in a window (for example 90)
- `F`: frame feature dimension (for example `1503` for xyz landmarks: `468 FaceMesh * 3 + 33 Pose * 3`, optionally doubled when velocity features are added)

Model:
1. BiLSTM encoder:
- hidden size: 192
- layers: 2
- dropout: 0.3 between recurrent layers
2. Temporal summary:
- use last recurrent output vector (sequence-level representation)
3. MLP classifier head:
- LayerNorm -> Linear(384, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 2)

Loss and optimization:
- CrossEntropyLoss (optionally class-weighted)
- AdamW optimizer
- ReduceLROnPlateau scheduler on validation F1
- Early stopping on validation F1

## 1.5) Body Language & Gesture Detection

The model incorporates both facial and body language cues:

**FaceMesh Facial Features** (468 landmarks):
- Jaw position and tension
- Eye openness and vergence
- Mouth shape and tightness
- Eyebrow position

**Pose Body Language** (33 landmarks):
- Shoulder elevation and symmetry
- Arm/elbow tension and position
- Spine curvature and posture
- Head orientation and micro-movements

**Stress Indicators Detected:**
- Elevated shoulders (tension)
- Jaw clenching (tight mouth, reduced opening)
- Eye narrowing (reduced openness)
- Reduced body sway (postural stiffness)
- Increased micro-tremors (fine movement irregularities)

These temporal patterns are captured by the BiLSTM, which learns to identify sequences indicative of stress states.

## 2) Preprocessing Pipeline

Implemented in `src/preprocess.py`:

1. Frame ordering:
- Sort by frame index per subject/recording.

2. Geometric normalization (if features are full xyz landmarks):
- Center each frame on nose tip (`landmark 1`).
- Scale by inter-ocular distance (`landmarks 33 and 263`).

3. Temporal smoothing:
- Moving average over frames (`smooth_window`, default 3).

4. Motion dynamics:
- Add per-frame velocity features using first-order temporal difference.

5. Sequence generation:
- Sliding windows with `sequence_length` and `stride`.

6. Train-only normalization:
- Z-score normalization fit only on train split, then applied to val/test.

7. Leakage-aware split:
- Optional group split by subject ID to avoid identity leakage.

## 2.5) Improved Synthetic Data Generation

For development and testing, use the improved synthetic data generator that creates realistic stress patterns:

```bash
.\.venv310\Scripts\python.exe tools/make_synthetic_data_v2.py
```

This generator:
- Combines FaceMesh (468 landmarks) + Pose (33 landmarks) features
- Synthesizes realistic stress patterns:
  - **Facial**: jaw clenching, eye narrowing, mouth tightness
  - **Body**: shoulder elevation, posture stiffness, tremors
- Creates temporal dynamics showing stress build-up and relaxation
- Generates balanced stress/non-stress data with natural dynamics
- Produces 6000+ frames of realistic training data

Output: `data/facemesh_sequences.csv` with 1503 features per frame (or 3006 with velocity)

## 3) Data Format

CSV columns expected:
- `subject_id`
- `frame`
- `label` (0/1)
- feature columns (all remaining numeric columns, or explicitly set via `feature_cols` in config)

**Feature Structure:**
- Combined FaceMesh + Pose landmarks: `(468 + 33) * 3 = 1503` features per frame
  - Features 0-1403: FaceMesh xyz coordinates (468 landmarks × 3)
  - Features 1404-1502: Pose xyz coordinates (33 landmarks × 3)
- With velocity: doubled to 3006 features
- Typical feature naming: `feat_0, feat_1, ..., feat_1502`

**Example CSV Row:**
```
subject_id,frame,label,feat_0,feat_1,feat_2,...,feat_1502
S001,0,0,0.123,0.456,0.789,...,0.234
```

## 4) Run

Activate Python 3.10 virtual environment first:

PowerShell:

```powershell
.\.venv310\Scripts\Activate.ps1
```

Command Prompt:

```bat
.venv310\Scripts\activate.bat
```

If you prefer not to activate, run commands with the explicit interpreter path:

```powershell
.\.venv310\Scripts\python.exe <command>
```

Install:

```bash
pip install -r requirements.txt
```

Train:

```bash
python train.py --config config.yaml
```

Outputs saved to `artifacts/`:
- `best_model.pt`
- `metrics.json`
- `scaler_mean.npy` and `scaler_scale.npy` (if z-score enabled)

## 4.1) Realtime Webcam Inference (OpenCV + MediaPipe FaceMesh + Pose)

Run realtime stress prediction from webcam:

```bash
.\.venv310\Scripts\python.exe realtime_infer.py --config config.yaml --checkpoint artifacts/best_model.pt --show-mesh
```

Optional flags:
- `--camera 0` camera index
- `--sequence-length 90` override sequence length at inference
- `--predict-every 3` run model every N frames
- `--threshold 0.5` stress decision threshold

Notes:
- Press `q` to quit.
- The script infers input feature size from the checkpoint. It requires `1503` features (`468 FaceMesh * 3 + 33 Pose * 3`) or `3006` features (with velocity).
- If your checkpoint was trained on non-FaceMesh+Pose synthetic features, realtime inference will fail by design with a clear error.
- If `artifacts/scaler_mean.npy` and `artifacts/scaler_scale.npy` exist, they are applied automatically at inference.
- Both FaceMesh and Pose landmarks are visualized when `--show-mesh` is enabled.

## 5) Key Hyperparameters To Tune

Most impactful for this task:

1. Sequence construction:
- `sequence_length`: 45, 60, 90, 120
- `stride`: 5, 10, 15, 30

2. Recurrent capacity:
- `hidden_dim`: 96, 128, 192, 256
- `num_layers`: 1, 2, 3
- `bidirectional`: true/false

3. Regularization:
- `dropout`: 0.1 to 0.5
- `weight_decay`: 1e-6 to 1e-3
- `grad_clip`: 0.5 to 2.0

4. Optimization:
- `learning_rate`: 1e-4 to 3e-3
- `batch_size`: 16, 32, 64
- scheduler and early-stopping patience

5. Feature engineering:
- velocity on/off
- smoothing window 1/3/5
- center-and-scale on/off (if landmarks are stable enough)

6. Imbalance handling:
- class weighting on/off
- optionally oversampling in training loader

## 6) Recommended Evaluation

- Subject-disjoint split or LOSO cross-validation.
- Primary metrics: F1, recall, precision, accuracy.
- Report per-subject performance variance, not only global average.

## 7) Extension Ideas

- Replace last-timestep pooling with attention pooling.
- Add CNN front-end for local temporal patterns before LSTM.
- Multi-task setting: stress + arousal regression.
- Domain adaptation across camera conditions and subjects.
