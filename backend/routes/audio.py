"""Audio endpoints — STT via Groq Whisper, TTS via edge-tts, SER via wav2vec2-IEMOCAP.

SER model: superb/wav2vec2-base-superb-er
  - Trained on IEMOCAP (same data as the SpeechBrain model)
  - 4 labels: hap, ang, sad, neu
  - Pure HuggingFace Transformers — no SpeechBrain, no k2, no symlinks
  - Runs on CPU (~360 MB), keeping VRAM free for Qwen
"""

import asyncio
import io
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..dependencies import get_current_user
from ..models.user import UserInDB
from ..utils.groq_client import get_groq_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audio", tags=["audio"])

_ALLOWED_VOICES = {
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "fr-FR-DeniseNeural",
    "fr-FR-HenriNeural",
    "ar-SA-ZariyahNeural",
    "ar-SA-HamedNeural",
}

_SER_MODEL_ID = "superb/wav2vec2-base-superb-er"
_SER_CACHE    = str(Path(__file__).resolve().parents[2] / "alia_nlp" / "models" / "ser_wav2vec2")

# Canonical 4-label emotion map (matches IEMOCAP conventions)
_EMOTION_LABELS = {
    "hap": "Happy",
    "ang": "Angry",
    "sad": "Sad",
    "neu": "Neutral",
    # Fallbacks in case the model uses full-word labels
    "happy":   "Happy",
    "angry":   "Angry",
    "sad":     "Sad",
    "neutral": "Neutral",
}

# ── SER singleton (lazy-loaded) ───────────────────────────────────────────────

_ser_lock           = threading.Lock()
_ser_model          = None
_ser_extractor      = None
_ser_load_attempted = False


def _load_ser_model() -> None:
    global _ser_model, _ser_extractor, _ser_load_attempted
    if _ser_load_attempted:
        return
    _ser_load_attempted = True
    try:
        from transformers import (  # noqa: PLC0415
            Wav2Vec2ForSequenceClassification,
            Wav2Vec2FeatureExtractor,
        )
        os.makedirs(_SER_CACHE, exist_ok=True)
        logger.info("Loading SER model %s (CPU, ~360 MB) …", _SER_MODEL_ID)
        _ser_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            _SER_MODEL_ID, cache_dir=_SER_CACHE
        )
        _ser_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            _SER_MODEL_ID, cache_dir=_SER_CACHE
        )
        _ser_model.eval()
        logger.info("SER model ready (CPU) — labels: %s", list(_ser_model.config.id2label.values()))
    except Exception as exc:
        logger.warning("SER model failed to load: %s — audio emotion disabled", exc)
        _ser_model     = None
        _ser_extractor = None


# ── Audio conversion ──────────────────────────────────────────────────────────

def _get_ffmpeg_exe() -> Optional[str]:
    import shutil  # noqa: PLC0415
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg  # noqa: PLC0415
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _convert_to_wav16k(audio_bytes: bytes) -> Optional[str]:
    """Convert any browser audio (WebM/Opus, mp4 …) → 16 kHz mono WAV temp file."""
    ffmpeg = _get_ffmpeg_exe()
    if ffmpeg is None:
        logger.debug("ffmpeg unavailable — audio emotion skipped")
        return None

    inp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as inp:
            inp.write(audio_bytes)
            inp_path = inp.name

        out_path = inp_path.replace(".webm", ".wav")
        subprocess.run(
            [ffmpeg, "-y", "-i", inp_path, "-ar", "16000", "-ac", "1", out_path],
            capture_output=True,
            check=True,
        )
        return out_path

    except Exception as exc:
        logger.debug("Audio conversion failed: %s", exc)
        return None

    finally:
        if inp_path:
            try:
                os.unlink(inp_path)
            except Exception:
                pass


# ── Emotion classification ────────────────────────────────────────────────────

def _classify_emotion(audio_bytes: bytes) -> Optional[dict]:
    """Run wav2vec2 SER on audio bytes. Synchronous — call via asyncio.to_thread."""
    with _ser_lock:
        _load_ser_model()

    if _ser_model is None or _ser_extractor is None:
        return None

    wav_path = _convert_to_wav16k(audio_bytes)
    if wav_path is None:
        return None

    try:
        import torch                     # noqa: PLC0415
        import soundfile as sf           # noqa: PLC0415
        import torchaudio.functional as F_audio  # noqa: PLC0415

        # soundfile reads WAV natively — no torchaudio backend / torchcodec needed
        audio_np, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        # audio_np: (samples, channels) → mono
        audio_np = audio_np.mean(axis=1)
        waveform = torch.from_numpy(audio_np)

        if sr != 16000:
            waveform = F_audio.resample(waveform.unsqueeze(0), sr, 16000).squeeze(0)

        inputs = _ser_extractor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            logits = _ser_model(**inputs).logits

        probs   = torch.softmax(logits, dim=-1)[0]
        top_idx = probs.argmax().item()
        id2label = _ser_model.config.id2label

        emotion    = id2label[top_idx].lower()
        confidence = float(probs[top_idx].item())
        all_scores = {id2label[i].lower(): round(float(probs[i].item()), 4)
                      for i in range(len(probs))}

        return {
            "emotion":       emotion,
            "emotion_label": _EMOTION_LABELS.get(emotion, emotion.capitalize()),
            "confidence":    round(confidence, 4),
            "source":        "wav2vec2-iemocap",
            "all_scores":    all_scores,
        }

    except Exception as exc:
        logger.warning("SER inference error: %s", exc)
        return None

    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass


def _classify_emotion_array(audio_np: "np.ndarray", sr: int) -> Optional[dict]:
    """
    Run SER on a raw float32 numpy array (already 16 kHz mono).
    Used by the audio explainability module for segment masking.
    """
    with _ser_lock:
        _load_ser_model()

    if _ser_model is None or _ser_extractor is None:
        return None

    try:
        import torch
        import numpy as np

        waveform = torch.from_numpy(audio_np.astype("float32"))
        if sr != 16000:
            import torchaudio.functional as F_audio
            waveform = F_audio.resample(waveform.unsqueeze(0), sr, 16000).squeeze(0)

        inputs = _ser_extractor(
            waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = _ser_model(**inputs).logits

        probs    = torch.softmax(logits, dim=-1)[0]
        top_idx  = probs.argmax().item()
        id2label = _ser_model.config.id2label
        emotion  = id2label[top_idx].lower()

        return {
            "emotion":       emotion,
            "emotion_label": _EMOTION_LABELS.get(emotion, emotion.capitalize()),
            "confidence":    round(float(probs[top_idx].item()), 4),
            "source":        "wav2vec2-iemocap",
            "all_scores":    {id2label[i].lower(): round(float(probs[i].item()), 4)
                              for i in range(len(probs))},
        }
    except Exception as exc:
        logger.warning("SER array inference error: %s", exc)
        return None


# ── STT + SER ─────────────────────────────────────────────────────────────────

class SpeakRequest(BaseModel):
    text: str
    voice: str = "en-US-JennyNeural"


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user),
):
    """Transcribe audio (Groq Whisper) and classify emotion (wav2vec2-IEMOCAP) in parallel.

    Returns { text, audio_affect }.
    audio_affect is None when the clip is too short or the SER model is unavailable.
    """
    client = get_groq_client()
    if client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Groq client not configured (missing GROQ_API_KEY)")

    audio_bytes  = await file.read()
    if len(audio_bytes) < 1000:
        return {"text": "", "audio_affect": None}

    filename     = file.filename or "audio.webm"
    content_type = file.content_type or "audio/webm"

    def _transcribe() -> str:
        result = client.audio.transcriptions.create(
            file=(filename, audio_bytes, content_type),
            model="whisper-large-v3",
            response_format="json",
        )
        return result.text.strip()

    # Whisper + SER run concurrently in the thread-pool
    text, audio_affect = await asyncio.gather(
        asyncio.to_thread(_transcribe),
        asyncio.to_thread(_classify_emotion, audio_bytes),
    )
    return {"text": text, "audio_affect": audio_affect}


# ── TTS ───────────────────────────────────────────────────────────────────────

@router.post("/speak")
async def speak(
    body: SpeakRequest,
    current_user: UserInDB = Depends(get_current_user),
):
    """Stream TTS audio via edge-tts (Microsoft neural voices, free, no key needed)."""
    try:
        import edge_tts  # noqa: PLC0415
    except ImportError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="edge-tts not installed — run: pip install edge-tts")

    voice = body.voice if body.voice in _ALLOWED_VOICES else "en-US-JennyNeural"
    text  = body.text[:2000].strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text is empty")

    async def _stream():
        communicate = edge_tts.Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    return StreamingResponse(
        _stream(),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache", "X-Voice": voice},
    )
