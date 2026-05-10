"""
Affect explainability endpoints.

POST /affect/explain-text
    Run LIME and/or SHAP on a text message and return token-level
    importance scores for each affect dimension.

POST /affect/explain-audio
    Run LIME and/or SHAP on an uploaded audio file and return
    segment-level importance scores plus text-level explanation
    of the transcription.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from ..dependencies import get_current_user
from ..models.user import UserInDB

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/affect", tags=["affect"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TextExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    mode: Literal["physician_portal", "medrep_training"] = "physician_portal"
    method: Literal["shap", "lime", "both"] = "both"
    top_k: int = Field(default=10, ge=3, le=20)
    targets: Optional[list[str]] = None  # subset of ["confidence_high","frustration","stress"]


# ---------------------------------------------------------------------------
# Text explanation
# ---------------------------------------------------------------------------

@router.post("/explain-text")
async def explain_text(
    body: TextExplainRequest,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Return LIME and/or SHAP token-level attributions for a text message.

    Each token gets a signed importance score: positive = pushes the
    affect dimension higher, negative = pushes it lower.

    Dimensions explained:
      - confidence_high  (rep speaking with high confidence)
      - frustration      (frustration signal detected)
      - stress           (stress signal detected)
    """
    from alia_nlp.src.layers.L7_affect.explainability import explain_text as _explain

    try:
        result = await asyncio.to_thread(
            _explain,
            body.text,
            body.mode,
            body.method,
            body.top_k,
            body.targets,
        )
    except Exception as exc:
        logger.error("Text explanation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {exc}",
        )
    return result


# ---------------------------------------------------------------------------
# Audio explanation
# ---------------------------------------------------------------------------

@router.post("/explain-audio")
async def explain_audio(
    file: UploadFile = File(...),
    method: Literal["shap", "lime", "both"] = Query(default="both"),
    n_segments: int = Query(default=10, ge=4, le=20),
    top_k: int = Query(default=6, ge=3, le=12),
    transcription: str = Query(default=""),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Run segment-level LIME/SHAP on an audio file.

    The audio is split into N equal time segments. Each segment is treated
    as a binary feature; the explainer silences subsets of segments and
    measures how the emotion-confidence score changes.

    If *transcription* is provided (from a prior /audio/transcribe call),
    word-level text LIME/SHAP is also included in the response under
    the "text_explanation" key.

    Returns segment importances (which part of the audio drove the emotion)
    and optionally word importances (which words drove sentiment in the text).
    """
    import numpy as np
    import soundfile as sf
    import io

    audio_bytes = await file.read()
    if len(audio_bytes) < 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file too short (< 1 KB).",
        )

    # Decode audio to float32 numpy array
    def _decode() -> tuple[np.ndarray, int]:
        from backend.routes.audio import _convert_to_wav16k
        wav_path = _convert_to_wav16k(audio_bytes)
        if wav_path is None:
            raise RuntimeError("Could not convert audio to WAV (ffmpeg unavailable?)")
        import os
        audio_np, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        os.unlink(wav_path)
        return audio_np.mean(axis=1), sr  # mono

    try:
        audio_np, sr = await asyncio.to_thread(_decode)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Audio decode failed: {exc}",
        )

    from backend.utils.audio_explain import explain_audio as _explain_audio

    try:
        result = await asyncio.to_thread(
            _explain_audio, audio_np, sr, transcription, method, n_segments, top_k
        )
    except Exception as exc:
        logger.error("Audio explanation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {exc}",
        )
    return result
