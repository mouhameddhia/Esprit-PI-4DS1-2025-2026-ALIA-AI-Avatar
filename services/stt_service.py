"""
Speech-to-Text Service using Whisper
Converts audio from microphone to text for AI processing
"""

import asyncio
import logging
import whisper
import numpy as np
import io
import wave
import os
import re
from typing import Optional, Dict, Tuple, List, Any

from utils.config import config

logger = logging.getLogger(__name__)

class STTService:
    """Service for Speech-to-Text using Whisper"""
    
    def __init__(self):
        self.model = None
        # Legacy ALIA_WHISPER_MODEL wins; else Settings.STT_WHISPER_MODEL (.env key STT_WHISPER_MODEL) or base.en.
        self.model_name = (
            os.environ.get("ALIA_WHISPER_MODEL", "").strip()
            or str(getattr(config, "STT_WHISPER_MODEL", "") or "").strip()
            or "base.en"
        )
        self.fast_decode: bool = bool(getattr(config, "STT_FAST_DECODE", True))
        self.skip_expensive_retry: bool = bool(getattr(config, "STT_SKIP_EXPENSIVE_RETRY", True))
        self.run_in_thread: bool = bool(getattr(config, "STT_RUN_IN_THREAD", True))
        self.default_language = os.environ.get("ALIA_STT_LANGUAGE", "en").strip() or "en"
        self.context_window_words = int((os.environ.get("ALIA_STT_CONTEXT_WORDS", "8") or "8").strip() or "8")
        self.initial_prompt = os.environ.get(
            "ALIA_STT_INITIAL_PROMPT",
            "Clear conversational speech in English. Transcribe exactly what is said.",
        ).strip()
        self.previous_transcript_tail = ""
        self.last_result_debug: Dict[str, Any] = {}
        # Options: tiny, base, small, medium, large
        
    async def initialize(self):
        """Initialize Whisper model"""
        try:
            logger.info(f"Loading Whisper '{self.model_name}' model...")
            # Load model (downloads on first run to ~/.cache/whisper/)
            try:
                self.model = whisper.load_model(self.model_name)
            except Exception as model_error:
                fallback_model = "base"
                logger.warning(
                    f"Failed to load Whisper '{self.model_name}' ({model_error}), falling back to '{fallback_model}'"
                )
                self.model = whisper.load_model(fallback_model)
                self.model_name = fallback_model
            logger.info(f"Whisper '{self.model_name}' model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if STT service is ready"""
        return self.model is not None

    def get_last_result_debug(self) -> Dict[str, Any]:
        """Return debug metrics from the most recent transcription."""
        return dict(self.last_result_debug)

    def _build_primary_decode_kwargs(
        self,
        selected_language: Optional[str],
        prompt_with_context: str,
    ) -> Dict[str, Any]:
        if self.fast_decode:
            decode_kwargs: Dict[str, Any] = {
                "fp16": False,
                "verbose": False,
                "temperature": 0.0,
                "task": "transcribe",
                "beam_size": 1,
                "best_of": 1,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.95,
                "compression_ratio_threshold": 3.0,
                "logprob_threshold": -2.0,
                "without_timestamps": True,
                "word_timestamps": False,
            }
        else:
            decode_kwargs = {
                "fp16": False,
                "verbose": False,
                "temperature": 0.0,
                "task": "transcribe",
                "beam_size": 6,
                "best_of": 6,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.95,
                "compression_ratio_threshold": 3.0,
                "logprob_threshold": -2.0,
                "without_timestamps": False,
                "word_timestamps": True,
            }
        if selected_language:
            decode_kwargs["language"] = selected_language
        if prompt_with_context:
            decode_kwargs["initial_prompt"] = prompt_with_context
        return decode_kwargs

    def _transcribe_audio_sync(
        self,
        decode_audio: np.ndarray,
        decode_duration: float,
        decode_rms: float,
        preprocess_stats: Dict[str, Any],
        selected_language: Optional[str],
    ) -> Optional[str]:
        """Blocking Whisper decode (+ optional expensive retry)."""
        prompt_with_context = self._build_prompt_with_context()
        decode_kwargs = self._build_primary_decode_kwargs(selected_language, prompt_with_context)

        logger.info(
            "Whisper params(primary): beam_size=%d best_of=%d temperature=%s no_speech_threshold=%.2f logprob_threshold=%.2f timestamps=%s fast=%s",
            int(decode_kwargs.get("beam_size", 0)),
            int(decode_kwargs.get("best_of", 0)),
            str(decode_kwargs.get("temperature")),
            float(decode_kwargs.get("no_speech_threshold", 0.0)),
            float(decode_kwargs.get("logprob_threshold", 0.0)),
            str(not bool(decode_kwargs.get("without_timestamps", True))),
            self.fast_decode,
        )

        primary_result = self.model.transcribe(decode_audio, **decode_kwargs)
        primary_text = (primary_result.get("text") or "").strip()
        primary_metrics = self._extract_result_metrics(
            primary_result,
            duration_sec=decode_duration,
            input_rms=decode_rms,
            preprocess_stats=preprocess_stats,
        )
        primary_uncertain_words = self._collect_uncertain_words(primary_result)
        primary_metrics["uncertain_word_count"] = float(len(primary_uncertain_words))

        final_text = primary_text
        final_metrics = primary_metrics
        final_uncertain_words = primary_uncertain_words
        decode_mode = "primary"

        primary_word_count = int(primary_metrics.get("word_count", 0))
        if (
            not self.skip_expensive_retry
            and (not primary_text or (primary_word_count <= 1 and decode_duration > 1.20))
        ):
            retry_kwargs = dict(decode_kwargs)
            retry_kwargs.pop("language", None)
            retry_kwargs["temperature"] = (0.0, 0.2)
            retry_kwargs["beam_size"] = 8
            retry_kwargs["best_of"] = 8
            # Retry path benefits from richer alignment when available.
            retry_kwargs["without_timestamps"] = False
            retry_kwargs["word_timestamps"] = True

            logger.info(
                "Whisper params(retry): beam_size=%d best_of=%d temperature=%s",
                int(retry_kwargs.get("beam_size", 0)),
                int(retry_kwargs.get("best_of", 0)),
                str(retry_kwargs.get("temperature")),
            )

            retry_result = self.model.transcribe(decode_audio, **retry_kwargs)
            retry_text = (retry_result.get("text") or "").strip()
            retry_metrics = self._extract_result_metrics(
                retry_result,
                duration_sec=decode_duration,
                input_rms=decode_rms,
                preprocess_stats=preprocess_stats,
            )
            retry_uncertain_words = self._collect_uncertain_words(retry_result)
            retry_metrics["uncertain_word_count"] = float(len(retry_uncertain_words))

            retry_words = int(retry_metrics.get("word_count", 0))
            primary_words = int(primary_metrics.get("word_count", 0))
            retry_logprob = float(retry_metrics.get("avg_logprob", -2.0))
            primary_logprob = float(primary_metrics.get("avg_logprob", -2.0))

            should_use_retry = False
            if not final_text and retry_text:
                should_use_retry = True
            elif retry_text and retry_words > primary_words and retry_logprob >= (primary_logprob - 0.35):
                should_use_retry = True

            if should_use_retry:
                final_text = retry_text
                final_metrics = retry_metrics
                final_uncertain_words = retry_uncertain_words
                decode_mode = "retry"

        if not final_text:
            return None

        self._update_context_memory(final_text, final_metrics)
        self.last_result_debug = {
            "text": final_text,
            "decode_mode": decode_mode,
            "avg_logprob": float(final_metrics.get("avg_logprob", -2.0)),
            "avg_no_speech": float(final_metrics.get("avg_no_speech", 1.0)),
            "word_count": int(final_metrics.get("word_count", 0)),
            "short_word_count": int(final_metrics.get("short_word_count", 0)),
            "words_per_second": float(final_metrics.get("words_per_second", 0.0)),
            "timed_word_count": int(final_metrics.get("timed_word_count", 0)),
            "long_gap_count": int(final_metrics.get("long_gap_count", 0)),
            "uncertain_word_count": int(final_metrics.get("uncertain_word_count", 0)),
            "uncertain_words": final_uncertain_words[:10],
            "speech_duration_sec": float(final_metrics.get("speech_duration_sec", 0.0)),
            "audio_duration_sec": float(final_metrics.get("audio_duration_sec", 0.0)),
            "input_rms": float(final_metrics.get("input_rms", 0.0)),
            "context_tail": self.previous_transcript_tail,
            "decode_params": {
                "fast_decode": self.fast_decode,
                "skip_expensive_retry": self.skip_expensive_retry,
                "beam_size": int(decode_kwargs.get("beam_size", 0)),
                "best_of": int(decode_kwargs.get("best_of", 0)),
                "retry_beam_size": 8 if not self.skip_expensive_retry else None,
                "retry_best_of": 8 if not self.skip_expensive_retry else None,
                "no_speech_threshold": float(decode_kwargs.get("no_speech_threshold", 0.0)),
                "logprob_threshold": float(decode_kwargs.get("logprob_threshold", 0.0)),
                "word_timestamps": bool(decode_kwargs.get("word_timestamps", False)),
            },
        }

        logger.info(
            "Transcribed final(%s): '%s' words=%d wps=%.2f logprob=%.3f no_speech=%.3f uncertain=%d",
            decode_mode,
            final_text,
            int(final_metrics.get("word_count", 0)),
            float(final_metrics.get("words_per_second", 0.0)),
            float(final_metrics.get("avg_logprob", -2.0)),
            float(final_metrics.get("avg_no_speech", 1.0)),
            int(final_metrics.get("uncertain_word_count", 0)),
        )
        if final_uncertain_words:
            logger.warning("Uncertain words: %s", ", ".join(final_uncertain_words[:10]))
        return final_text

    async def transcribe_audio(self, audio_data: bytes, language: Optional[str] = None) -> Optional[str]:
        """
        Transcribe audio bytes to text
        
        Args:
            audio_data: Raw audio bytes (WAV format preferred)
            language: Language code. If None, Whisper auto-detects language.
            
        Returns:
            Transcribed text or None if failed
        """
        if not self.is_ready():
            logger.error("STT service not initialized")
            return None
        
        try:
            logger.info(f"Transcribing audio ({len(audio_data)} bytes)...")
            self.last_result_debug = {}

            audio_array = self._decode_audio_bytes_to_float32(audio_data)
            if audio_array is None or audio_array.size == 0:
                logger.warning("Decoded audio is empty")
                return None

            raw_duration = audio_array.size / 16000.0
            raw_rms = float(np.sqrt(np.mean(np.square(audio_array)))) if audio_array.size > 0 else 0.0
            raw_peak = float(np.max(np.abs(audio_array))) if audio_array.size > 0 else 0.0
            if raw_rms < 0.010:
                logger.warning("Input audio is quiet before Whisper: rms=%.5f duration=%.3fs", raw_rms, raw_duration)

            processed_audio, preprocess_stats = self._adaptive_preprocess_audio(audio_array)
            trim_stats = {
                "trimmed": False,
                "removed_duration_sec": 0.0,
                "active_duration_sec": raw_duration,
                "threshold": 0.0,
            }
            decode_audio = processed_audio

            decode_duration = decode_audio.size / 16000.0 if decode_audio is not None else 0.0
            decode_rms = float(np.sqrt(np.mean(np.square(decode_audio)))) if decode_audio is not None and decode_audio.size > 0 else 0.0
            decode_peak = float(np.max(np.abs(decode_audio))) if decode_audio is not None and decode_audio.size > 0 else 0.0

            logger.info(
                "STT input: samples=%d duration=%.3fs rms=%.5f peak=%.4f",
                int(audio_array.size),
                raw_duration,
                raw_rms,
                raw_peak,
            )
            logger.info(
                "STT preprocess: decode_samples=%d decode_duration=%.3fs decode_rms=%.5f decode_peak=%.4f gain=%.2f comp=%.2f noise_floor=%.5f trim_removed=%.3fs",
                int(decode_audio.size),
                decode_duration,
                decode_rms,
                decode_peak,
                float(preprocess_stats.get("adaptive_gain", 1.0)),
                float(preprocess_stats.get("compression_reduction", 1.0)),
                float(preprocess_stats.get("noise_floor", 0.0)),
                float(trim_stats.get("removed_duration_sec", 0.0)),
            )

            if decode_audio is None or decode_audio.size < 240:
                logger.warning("Decoded audio too short after preprocessing")
                return None

            selected_language = language if language else self.default_language
            if self.run_in_thread:
                return await asyncio.to_thread(
                    self._transcribe_audio_sync,
                    decode_audio,
                    decode_duration,
                    decode_rms,
                    preprocess_stats,
                    selected_language,
                )
            return self._transcribe_audio_sync(
                decode_audio,
                decode_duration,
                decode_rms,
                preprocess_stats,
                selected_language,
            )
                    
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    def _decode_audio_bytes_to_float32(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Decode incoming audio bytes to mono float32 waveform in [-1, 1].
        Supports:
        - WAV container bytes
        - Raw PCM s16le mono bytes
        """
        if not audio_data:
            return None

        # Try WAV first.
        try:
            with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())

            if sample_width != 2:
                logger.warning(f"Unsupported WAV sample width: {sample_width}")
                return None

            pcm = np.frombuffer(frames, dtype=np.int16)
            if channels > 1:
                pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)

            audio = pcm.astype(np.float32) / 32768.0

            # Whisper expects 16k for best results; crude resample if needed.
            if frame_rate and frame_rate != 16000 and audio.size > 0:
                duration = audio.size / float(frame_rate)
                target_size = max(1, int(duration * 16000.0))
                x_old = np.linspace(0.0, duration, num=audio.size, endpoint=False)
                x_new = np.linspace(0.0, duration, num=target_size, endpoint=False)
                audio = np.interp(x_new, x_old, audio).astype(np.float32)

            return audio
        except Exception:
            pass

        # Fallback: treat as raw PCM16 mono at 16k.
        try:
            pcm = np.frombuffer(audio_data, dtype=np.int16)
            if pcm.size == 0:
                return None
            audio = pcm.astype(np.float32) / 32768.0
            return audio
        except Exception as e:
            logger.error(f"Failed to decode audio bytes: {e}")
            return None

    def _adaptive_preprocess_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[np.ndarray, Dict[str, float]]:
        if audio.size == 0:
            return audio, {
                "input_rms": 0.0,
                "input_peak": 0.0,
                "adaptive_gain": 1.0,
                "compression_reduction": 1.0,
                "noise_floor": 0.0,
            }

        signal = audio.astype(np.float32, copy=True)
        signal = signal - float(np.mean(signal))

        input_rms = float(np.sqrt(np.mean(np.square(signal))) + 1e-8)
        input_peak = float(np.max(np.abs(signal)) + 1e-8)

        noise_floor = float(np.percentile(np.abs(signal), 20))

        # Transparent adaptive normalization: boost quiet speech without heavy filtering.
        target_rms = 0.060 if input_rms < 0.015 else 0.052
        adaptive_gain = float(np.clip(target_rms / max(input_rms, 1e-6), 1.0, 3.0))
        if input_peak * adaptive_gain > 0.985:
            adaptive_gain = float(min(adaptive_gain, 0.985 / max(input_peak, 1e-6)))

        processed = signal * adaptive_gain
        soft_clip_threshold = 0.985
        over = np.abs(processed) > soft_clip_threshold
        if np.any(over):
            clipped = processed[over]
            clipped_abs = np.abs(clipped)
            overshoot = (clipped_abs - soft_clip_threshold) / max(1e-6, 1.0 - soft_clip_threshold)
            smoothed = soft_clip_threshold + (1.0 - soft_clip_threshold) * np.tanh(overshoot)
            processed[over] = np.sign(clipped) * smoothed

        processed = np.clip(processed, -1.0, 1.0).astype(np.float32)

        return processed, {
            "input_rms": input_rms,
            "input_peak": input_peak,
            "adaptive_gain": adaptive_gain,
            "makeup_gain": 1.0,
            "compression_reduction": 1.0,
            "noise_floor": noise_floor,
            "speech_threshold": 0.0,
        }

    def _trim_silence(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[np.ndarray, Dict[str, float]]:
        if audio.size < 400:
            return audio, {
                "trimmed": False,
                "removed_duration_sec": 0.0,
                "active_duration_sec": audio.size / float(sample_rate),
                "threshold": 0.0,
            }

        abs_audio = np.abs(audio)
        window = max(64, int(sample_rate * 0.01))
        kernel = np.ones(window, dtype=np.float32) / float(window)
        envelope = np.convolve(abs_audio, kernel, mode="same")

        noise_floor = float(np.percentile(envelope, 12))
        threshold = max(0.0012, noise_floor * 1.20)
        active = np.where(envelope > threshold)[0]
        if active.size == 0:
            return audio, {
                "trimmed": False,
                "removed_duration_sec": 0.0,
                "active_duration_sec": 0.0,
                "threshold": float(threshold),
            }

        pad_before = int(sample_rate * 0.30)
        pad_after = int(sample_rate * 0.45)
        start = max(0, int(active[0]) - pad_before)
        end = min(audio.size, int(active[-1]) + pad_after)
        trimmed = audio[start:end]
        if trimmed.size <= 0:
            return audio, {
                "trimmed": False,
                "removed_duration_sec": 0.0,
                "active_duration_sec": 0.0,
                "threshold": float(threshold),
            }

        if trimmed.size < int(sample_rate * 0.35):
            return audio, {
                "trimmed": False,
                "removed_duration_sec": 0.0,
                "active_duration_sec": active.size / float(sample_rate),
                "threshold": float(threshold),
            }

        removed = max(0, audio.size - trimmed.size)
        removed_ratio = float(removed) / float(max(1, audio.size))
        if removed_ratio > 0.55:
            # Keep full audio when trim appears too aggressive (common in fast/quiet speech).
            return audio, {
                "trimmed": False,
                "removed_duration_sec": 0.0,
                "active_duration_sec": active.size / float(sample_rate),
                "threshold": float(threshold),
            }

        return trimmed, {
            "trimmed": bool(removed > 0),
            "removed_duration_sec": removed / float(sample_rate),
            "active_duration_sec": active.size / float(sample_rate),
            "threshold": float(threshold),
        }

    def _has_heavy_repetition(self, text: str) -> bool:
        tokens = [t for t in re.findall(r"[a-zA-Z']+", (text or "").lower()) if t]
        if len(tokens) < 4:
            return False

        max_run = 1
        current_run = 1
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        if max_run >= 3:
            return True

        unique_tokens = set(tokens)
        dominant_count = max(tokens.count(t) for t in unique_tokens)
        return dominant_count / float(len(tokens)) > 0.55

    def _is_low_quality_transcription(self, text: str, metrics: Dict[str, float]) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return True

        if self._has_heavy_repetition(normalized):
            return True

        avg_logprob = float(metrics.get("avg_logprob", -1.5))
        avg_no_speech = float(metrics.get("avg_no_speech", 0.0))
        speech_duration = float(metrics.get("speech_duration_sec", metrics.get("audio_duration_sec", 0.0)))
        word_count = int(metrics.get("word_count", 0))
        uncertain_words = int(metrics.get("uncertain_word_count", 0))

        if avg_logprob < -1.55 and speech_duration > 1.0 and word_count <= 1:
            return True
        if avg_no_speech > 0.90 and speech_duration > 0.8 and word_count == 0:
            return True
        if uncertain_words >= max(3, word_count):
            return True

        return False

    def _extract_result_metrics(
        self,
        result: Optional[dict],
        duration_sec: float = 0.0,
        input_rms: float = 0.0,
        preprocess_stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        segments = (result or {}).get("segments") or []
        log_probs = [s.get("avg_logprob") for s in segments if s.get("avg_logprob") is not None]
        no_speech_probs = [s.get("no_speech_prob") for s in segments if s.get("no_speech_prob") is not None]
        token_count = 0
        word_count = 0
        short_word_count = 0
        timed_word_count = 0
        long_gap_count = 0
        prev_word_end = None
        speech_start = None
        speech_end = None

        for segment in segments:
            tokens = segment.get("tokens") or []
            token_count += len(tokens)
            if speech_start is None and segment.get("start") is not None:
                speech_start = float(segment.get("start"))
            if segment.get("end") is not None:
                speech_end = float(segment.get("end"))

            segment_words = segment.get("words") or []
            for word_info in segment_words:
                word_text = str(word_info.get("word") or "").strip()
                if not word_text:
                    continue
                word_count += 1

                normalized = re.sub(r"[^a-zA-Z']", "", word_text).strip("'").lower()
                if normalized and len(normalized) <= 2:
                    short_word_count += 1

                word_start = word_info.get("start")
                word_end = word_info.get("end")
                if word_start is not None and word_end is not None:
                    timed_word_count += 1
                    word_start = float(word_start)
                    word_end = float(word_end)
                    if prev_word_end is not None and (word_start - prev_word_end) > 0.24:
                        long_gap_count += 1
                    prev_word_end = word_end

        full_text = str((result or {}).get("text") or "").strip()
        fallback_words = [w for w in re.findall(r"[a-zA-Z']+", full_text.lower()) if w]
        if word_count <= 0 and fallback_words:
            word_count = len(fallback_words)
            short_word_count = len([w for w in fallback_words if len(w) <= 2])

        speech_duration = 0.0
        if speech_start is not None and speech_end is not None and speech_end >= speech_start:
            speech_duration = float(speech_end - speech_start)
        if speech_duration <= 0.0:
            speech_duration = float(duration_sec)

        words_per_second = float(word_count) / max(speech_duration, 1e-3)
        preprocess_stats = preprocess_stats or {}

        return {
            "avg_logprob": float(np.mean(log_probs)) if log_probs else -1.2,
            "avg_no_speech": float(np.mean(no_speech_probs)) if no_speech_probs else 0.0,
            "token_count": float(token_count),
            "segment_count": float(len(segments)),
            "word_count": float(word_count),
            "short_word_count": float(short_word_count),
            "timed_word_count": float(timed_word_count),
            "long_gap_count": float(long_gap_count),
            "words_per_second": float(words_per_second),
            "speech_duration_sec": float(speech_duration),
            "audio_duration_sec": float(duration_sec),
            "input_rms": float(input_rms),
            "adaptive_gain": float(preprocess_stats.get("adaptive_gain", 1.0)),
            "compression_reduction": float(preprocess_stats.get("compression_reduction", 1.0)),
            "noise_floor": float(preprocess_stats.get("noise_floor", 0.0)),
        }

    def _score_transcription_candidate(self, text: str, metrics: Dict[str, float]) -> float:
        if not text:
            return -1e9

        avg_logprob = float(metrics.get("avg_logprob", -1.2))
        avg_no_speech = float(metrics.get("avg_no_speech", 0.0))
        word_count = float(metrics.get("word_count", 0.0))
        short_word_count = float(metrics.get("short_word_count", 0.0))
        speech_duration = float(metrics.get("speech_duration_sec", metrics.get("audio_duration_sec", 0.0)))
        uncertain_count = float(metrics.get("uncertain_word_count", 0.0))
        long_gap_count = float(metrics.get("long_gap_count", 0.0))

        expected_min_words = max(1.0, speech_duration * 1.25)
        under_length_penalty = max(0.0, expected_min_words - word_count) * 0.18 if speech_duration > 0.8 else 0.0
        short_word_bonus = min(0.45, short_word_count * 0.05)
        length_bonus = min(1.0, 0.017 * len(text))
        uncertainty_penalty = uncertain_count * 0.10
        drop_gap_penalty = long_gap_count * 0.09

        return (
            avg_logprob
            - (0.55 * avg_no_speech)
            + length_bonus
            + short_word_bonus
            - under_length_penalty
            - uncertainty_penalty
            - drop_gap_penalty
        )

    def _collect_uncertain_words(self, result: Optional[dict]) -> List[str]:
        uncertain: List[str] = []
        segments = (result or {}).get("segments") or []
        for segment in segments:
            for word_info in segment.get("words") or []:
                word_text = str(word_info.get("word") or "").strip()
                probability = word_info.get("probability")
                if not word_text or probability is None:
                    continue
                if float(probability) < 0.45:
                    uncertain.append(f"{word_text}({float(probability):.2f})")
        return uncertain[:20]

    def _build_prompt_with_context(self) -> str:
        context_tail = (self.previous_transcript_tail or "").strip()
        if context_tail:
            if self.initial_prompt:
                return f"{self.initial_prompt} Context words for continuity only (do not copy unless heard): {context_tail}"
            return f"Context words for continuity only (do not copy unless heard): {context_tail}"
        return self.initial_prompt

    def _update_context_memory(self, text: str, metrics: Dict[str, float]):
        normalized = (text or "").strip()
        if not normalized:
            return

        # Update context only from reasonable-confidence outputs.
        if float(metrics.get("avg_logprob", -2.0)) < -1.2:
            return
        if float(metrics.get("avg_no_speech", 1.0)) > 0.6:
            return
        if int(metrics.get("word_count", 0)) < 3:
            return

        words = [w for w in re.findall(r"[a-zA-Z']+", normalized.lower()) if w]
        if not words:
            return
        window_size = max(4, min(16, int(self.context_window_words)))
        self.previous_transcript_tail = " ".join(words[-window_size:])
    
    async def transcribe_file(self, file_path: str, language: str = "en") -> Optional[str]:
        """
        Transcribe audio file to text
        
        Args:
            file_path: Path to audio file
            language: Language code (default: "en")
            
        Returns:
            Transcribed text or None if failed
        """
        if not self.is_ready():
            logger.error("STT service not initialized")
            return None
        
        try:
            logger.info(f"Transcribing file: {file_path}")
            result = self.model.transcribe(
                file_path,
                language=language,
                fp16=False,
                verbose=False
            )
            
            text = result["text"].strip()
            logger.info(f"Transcribed: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.model:
            del self.model
            self.model = None
        logger.info("STT service cleaned up")


# Singleton instance
stt_service = STTService()
