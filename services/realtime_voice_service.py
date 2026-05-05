"""
Real-Time Voice Service using WebSocket
Handles bidirectional streaming: voice → text → AI → audio → client
"""

import logging
import asyncio
import json
import base64
import uuid
import wave
import math
import struct
import subprocess
import os
import io
import numpy as np
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

from utils.config import config
# Import service instances (will be set by main.py)
stt_service = None
llm_service = None
tts_service = None
lipsync_service = None

# Import after services are defined
from utils.alia_morphtargets import get_morph_for_viseme

logger = logging.getLogger(__name__)
DEFAULT_STT_LANGUAGE = (os.environ.get("ALIA_STT_LANGUAGE", "en") or "en").strip()

def set_services(stt, llm, tts, lipsync):
    """Set service instances (called from main.py)"""
    global stt_service, llm_service, tts_service, lipsync_service
    stt_service = stt
    llm_service = llm
    tts_service = tts
    lipsync_service = lipsync

class RealtimeVoiceService:
    """
    Manages real-time voice conversation pipeline:
    1. Receive audio chunks from client (WebSocket)
    2. Transcribe to text (Whisper STT)
    3. Generate AI response (Ollama LLM)
    4. Generate speech (Piper TTS)
    5. Extract lip sync data (Visemes)
    6. Stream back audio + visemes in chunks
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, dict] = {}

    @staticmethod
    def _ensure_wav_bytes(audio_data: bytes, sample_rate: int, channels: int) -> bytes:
        """Return valid WAV bytes. If input is raw PCM16, wrap it into a WAV container."""
        if not audio_data:
            return b""

        # Already WAV
        if len(audio_data) >= 12 and audio_data[:4] == b"RIFF" and audio_data[8:12] == b"WAVE":
            return audio_data

        sample_rate = int(sample_rate or 16000)
        channels = int(channels or 1)
        channels = max(1, channels)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # PCM16
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        return buffer.getvalue()

    @staticmethod
    def _analyze_pcm_metrics(audio_data: bytes, sample_rate: int, channels: int) -> Dict[str, float]:
        if not audio_data:
            return {
                "duration_sec": 0.0,
                "samples": 0.0,
                "rms": 0.0,
                "peak": 0.0,
                "clipping_ratio": 0.0,
                "zero_crossing_rate": 0.0,
            }

        pcm = np.frombuffer(audio_data, dtype=np.int16)
        if pcm.size == 0:
            return {
                "duration_sec": 0.0,
                "samples": 0.0,
                "rms": 0.0,
                "peak": 0.0,
                "clipping_ratio": 0.0,
                "zero_crossing_rate": 0.0,
            }

        channels = max(1, int(channels or 1))
        if channels > 1 and (pcm.size % channels) == 0:
            pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)

        audio = pcm.astype(np.float32) / 32768.0
        duration_sec = float(audio.size) / float(max(1, int(sample_rate or 16000)))
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size > 0 else 0.0
        peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        clipping_ratio = float(np.mean(np.abs(audio) >= 0.985)) if audio.size > 0 else 0.0
        zero_crossing_rate = 0.0
        if audio.size > 1:
            zero_crossing_rate = float(np.mean(np.diff(np.signbit(audio)).astype(np.float32)))

        return {
            "duration_sec": duration_sec,
            "samples": float(audio.size),
            "rms": rms,
            "peak": peak,
            "clipping_ratio": clipping_ratio,
            "zero_crossing_rate": zero_crossing_rate,
        }

    @staticmethod
    def _save_input_wav_for_debug(wav_audio: bytes, session: dict, duration_sec: float):
        if not wav_audio:
            return

        save_enabled = (os.environ.get("ALIA_SAVE_INPUT_WAV", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
        if not save_enabled:
            return

        output_dir = Path(__file__).parent.parent / "audio_debug_input"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        client_id = str(session.get("id", "client"))[:12]
        file_name = f"turn_{stamp}_{client_id}.wav"
        output_path = output_dir / file_name
        output_path.write_bytes(wav_audio)
        logger.info("Saved input WAV for inspection: %s (duration=%.3fs)", str(output_path), duration_sec)
    
    async def handle_voice_stream(self, websocket, client_id: str):
        """
        Handle WebSocket connection for real-time voice
        
        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
        """
        session = {
            "id": client_id,
            "audio_buffer": bytearray(),
            "is_recording": False,
            "last_activity": datetime.now(),
            "sample_rate": 16000,
            "channels": 1,
            "encoding": "pcm_s16le",
            "language": DEFAULT_STT_LANGUAGE,
            "input_gain": 1.0,
            "noise_floor_rms": 0.0,
            "mic_device": "<unknown>",
            "competency_level": None,
        }
        self.active_sessions[client_id] = session
        
        try:
            logger.info(f"Voice session started: {client_id}")
            
            # Send welcome message
            await websocket.send_json({
                "type": "connected",
                "message": "Real-time voice session ready",
                "client_id": client_id
            })
            
            # Message loop
            async for message in websocket.iter_text():
                try:
                    data = json.loads(message)
                    await self._handle_message(websocket, session, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {client_id}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                    
        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
        finally:
            # Cleanup
            if client_id in self.active_sessions:
                del self.active_sessions[client_id]
            logger.info(f"Voice session ended: {client_id}")
    
    async def _handle_message(self, websocket, session: dict, data: dict):
        """Handle incoming WebSocket message"""
        msg_type = data.get("type")
        
        if msg_type == "audio_chunk":
            # Receive audio chunk from client
            await self._handle_audio_chunk(websocket, session, data)
            
        elif msg_type == "audio_end":
            # Client finished speaking - process full audio
            await self._process_complete_audio(websocket, session)
            
        elif msg_type == "text_message":
            # Direct text message (skip STT)
            text = data.get("message", "")
            if data.get("competency_level"):
                session["competency_level"] = str(data.get("competency_level"))
            await self._process_text_message(websocket, session, text)

        elif msg_type == "set_competency_level":
            incoming_level = str(data.get("competency_level", "")).strip().upper()
            if incoming_level in {"BEGINNER", "JUNIOR", "CONFIRMED", "EXPERT"}:
                session["competency_level"] = incoming_level
                await websocket.send_json({"type": "competency_level_set", "competency_level": incoming_level})
                logger.info("Voice session %s competency level set to %s", session.get("id"), incoming_level)
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Invalid competency_level. Use BEGINNER/JUNIOR/CONFIRMED/EXPERT",
                    }
                )

        elif msg_type == "interrupt":
            # Client barge-in interrupt
            session["audio_buffer"].clear()
            await websocket.send_json({"type": "interrupted", "ok": True})
            
        elif msg_type == "ping":
            # Keep-alive
            await websocket.send_json({"type": "pong"})
            
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    async def _handle_audio_chunk(self, websocket, session: dict, data: dict):
        """Receive and buffer audio chunk"""
        try:
            # Audio is base64 encoded. Accept both "audio" and "data" keys.
            audio_b64 = data.get("audio") or data.get("data") or ""
            if not audio_b64:
                return

            # Keep latest sender-provided stream metadata when available.
            if data.get("sample_rate") is not None:
                session["sample_rate"] = int(data.get("sample_rate") or 16000)
            if data.get("channels") is not None:
                session["channels"] = int(data.get("channels") or 1)
            if data.get("encoding"):
                session["encoding"] = str(data.get("encoding"))
            if data.get("language"):
                session["language"] = str(data.get("language"))
            if data.get("input_gain") is not None:
                session["input_gain"] = float(data.get("input_gain") or 1.0)
            if data.get("noise_floor_rms") is not None:
                session["noise_floor_rms"] = float(data.get("noise_floor_rms") or 0.0)
            if data.get("mic_device"):
                session["mic_device"] = str(data.get("mic_device"))
            if data.get("competency_level"):
                session["competency_level"] = str(data.get("competency_level"))

            audio_bytes = base64.b64decode(audio_b64)
            
            # Append to buffer
            session["audio_buffer"].extend(audio_bytes)
            session["last_activity"] = datetime.now()
            
            # Send ACK
            await websocket.send_json({
                "type": "audio_ack",
                "buffered_bytes": len(session["audio_buffer"]),
                "sample_rate": session.get("sample_rate", 16000),
                "channels": session.get("channels", 1),
            })
            
        except Exception as e:
            logger.error(f"Audio chunk error: {e}")
    
    async def _process_complete_audio(self, websocket, session: dict):
        """Process complete audio buffer - full pipeline"""
        try:
            audio_data = bytes(session["audio_buffer"])
            session["audio_buffer"].clear()

            metrics = self._analyze_pcm_metrics(
                audio_data,
                sample_rate=session.get("sample_rate", 16000),
                channels=session.get("channels", 1),
            )
            logger.info(
                "Turn audio stats: bytes=%d samples=%d duration=%.3fs sr=%s ch=%s rms=%.5f peak=%.4f clipping=%.3f%% zcr=%.4f input_gain=%.2f noise_floor=%.5f mic_device=%s",
                len(audio_data),
                int(metrics.get("samples", 0.0)),
                float(metrics.get("duration_sec", 0.0)),
                session.get("sample_rate", 16000),
                session.get("channels", 1),
                float(metrics.get("rms", 0.0)),
                float(metrics.get("peak", 0.0)),
                float(metrics.get("clipping_ratio", 0.0)) * 100.0,
                float(metrics.get("zero_crossing_rate", 0.0)),
                float(session.get("input_gain", 1.0)),
                float(session.get("noise_floor_rms", 0.0)),
                str(session.get("mic_device", "<unknown>")),
            )

            if int(session.get("sample_rate", 16000)) != 16000:
                logger.warning(
                    "Incoming stream is not 16kHz (%s Hz). Backend will resample to 16kHz for Whisper.",
                    session.get("sample_rate", 16000),
                )

            if float(metrics.get("rms", 0.0)) < 0.0045:
                logger.warning(
                    "Near-silence turn detected: rms=%.5f duration=%.3fs bytes=%d",
                    float(metrics.get("rms", 0.0)),
                    float(metrics.get("duration_sec", 0.0)),
                    len(audio_data),
                )
            
            min_bytes = int(max(600, session.get("sample_rate", 16000) * session.get("channels", 1) * 2 * 0.12))
            if len(audio_data) < min_bytes:
                logger.warning("Very short turn detected (%d bytes). Attempting STT anyway for fast command support.", len(audio_data))
                if len(audio_data) < 360:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Audio too short; hold push-to-talk a bit longer"
                    })
                    return
            
            # Normalize incoming stream bytes to a valid WAV payload for Whisper.
            # Unreal pushes PCM frames chunk-by-chunk, not a WAV container.
            wav_audio = self._ensure_wav_bytes(
                audio_data,
                sample_rate=session.get("sample_rate", 16000),
                channels=session.get("channels", 1),
            )
            self._save_input_wav_for_debug(
                wav_audio,
                session,
                float(metrics.get("duration_sec", 0.0)),
            )

            # Step 1: Speech-to-Text
            await websocket.send_json({"type": "status", "message": "Transcribing..."})
            text = await stt_service.transcribe_audio(
                wav_audio,
                language=session.get("language") or None,
            )

            if not text:
                logger.warning("WAV-normalized transcription returned empty; retrying direct PCM path")
                text = await stt_service.transcribe_audio(
                    audio_data,
                    language=session.get("language") or None,
                )

            stt_debug = {}
            if hasattr(stt_service, "get_last_result_debug"):
                try:
                    stt_debug = stt_service.get_last_result_debug() or {}
                except Exception:
                    stt_debug = {}

            if stt_debug:
                logger.info(
                    "STT debug: mode=%s wps=%.2f timed_words=%d uncertain=%d gaps=%d speech_dur=%.3fs params=%s",
                    str(stt_debug.get("decode_mode", "unknown")),
                    float(stt_debug.get("words_per_second", 0.0)),
                    int(stt_debug.get("timed_word_count", 0)),
                    int(stt_debug.get("uncertain_word_count", 0)),
                    int(stt_debug.get("long_gap_count", 0)),
                    float(stt_debug.get("speech_duration_sec", 0.0)),
                    str(stt_debug.get("decode_params", {})),
                )
                uncertain_words = stt_debug.get("uncertain_words") or []
                if uncertain_words:
                    logger.warning("STT uncertain words: %s", ", ".join([str(w) for w in uncertain_words[:10]]))
            
            if not text:
                fallback_text = "[inaudible]"
                await websocket.send_json({"type": "transcription", "text": fallback_text})
                await websocket.send_json({
                    "type": "status",
                    "message": "Low confidence transcription - asking user to repeat"
                })
                await self._process_text_message(
                    websocket,
                    session,
                    "User speech was unclear. Ask them politely to repeat in one short sentence.",
                )
                return
            
            # Send transcription to client
            await websocket.send_json({
                "type": "transcription",
                "text": text
            })
            
            # Step 2: Process text message
            await self._process_text_message(websocket, session, text)
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Processing failed: {str(e)}"
            })
    
    async def _process_text_message(self, websocket, session: dict, text: str):
        """Process text message - OPTIMIZED FOR LOW LATENCY with streaming"""
        try:
            # Step 1: Generate AI response (NON-STREAMING for simplicity with short responses)
            # For 1-2 sentence responses, streaming overhead isn't worth it
            await websocket.send_json({"type": "status", "message": "Thinking..."})
            
            try:
                rag_timeout = max(15.0, float(getattr(config, "RAG_TIMEOUT_SECONDS", 180)) + 30.0)
                ai_response_data = await asyncio.wait_for(
                    llm_service.generate_response(
                        text,
                        conversation_id=session.get("id"),
                        use_rag=None,
                        competency_level=session.get("competency_level"),
                    ),
                    timeout=rag_timeout,
                )
                ai_response = ai_response_data["message"]  # Extract just the text
            except asyncio.TimeoutError:
                logger.warning("LLM timeout after %.1fs; using fast fallback response", max(15.0, float(getattr(config, "RAG_TIMEOUT_SECONDS", 180)) + 30.0))
                ai_response = "I am ALIA. I can help with medical representative training."
            
            # Send AI text response IMMEDIATELY
            await websocket.send_json({
                "type": "ai_response",
                "text": ai_response
            })

            # Send expression state hint for client-side debugging/telemetry.
            try:
                expression = lipsync_service.analyze_expression_state(ai_response)
                await websocket.send_json({
                    "type": "expression_hint",
                    "state": expression.get("state", "neutral"),
                    "is_question": bool(expression.get("is_question", False)),
                    "has_emphasis": bool(expression.get("has_emphasis", False)),
                    "is_reassuring": bool(expression.get("is_reassuring", False)),
                    "has_caution": bool(expression.get("has_caution", False)),
                    "has_thinking": bool(expression.get("has_thinking", False)),
                    "has_surprise": bool(expression.get("has_surprise", False)),
                })
            except Exception as hint_error:
                logger.warning(f"Expression hint generation failed: {hint_error}")

            # SAFE MODE: bypass external TTS/Piper path entirely and stream a generated WAV.
            # This guarantees audio bytes are always produced even on machines where Piper fails.
            force_safe_audio = os.environ.get("ALIA_FORCE_SAFE_AUDIO", "0").strip().lower() in {"1", "true", "yes", "on"}
            if force_safe_audio:
                audio_bytes = self._build_safe_wav_bytes(ai_response)
                visemes = self._build_safe_visemes_from_wav(audio_bytes)

                await websocket.send_json({
                    "type": "visemes",
                    "data": visemes
                })

                await self._stream_response_fast(websocket, audio_bytes, "safe_generated.wav")
                return
            
            # Step 2: Generate TTS audio with retry/fallback text for robustness.
            audio_file = await self._generate_tts_with_fallback(ai_response)

            if not audio_file:
                logger.warning("TTS failed completely; generating emergency audio fallback")
                audio_file = await self._generate_emergency_audio(ai_response)

            if not audio_file:
                raise Exception("TTS generation failed after fallback")
            
            # Get full path to audio file
            audio_path = tts_service.output_dir / audio_file
            
            # Read audio file
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            
            # Step 3: Prefer audio-aligned visemes for natural mouth motion.
            # If extraction takes too long, fall back to fast realtime visemes.
            try:
                visemes = await asyncio.wait_for(
                    lipsync_service.extract_visemes(str(audio_path), text=ai_response),
                    timeout=2.5,
                )
                logger.info(f"Generated {len(visemes)} quality visemes")
            except asyncio.TimeoutError:
                logger.warning("Quality viseme extraction timed out; using realtime fallback")
                visemes = await lipsync_service.extract_visemes_realtime(str(audio_path), text=ai_response)
            except Exception as e:
                logger.error(f"Lip sync generation failed: {e}")
                visemes = []

            # Ensure viseme track spans the full audio duration so mouth motion
            # does not stop before playback ends.
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                    audio_duration = wav_file.getnframes() / float(wav_file.getframerate())
                if not visemes or visemes[-1].get("time", 0.0) < (audio_duration - 0.02):
                    visemes.append({"time": round(audio_duration, 4), "morph": "A25_Jaw_Open", "value": 0.0})
                    visemes.append({"time": round(audio_duration, 4), "morph": "V_Open", "value": 0.0})
            except Exception:
                pass
            
            # Send visemes BEFORE audio
            await websocket.send_json({
                "type": "visemes",
                "data": visemes
            })
            
            # Step 4: Now stream audio (visemes already received by client)
            await self._stream_response_fast(websocket, audio_bytes, audio_file)
            
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to generate response: {str(e)}"
            })

    def _build_safe_wav_bytes(self, text: str) -> bytes:
        """Generate a short WAV tone in memory as guaranteed audio output."""
        sample_rate = 22050
        duration_sec = max(0.8, min(2.4, 0.4 + (len(text or "") / 50.0)))
        frequency = 440.0
        amplitude = 0.2
        total_samples = int(sample_rate * duration_sec)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            frames = bytearray()
            for n in range(total_samples):
                value = amplitude * math.sin(2.0 * math.pi * frequency * (n / sample_rate))
                sample = int(max(-1.0, min(1.0, value)) * 32767)
                frames.extend(struct.pack("<h", sample))
            wav_file.writeframes(frames)

        return buffer.getvalue()

    def _build_safe_visemes_from_wav(self, wav_bytes: bytes):
        """Create simple jaw-open visemes matched to generated safe audio duration."""
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                duration = wav_file.getnframes() / float(wav_file.getframerate())
        except Exception:
            duration = 1.0

        visemes = []
        t = 0.0
        step = 0.08
        toggle = False
        while t < duration:
            jaw = 0.45 if toggle else 0.18
            visemes.append({"time": round(t, 4), "morph": "A25_Jaw_Open", "value": jaw})
            visemes.append({"time": round(t, 4), "morph": "V_Open", "value": round(jaw * 0.7, 4)})
            toggle = not toggle
            t += step

        visemes.append({"time": round(duration, 4), "morph": "A25_Jaw_Open", "value": 0.0})
        visemes.append({"time": round(duration, 4), "morph": "V_Open", "value": 0.0})
        return visemes

    async def _generate_tts_with_fallback(self, ai_response: str):
        """Try full response TTS first, then retry with shorter sanitized text."""
        force_emergency = os.environ.get("ALIA_FORCE_EMERGENCY_TTS", "").strip().lower() in {"1", "true", "yes", "on"}
        if force_emergency:
            logger.warning("ALIA_FORCE_EMERGENCY_TTS is enabled; skipping primary TTS service")
            return await self._generate_emergency_audio(ai_response)

        # Primary attempt with full model response.
        try:
            return await tts_service.text_to_speech(ai_response)
        except Exception as primary_error:
            logger.warning(f"Primary TTS failed, retrying with fallback text: {primary_error}")

        # Fallback 1: first sentence only.
        fallback_text = (ai_response or "").strip().replace("\r", " ").replace("\n", " ")
        if not fallback_text:
            fallback_text = "Hello. I am ALIA. How can I help you today?"

        sentence = fallback_text.split(".")[0].strip()
        if sentence:
            fallback_text = sentence + "."

        # Keep fallback prompt short and stable for Piper.
        fallback_text = " ".join(fallback_text.split())
        fallback_text = fallback_text[:180] if len(fallback_text) > 180 else fallback_text

        try:
            return await tts_service.text_to_speech(fallback_text)
        except Exception as secondary_error:
            logger.error(f"Fallback TTS failed: {secondary_error}")
            logger.warning("Falling back to emergency audio generation")
            return await self._generate_emergency_audio(fallback_text)

    async def _generate_emergency_audio(self, text: str):
        """Last-resort audio generation so client receives audio even if Piper fails."""
        output_dir = None
        if tts_service is not None and getattr(tts_service, "output_dir", None):
            output_dir = tts_service.output_dir
        else:
            output_dir = Path(__file__).parent.parent / "audio_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Try Windows built-in SpeechSynthesizer directly.
        try:
            output_name = f"speech_emergency_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            output_path = output_dir / output_name

            safe_text = (text or "Hello. I am ALIA.").replace('"', "'")
            safe_text = safe_text.replace("ALIA", "Alia")
            safe_text = " ".join(safe_text.replace("\r", " ").replace("\n", " ").split())[:180]

            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$voices = $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }; "
                "$preferred = @('Aria','Jenny','Zira','Sonia','Eva','David'); "
                "foreach($p in $preferred){ $m = $voices | Where-Object { $_ -like ('*' + $p + '*') } | Select-Object -First 1; if($m){ $s.SelectVoice($m); break } }; "
                "$s.Rate = -1; $s.Volume = 100; "
                f"$s.SetOutputToWaveFile(\"{str(output_path)}\"); "
                f"$s.Speak(\"{safe_text}\"); "
                "$s.Dispose();"
            )

            proc = await asyncio.create_subprocess_exec(
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 44:
                logger.info(f"Emergency SAPI audio generated: {output_name}")
                return output_name

            logger.warning(
                f"Emergency SAPI generation failed code={proc.returncode} stderr={stderr.decode('utf-8', errors='ignore')[:240]}"
            )
        except Exception as e:
            logger.warning(f"Emergency SAPI exception: {e}")

        # 2) Guaranteed tone WAV fallback.
        try:
            output_name = f"speech_tone_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            output_path = output_dir / output_name

            sample_rate = 22050
            duration_sec = 1.2
            frequency = 440.0
            amplitude = 0.2
            total_samples = int(sample_rate * duration_sec)

            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)

                frames = bytearray()
                for n in range(total_samples):
                    value = amplitude * math.sin(2.0 * math.pi * frequency * (n / sample_rate))
                    sample = int(max(-1.0, min(1.0, value)) * 32767)
                    frames.extend(struct.pack("<h", sample))
                wav_file.writeframes(frames)

            if output_path.exists() and output_path.stat().st_size > 44:
                logger.info(f"Emergency tone audio generated: {output_name}")
                return output_name
        except Exception as e:
            logger.error(f"Emergency tone generation failed: {e}")

        return None
    
    async def _generate_and_send_lipsync(self, websocket, audio_path: str):
        """Generate and send lip sync data in background (non-blocking)"""
        try:
            visemes = await lipsync_service.extract_visemes(audio_path)
            
            # Send visemes when ready (audio already streaming)
            await websocket.send_json({
                "type": "visemes",
                "data": visemes
            })
            
            logger.info(f"Sent {len(visemes)} visemes (async)")
            
        except Exception as e:
            logger.error(f"Async lip sync error: {e}")
            # Don't fail - just skip lip sync
    
    async def _stream_response(self, websocket, audio_bytes: bytes, 
                               visemes: list, audio_filename: str):
        """Stream audio and visemes to client in chunks"""
        try:
            # Send metadata first
            await websocket.send_json({
                "type": "response_start",
                "audio_size": len(audio_bytes),
                "viseme_count": len(visemes),
                "audio_file": audio_filename
            })
            
            # Send visemes (small, send all at once)
            await websocket.send_json({
                "type": "visemes",
                "data": visemes
            })
            
            # Stream audio in chunks (16KB chunks for low latency)
            chunk_size = 16384  # 16KB
            total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                
                chunk_num = i // chunk_size
                
                await websocket.send_json({
                    "type": "audio_chunk",
                    "chunk": chunk_num,
                    "total_chunks": total_chunks,
                    "data": chunk_b64,
                    "is_last": (i + chunk_size >= len(audio_bytes))
                })
                
                # Small delay to avoid overwhelming client
                await asyncio.sleep(0.01)
            
            # Send completion
            await websocket.send_json({
                "type": "response_complete",
                "message": "Audio and visemes sent"
            })
            
            logger.info(f"Streamed {len(audio_bytes)} bytes audio + {len(visemes)} visemes")
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
    
    async def _stream_response_fast(self, websocket, audio_bytes: bytes, audio_filename: str):
        """OPTIMIZED: Stream audio immediately without waiting for lip sync"""
        try:
            # Send metadata first
            await websocket.send_json({
                "type": "response_start",
                "audio_size": len(audio_bytes),
                "audio_file": audio_filename
            })
            
            # Use larger chunks to reduce overhead and improve continuity.
            chunk_size = 32768
            total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                
                chunk_num = i // chunk_size
                
                await websocket.send_json({
                    "type": "audio_chunk",
                    "chunk": chunk_num,
                    "total_chunks": total_chunks,
                    "data": chunk_b64,
                    "is_last": (i + chunk_size >= len(audio_bytes))
                })
                
                # No artificial delay: let the socket/UE queue naturally.
            
            # Send completion
            await websocket.send_json({
                "type": "response_complete",
                "message": "Audio streaming complete"
            })
            
            logger.info(f"Fast-streamed {len(audio_bytes)} bytes audio")
            
        except Exception as e:
            logger.error(f"Fast streaming error: {e}")
            raise


# Singleton instance
realtime_voice_service = RealtimeVoiceService()
