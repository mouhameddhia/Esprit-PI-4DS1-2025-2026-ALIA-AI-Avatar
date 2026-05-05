"""
TTS Service - Piper Text-to-Speech Integration
Converts text to speech using offline Piper TTS
"""

import asyncio
import logging
import subprocess
import os
from pathlib import Path
from typing import Optional
import uuid
from datetime import datetime
import wave
import math
import struct

from utils.config import config

logger = logging.getLogger(__name__)

class TTSService:
    """Service for text-to-speech using Piper"""
    
    def __init__(self):
        self.piper_executable = config.PIPER_EXECUTABLE
        self.voice_model = config.PIPER_VOICE_MODEL
        self.output_dir = Path(config.AUDIO_OUTPUT_DIR)
        self.sample_rate = 22050  # Piper default
        self._windows_tts_fallback = True
        # Piper crashes on this machine (0xc0000142). Use Windows TTS first.
        self._prefer_windows_tts = os.name == "nt"
        # Keep Piper disabled on Windows unless explicitly re-enabled in code.
        env_disable = os.environ.get("ALIA_DISABLE_PIPER", "").strip().lower()
        self._disable_piper_on_windows = (os.name == "nt") or (env_disable in {"1", "true", "yes", "on"})
        
    async def initialize(self):
        """Initialize TTS service"""
        try:
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Audio output directory: {self.output_dir}")

            if self._disable_piper_on_windows:
                logger.warning("Piper is disabled on Windows due to startup crash (0x0000142). Using Windows TTS fallback.")
                return
            
            # Check if Piper executable exists
            if not Path(self.piper_executable).exists():
                logger.error(f"Piper executable not found at: {self.piper_executable}")
                logger.info("Please download Piper from: https://github.com/rhasspy/piper/releases")
                return
            
            # Check if voice model exists
            if not Path(self.voice_model).exists():
                logger.error(f"Voice model not found at: {self.voice_model}")
                logger.info("Please download a voice model from: https://github.com/rhasspy/piper/releases")
                return
                
            logger.info(f"Piper TTS initialized with model: {self.voice_model}")
            
        except Exception as e:
            logger.error(f"Error initializing TTS service: {e}")
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        if self._disable_piper_on_windows:
            return True

        return (
            Path(self.piper_executable).exists() and 
            Path(self.voice_model).exists()
        )
    
    async def text_to_speech(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Convert text to speech and save as WAV file
        
        Args:
            text: Text to convert
            conversation_id: Optional conversation ID for filename
            output_filename: Optional custom filename
            
        Returns:
            Filename of generated audio file
        """
        try:
            start_time = datetime.now()

            # Hard safe mode: bypass Piper/SAPI and always generate a guaranteed local WAV.
            force_emergency = os.environ.get("ALIA_FORCE_EMERGENCY_TTS", "").strip().lower() in {"1", "true", "yes", "on"}

            # Generate filename
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                conv_id = conversation_id[:8] if conversation_id else str(uuid.uuid4())[:8]
                output_filename = f"speech_{conv_id}_{timestamp}.wav"

            output_path = self.output_dir / output_filename

            if force_emergency:
                tone_ok = await self._generate_tone_fallback(output_path)
                if tone_ok:
                    duration = (datetime.now() - start_time).total_seconds()
                    audio_duration = self._get_audio_duration(str(output_path))
                    logger.warning(
                        f"[ALIA_TTS_SAFE_MODE] Generated emergency tone in {duration:.2f}s (audio duration: {audio_duration:.2f}s): {output_filename}"
                    )
                    return output_filename
                raise Exception("ALIA_TTS_SAFE_MODE failed to generate emergency WAV")

            # Prepare text variants for retries.
            base_text = str(text) if not isinstance(text, str) else text
            base_text = " ".join(base_text.replace("\r", " ").replace("\n", " ").split())
            if not base_text:
                base_text = "Hello."

            retry_texts = [
                base_text,
                base_text[:220],
                "Hello. I am ALIA. How can I help you today?",
            ]

            # Primary path on Windows: avoid Piper launch issues by using SAPI directly.
            if self._prefer_windows_tts and self._windows_tts_fallback:
                sapi_ok = await self._run_windows_tts_fallback(base_text, output_path)
                if sapi_ok:
                    duration = (datetime.now() - start_time).total_seconds()
                    audio_duration = self._get_audio_duration(str(output_path))
                    logger.info(
                        f"Speech generated via SAPI primary in {duration:.2f}s (audio duration: {audio_duration:.2f}s): {output_filename}"
                    )
                    return output_filename

                tone_ok = await self._generate_tone_fallback(output_path)
                if tone_ok:
                    duration = (datetime.now() - start_time).total_seconds()
                    audio_duration = self._get_audio_duration(str(output_path))
                    logger.warning(
                        f"SAPI failed; generated tone fallback in {duration:.2f}s (audio duration: {audio_duration:.2f}s): {output_filename}"
                    )
                    return output_filename

                if self._disable_piper_on_windows:
                    raise Exception("Windows TTS failed and Piper is disabled on this machine")

                logger.warning("SAPI primary synthesis failed; falling back to Piper retries")

            last_error = None
            for attempt, attempt_text in enumerate(retry_texts, start=1):
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except Exception:
                        pass

                logger.info(
                    f"Generating speech attempt {attempt}/{len(retry_texts)} for text: {attempt_text[:80]}..."
                )

                return_code, stdout_text, stderr_text = await self._run_piper_once(attempt_text, output_path)

                # Retry if Piper failed.
                if return_code != 0:
                    last_error = (
                        f"Piper exit code {return_code}. stderr={stderr_text[:400]} stdout={stdout_text[:200]}"
                    )
                    logger.warning(last_error)
                    await asyncio.sleep(0.2)
                    continue

                # Wait briefly for file flush/visibility on Windows.
                for _ in range(5):
                    if output_path.exists() and output_path.stat().st_size > 44:
                        duration = (datetime.now() - start_time).total_seconds()
                        audio_duration = self._get_audio_duration(str(output_path))
                        logger.info(
                            f"Speech generated in {duration:.2f}s (audio duration: {audio_duration:.2f}s): {output_filename}"
                        )
                        return output_filename
                    await asyncio.sleep(0.1)

                last_error = (
                    f"Piper returned success but output file is missing/empty at {output_path}. "
                    f"stderr={stderr_text[:300]}"
                )
                logger.warning(last_error)

            # Final fallback on Windows: built-in SpeechSynthesizer.
            if self._windows_tts_fallback:
                logger.warning("Piper failed; trying Windows SAPI fallback synthesis")
                fallback_ok = await self._run_windows_tts_fallback(base_text, output_path)
                if fallback_ok:
                    duration = (datetime.now() - start_time).total_seconds()
                    audio_duration = self._get_audio_duration(str(output_path))
                    logger.info(
                        f"Speech generated via SAPI fallback in {duration:.2f}s (audio duration: {audio_duration:.2f}s): {output_filename}"
                    )
                    return output_filename

                tone_ok = await self._generate_tone_fallback(output_path)
                if tone_ok:
                    duration = (datetime.now() - start_time).total_seconds()
                    audio_duration = self._get_audio_duration(str(output_path))
                    logger.warning(
                        f"Piper+SAPI failed; generated tone fallback in {duration:.2f}s (audio duration: {audio_duration:.2f}s): {output_filename}"
                    )
                    return output_filename

            # Final hard fallback before giving up.
            tone_ok = await self._generate_tone_fallback(output_path)
            if tone_ok:
                duration = (datetime.now() - start_time).total_seconds()
                audio_duration = self._get_audio_duration(str(output_path))
                logger.warning(
                    f"Final hard fallback tone generated in {duration:.2f}s (audio duration: {audio_duration:.2f}s): {output_filename}"
                )
                return output_filename

            raise Exception(last_error or "Audio file was not generated")

        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}", exc_info=True)
            raise

    async def _run_piper_once(self, text: str, output_path: Path):
        """Run one Piper synthesis attempt and return (returncode, stdout, stderr)."""
        command = [
            self.piper_executable,
            "--model", self.voice_model,
            "--output_file", str(output_path),
        ]

        import concurrent.futures

        loop = asyncio.get_event_loop()

        def run_piper():
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate(input=text.encode("utf-8"))
            return process.returncode, stdout, stderr

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return_code, stdout, stderr = await loop.run_in_executor(pool, run_piper)

        stdout_text = stdout.decode("utf-8", errors="ignore") if stdout else ""
        stderr_text = stderr.decode("utf-8", errors="ignore") if stderr else ""
        return return_code, stdout_text, stderr_text

    async def _run_windows_tts_fallback(self, text: str, output_path: Path) -> bool:
        """Generate WAV using Windows built-in SpeechSynthesizer via PowerShell."""
        try:
            import concurrent.futures

            safe_text = text.replace('"', "'")
            safe_text = safe_text.replace("ALIA", "Alia")
            safe_path = str(output_path).replace('"', "")

            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$voices = $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }; "
                "$preferred = @('Aria','Jenny','Zira','Sonia','Eva','David'); "
                "foreach($p in $preferred){ $m = $voices | Where-Object { $_ -like ('*' + $p + '*') } | Select-Object -First 1; if($m){ $s.SelectVoice($m); break } }; "
                "$s.Rate = -1; "
                "$s.Volume = 100; "
                f"$s.SetOutputToWaveFile(\"{safe_path}\"); "
                f"$s.Speak(\"{safe_text}\"); "
                "$s.Dispose();"
            )

            command = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps_script,
            ]

            loop = asyncio.get_event_loop()

            def run_ps():
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = process.communicate(timeout=30)
                return process.returncode, stdout, stderr

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return_code, stdout, stderr = await loop.run_in_executor(pool, run_ps)

            if return_code != 0:
                stderr_text = stderr.decode("utf-8", errors="ignore") if stderr else ""
                logger.error(f"Windows SAPI fallback failed: {stderr_text[:400]}")
                return False

            return output_path.exists() and output_path.stat().st_size > 44

        except Exception as e:
            logger.error(f"Windows SAPI fallback exception: {e}")
            return False

    async def _generate_tone_fallback(self, output_path: Path) -> bool:
        """Generate a short WAV tone as a guaranteed last-resort fallback."""
        try:
            sample_rate = self.sample_rate
            duration_sec = 0.9
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

            return output_path.exists() and output_path.stat().st_size > 44
        except Exception as e:
            logger.error(f"Tone fallback generation failed: {e}")
            return False
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of WAV file in seconds"""
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            return 0.0
    
    async def text_to_speech_streaming(self, text: str):
        """
        Stream audio generation (for future implementation)
        
        Piper doesn't natively support streaming, but this could be
        implemented by:
        1. Chunking text into sentences
        2. Generating each chunk separately
        3. Streaming chunks to client
        """
        # TODO: Implement streaming TTS
        pass
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("TTS service cleaned up")
