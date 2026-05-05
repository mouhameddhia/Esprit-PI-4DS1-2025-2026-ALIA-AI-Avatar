"""
Lip Sync Service - CC4 V_ Viseme Approach
Uses Rhubarb Lip Sync mapped to Reallusion CC4 native viseme morphs.
CC4 V_ morphs are artist-designed complete mouth shapes - one morph = one phoneme pose.
This is far more reliable than blending 6+ ARKit morphs with guessed weights.
"""

import logging
import wave
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Optional
import json
import asyncio
import subprocess
import concurrent.futures

from utils.config import config

logger = logging.getLogger(__name__)


# ============================================================
# CC4 NATIVE VISEME MAPPING
# Each Rhubarb shape maps to ONE primary CC4 V_ morph + jaw open
# V_ morphs are complete mouth shapes designed by Reallusion
# ============================================================

RHUBARB_TO_CC4 = {
    # Calibrated from ALIA1_1__Lipsync_F_Motion extracted curve peaks.
    # Keep mapping mostly on CC4 V_ curves to match authored animation style.

    # A = open vowel (ah, father)
    'A': {'V_Open': 0.64, 'A25_Jaw_Open': 0.18},

    # B = bilabial (m, b, p)
    'B': {'V_Explosive': 0.50, 'Mouth_Plosive': 0.18},

    # C = mid vowel (eh, bed)
    'C': {'V_Wide': 0.68, 'A25_Jaw_Open': 0.12},

    # D = front vowel (ee, teeth)
    'D': {'V_Wide': 0.94, 'A25_Jaw_Open': 0.08},

    # E = rounded vowel (oh, go)
    'E': {'V_Tight_O': 0.80, 'Mouth_Pucker_Open': 0.18, 'A25_Jaw_Open': 0.10},

    # F = close rounded (oo, too)
    'F': {'V_Lip_Open': 0.98, 'A25_Jaw_Open': 0.06},

    # G = labiodental (f, v)
    'G': {'V_Dental_Lip': 0.50, 'A25_Jaw_Open': 0.08},

    # H = alveolar (l, d, t)
    'H': {'V_Tight': 0.95, 'A25_Jaw_Open': 0.10},

    # X = rest / silence
    'X': {},
}

# Calm speaking style profile: smoother transitions and less aggressive peaks.
CALM_STYLE = {
    "peak_scale": 0.82,
    "onset_scale": 0.45,
    "hold_scale": 0.72,
    "min_jaw_rest": 0.05,
    "min_lips_rest": 0.14,
    "min_mouth_open_rest": 0.05,
    "lips_part_floor": 0.20,
    "mouth_open_floor": 0.12,
    "lips_part_peak_scale": 0.58,
    "mouth_open_peak_scale": 0.42,
    "teeth_hint": 0.22,
    "attack_max": 0.06,
    "attack_ratio": 0.32,
    "release_tail": 0.02,
    "coarticulation_blend": 0.16,
    "plosive_attack_scale": 0.72,
    "plosive_peak_boost": 1.10,
    "vowel_hold_boost": 1.08,
    "formal_expression_scale": 0.90,
    "rapid_chars_per_sec": 14.0,
    "rapid_scale_floor": 0.78,
    "comma_pause_sec": 0.035,
    "sentence_pause_sec": 0.075,
}

# Nonverbal facial behavior tuned for realistic professional trainer delivery.
NONVERBAL_STYLE = {
    "blink_min_interval": 2.6,
    "blink_max_interval": 4.2,
    "blink_close": 0.76,
    "blink_half": 0.22,
    "blink_duration": 0.14,
    "brow_neutral": 0.04,
    "brow_explain": 0.08,
    "brow_question": 0.14,
    "smile_reassure": 0.10,
    "smile_emphasis": 0.06,
}

EXPRESSION_PROFILES = {
    "neutral": {
        "brow_inner": 0.07,
        "brow_down": 0.00,
        "smile": 0.02,
        "blink_scale": 1.00,
    },
    "reassuring": {
        "brow_inner": 0.09,
        "brow_down": 0.00,
        "smile": 0.11,
        "blink_scale": 1.05,
    },
    "thinking": {
        "brow_inner": 0.05,
        "brow_down": 0.05,
        "smile": 0.00,
        "blink_scale": 0.90,
    },
    "caution": {
        "brow_inner": 0.03,
        "brow_down": 0.08,
        "smile": 0.00,
        "blink_scale": 0.85,
    },
    "surprise": {
        "brow_inner": 0.16,
        "brow_down": 0.00,
        "smile": 0.03,
        "blink_scale": 0.75,
    },
    "question": {
        "brow_inner": 0.13,
        "brow_down": 0.00,
        "smile": 0.02,
        "blink_scale": 0.95,
    },
}

# Text-to-viseme mapping for when we know the text (more accurate than audio analysis)
# Maps character patterns to Rhubarb-equivalent shapes
LETTER_TO_SHAPE = {
    # Vowels
    'a': 'A', 'e': 'C', 'i': 'D', 'o': 'E', 'u': 'F',
    # Bilabials
    'b': 'B', 'm': 'B', 'p': 'B',
    # Labiodentals
    'f': 'G', 'v': 'G',
    # Wide/smile consonants
    's': 'D', 'z': 'D',
    # Tongue/alveolar
    'l': 'H', 'd': 'H', 't': 'H', 'n': 'H',
    # Velar/back
    'k': 'E', 'g': 'E',
    # Affricates
    'j': 'D', 'c': 'C',
    # Rounded
    'w': 'F', 'r': 'E',
    # Others
    'h': 'A', 'y': 'D', 'x': 'H', 'q': 'E',
}


class LipSyncService:
    """Lip sync using CC4 native V_ visemes for reliable results"""

    def __init__(self):
        self._rhubarb_available = False

    async def initialize(self):
        if config.USE_RHUBARB and config.RHUBARB_EXECUTABLE:
            rhubarb_path = Path(config.RHUBARB_EXECUTABLE)
            self._rhubarb_available = rhubarb_path.exists()
            if self._rhubarb_available:
                logger.info(f"Rhubarb available: {rhubarb_path}")
            else:
                logger.warning(f"Rhubarb not found: {rhubarb_path}")
        logger.info("Lip sync service initialized (CC4 V_ viseme mode)")

    def is_ready(self) -> bool:
        return True

    def analyze_expression_state(self, text: str) -> Dict[str, object]:
        """Public helper for realtime service to inspect the current expression state."""
        normalized = (text or "").strip().lower()
        return self._analyze_expression_intent(normalized)

    async def extract_visemes(self, audio_path: str, text: str = "") -> List[Dict]:
        """
        Extract visemes. Priority:
        1. Rhubarb (phoneme-accurate from audio)
        2. Text-based (if text provided, estimate from letters)
        3. Energy-based fallback
        """
        try:
            logger.info(f"Extracting visemes from: {audio_path}")

            # Get audio duration for all methods
            audio_duration = self._get_audio_duration(audio_path)

            # Try Rhubarb first (best quality)
            if self._rhubarb_available and config.USE_RHUBARB:
                result = await self._extract_rhubarb(audio_path)
                if result:
                    if text:
                        result = self._apply_text_style_polish(result, text, audio_duration)
                    logger.info(f"Rhubarb: {len(result)} keyframes for {audio_duration:.2f}s audio")
                    return result
                logger.warning("Rhubarb failed, trying text-based")

            # Text-based (good for TTS since we know exact text)
            if text:
                result = self._extract_from_text(text, audio_duration)
                if result:
                    result = self._apply_text_style_polish(result, text, audio_duration)
                    logger.info(f"Text-based: {len(result)} keyframes for '{text[:30]}...'")
                    return result

            # Energy fallback
            return await self._extract_energy(audio_path)

        except Exception as e:
            logger.error(f"Viseme extraction error: {e}", exc_info=True)
            return []

    async def extract_visemes_realtime(self, audio_path: str, text: str = "") -> List[Dict]:
        """
        Low-latency viseme path for live conversations.
        Prefers text-based keyframes to avoid Rhubarb processing delay.
        """
        try:
            audio_duration = self._get_audio_duration(audio_path)

            if text:
                result = self._extract_from_text(text, audio_duration)
                if result:
                    result = self._apply_text_style_polish(result, text, audio_duration)
                    logger.info(f"Realtime visemes (text-based): {len(result)} keyframes")
                    return result

            # If no text is provided, fall back to energy-based extraction.
            return await self._extract_energy(audio_path)
        except Exception as e:
            logger.error(f"Realtime viseme extraction error: {e}", exc_info=True)
            return []

    # ============================================================
    # RHUBARB (PRIMARY - BEST QUALITY)
    # ============================================================

    async def _extract_rhubarb(self, audio_path: str) -> Optional[List[Dict]]:
        try:
            command = [
                str(config.RHUBARB_EXECUTABLE),
                audio_path,
                "-f", "json",
                "--extendedShapes", "GHX",
            ]

            loop = asyncio.get_event_loop()

            def run():
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                return proc.communicate(timeout=30)

            with concurrent.futures.ThreadPoolExecutor() as pool:
                stdout, stderr = await loop.run_in_executor(pool, run)

            if not stdout:
                logger.error(f"Rhubarb no output. stderr: {stderr.decode()}")
                return None

            result = json.loads(stdout.decode())
            cues = result.get("mouthCues", [])
            if not cues:
                return None

            return self._convert_rhubarb(cues)

        except Exception as e:
            logger.error(f"Rhubarb error: {e}")
            return None

    def _convert_rhubarb(self, cues: List[Dict]) -> List[Dict]:
        """Convert Rhubarb cues to CC4 V_ viseme keyframes"""
        visemes = []

        for i, cue in enumerate(cues):
            shape = cue.get("value", "X")
            start = float(cue.get("start", 0))
            end = float(cue.get("end", start + 0.05))
            duration = max(0.01, end - start)
            base_morphs = RHUBARB_TO_CC4.get(shape, {})

            next_shape = "X"
            if i + 1 < len(cues):
                next_shape = str(cues[i + 1].get("value", "X"))

            next_morphs = RHUBARB_TO_CC4.get(next_shape, {})
            morphs = self._apply_coarticulation_blend(base_morphs, next_morphs)
            morphs = self._merge_support_morphs(morphs)

            if not morphs:
                # Rest shape - zero everything
                visemes.append({"time": round(start, 4), "morph": "A25_Jaw_Open", "value": 0.0})
                visemes.append({"time": round(start, 4), "morph": "V_Open", "value": 0.0})
                visemes.append({"time": round(start, 4), "morph": "Mouth_Lips_Part", "value": 0.0})
                visemes.append({"time": round(start, 4), "morph": "Mouth_Open", "value": 0.0})
                visemes.append({"time": round(start, 4), "morph": "Mouth_Plosive", "value": 0.0})
                visemes.append({"time": round(start, 4), "morph": "Mouth_Pucker_Open", "value": 0.0})
                continue

            # Calm delivery: slower ramp and softer peak than energetic speech.
            attack = min(CALM_STYLE["attack_max"], duration * CALM_STYLE["attack_ratio"])
            peak_scale = CALM_STYLE["peak_scale"]
            onset_scale = CALM_STYLE["onset_scale"]
            hold_scale = CALM_STYLE["hold_scale"]

            # Reduce over-articulation for very short/rapid cues.
            if duration < 0.055:
                peak_scale *= 0.84
                hold_scale *= 0.78

            # Shape-aware timing/intensity for more realistic articulation.
            if shape == 'B':
                attack = max(0.012, attack * CALM_STYLE["plosive_attack_scale"])
                peak_scale = min(1.0, peak_scale * CALM_STYLE["plosive_peak_boost"])
            elif self._is_vowel_shape(shape):
                hold_scale = min(1.0, hold_scale * CALM_STYLE["vowel_hold_boost"])

            # Peak keyframes
            peak_time = round(start + attack, 4)
            for morph_name, weight in morphs.items():
                start_weight = max(0.0, min(1.0, weight * onset_scale))
                peak_weight = max(0.0, min(1.0, weight * peak_scale))

                # Start with a gentle onset for calm speech.
                visemes.append({
                    "time": round(start, 4),
                    "morph": morph_name,
                    "value": round(start_weight, 4),
                })
                # Peak
                visemes.append({
                    "time": peak_time,
                    "morph": morph_name,
                    "value": round(peak_weight, 4),
                })

            # Hold for long sounds
            if duration > 0.15:
                hold_time = round(start + duration * 0.7, 4)
                for morph_name, weight in morphs.items():
                    hold_weight = max(0.0, min(1.0, weight * hold_scale))
                    visemes.append({
                        "time": hold_time,
                        "morph": morph_name,
                        "value": round(hold_weight, 4),
                    })

            # Release - ramp down at end of cue
            for morph_name, weight in morphs.items():
                # Keep jaw slightly open between speech sounds
                if 'Jaw' in morph_name:
                    rest_val = CALM_STYLE["min_jaw_rest"]
                elif 'Lips_Part' in morph_name:
                    rest_val = CALM_STYLE["min_lips_rest"]
                elif 'Mouth_Open' in morph_name:
                    rest_val = CALM_STYLE["min_mouth_open_rest"]
                else:
                    rest_val = 0.0
                # But if next cue is NOT rest, keep some openness
                if i + 1 < len(cues) and cues[i + 1].get("value") != "X":
                    if 'Jaw' in morph_name:
                        rest_val = max(rest_val, 0.07)
                    elif 'Lips_Part' in morph_name:
                        rest_val = max(rest_val, 0.16)
                    elif 'Mouth_Open' in morph_name:
                        rest_val = max(rest_val, 0.08)
                visemes.append({
                    "time": round(end + CALM_STYLE["release_tail"], 4),
                    "morph": morph_name,
                    "value": rest_val,
                })

        # Final rest
        if cues:
            final = float(cues[-1].get("end", 0)) + 0.08
            visemes.append({"time": round(final, 4), "morph": "A25_Jaw_Open", "value": 0.0})
            visemes.append({"time": round(final, 4), "morph": "Mouth_Lips_Part", "value": 0.0})
            visemes.append({"time": round(final, 4), "morph": "Mouth_Open", "value": 0.0})

        visemes.sort(key=lambda v: v["time"])
        return visemes

    # ============================================================
    # TEXT-BASED (for when we know what's being said)
    # ============================================================

    def _extract_from_text(self, text: str, audio_duration: float) -> List[Dict]:
        """
        Generate visemes from text. Since we use TTS, we KNOW the exact text.
        This gives better timing than audio analysis because:
        - Each letter maps to a known mouth shape
        - Duration is proportional to audio length
        """
        if not text or audio_duration <= 0:
            return []

        # Clean text -> only letters and spaces
        clean = ''.join(c.lower() for c in text if c.isalpha() or c == ' ')
        if not clean:
            return []

        # Calculate time per character (rough but effective for TTS)
        total_chars = len(clean.replace(' ', ''))
        if total_chars == 0:
            return []

        # Leave 0.1s margin at start/end
        usable_duration = max(audio_duration - 0.2, audio_duration * 0.8)
        time_per_char = usable_duration / total_chars

        visemes = []
        current_time = 0.1  # Start offset
        prev_shape = None

        for idx, char in enumerate(clean):
            if char == ' ':
                # Brief pause between words
                visemes.append({
                    "time": round(current_time, 4),
                    "morph": "A25_Jaw_Open",
                    "value": CALM_STYLE["min_jaw_rest"],
                })
                visemes.append({
                    "time": round(current_time, 4),
                    "morph": "Mouth_Lips_Part",
                    "value": CALM_STYLE["min_lips_rest"],
                })
                visemes.append({
                    "time": round(current_time, 4),
                    "morph": "Mouth_Open",
                    "value": CALM_STYLE["min_mouth_open_rest"],
                })
                current_time += time_per_char * 0.5
                prev_shape = None
                continue

            shape = LETTER_TO_SHAPE.get(char, 'C')
            base_morphs = RHUBARB_TO_CC4.get(shape, {})

            next_shape = self._next_text_shape(clean, idx)
            next_morphs = RHUBARB_TO_CC4.get(next_shape, {})
            morphs = self._apply_coarticulation_blend(base_morphs, next_morphs)
            morphs = self._merge_support_morphs(morphs)

            if not morphs:
                current_time += time_per_char
                prev_shape = shape
                continue

            # Calm style envelope for each phoneme segment.
            segment = max(time_per_char, 0.04)
            attack_t = current_time
            peak_t = current_time + min(CALM_STYLE["attack_max"], segment * CALM_STYLE["attack_ratio"])
            release_t = current_time + segment

            local_peak_scale = CALM_STYLE["peak_scale"]
            if shape == 'B':
                peak_t = current_time + min(0.03, segment * 0.22)
                local_peak_scale = min(1.0, local_peak_scale * CALM_STYLE["plosive_peak_boost"])

            if shape != prev_shape:
                for morph_name, weight in morphs.items():
                    peak_weight = max(0.0, min(1.0, weight * local_peak_scale))
                    start_weight = max(0.0, min(1.0, weight * CALM_STYLE["onset_scale"]))

                    visemes.append({
                        "time": round(attack_t, 4),
                        "morph": morph_name,
                        "value": round(start_weight, 4),
                    })
                    visemes.append({
                        "time": round(peak_t, 4),
                        "morph": morph_name,
                        "value": round(peak_weight, 4),
                    })

                    if 'Jaw' in morph_name:
                        rest_val = CALM_STYLE["min_jaw_rest"]
                    elif 'Lips_Part' in morph_name:
                        rest_val = CALM_STYLE["min_lips_rest"]
                    elif 'Mouth_Open' in morph_name:
                        rest_val = CALM_STYLE["min_mouth_open_rest"]
                    else:
                        rest_val = 0.0

                    visemes.append({
                        "time": round(release_t, 4),
                        "morph": morph_name,
                        "value": rest_val,
                    })

            current_time += time_per_char
            prev_shape = shape

        # End at rest
        visemes.append({
            "time": round(min(current_time, audio_duration), 4),
            "morph": "A25_Jaw_Open",
            "value": 0.0,
        })

        visemes.sort(key=lambda v: v["time"])
        return visemes

    def _apply_text_style_polish(self, visemes: List[Dict], text: str, audio_duration: float) -> List[Dict]:
        """
        Final polish layer for medical-trainer style:
        - restrained formal expressions
        - punctuation-aware micro pauses
        - reduced over-articulation at high speaking rates
        """
        if not visemes:
            return visemes

        # Compute approximate speaking rate from letters only.
        letter_count = len(re.findall(r"[A-Za-zÀ-ÿ]", text or ""))
        usable_duration = max(0.25, audio_duration if audio_duration > 0 else 2.0)
        chars_per_sec = letter_count / usable_duration if letter_count > 0 else 0.0

        # Rate-aware scaling: keep natural clarity on rapid phrases.
        if chars_per_sec <= CALM_STYLE["rapid_chars_per_sec"]:
            rate_scale = 1.0
        else:
            excess = chars_per_sec - CALM_STYLE["rapid_chars_per_sec"]
            rate_scale = max(CALM_STYLE["rapid_scale_floor"], 1.0 - (0.03 * excess))

        formal_scale = CALM_STYLE["formal_expression_scale"]

        def _scaled(morph: str, value: float) -> float:
            if value <= 0.0:
                return value

            # Keep mouth readability morphs less attenuated.
            if morph in {"A25_Jaw_Open", "Mouth_Lips_Part", "Mouth_Open"}:
                s = max(0.90, rate_scale)
            else:
                s = formal_scale * rate_scale

            return max(0.0, min(1.0, value * s))

        polished = []
        for item in visemes:
            morph = str(item.get("morph", ""))
            t = float(item.get("time", 0.0))
            v = float(item.get("value", 0.0))
            polished.append({
                "time": round(t, 4),
                "morph": morph,
                "value": round(_scaled(morph, v), 4),
            })

        # Add punctuation-aware micro pauses by easing toward neutral openness.
        pause_marks = []
        if text and letter_count > 0 and usable_duration > 0:
            letter_idx = 0
            for ch in text:
                if re.match(r"[A-Za-zÀ-ÿ]", ch):
                    letter_idx += 1
                elif ch in {',', ';', ':'}:
                    t = min(usable_duration, (letter_idx / max(1, letter_count)) * usable_duration)
                    pause_marks.append((t, CALM_STYLE["comma_pause_sec"]))
                elif ch in {'.', '!', '?'}:
                    t = min(usable_duration, (letter_idx / max(1, letter_count)) * usable_duration)
                    pause_marks.append((t, CALM_STYLE["sentence_pause_sec"]))

        for t, pause_dur in pause_marks:
            t0 = max(0.0, t)
            t1 = min(usable_duration, t + pause_dur)
            polished.append({"time": round(t0, 4), "morph": "A25_Jaw_Open", "value": 0.045})
            polished.append({"time": round(t0, 4), "morph": "Mouth_Lips_Part", "value": 0.12})
            polished.append({"time": round(t0, 4), "morph": "Mouth_Open", "value": 0.045})
            polished.append({"time": round(t1, 4), "morph": "A25_Jaw_Open", "value": CALM_STYLE["min_jaw_rest"]})
            polished.append({"time": round(t1, 4), "morph": "Mouth_Lips_Part", "value": CALM_STYLE["min_lips_rest"]})
            polished.append({"time": round(t1, 4), "morph": "Mouth_Open", "value": CALM_STYLE["min_mouth_open_rest"]})

        # Add subtle speaking expressions (blink + brows + restrained affect).
        polished.extend(self._build_nonverbal_track(text, usable_duration))

        polished.sort(key=lambda v: v["time"])
        return polished

    def _build_nonverbal_track(self, text: str, duration: float) -> List[Dict]:
        if duration <= 0.25:
            return []

        cues = []
        normalized = (text or "").strip().lower()

        intent = self._analyze_expression_intent(normalized)
        state = intent["state"]
        profile = EXPRESSION_PROFILES.get(state, EXPRESSION_PROFILES["neutral"])

        brow_base = profile["brow_inner"]
        brow_down = profile["brow_down"]
        smile_base = profile["smile"]
        blink_scale = profile["blink_scale"]

        if intent["is_reassuring"]:
            smile_base = max(smile_base, NONVERBAL_STYLE["smile_reassure"])
        if intent["has_emphasis"]:
            smile_base = max(smile_base, NONVERBAL_STYLE["smile_emphasis"])

        logger.info(f"Nonverbal state={state} reassure={intent['is_reassuring']} emphasis={intent['has_emphasis']} question={intent['is_question']}")

        # Gentle baseline brows and expression anchors across the utterance.
        cues.append({"time": 0.0, "morph": "A01_Brow_Inner_Up", "value": round(brow_base, 4)})
        cues.append({"time": 0.0, "morph": "A02_Brow_Down_Left", "value": round(brow_down, 4)})
        cues.append({"time": 0.0, "morph": "A03_Brow_Down_Right", "value": round(brow_down, 4)})
        cues.append({"time": 0.0, "morph": "A38_Mouth_Smile_Left", "value": round(smile_base, 4)})
        cues.append({"time": 0.0, "morph": "A39_Mouth_Smile_Right", "value": round(smile_base, 4)})

        # Slight expression lift near the end for sentence completion.
        end_lift_t = max(0.0, duration - 0.35)
        cues.append({"time": round(end_lift_t, 4), "morph": "A01_Brow_Inner_Up", "value": round(min(0.18, brow_base + 0.03), 4)})

        # If question, add final eyebrow raise.
        if intent["is_question"]:
            q_t = max(0.0, duration - 0.22)
            cues.append({"time": round(q_t, 4), "morph": "A01_Brow_Inner_Up", "value": round(min(0.2, brow_base + 0.04), 4)})

        if state == "surprise":
            s_t = min(max(0.25, duration * 0.28), max(0.3, duration - 0.4))
            cues.append({"time": round(s_t, 4), "morph": "A01_Brow_Inner_Up", "value": 0.20})
            cues.append({"time": round(s_t, 4), "morph": "Mouth_Open", "value": max(CALM_STYLE["mouth_open_floor"], 0.16)})

        # Natural blinks (deterministic spacing from text hash to avoid robotic regularity).
        seed = (sum(ord(c) for c in normalized) % 997) / 997.0
        interval_span = NONVERBAL_STYLE["blink_max_interval"] - NONVERBAL_STYLE["blink_min_interval"]
        base_interval = (NONVERBAL_STYLE["blink_min_interval"] + interval_span * seed) / max(0.75, blink_scale)

        t = 0.7 + (0.25 * seed)
        blink_idx = 0
        while t < (duration - 0.2):
            # Alternate tiny interval modulation per blink.
            mod = 0.22 if (blink_idx % 2 == 0) else -0.18
            interval = max(
                NONVERBAL_STYLE["blink_min_interval"],
                min(NONVERBAL_STYLE["blink_max_interval"], base_interval + mod),
            )

            close_t = t + 0.05
            open_t = t + NONVERBAL_STYLE["blink_duration"]

            # Bilateral blink with quick close/open profile.
            cues.append({"time": round(t, 4), "morph": "A14_Eye_Blink_Left", "value": NONVERBAL_STYLE["blink_half"]})
            cues.append({"time": round(t, 4), "morph": "A15_Eye_Blink_Right", "value": NONVERBAL_STYLE["blink_half"]})
            cues.append({"time": round(close_t, 4), "morph": "A14_Eye_Blink_Left", "value": NONVERBAL_STYLE["blink_close"]})
            cues.append({"time": round(close_t, 4), "morph": "A15_Eye_Blink_Right", "value": NONVERBAL_STYLE["blink_close"]})
            cues.append({"time": round(open_t, 4), "morph": "A14_Eye_Blink_Left", "value": 0.0})
            cues.append({"time": round(open_t, 4), "morph": "A15_Eye_Blink_Right", "value": 0.0})

            t += interval
            blink_idx += 1

        # Return to neutral at the very end.
        cues.append({"time": round(duration, 4), "morph": "A01_Brow_Inner_Up", "value": NONVERBAL_STYLE["brow_neutral"]})
        cues.append({"time": round(duration, 4), "morph": "A02_Brow_Down_Left", "value": 0.0})
        cues.append({"time": round(duration, 4), "morph": "A03_Brow_Down_Right", "value": 0.0})
        cues.append({"time": round(duration, 4), "morph": "A38_Mouth_Smile_Left", "value": 0.0})
        cues.append({"time": round(duration, 4), "morph": "A39_Mouth_Smile_Right", "value": 0.0})

        return cues

    def _analyze_expression_intent(self, normalized_text: str) -> Dict[str, object]:
        text = normalized_text or ""

        is_question = "?" in text
        has_emphasis = "!" in text or any(w in text for w in ["important", "urgent", "attention", "critical", "essentiel", "important"])
        is_reassuring = any(w in text for w in ["safe", "tolere", "simple", "clear", "benefit", "support", "confiance", "rassurant", "calme"])
        has_caution = any(w in text for w in ["risk", "warning", "precaution", "contre", "danger", "attention", "adverse", "side effect", "surveillance"])
        has_thinking = any(w in text for w in ["analy", "consider", "let us see", "hmm", "peut", "peut-etre", "examinons", "voyons"])
        has_surprise = any(w in text for w in ["surprising", "unexpected", "wow", "incroyable", "etonnant"]) or ("!" in text and "?" in text)

        state = "neutral"
        if has_caution:
            state = "caution"
        elif has_surprise:
            state = "surprise"
        elif has_thinking:
            state = "thinking"
        elif is_question:
            state = "question"
        elif is_reassuring:
            state = "reassuring"

        return {
            "state": state,
            "is_question": is_question,
            "has_emphasis": has_emphasis,
            "is_reassuring": is_reassuring,
            "has_caution": has_caution,
            "has_thinking": has_thinking,
            "has_surprise": has_surprise,
        }

    def _is_vowel_shape(self, shape: str) -> bool:
        return shape in {'A', 'C', 'D', 'E', 'F'}

    def _next_text_shape(self, clean_text: str, index: int) -> str:
        for j in range(index + 1, len(clean_text)):
            c = clean_text[j]
            if c.isalpha():
                return LETTER_TO_SHAPE.get(c, 'C')
        return 'X'

    def _apply_coarticulation_blend(self, current_morphs: Dict[str, float], next_morphs: Dict[str, float]) -> Dict[str, float]:
        """
        Blend a small portion of the next phoneme into the current one.
        This reduces robotic, discrete mouth snaps and improves realism.
        """
        if not current_morphs:
            return dict(current_morphs)

        blended = dict(current_morphs)
        blend = CALM_STYLE["coarticulation_blend"]
        for morph_name, next_value in next_morphs.items():
            carry = max(0.0, min(1.0, next_value * blend))
            blended[morph_name] = max(blended.get(morph_name, 0.0), carry)

        return blended

    def _merge_support_morphs(self, primary_morphs: Dict[str, float]) -> Dict[str, float]:
        """
        Add support morphs so lips visibly separate while jaw moves.
        This keeps mouth opening readable and avoids a "sealed lips" look.
        """
        merged = dict(primary_morphs)
        if not merged:
            return merged

        open_driver = max(
            merged.get("V_Open", 0.0),
            merged.get("A25_Jaw_Open", 0.0),
            merged.get("V_Wide", 0.0) * 0.7,
            merged.get("V_Lip_Open", 0.0) * 0.7,
            merged.get("V_Tight_O", 0.0) * 0.6,
        )

        lips_part = max(CALM_STYLE["lips_part_floor"], open_driver * CALM_STYLE["lips_part_peak_scale"])
        mouth_open = max(CALM_STYLE["mouth_open_floor"], open_driver * CALM_STYLE["mouth_open_peak_scale"])

        merged["Mouth_Lips_Part"] = max(merged.get("Mouth_Lips_Part", 0.0), min(0.80, lips_part))
        merged["Mouth_Open"] = max(merged.get("Mouth_Open", 0.0), min(0.70, mouth_open))

        # Subtle teeth visibility support on wide/dental/tight shapes.
        if merged.get("V_Dental_Lip", 0.0) > 0.08 or merged.get("V_Tight", 0.0) > 0.55 or merged.get("V_Wide", 0.0) > 0.55:
            merged["V_Dental_Lip"] = max(merged.get("V_Dental_Lip", 0.0), CALM_STYLE["teeth_hint"])

        return merged

    # ============================================================
    # ENERGY FALLBACK (when no Rhubarb and no text)
    # ============================================================

    async def _extract_energy(self, audio_path: str) -> List[Dict]:
        """Simple energy-based: just jaw open proportional to loudness"""
        try:
            with wave.open(audio_path, 'rb') as wav:
                sr = wav.getframerate()
                n = wav.getnframes()
                raw = wav.readframes(n)
                bps = wav.getsampwidth()

            if bps == 2:
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            win = int(sr * 0.05)   # 50ms
            hop = int(sr * 0.025)  # 25ms

            visemes = []
            smooth = 0.0

            for i in range(0, len(audio) - win, hop):
                w = audio[i:i + win]
                energy = float(np.sqrt(np.mean(w ** 2)))
                smooth = smooth * 0.4 + energy * 0.6
                t = float(i / sr)

                # Map energy to jaw opening (0 to 0.7)
                jaw = min(smooth * 8.0, 0.7)

                if jaw > 0.05:
                    visemes.append({"time": round(t, 4), "morph": "A25_Jaw_Open", "value": round(jaw, 4)})
                    visemes.append({"time": round(t, 4), "morph": "V_Open", "value": round(jaw * 0.6, 4)})
                else:
                    visemes.append({"time": round(t, 4), "morph": "A25_Jaw_Open", "value": 0.0})

            # Final close
            final_t = float(len(audio) / sr)
            visemes.append({"time": round(final_t, 4), "morph": "A25_Jaw_Open", "value": 0.0})

            logger.info(f"Energy fallback: {len(visemes)} keyframes")
            return visemes

        except Exception as e:
            logger.error(f"Energy extraction error: {e}")
            return []

    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            with wave.open(audio_path, 'rb') as wav:
                return wav.getnframes() / float(wav.getframerate())
        except:
            return 0.0

    async def cleanup(self):
        logger.info("Lip sync service cleaned up")
