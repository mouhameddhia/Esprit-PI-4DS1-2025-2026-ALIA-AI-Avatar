"""
Quick terminal test for voice emotion detection.
Speak into your mic, press Enter to stop, and see the emotion result.

Usage:
    python test_voice_emotion.py
"""
import os, sys, time, threading, wave, tempfile
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

def record_until_enter(wav_path: str, sample_rate: int = 16000) -> None:
    """Record from mic until user presses Enter."""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
    except ImportError:
        print("pyaudio not installed. Install with: pip install pyaudio")
        sys.exit(1)

    frames = []
    stop_event = threading.Event()

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024,
    )

    def _record():
        while not stop_event.is_set():
            frames.append(stream.read(1024, exception_on_overflow=False))

    t = threading.Thread(target=_record, daemon=True)
    t.start()

    print("\n  🎤  Recording... press Enter to stop\n")
    input()
    stop_event.set()
    t.join()

    stream.stop_stream()
    stream.close()
    pa.terminate()

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))


def main():
    from backend.routes.audio import _classify_emotion

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    record_until_enter(tmp.name)

    print("  Analyzing emotion...")
    with open(tmp.name, "rb") as f:
        audio_bytes = f.read()
    os.unlink(tmp.name)

    result = _classify_emotion(audio_bytes)

    if not result:
        print("\n  Could not classify (audio too short or model unavailable)")
        return

    labels = {"neu": "😐 Neutral", "hap": "😊 Happy", "ang": "😤 Angry", "sad": "😔 Sad"}
    emo   = labels.get(result["emotion"], result["emotion"])
    conf  = result["confidence"] * 100
    bar   = "█" * int(conf / 5) + "░" * (20 - int(conf / 5))

    print(f"\n  Result ──────────────────────────────")
    print(f"  Emotion    : {emo}")
    print(f"  Confidence : {bar}  {conf:.1f}%")
    print(f"  Source     : {result['source']}")
    print(f"\n  All scores:")
    for k, v in result["all_scores"].items():
        bar2 = "█" * int(v * 40) + "░" * (40 - int(v * 40))
        print(f"    {k}  {bar2}  {v:.4f}")
    print()


if __name__ == "__main__":
    main()
