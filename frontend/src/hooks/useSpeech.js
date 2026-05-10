import { useState, useRef, useCallback } from 'react';

const API_BASE = 'http://localhost:8000';

/**
 * Shared hook for STT (Groq Whisper) and TTS (edge-tts).
 *
 * STT flow:  startRecording() → user speaks → stopRecording() → returns transcribed text
 * TTS flow:  speak(text, voice, msgId) → plays audio; call again with same msgId to stop
 */
export function useSpeech() {
  const [isRecording,    setIsRecording]    = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [speakingId,     setSpeakingId]     = useState(null); // msg id currently being spoken

  const mediaRecorderRef = useRef(null);
  const chunksRef        = useRef([]);
  const audioRef         = useRef(null);

  // ── STT ────────────────────────────────────────────────────────────────────

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : 'audio/mp4';

      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.start(100);
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch (err) {
      console.error('Microphone access denied:', err);
    }
  }, []);

  /** Stop recording, transcribe, and return the text (or '' on failure). */
  const stopRecording = useCallback(() => {
    return new Promise((resolve) => {
      const recorder = mediaRecorderRef.current;
      if (!recorder || recorder.state === 'inactive') { resolve(''); return; }

      recorder.onstop = async () => {
        recorder.stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType });
        chunksRef.current = [];
        setIsRecording(false);
        setIsTranscribing(true);

        try {
          const token    = localStorage.getItem('token');
          const formData = new FormData();
          formData.append('file', blob, 'audio.webm');

          const r    = await fetch(`${API_BASE}/audio/transcribe`, {
            method: 'POST',
            headers: { Authorization: `Bearer ${token}` },
            body: formData,
          });
          const data = await r.json().catch(() => ({}));
          resolve({ text: data.text || '', audioAffect: data.audio_affect || null });
        } catch {
          resolve({ text: '', audioAffect: null });
        } finally {
          setIsTranscribing(false);
        }
      };

      recorder.stop();
    });
  }, []);

  // ── TTS ────────────────────────────────────────────────────────────────────

  /** Speak text using edge-tts via the backend. Calling with the same msgId stops playback. */
  const speak = useCallback(async (text, voice = 'en-US-JennyNeural', msgId = null) => {
    // Toggle off if the same message is already speaking
    if (speakingId === msgId && audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
      setSpeakingId(null);
      return;
    }

    // Stop whatever was playing
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setSpeakingId(null);

    try {
      const token = localStorage.getItem('token');
      const r = await fetch(`${API_BASE}/audio/speak`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ text, voice }),
      });
      if (!r.ok) return;

      const blob = await r.blob();
      const url  = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioRef.current = audio;
      setSpeakingId(msgId);

      audio.onended = () => {
        URL.revokeObjectURL(url);
        audioRef.current = null;
        setSpeakingId(null);
      };
      audio.play();
    } catch (err) {
      console.error('TTS error:', err);
      setSpeakingId(null);
    }
  }, [speakingId]);

  return {
    // STT
    isRecording,
    isTranscribing,
    startRecording,
    stopRecording,
    // TTS
    speakingId,
    speak,
  };
}
