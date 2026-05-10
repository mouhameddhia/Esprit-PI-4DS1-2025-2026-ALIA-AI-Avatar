import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Mic, XCircle, Activity, CheckCircle2, ChevronLeft, Volume2, VolumeX, Loader2 } from 'lucide-react';
import { useSpeech } from '../../hooks/useSpeech';
import AffectPanel from '../shared/AffectPanel';
import './MedRepPortal.css';
import './MedRepSimulation.css';

const API_BASE = 'http://localhost:8000';

const MedRepSimulation = () => {
  const navigate   = useNavigate();
  const location   = useLocation();
  const persona    = location.state?.persona  || { name: 'Dr. Skeptical', specialty: 'Cardiology', style: 'Critical, Evidence-Focused' };
  const product    = location.state?.product  || { name: 'Cardivex' };

  const [messages, setMessages] = useState([{
    id: 1,
    sender: 'doctor',
    text: `Hello, I'm ${persona.name}, ${persona.specialty}. I understand you'd like to present ${product.name}. What can you tell me about its clinical outcomes?`,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  }]);
  const [inputText,    setInputText]    = useState('');
  const [isLoading,    setIsLoading]    = useState(false);
  const [sessionId,    setSessionId]    = useState(null);
  const [currentAffect, setCurrentAffect] = useState(null);
  const [isEndingSession, setIsEndingSession] = useState(false);
  const [feedback, setFeedback]           = useState(null);
  const chatEndRef = useRef(null);

  const { isRecording, isTranscribing, startRecording, stopRecording, speakingId, speak } = useSpeech();
  const [audioAffect, setAudioAffect] = useState(null);

  const handleEndSession = async () => {
    if (!sessionId) { navigate('/rep/dashboard'); return; }
    setIsEndingSession(true);
    try {
      const token = localStorage.getItem('token');
      const r = await fetch(`${API_BASE}/chat/sessions/${sessionId}/finalize`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await r.json().catch(() => ({}));
      if (r.ok) {
        setFeedback(data);
      } else {
        navigate('/rep/dashboard');
      }
    } catch {
      navigate('/rep/dashboard');
    } finally {
      setIsEndingSession(false);
    }
  };

  const handleMicClick = async () => {
    if (isRecording) {
      const { text, audioAffect: af } = await stopRecording();
      if (text) setInputText((prev) => (prev ? `${prev} ${text}` : text));
      if (af)   setAudioAffect(af);
    } else {
      await startRecording();
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const text  = inputText.trim();
    const token = localStorage.getItem('token');

    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages(prev => [...prev, userMsg]);
    setInputText('');
    setIsLoading(true);

    try {
      const r = await fetch(`${API_BASE}/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          session_id: sessionId,
          content: text,
          mode: 'medrep_training',
          audio_affect: audioAffect,
        }),
      });
      setAudioAffect(null);

      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || 'Chat request failed');

      setSessionId(data.session_id);
      if (data.affect) setCurrentAffect(data.affect);

      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        sender: 'doctor',
        text: data.reply,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }]);
    } catch (err) {
      console.error('Simulation chat error:', err);
      setMessages(prev => prev.filter(m => m.id !== userMsg.id));
    } finally {
      setIsLoading(false);
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    show:   { opacity: 1, y: 0 },
  };

  return (
    <div className="portal-container" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div className="portal-bg-aura" />

      {/* Navbar */}
      <nav className="portal-navbar" style={{ position: 'sticky', top: 0, zIndex: 50 }}>
        <div className="portal-nav-left">
          <button onClick={() => navigate('/rep/training')} className="dashboard-back-btn" style={{ margin: 0 }}>
            <ChevronLeft size={18} />
            Cancel Session
          </button>
        </div>
        <div className="portal-nav-text" style={{ textAlign: 'center', flex: 1 }}>
          <span className="portal-nav-title">ALIA Simulation Environment</span>
          <span className="portal-nav-subtitle">Live Interactive Evaluation</span>
        </div>
        <div className="portal-nav-right" style={{ visibility: 'hidden' }} />
      </nav>

      <main className="sim-container relative z-10">

        {/* ── Left Sidebar ── */}
        <aside className="sim-sidebar">
          <div className="sim-avatar-wrapper">
            <div className="avatar-circle">
              <img
                src="https://images.unsplash.com/photo-1559839734-2b71ea197ec2?auto=format&fit=crop&q=80&w=400&h=400"
                alt={persona.name}
              />
              <div className="avatar-status-overlay">
                <span className="status-dot active" /> Live AI Avatar
              </div>
            </div>
            <div className="sim-doctor-info">
              <h2>{persona.name}</h2>
              <p>{persona.specialty} | {persona.style}</p>
            </div>
          </div>

          {/* ── Affect Panel ── */}
          <div className="sim-metrics-box">
            <h3 style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Activity size={16} color="#7c3aed" />
              NLP Affect Analysis
            </h3>
            {currentAffect ? (
              <AffectPanel affect={currentAffect} audioAffect={audioAffect} mode="medrep_training" />
            ) : (
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Send a message to see live affect analysis from the NLP pipeline.
              </p>
            )}
          </div>

          <div className="sim-product-info" style={{ marginTop: 'auto' }}>
            <div className="flow-node product" style={{ width: '100%' }}>
              <CheckCircle2 size={18} />
              <span>Targeting: {product.name}</span>
            </div>
          </div>

          <div className="end-session-row">
            <button
              className="btn-end-session"
              onClick={handleEndSession}
              disabled={isEndingSession}
            >
              <XCircle size={18} />
              {isEndingSession ? 'Generating feedback…' : 'End Session & Get Feedback'}
            </button>
          </div>
        </aside>

        {/* ── Feedback Panel (shown after session ends) ── */}
        {feedback && (
          <section className="sim-chat-area sim-feedback-panel">
            <div className="feedback-header">
              <h2>Session Feedback</h2>
              {feedback.competency_level && (
                <span className="feedback-level-badge">{feedback.competency_level}</span>
              )}
              {feedback.evaluation_score != null && (
                <span className="feedback-score">{feedback.evaluation_score.toFixed(1)} / 10</span>
              )}
            </div>

            <div className="feedback-summary">
              <h3>Summary</h3>
              <p>{feedback.summary}</p>
            </div>

            {Object.keys(feedback.evaluation_dimensions || {}).length > 0 && (
              <div className="feedback-dimensions">
                <h3>Dimensions</h3>
                <div className="dimensions-grid">
                  {Object.entries(feedback.evaluation_dimensions).map(([dim, score]) => (
                    <div key={dim} className="dimension-item">
                      <span className="dim-label">{dim.replace(/_/g, ' ')}</span>
                      <div className="dim-bar-wrap">
                        <div className="dim-bar" style={{ width: `${score * 10}%`, background: score >= 8 ? '#22c55e' : score >= 7 ? '#7c3aed' : '#ef4444' }} />
                      </div>
                      <span className="dim-score">{score}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {feedback.evaluation_strengths?.length > 0 && (
              <div className="feedback-section">
                <h3>✓ Strengths</h3>
                <ul>{feedback.evaluation_strengths.map((s, i) => <li key={i}>{s.replace(/_/g, ' ')}</li>)}</ul>
              </div>
            )}

            {feedback.evaluation_gaps?.length > 0 && (
              <div className="feedback-section">
                <h3>△ Areas to improve</h3>
                <ul>{feedback.evaluation_gaps.map((g, i) => <li key={i}>{g.replace(/_/g, ' ')}</li>)}</ul>
              </div>
            )}

            {feedback.evaluation_notes?.length > 0 && (
              <div className="feedback-section">
                <h3>Notes</h3>
                <ul>{feedback.evaluation_notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
              </div>
            )}

            <button className="btn-end-session" style={{ marginTop: '1.5rem' }} onClick={() => navigate('/rep/dashboard')}>
              Back to Dashboard
            </button>
          </section>
        )}

        {/* ── Chat Area ── */}
        {!feedback && <section className="sim-chat-area">
          <div className="chat-history">
            <AnimatePresence>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  variants={itemVariants}
                  initial="hidden"
                  animate="show"
                  className={`message-bubble ${msg.sender}`}
                >
                  <p>{msg.text}</p>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: msg.sender === 'user' ? 'flex-end' : 'space-between', marginTop: '0.5rem' }}>
                    <span style={{ fontSize: '0.7rem', opacity: 0.6 }}>{msg.timestamp}</span>
                    {msg.sender === 'doctor' && (
                      <button
                        type="button"
                        onClick={() => speak(msg.text, 'en-US-GuyNeural', msg.id)}
                        className="speak-btn"
                        aria-label={speakingId === msg.id ? 'Stop speaking' : 'Read aloud'}
                        title={speakingId === msg.id ? 'Stop' : 'Read physician reply'}
                      >
                        {speakingId === msg.id ? <VolumeX size={13} /> : <Volume2 size={13} />}
                      </button>
                    )}
                  </div>
                </motion.div>
              ))}
              {isLoading && (
                <motion.div
                  key="typing"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="message-bubble doctor"
                >
                  <div className="typing-dots"><span /><span /><span /></div>
                </motion.div>
              )}
            </AnimatePresence>
            <div ref={chatEndRef} />
          </div>

          <div className="chat-input-wrapper">
            <button
              className={`voice-btn${isRecording ? ' recording' : ''}`}
              onClick={handleMicClick}
              disabled={isTranscribing || isLoading}
              aria-label={isRecording ? 'Stop recording' : 'Start voice input'}
              title={isRecording ? 'Stop & transcribe' : 'Speak your response'}
              style={{
                background: isRecording ? 'rgba(239,68,68,0.12)' : undefined,
                border: isRecording ? '1px solid #ef4444' : undefined,
              }}
            >
              {isTranscribing
                ? <Loader2 size={20} className="spin" />
                : <Mic size={20} color={isRecording ? '#ef4444' : undefined} />}
            </button>
            <input
              type="text"
              className="chat-input-field"
              placeholder="Type your response to the physician..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
              disabled={isLoading}
            />
            <button className="btn-send" onClick={handleSendMessage} disabled={isLoading}>
              <Send size={18} />
            </button>
          </div>

          <div style={{ textAlign: 'center', paddingBottom: '0.75rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            NLP pipeline analyzing clarity, affect, and communication patterns in real-time
          </div>
        </section>}

      </main>
    </div>
  );
};

export default MedRepSimulation;
