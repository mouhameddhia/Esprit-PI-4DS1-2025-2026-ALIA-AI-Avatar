import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Mic, Power, XCircle, User, Activity, CheckCircle2, ChevronLeft, Sparkles, RefreshCw } from 'lucide-react';
import './MedRepPortal.css';
import './MedRepSimulation.css';

const API_BASE = 'http://localhost:8000';

function parseApiError(data) {
  if (!data) return 'Request failed';
  if (typeof data.detail === 'string') return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail.map((e) => (typeof e === 'string' ? e : e.msg || JSON.stringify(e))).join(', ');
  }
  return 'Request failed';
}

function toPercentFromTen(score) {
  const value = Number(score);
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value * 10)));
}

const DEFAULT_PERSONA = {
  name: 'Dr. Skeptical',
  specialty: 'Cardiology',
  traits: 'Critical, Evidence-Focused',
};

const DEFAULT_PRODUCT = {
  name: 'CardioGuard',
  category: 'Cardiovascular',
  indication: 'Treatment of hypertension and heart failure',
};

function buildDoctorOpening(persona, product) {
  return {
    id: 1,
    sender: 'doctor',
    text: `Hello, I'm ${persona.name}, ${persona.specialty}. I understand you'd like to discuss ${product.name}. What can you tell me about the specific clinical outcomes and safety profile?`,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  };
}

const MedRepSimulation = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const scenario = useMemo(() => {
    const trainingScenario = location.state?.scenario || location.state || {};
    return {
      persona: trainingScenario.persona || DEFAULT_PERSONA,
      product: trainingScenario.product || DEFAULT_PRODUCT,
    };
  }, [location.state]);

  const [messages, setMessages] = useState(() => [buildDoctorOpening(scenario.persona, scenario.product)]);
  const [inputText, setInputText] = useState('');
  const [metrics, setMetrics] = useState({ clarity: 10, accuracy: 5, persuasion: 0 });
  const [nlpDebug, setNlpDebug] = useState(null);
  const [nlpDebugError, setNlpDebugError] = useState('');
  const [nlpDebugLoading, setNlpDebugLoading] = useState(false);
  const chatEndRef = useRef(null);

  const latestUserMessage = useMemo(
    () => [...messages].reverse().find((message) => message.sender === 'user') || null,
    [messages],
  );

  const analysisSeed = useMemo(() => {
    const draft = inputText.trim();
    if (draft) return draft;
    if (latestUserMessage?.text) return latestUserMessage.text;
    return `I'm a medical representative discussing ${scenario.product.name} with ${scenario.persona.name}, who wants evidence and safety detail.`;
  }, [inputText, latestUserMessage, scenario.persona.name, scenario.product.name]);

  // Auto-scroll to bottom of chat
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    setMessages([buildDoctorOpening(scenario.persona, scenario.product)]);
    setInputText('');
    setNlpDebug(null);
    setNlpDebugError('');
  }, [scenario.persona, scenario.product]);

  // Simulate AI "analyzing" to increase metrics slowly as user types/interacts
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        clarity: Math.min(prev.clarity + (inputText.length > 5 ? 2 : 0), 92),
        accuracy: Math.min(prev.accuracy + (inputText.length > 10 ? 1 : 0), 85),
        persuasion: Math.min(prev.persuasion + (inputText.length > 15 ? 3 : 0), 78)
      }));
    }, 3000);
    return () => clearInterval(interval);
  }, [inputText]);

    const refreshNlpDebug = useCallback(async (overridePrompt) => {
      const prompt = (overridePrompt ?? analysisSeed).trim();
      const token = localStorage.getItem('token');

      if (!prompt) {
        setNlpDebug(null);
        setNlpDebugError('Type a response or send a message to inspect the trace.');
        return;
      }

      if (!token) {
        setNlpDebug(null);
        setNlpDebugError('Sign in to inspect the training analysis.');
        return;
      }

      setNlpDebugLoading(true);
      setNlpDebugError('');
      try {
        const response = await fetch('http://localhost:8000/chat/nlp-debug', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            content: prompt,
            mode: 'medrep_training',
          }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(data.detail || 'Failed to generate simulation analysis');
        }

        setNlpDebug(data.analysis || null);
      } catch (error) {
        setNlpDebug(null);
        setNlpDebugError(error.message || 'Failed to generate simulation analysis');
      } finally {
        setNlpDebugLoading(false);
      }
    }, [analysisSeed]);

    useEffect(() => {
      refreshNlpDebug();
    }, [refreshNlpDebug]);

    const renderChips = (items, emptyLabel = 'none') => {
      const values = (Array.isArray(items) ? items : []).filter((item) => typeof item === 'string' && item.trim());
      if (!values.length) {
        return <span className="sim-debug-empty-value">{emptyLabel}</span>;
      }

      return (
        <div className="sim-debug-chip-row">
          {values.map((value) => (
            <span key={value} className="sim-debug-chip">{value}</span>
          ))}
        </div>
      );
    };

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;
    const token = localStorage.getItem('token');
    const repText = inputText.trim();

    const newMessage = {
      id: Date.now(),
      sender: 'user',
      text: repText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, newMessage]);
    setInputText('');
    setNlpDebug(null);
    setNlpDebugError('');

    if (token) {
      try {
        const response = await fetch(`${API_BASE}/chat/rep-score`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ content: repText }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(parseApiError(data));
        }

        setMetrics({
          clarity: toPercentFromTen(data.clarity_score),
          accuracy: toPercentFromTen(data.confidence_score),
          persuasion: toPercentFromTen(data.persuasion_score),
        });
      } catch (error) {
        console.warn('Failed to score representative response:', error.message || error);
      }
    }

    // Simulate AI Physician thinking and responding
    setTimeout(() => {
      const response = {
        id: Date.now() + 1,
        sender: 'doctor',
        text: "That's an interesting point. However, the data I've seen suggests a higher incidence of side effects compared to traditional ACE inhibitors. How does your product address renal safety concerns in diabetic patients?",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, response]);
    }, 2000);

    refreshNlpDebug(repText);
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <div className="portal-container" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div className="portal-bg-aura"></div>
      
      {/* Top Header/Nav */}
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
        <div className="portal-nav-right" style={{ visibility: 'hidden' }}>
          {/* Placeholder for balance */}
        </div>
      </nav>

      <main className="sim-container relative z-10">
        
        {/* Left Sidebar: Profile & Metrics */}
        <aside className="sim-sidebar">
          <div className="sim-avatar-wrapper">
            <div className="avatar-circle">
              <img 
                src="https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&q=80&w=400&h=400" 
                alt="Dr. Skeptical" 
              />
              <div className="avatar-status-overlay">
                <span className="status-dot active"></span> Live AI Avatar
              </div>
            </div>
            <div className="sim-doctor-info">
              <h2>Dr. Skeptical</h2>
              <p>Cardiology | Critical, Evidence-Focused</p>
            </div>
          </div>

          <div className="sim-metrics-box">
            <h3 style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Activity size={16} color="#7c3aed" /> 
              Live Performance Metrics
            </h3>
            
            <div className="metric-row">
              <div className="metric-label-row">
                <span>Clarity</span>
                <span>{metrics.clarity}%</span>
              </div>
              <div className="metric-bar-bg">
                <motion.div initial={{ width: 0 }} animate={{ width: `${metrics.clarity}%` }} className="metric-bar-fill" />
              </div>
            </div>

            <div className="metric-row">
              <div className="metric-label-row">
                <span>Accuracy</span>
                <span>{metrics.accuracy}%</span>
              </div>
              <div className="metric-bar-bg">
                <motion.div initial={{ width: 0 }} animate={{ width: `${metrics.accuracy}%` }} className="metric-bar-fill" style={{ background: '#2dd4bf' }} />
              </div>
            </div>

            <div className="metric-row">
              <div className="metric-label-row">
                <span>Persuasion</span>
                <span>{metrics.persuasion}%</span>
              </div>
              <div className="metric-bar-bg">
                <motion.div initial={{ width: 0 }} animate={{ width: `${metrics.persuasion}%` }} className="metric-bar-fill" style={{ background: '#f59e0b' }} />
              </div>
            </div>
          </div>

          <div className="sim-debug-panel" aria-label="Simulation explainability panel">
            <div className="sim-debug-header">
              <div>
                <p className="sim-debug-kicker"><Sparkles size={12} /> Training debug</p>
                <h3>Explainability snapshot</h3>
              </div>
              <button type="button" className="sim-debug-refresh" onClick={refreshNlpDebug} disabled={nlpDebugLoading}>
                <RefreshCw size={15} />
                {nlpDebugLoading ? 'Analyzing…' : 'Refresh'}
              </button>
            </div>

            {nlpDebugError ? <div className="sim-debug-error">{nlpDebugError}</div> : null}

            {nlpDebug ? (
              <div className="sim-debug-body">
                <div className="sim-debug-summary-row">
                  <div className="sim-debug-summary-item">
                    <span className="sim-debug-label">Intent</span>
                    <strong>{nlpDebug.intent || 'other'}</strong>
                  </div>
                  <div className="sim-debug-summary-item">
                    <span className="sim-debug-label">Confidence</span>
                    <strong>{Math.round((nlpDebug.confidence || 0) * 100)}%</strong>
                  </div>
                  <div className="sim-debug-summary-item">
                    <span className="sim-debug-label">Clarification</span>
                    <strong>{nlpDebug.needs_clarification ? 'Needed' : 'Not needed'}</strong>
                  </div>
                </div>

                <div className="sim-debug-metric">
                  <span className="sim-debug-label">Why it was chosen</span>
                  <p>{nlpDebug.explainability?.why_class_was_chosen || nlpDebug.explainability?.reasoning || 'No explainability details returned.'}</p>
                </div>

                <div className="sim-debug-metric">
                  <span className="sim-debug-label">Rewritten query</span>
                  <p>{nlpDebug.rewritten_query || analysisSeed}</p>
                </div>

                <div className="sim-debug-grid">
                  <div className="sim-debug-metric">
                    <span className="sim-debug-label">Safety flags</span>
                    {renderChips(nlpDebug.safety_flags)}
                  </div>
                  <div className="sim-debug-metric">
                    <span className="sim-debug-label">Secondary tags</span>
                    {renderChips(nlpDebug.secondary_tags)}
                  </div>
                </div>

                <div className="sim-debug-metric">
                  <span className="sim-debug-label">Entity map</span>
                  {Object.keys(nlpDebug.entity_map || {}).length ? (
                    <div className="sim-debug-entity-list">
                      {Object.entries(nlpDebug.entity_map || {}).map(([entityType, values]) => (
                        <div key={entityType} className="sim-debug-entity-group">
                          <span className="sim-debug-entity-label">{entityType}</span>
                          {renderChips(values)}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <span className="sim-debug-empty-value">none</span>
                  )}
                </div>

                <div className="sim-debug-grid">
                  <div className="sim-debug-metric">
                    <span className="sim-debug-label">Influential keywords</span>
                    {renderChips(nlpDebug.explainability?.influential_keywords)}
                  </div>
                  <div className="sim-debug-metric">
                    <span className="sim-debug-label">Missing expected concepts</span>
                    {renderChips(nlpDebug.explainability?.missing_expected_concepts, 'none')}
                  </div>
                </div>
              </div>
            ) : (
              <div className="sim-debug-empty-state">
                The trace updates from your draft or most recent rep message.
              </div>
            )}
          </div>

          <div className="sim-product-info" style={{ marginTop: 'auto' }}>
            <div className="flow-node product" style={{ width: '100%' }}>
              <CheckCircle2 size={18} />
              <span>Targeting: {scenario.product.name}</span>
            </div>
          </div>

          <div className="end-session-row">
            <button className="btn-end-session" onClick={() => navigate('/rep/dashboard')}>
              <XCircle size={18} />
              End Session & Get Feedback
            </button>
          </div>
        </aside>

        {/* Right Area: Chat Simulation */}
        <section className="sim-chat-area">
          <div className="chat-history">
            <AnimatePresence>
              {messages.map((msg) => (
                <motion.div 
                  key={msg.id}
                  initial={{ opacity: 0, y: 20, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  className={`message-bubble ${msg.sender}`}
                >
                  <p>{msg.text}</p>
                  <span style={{ fontSize: '0.7rem', opacity: 0.6, display: 'block', marginTop: '0.5rem', textAlign: msg.sender === 'user' ? 'right' : 'left' }}>
                    {msg.timestamp}
                  </span>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={chatEndRef} />
          </div>

          <div className="chat-input-wrapper">
            <button className="voice-btn">
              <Mic size={20} />
            </button>
            <input 
              type="text" 
              className="chat-input-field" 
              placeholder="Type your response to the physician..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <button className="btn-send" onClick={handleSendMessage}>
              <Send size={18} />
            </button>
          </div>
          
          <div style={{ textAlign: 'center', paddingBottom: '0.75rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            AI is analyzing clarity, accuracy, and persuasion in real-time
          </div>
        </section>

      </main>
    </div>
  );
};

export default MedRepSimulation;
