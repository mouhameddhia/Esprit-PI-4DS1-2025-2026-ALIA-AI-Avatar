import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LogOut,
  History,
  ChevronRight,
  Send,
  Mic,
  Activity,
  ArrowRight,
  ShieldCheck,
  Package,
  MessageSquarePlus,
} from 'lucide-react';
import './PhysicianPortal.css';
import '../medrep/MedRepPortal.css';
import '../medrep/MedRepSimulation.css';

const API_BASE = 'http://localhost:8000';
const SESSION_STORAGE_KEY = 'alia_physician_session_id';

const NOUR_AVATAR_PRIMARY =
  'https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&q=80&w=400&h=400';
/** Shown if the primary URL fails (network / referrer / hotlink limits). */
const NOUR_AVATAR_FALLBACK =
  'https://api.dicebear.com/7.x/avataaars/svg?seed=Nour&backgroundColor=b6e3f4&radius=50';

const WELCOME_TEXT =
  "Hello Doctor! I'm Nour, your AI pharmaceutical representative. How can I assist you today? You can ask me about any of our products, clinical data, dosing guidelines, or request information about upcoming webinars.";

function buildWelcomeMessage() {
  return {
    id: 'welcome',
    sender: 'doctor',
    text: WELCOME_TEXT,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  };
}

function mapApiMessagesToUi(messages, sessionId) {
  if (!messages?.length) return [buildWelcomeMessage()];
  return messages.map((m, i) => ({
    id: `${sessionId}-${i}-${m.at}`,
    sender: m.role === 'user' ? 'user' : 'doctor',
    text: m.content,
    timestamp: new Date(m.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  }));
}

function parseApiError(data) {
  if (!data) return 'Request failed';
  if (typeof data.detail === 'string') return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail.map((e) => (typeof e === 'string' ? e : e.msg || JSON.stringify(e))).join(', ');
  }
  return 'Request failed';
}

const PhysicianPortal = () => {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([buildWelcomeMessage()]);
  const [inputText, setInputText] = useState('');
  const [activeSection, setActiveSection] = useState('chat');
  const [sessionId, setSessionId] = useState(null);
  const [chatLoading, setChatLoading] = useState(false);
  const [sessionHistory, setSessionHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [avatarSrc, setAvatarSrc] = useState(NOUR_AVATAR_PRIMARY);
  const chatHistoryRef = useRef(null);

  const products = [
    { id: 'pr1', name: 'CardioGuard', desc: 'ACE Inhibitor' },
    { id: 'pr2', name: 'NeuroShield', desc: 'Anticonvulsant' },
    { id: 'pr3', name: 'DiabetoCare Plus', desc: 'GLP-1 Agonist' },
  ];

  const webinars = [
    { date: 'OCT 24', title: 'Advances in Hypertension', time: '14:00 GMT' },
    { date: 'NOV 12', title: 'Managing Diabetic Renal Risk', time: '10:00 GMT' },
  ];

  const scrollToBottom = () => {
    const el = chatHistoryRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const token = localStorage.getItem('token');
    const sid = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (!token || !sid) return;

    (async () => {
      try {
        const r = await fetch(`${API_BASE}/chat/sessions/${sid}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!r.ok) {
          sessionStorage.removeItem(SESSION_STORAGE_KEY);
          return;
        }
        const data = await r.json();
        setSessionId(sid);
        setMessages(mapApiMessagesToUi(data.messages, sid));
      } catch {
        sessionStorage.removeItem(SESSION_STORAGE_KEY);
      }
    })();
  }, []);

  useEffect(() => {
    if (activeSection !== 'history') return;
    const token = localStorage.getItem('token');
    if (!token) return;
    setHistoryLoading(true);
    fetch(`${API_BASE}/chat/sessions`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => (r.ok ? r.json() : []))
      .then(setSessionHistory)
      .catch(() => setSessionHistory([]))
      .finally(() => setHistoryLoading(false));
  }, [activeSection]);

  const handleSignOut = async () => {
    const token = localStorage.getItem('token');
    const sid = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (token && sid) {
      try {
        await fetch(`${API_BASE}/chat/sessions/${sid}/finalize`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
        });
      } catch {
        /* ignore */
      }
    }
    sessionStorage.removeItem(SESSION_STORAGE_KEY);
    localStorage.removeItem('token');
    navigate('/login');
  };

  const startNewChat = useCallback(() => {
    sessionStorage.removeItem(SESSION_STORAGE_KEY);
    setSessionId(null);
    setMessages([buildWelcomeMessage()]);
    setInputText('');
    setActiveSection('chat');
  }, []);

  const handleSendMessage = async () => {
    if (!inputText.trim() || chatLoading) return;
    const token = localStorage.getItem('token');
    if (!token) {
      alert('Please sign in to chat.');
      navigate('/login');
      return;
    }

    const text = inputText.trim();
    setInputText('');

    const userMsg = {
      id: `u-${Date.now()}`,
      sender: 'user',
      text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    setMessages((prev) => [...prev, userMsg]);
    setChatLoading(true);

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
          mode: 'physician_portal',
        }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        throw new Error(parseApiError(data));
      }
      setSessionId(data.session_id);
      sessionStorage.setItem(SESSION_STORAGE_KEY, data.session_id);
      const reply = {
        id: `a-${Date.now()}`,
        sender: 'doctor',
        text: data.reply,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, reply]);
    } catch (err) {
      alert(err.message || 'Failed to reach chat service');
      setMessages((prev) => prev.filter((m) => m.id !== userMsg.id));
    } finally {
      setChatLoading(false);
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.1 } },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 },
  };

  const modeLabel = (mode) => {
    if (mode === 'physician_portal') return 'Physician — AI rep (Nour)';
    if (mode === 'medrep_training') return 'Med rep training';
    return mode || 'Chat';
  };

  return (
    <div className="portal-container physician-portal-page">
      <div className="portal-bg-wrap" aria-hidden="true">
        <div className="portal-bg-aura" />
        <div className="portal-bg-aura-2" style={{ top: '60%' }} />
      </div>

      <nav className="portal-navbar">
        <div className="portal-nav-left">
          <div className="portal-nav-logo" style={{ background: '#7c3aed' }}>
            P
          </div>
          <div className="portal-nav-text">
            <span className="portal-nav-title">ALIA - Physician Portal</span>
            <span className="portal-nav-subtitle">Your AI Medical Representative</span>
          </div>
        </div>
        <div className="portal-nav-right">
          <button className="portal-signout-btn" type="button" onClick={handleSignOut}>
            <LogOut size={18} />
            <span>Sign Out</span>
          </button>
        </div>
      </nav>

      <motion.main
        className="physician-container relative z-10"
        variants={containerVariants}
        initial="hidden"
        animate="show"
      >
        <div className="physician-content-grid">
          <motion.aside variants={itemVariants} className="ai-rep-card">
            <div className="ai-rep-avatar">
              <img
                src={avatarSrc}
                alt="Nour AI"
                referrerPolicy="no-referrer"
                loading="eager"
                decoding="async"
                onError={() => setAvatarSrc((s) => (s === NOUR_AVATAR_FALLBACK ? s : NOUR_AVATAR_FALLBACK))}
              />
              <div className="ai-rep-status">Available 24/7</div>
            </div>

            <div className="ai-rep-info">
              <h2>Nour</h2>
              <p>Your AI Pharmaceutical Rep</p>
            </div>

            <div className="physician-stats">
              <div
                className={`physician-stat-item ${activeSection === 'history' ? 'active' : ''}`}
                onClick={() => setActiveSection('history')}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === 'Enter' && setActiveSection('history')}
              >
                <History size={20} color="#7c3aed" />
                <div style={{ textAlign: 'left' }}>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Interaction History</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    Saved chats & summaries
                  </div>
                </div>
                <ChevronRight size={16} style={{ marginLeft: 'auto' }} />
              </div>

              <div
                className={`physician-stat-item ${activeSection === 'products' ? 'active' : ''}`}
                onClick={() => setActiveSection('products')}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === 'Enter' && setActiveSection('products')}
              >
                <Package size={20} color="#7c3aed" />
                <div style={{ textAlign: 'left' }}>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Browse Products</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    Dosing & Indication guides
                  </div>
                </div>
                <ChevronRight size={16} style={{ marginLeft: 'auto' }} />
              </div>
            </div>

            <div style={{ marginTop: '2rem', textAlign: 'left' }}>
              <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem' }}>Upcoming Webinars</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {webinars.map((w, idx) => (
                  <div key={idx} className="webinar-card">
                    <span className="webinar-date">{w.date}</span>
                    <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>{w.title}</span>
                    <button type="button" className="btn-notify">
                      Get Notified
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </motion.aside>

          <motion.section variants={itemVariants} className="physician-chat-box">
            {activeSection === 'history' ? (
              <div className="history-view">
                <div className="history-header">
                  <div>
                    <h2>Interaction History</h2>
                    <p>Your conversations with Nour (saved when you sign out)</p>
                  </div>
                  <button type="button" className="history-back" onClick={() => setActiveSection('chat')}>
                    <ArrowRight size={16} style={{ transform: 'rotate(180deg)' }} />
                    Back to Chat
                  </button>
                </div>

                <div className="history-list">
                  {historyLoading ? (
                    <p style={{ color: 'var(--text-secondary)', padding: '1rem' }}>Loading…</p>
                  ) : sessionHistory.length === 0 ? (
                    <p style={{ color: 'var(--text-secondary)', padding: '1rem' }}>
                      No saved sessions yet. Chat with Nour, then sign out to generate a summary.
                    </p>
                  ) : (
                    sessionHistory.map((item) => (
                      <div key={item.id} className="history-card">
                        <div className="history-card-header">
                          <div className="history-avatar-circle">N</div>
                          <div>
                            <h3>{modeLabel(item.mode)}</h3>
                            <span className="history-date">
                              {item.updated_at
                                ? new Date(item.updated_at).toLocaleString()
                                : ''}
                            </span>
                          </div>
                          <span
                            className="history-pill"
                            style={{
                              fontSize: '0.75rem',
                              opacity: 0.9,
                            }}
                          >
                            {item.status === 'closed' ? 'Summarized' : 'In progress'}
                          </span>
                        </div>
                        <div className="history-card-body">
                          <div className="history-card-section">
                            <strong>Summary / preview:</strong>
                            <span>{item.summary || item.preview || '—'}</span>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            ) : activeSection === 'products' ? (
              <div className="products-view">
                <div className="history-header">
                  <div>
                    <h2>Browse Products</h2>
                    <p>Dosing, indications, and clinical summaries</p>
                  </div>
                  <button type="button" className="history-back" onClick={() => setActiveSection('chat')}>
                    <ArrowRight size={16} style={{ transform: 'rotate(180deg)' }} />
                    Back to Chat
                  </button>
                </div>
                <div className="products-grid">
                  {products.map((product) => (
                    <div key={product.id} className="product-card">
                      <h3>{product.name}</h3>
                      <p>{product.desc}</p>
                      <button type="button" className="btn-notify">
                        View Details
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <>
                <div className="chat-header" style={{ flexWrap: 'wrap', gap: '0.75rem' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <Activity size={20} color="#7c3aed" />
                      <span style={{ fontWeight: 700 }}>Ask Our AI Representative</span>
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                      Powered by Groq — voice or text
                    </div>
                  </div>
                  <button
                    type="button"
                    className="history-back"
                    onClick={startNewChat}
                    style={{ marginLeft: 'auto' }}
                  >
                    <MessageSquarePlus size={16} />
                    New chat
                  </button>
                </div>

                <div className="chat-history" ref={chatHistoryRef}>
                  <AnimatePresence>
                    {messages.map((msg) => (
                      <motion.div
                        key={msg.id}
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`message-bubble ${msg.sender === 'doctor' ? 'doctor' : 'user'}`}
                        style={{ maxWidth: '75%' }}
                      >
                        <p style={{ fontSize: '0.95rem' }}>{msg.text}</p>
                        <span
                          style={{
                            fontSize: '0.7rem',
                            opacity: 0.6,
                            display: 'block',
                            marginTop: '0.5rem',
                          }}
                        >
                          {msg.timestamp}
                        </span>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>

                <div className="chat-input-wrapper" style={{ padding: '1.25rem 2rem' }}>
                  <button
                    type="button"
                    className="voice-btn"
                    style={{ background: 'transparent', border: '1px solid var(--glass-border)' }}
                    aria-label="Voice input (coming soon)"
                  >
                    <Mic size={20} />
                  </button>
                  <input
                    type="text"
                    className="chat-input-field"
                    placeholder="Ask Nour about clinical data or product info..."
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSendMessage();
                      }
                    }}
                    disabled={chatLoading}
                  />
                  <button
                    type="button"
                    className="btn-send"
                    onClick={handleSendMessage}
                    disabled={chatLoading}
                    style={{ width: '50px', height: '50px' }}
                  >
                    <Send size={20} />
                  </button>
                </div>

                <div
                  style={{
                    textAlign: 'center',
                    paddingBottom: '0.75rem',
                    fontSize: '0.7rem',
                    color: 'var(--text-secondary)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.5rem',
                  }}
                >
                  <ShieldCheck size={12} /> Educational use — not a substitute for clinical judgment
                </div>
              </>
            )}
          </motion.section>
        </div>
      </motion.main>
    </div>
  );
};

export default PhysicianPortal;
