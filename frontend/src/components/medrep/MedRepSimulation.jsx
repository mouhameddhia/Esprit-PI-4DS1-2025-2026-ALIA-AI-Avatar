import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Mic, Power, XCircle, User, Activity, CheckCircle2, ChevronLeft } from 'lucide-react';
import './MedRepPortal.css';
import './MedRepSimulation.css';

const MedRepSimulation = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      sender: 'doctor', 
      text: "Hello, I'm Dr. Skeptical, Cardiology. I understand you'd like to discuss CardioGuard. What can you tell me about the specific clinical outcomes for patients with Stage 2 Hypertension?",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [metrics, setMetrics] = useState({ clarity: 10, accuracy: 5, persuasion: 0 });
  const chatEndRef = useRef(null);

  // Auto-scroll to bottom of chat
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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

  const handleSendMessage = () => {
    if (!inputText.trim()) return;

    const newMessage = {
      id: Date.now(),
      sender: 'user',
      text: inputText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages([...messages, newMessage]);
    setInputText('');

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

          <div className="sim-product-info" style={{ marginTop: 'auto' }}>
            <div className="flow-node product" style={{ width: '100%' }}>
              <CheckCircle2 size={18} />
              <span>Targeting: CardioGuard</span>
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
