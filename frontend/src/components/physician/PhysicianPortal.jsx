import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  LogOut, 
  MessageSquare, 
  Search, 
  Calendar, 
  History, 
  ChevronRight, 
  Send, 
  Mic, 
  Activity, 
  Info, 
  ArrowRight, 
  ShieldCheck, 
  Package, 
  Star 
} from 'lucide-react';
import './PhysicianPortal.css';
import '../medrep/MedRepPortal.css';
import '../medrep/MedRepSimulation.css'; // Reusing chat bubbles

const PhysicianPortal = () => {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      sender: 'doctor', 
      text: "Hello Doctor! I'm Nour, your AI pharmaceutical representative. How can I assist you today? You can ask me about any of our products, clinical data, dosing guidelines, or request information about upcoming webinars.",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [activeSection, setActiveSection] = useState('chat');
  const chatEndRef = useRef(null);

  const historyItems = [
    {
      id: 'h1',
      name: 'Ahmed Hassan',
      initial: 'A',
      date: '2026-01-28',
      product: 'CardioGuard',
      notes: 'Discussed dosing for elderly patients with renal impairment',
      followUp: 'Follow-up Requested',
      rating: 4,
      requested: true
    },
    {
      id: 'h2',
      name: 'Layla Mahmoud',
      initial: 'L',
      date: '2026-01-22',
      product: 'NeuroShield',
      notes: 'Provided clinical trial data for partial seizures',
      followUp: 'Request Follow-up',
      rating: 5,
      requested: false
    }
  ];

  const products = [
    { id: 'pr1', name: 'CardioGuard', desc: 'ACE Inhibitor' },
    { id: 'pr2', name: 'NeuroShield', desc: 'Anticonvulsant' },
    { id: 'pr3', name: 'DiabetoCare Plus', desc: 'GLP-1 Agonist' }
  ];

  const webinars = [
    { date: 'OCT 24', title: 'Advances in Hypertension', time: '14:00 GMT' },
    { date: 'NOV 12', title: 'Managing Diabetic Renal Risk', time: '10:00 GMT' }
  ];

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = () => {
    if (!inputText.trim()) return;

    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text: inputText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages([...messages, userMsg]);
    setInputText('');

    // AI Rep Response simulation
    setTimeout(() => {
      const response = {
        id: Date.now() + 1,
        sender: 'doctor',
        text: "Certainly, Doctor. I can provide the latest peer-reviewed clinical data for that specific indication. Would you like me to send the PDF abstract to your registered email, or should I summarize the key efficacy findings here?",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, response]);
    }, 1500);
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <div className="portal-container" style={{ position: 'relative' }}>
      <div className="portal-bg-aura"></div>
      <div className="portal-bg-aura-2" style={{ top: '60%' }}></div>

      <nav className="portal-navbar">
        <div className="portal-nav-left">
          <div className="portal-nav-logo" style={{ background: '#7c3aed' }}>P</div>
          <div className="portal-nav-text">
            <span className="portal-nav-title">ALIA - Physician Portal</span>
            <span className="portal-nav-subtitle">Your AI Medical Representative</span>
          </div>
        </div>
        <div className="portal-nav-right">
          <button className="portal-signout-btn" onClick={() => navigate('/login')}>
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
          
          {/* Left: AI Rep Info & Quick Access */}
          <motion.aside variants={itemVariants} className="ai-rep-card">
            <div className="ai-rep-avatar">
              <img 
                src="https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&q=80&w=400&h=400" 
                alt="Nour AI" 
              />
              <div className="ai-rep-status">Available 24/7</div>
            </div>
            
            <div className="ai-rep-info">
              <h2>Nour</h2>
              <p>Your AI Pharmaceutical Rep</p>
            </div>

            <div className="physician-stats">
              <div className={`physician-stat-item ${activeSection === 'history' ? 'active' : ''}`} onClick={() => setActiveSection('history')}>
                <History size={20} color="#7c3aed" />
                <div style={{ textAlign: 'left' }}>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Interaction History</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>View past consultations</div>
                </div>
                <ChevronRight size={16} style={{ marginLeft: 'auto' }} />
              </div>

              <div className={`physician-stat-item ${activeSection === 'products' ? 'active' : ''}`} onClick={() => setActiveSection('products')}>
                <Package size={20} color="#7c3aed" />
                <div style={{ textAlign: 'left' }}>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Browse Products</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Dosing & Indication guides</div>
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
                    <button className="btn-notify">Get Notified</button>
                  </div>
                ))}
              </div>
            </div>
          </motion.aside>

          {/* Right: AI Chat / History / Products Interface */}
          <motion.section variants={itemVariants} className="physician-chat-box">
            {activeSection === 'history' ? (
              <div className="history-view">
                <div className="history-header">
                  <div>
                    <h2>Interaction History</h2>
                    <p>Past meetings with medical representatives</p>
                  </div>
                  <button className="history-back" onClick={() => setActiveSection('chat')}>
                    <ArrowRight size={16} style={{ transform: 'rotate(180deg)' }} />
                    Back to Chat
                  </button>
                </div>

                <div className="history-list">
                  {historyItems.map((item) => (
                    <div key={item.id} className="history-card">
                      <div className="history-card-header">
                        <div className="history-avatar-circle">{item.initial}</div>
                        <div>
                          <h3>{item.name}</h3>
                          <span className="history-date">{item.date}</span>
                        </div>
                        <div className="history-stars">
                          {Array.from({ length: item.rating }).map((_, idx) => (
                            <Star key={idx} size={16} />
                          ))}
                        </div>
                      </div>

                      <div className="history-card-body">
                        <div className="history-card-section">
                          <strong>Product Discussed:</strong>
                          <span>{item.product}</span>
                        </div>
                        <div className="history-card-section">
                          <strong>Notes:</strong>
                          <span>{item.notes}</span>
                        </div>
                      </div>

                      <div className="history-card-actions">
                        <button className={`history-pill ${item.requested ? 'requested' : 'action'}`}>
                          {item.followUp}
                        </button>
                        <button className="history-rate">Rate</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : activeSection === 'products' ? (
              <div className="products-view">
                <div className="history-header">
                  <div>
                    <h2>Browse Products</h2>
                    <p>Dosing, indications, and clinical summaries</p>
                  </div>
                  <button className="history-back" onClick={() => setActiveSection('chat')}>
                    <ArrowRight size={16} style={{ transform: 'rotate(180deg)' }} />
                    Back to Chat
                  </button>
                </div>
                <div className="products-grid">
                  {products.map((product) => (
                    <div key={product.id} className="product-card">
                      <h3>{product.name}</h3>
                      <p>{product.desc}</p>
                      <button className="btn-notify">View Details</button>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <>
                <div className="chat-header">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <Activity size={20} color="#7c3aed" />
                    <span style={{ fontWeight: 700 }}>Ask Our AI Representative</span>
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Voice or text - get instant, compliant product information
                  </div>
                </div>

                <div className="chat-history">
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
                        <span style={{ fontSize: '0.7rem', opacity: 0.6, display: 'block', marginTop: '0.5rem' }}>
                          {msg.timestamp}
                        </span>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                  <div ref={chatEndRef} />
                </div>

                <div className="chat-input-wrapper" style={{ padding: '1.25rem 2rem' }}>
                  <button className="voice-btn" style={{ background: 'transparent', border: '1px solid var(--glass-border)' }}>
                    <Mic size={20} />
                  </button>
                  <input 
                    type="text" 
                    className="chat-input-field" 
                    placeholder="Ask Nour about clinical data or product info..."
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  />
                  <button className="btn-send" onClick={handleSendMessage} style={{ width: '50px', height: '50px' }}>
                    <Send size={20} />
                  </button>
                </div>
                
                <div style={{ textAlign: 'center', paddingBottom: '0.75rem', fontSize: '0.7rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                  <ShieldCheck size={12} /> Compliance-verified AI responses
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
