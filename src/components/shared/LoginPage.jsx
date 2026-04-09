import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, LogIn, X } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import './LoginPage.css';

const LoginPage = ({ onClose }) => {
  const [role, setRole] = useState('Medical Rep');
  const navigate = useNavigate();

  const handleLogin = (e) => {
    e.preventDefault();
    if (role === 'Medical Rep') {
      navigate('/portal');
    } else if (role === 'Physician') {
      navigate('/physician/portal');
    }
  };

  return (
    <AnimatePresence>
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="login-page-container"
      >
        <button className="login-close-btn" onClick={onClose} aria-label="Close">
          <X size={24} />
        </button>

        <div className="login-left-panel">
          <div className="login-left-content">
            <motion.div 
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.1 }}
              className="login-logo-container"
            >
              <div className="login-logo-icon">
                <Brain size={36} />
              </div>
              <div className="login-logo-text">
                <h1>ALIA</h1>
                <span>AI Avatar Platform</span>
              </div>
            </motion.div>

            <motion.h2 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="login-title"
            >
              Welcome to the Future of<br />
              <span>Pharmaceutical Training</span>
            </motion.h2>
            
            <motion.p 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="login-subtitle"
            >
              Interact with Nour, your intelligent AI representative, for training simulations and instant product knowledge.
            </motion.p>

            <motion.div 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="login-badges"
            >
              <span className="login-badge">
                <span className="login-badge-dot" style={{ backgroundColor: '#7c3aed' }}></span> Interactive AI Training
              </span>
              <span className="login-badge">
                <span className="login-badge-dot" style={{ backgroundColor: '#10b981' }}></span> Real-time Analytics
              </span>
              <span className="login-badge">
                <span className="login-badge-dot" style={{ backgroundColor: '#ea580c' }}></span> Smart Pairing System
              </span>
            </motion.div>
          </div>
        </div>

        <div className="login-right-panel">
           <motion.div 
             initial={{ opacity: 0, scale: 0.95 }}
             animate={{ opacity: 1, scale: 1 }}
             transition={{ delay: 0.2, type: 'spring', stiffness: 100 }}
             className="login-card"
           >
            <h2 className="login-card-title">Sign In</h2>

            <form onSubmit={handleLogin}>
              <div className="login-form-group">
                <label className="login-form-label">I am a:</label>
                <div className="login-role-toggles">
                  <button
                    type="button"
                    onClick={() => setRole('Medical Rep')}
                    className={`login-role-btn ${role === 'Medical Rep' ? 'active' : ''}`}
                  >
                    Medical Rep
                  </button>
                  <button
                    type="button"
                    onClick={() => setRole('Physician')}
                    className={`login-role-btn ${role === 'Physician' ? 'active' : ''}`}
                  >
                    Physician
                  </button>
                </div>
              </div>

              <div className="login-form-group">
                <label className="login-form-label">Email</label>
                <input 
                  type="email" 
                  placeholder="your.email@example.com" 
                  className="login-input"
                />
              </div>

              <div className="login-form-group">
                <label className="login-form-label">Password</label>
                <input 
                  type="password" 
                  placeholder="••••••••" 
                  className="login-input"
                />
              </div>

              <button type="submit" className="login-submit-btn">
                <LogIn size={18} /> Sign In
              </button>
            </form>

            <p className="login-footer-text">
              Don't have an account? <Link to="/signup">Sign Up</Link>
            </p>
          </motion.div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default LoginPage;
