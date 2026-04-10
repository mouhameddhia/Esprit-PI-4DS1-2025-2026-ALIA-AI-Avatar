import React from 'react';
import { LogOut, GraduationCap, BarChart2, ArrowRight, Mic } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import './MedRepPortal.css';

const MedRepPortal = () => {
  const navigate = useNavigate();

  const handleSignOut = () => {
    navigate('/login');
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.2 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 30 },
    show: { opacity: 1, y: 0, transition: { duration: 0.7, ease: [0.16, 1, 0.3, 1] } }
  };

  return (
    <div className="portal-container">
      {/* Animated Background Aura */}
      <div className="portal-bg-aura"></div>
      <div className="portal-bg-aura-2"></div>

      {/* Navbar */}
      <nav className="portal-navbar">
        <div className="portal-nav-left">
          <div className="portal-nav-logo">A</div>
          <div className="portal-nav-text">
            <span className="portal-nav-title">ALIA Platform</span>
            <span className="portal-nav-subtitle">Medical Representative Portal</span>
          </div>
        </div>
        <div className="portal-nav-right">
          <button className="portal-signout-btn" onClick={handleSignOut}>
            <LogOut size={18} />
            <span>Sign Out</span>
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <main className="portal-content">
        <motion.div
          className="portal-inner-content"
          variants={containerVariants}
          initial="hidden"
          animate="show"
        >
          {/* Avatar Section */}
          <motion.div variants={itemVariants} className="portal-avatar-section">
            <div className="portal-avatar-wrapper">
              <div className="portal-avatar-inner">
                <img
                  src="https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&q=80&w=400&h=400"
                  alt="ALIA AI"
                  className="portal-avatar-img"
                />
                <div className="portal-avatar-overlay">
                  <span>Hello, Im ALIA.</span>
                </div>
              </div>
            </div>
            <div className="portal-avatar-status">
              <div className="portal-status-dot"></div>
              <span>Active</span>
            </div>
          </motion.div>

          {/* Header Strings */}
          <motion.div variants={itemVariants} className="portal-header">
            <h1>Welcome back, Representative</h1>
            <p>Select a mode below to continue enhancing your skills or reviewing your data.</p>
          </motion.div>

          {/* Cards Grid */}
          <div className="portal-cards-grid">
            {/* Training Mode Card */}
            <motion.div variants={itemVariants} className="portal-card training" onClick={() => navigate('/rep/training')} style={{ cursor: 'pointer' }}>
              <div className="portal-card-icon mode-1">
                <Mic size={24} color="#7c3aed" />
              </div>
              <h3>Training Mode</h3>
              <p>Practice realistic, dynamic sales pitches with our AI-simulated physicians to sharpen your real-world communication.</p>
              <div className="portal-card-link">
                Enter Simulation <ArrowRight size={18} />
              </div>
            </motion.div>

            {/* Analytics & Pairing Card */}
            <motion.div variants={itemVariants} className="portal-card analytics" onClick={() => navigate('/analytics/pairing')} style={{ cursor: 'pointer' }}>
              <div className="portal-card-icon mode-2" style={{ background: 'rgba(45, 212, 191, 0.1)', width: '48px', height: '48px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem' }}>
                <BarChart2 size={24} color="#2dd4bf" />
              </div>
              <h3>Analytics & Pairing</h3>
              <p>View AI-generated insights, track your conversational skills, and access tailored product-rep recommendations.</p>
              <div className="portal-card-link">
                View Insights <ArrowRight size={18} />
              </div>
            </motion.div>
          </div>

          {/* Bottom Dashboard Button */}
          <motion.div variants={itemVariants}>
            <button className="portal-dashboard-btn" onClick={() => navigate('/rep/dashboard')}>
              <BarChart2 size={20} />
              View Full Performance Dashboard
            </button>
          </motion.div>
        </motion.div>
      </main>
    </div>
  );
};

export default MedRepPortal;
