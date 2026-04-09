import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, User, Package, Play, BarChart2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import './MedRepPortal.css'; // Ambient portal styling
import './MedRepTraining.css';

const personas = [
  { id: 'p1', name: 'Dr. Skeptical', specialty: 'Cardiology', difficulty: 'hard', traits: 'Critical, Evidence-Focused', desc: 'Challenges claims, demands robust clinical data' },
  { id: 'p2', name: 'Dr. Busy', specialty: 'General Practice', difficulty: 'medium', traits: 'Time-Pressed, Direct', desc: 'Wants quick, practical information' },
  { id: 'p3', name: 'Dr. Friendly', specialty: 'Endocrinology', difficulty: 'easy', traits: 'Collaborative, Open', desc: 'Receptive to new treatments, asks clarifying questions' },
  { id: 'p4', name: 'Dr. Academic', specialty: 'Neurology', difficulty: 'hard', traits: 'Research-Oriented, Detail-Focused', desc: 'Wants mechanism of action, study methodology details' },
  { id: 'p5', name: 'Dr. Conservative', specialty: 'Oncology', difficulty: 'medium', traits: 'Cautious, Safety-Focused', desc: 'Prioritizes patient safety, wants comprehensive side effect profiles' }
];

const products = [
  { id: 'pr1', name: 'CardioGuard', category: 'Cardiovascular', desc: 'Advanced ACE inhibitor for hypertension management', indication: 'Treatment of hypertension and heart failure' },
  { id: 'pr2', name: 'NeuroShield', category: 'Neurology', desc: 'Next-generation anticonvulsant for epilepsy', indication: 'Partial and generalized seizures in adults' },
  { id: 'pr3', name: 'DiabetoCare Plus', category: 'Endocrinology', desc: 'Long-acting GLP-1 receptor agonist', indication: 'Type 2 diabetes mellitus management' },
  { id: 'pr4', name: 'RespiClear', category: 'Respiratory', desc: 'Advanced bronchodilator combination therapy', indication: 'COPD and severe asthma treatment' },
  { id: 'pr5', name: 'OncoPro', category: 'Oncology', desc: 'Targeted immunotherapy for solid tumors', indication: 'Metastatic melanoma and lung cancer' },
  { id: 'pr6', name: 'PainRelief XR', category: 'Pain Management', desc: 'Extended-release analgesic formulation', indication: 'Moderate to severe chronic pain' }
];

const MedRepTraining = () => {
  const navigate = useNavigate();
  const [selectedPersona, setSelectedPersona] = useState(null);
  const [selectedProduct, setSelectedProduct] = useState(null);

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } }
  };

  return (
    <div className="portal-container" style={{ position: 'relative', overflowX: 'hidden' }}>
      {/* Background Auras */}
      <div className="portal-bg-aura"></div>
      <div className="portal-bg-aura-2" style={{ top: '60%' }}></div>

      <motion.main 
        className="training-main relative z-10"
        variants={containerVariants}
        initial="hidden"
        animate="show"
      >
        {/* Navigation Bar */}
        <motion.div variants={itemVariants} className="training-nav-bar">
          <button className="dashboard-back-btn" onClick={() => navigate('/portal')}>
            <ArrowLeft size={16} /> Back
          </button>
          
          <button className="dashboard-back-btn" onClick={() => navigate('/rep/dashboard')} style={{ background: 'var(--glass-bg)' }}>
            <BarChart2 size={16} /> My Dashboard
          </button>
        </motion.div>

        {/* Header */}
        <motion.div variants={itemVariants} className="training-header">
          <h1>Rep Evaluation & Training</h1>
          <p>Select a product and doctor persona to begin your adaptive training session</p>
        </motion.div>

        {/* Doctor Persona Selection */}
        <motion.section variants={itemVariants} className="training-section">
          <h2 className="training-section-title">Select Doctor Persona</h2>
          <div className="training-grid">
            {personas.map((persona) => (
              <motion.div 
                key={persona.id} 
                className={`training-card ${selectedPersona === persona.id ? 'selected' : ''}`}
                onClick={() => setSelectedPersona(persona.id)}
                whileTap={{ scale: 0.98 }}
              >
                <div className="card-header">
                  <div className="icon-box"><User size={24} /></div>
                  <span className={`diff-badge diff-${persona.difficulty}`}>{persona.difficulty}</span>
                </div>
                <div>
                  <h3 className="card-title">{persona.name}</h3>
                  <p className="card-subtitle">{persona.specialty}</p>
                  <p className="card-traits">{persona.traits}</p>
                </div>
                <p className="card-desc">{persona.desc}</p>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Product Selection */}
        <motion.section variants={itemVariants} className="training-section" style={{ marginTop: '2rem', marginBottom: '8rem' }}>
          <h2 className="training-section-title">Select a Product</h2>
          <div className="training-grid">
            {products.map((product) => (
              <motion.div 
                key={product.id} 
                className={`training-card ${selectedProduct === product.id ? 'selected' : ''}`}
                onClick={() => setSelectedProduct(product.id)}
                whileTap={{ scale: 0.98 }}
              >
                <div className="card-header">
                  <div className="icon-box"><Package size={24} /></div>
                </div>
                <div>
                  <h3 className="card-title">{product.name}</h3>
                  <p className="card-subtitle">{product.category}</p>
                </div>
                <p className="card-desc">{product.desc}</p>
                
                <div className="product-divider"></div>
                <div>
                  <div className="product-label">Indication</div>
                  <p className="card-desc" style={{ marginTop: '0', color: 'var(--text-primary)' }}>{product.indication}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.section>

      </motion.main>

      {/* Floating Action Button - Pops up when both are selected */}
      <AnimatePresence>
        {selectedPersona && selectedProduct && (
          <motion.button
            initial={{ y: 100, opacity: 0, x: '-50%' }}
            animate={{ y: 0, opacity: 1, x: '-50%' }}
            exit={{ y: 100, opacity: 0, x: '-50%' }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            className="training-start-cta"
            onClick={() => navigate('/rep/simulation')}
          >
            <Play fill="currentColor" size={20} />
            Launch Audio/Video Training 
          </motion.button>
        )}
      </AnimatePresence>
    </div>
  );
};

export default MedRepTraining;
