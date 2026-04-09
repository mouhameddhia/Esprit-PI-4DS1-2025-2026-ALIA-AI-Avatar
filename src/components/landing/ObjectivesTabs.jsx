import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Stethoscope, Presentation, Activity, Globe } from 'lucide-react';

const ObjectivesTabs = () => {
  const [activeTab, setActiveTab] = useState(0);

  const tabs = [
    {
      id: 0,
      title: "Mode 1: Training & Simulation",
      color: "var(--mode-training)",
      icon: <Stethoscope />,
      bo: ["Improve rep confidence", "Reduce training time by 30%", "Track longitudinal progress"],
      dso: ["NLP for medical entity recognition", "Sentiment analysis on rep tone", "Dynamic difficulty adjustment"]
    },
    {
      id: 1,
      title: "Mode 2: Client Engagement",
      color: "var(--mode-engagement)",
      icon: <Presentation />,
      bo: ["Boost doctor engagement rates", "Ensure 100% compliant information", "Automated CRM logging"],
      dso: ["Real-time language adaptation (FR/EN/AR/ES)", "Avatar lip-sync latency < 200ms", "Knowledge graph retrieval (RAG)"]
    }
  ];

  return (
    <section className="container" style={{ padding: '6rem 2rem' }}>
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <h2>Business & Data Science Objectives</h2>
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginBottom: '3rem' }}>
        {tabs.map((tab, i) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(i)}
            className="btn"
            style={{
              position: 'relative',
              background: 'transparent',
              border: `1px solid ${tab.color}`,
              color: activeTab === i ? '#1e293b' : tab.color,
              overflow: 'hidden'
            }}
          >
            {activeTab === i && (
              <motion.div
                layoutId="activeTabIndicator"
                style={{
                  position: 'absolute',
                  inset: 0,
                  backgroundColor: tab.color,
                  zIndex: 0,
                  borderRadius: 'inherit'
                }}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
              />
            )}
            <span style={{ position: 'relative', zIndex: 10, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              {tab.icon} {tab.title}
            </span>
          </button>
        ))}
      </div>

      <div style={{ position: 'relative', minHeight: '300px' }}>
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="glass-panel"
            style={{
              padding: '3rem',
              borderLeft: `4px solid ${tabs[activeTab].color}`
            }}
          >
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '3rem' }}>
              <div>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', color: tabs[activeTab].color }}>
                  <Activity size={24} /> Business Objectives
                </h3>
                <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {tabs[activeTab].bo.map((item, idx) => (
                    <li key={idx} style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '8px' }}>
                      <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: tabs[activeTab].color }}></div>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
              
              <div>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', color: tabs[activeTab].color }}>
                  <Globe size={24} /> Data Science Objectives
                </h3>
                <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {tabs[activeTab].dso.map((item, idx) => (
                    <li key={idx} style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '8px' }}>
                      <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: tabs[activeTab].color }}></div>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  );
};

export default ObjectivesTabs;
