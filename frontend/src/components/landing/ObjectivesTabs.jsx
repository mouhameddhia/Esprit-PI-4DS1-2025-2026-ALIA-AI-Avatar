import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Stethoscope, Presentation, Activity, Globe } from 'lucide-react';

export default function ObjectivesTabs() {
  const [activeTab, setActiveTab] = useState(0);

  const tabs = [
    {
      id: 0,
      title: 'Mode 1: Train the rep',
      short: 'Training',
      color: 'var(--mode-training)',
      icon: <Stethoscope size={20} />,
      bo: [
        'Build confidence through repeated, graded simulations',
        'Shorten time-to-readiness with structured scenarios',
        'Track clarity, accuracy, and persuasion over time',
      ],
      dso: [
        'NLP for medical entities and objection patterns',
        'Sentiment & tone signals for coaching',
        'Adaptive difficulty based on performance',
      ],
    },
    {
      id: 1,
      title: 'Mode 2: Nour as the rep',
      short: 'Engagement',
      color: 'var(--mode-engagement)',
      icon: <Presentation size={20} />,
      bo: [
        'Give HCPs compliant, on-demand product context',
        'Capture structured interaction data for the field force',
        'Align messaging with approved claims and materials',
      ],
      dso: [
        'Multilingual responses (e.g. FR / EN / AR / ES)',
        'RAG over curated product knowledge',
        'Low-latency conversational UX',
      ],
    },
  ];

  return (
    <section id="modes" className="landing-section container">
      <motion.div
        className="landing-section-head"
        initial={{ opacity: 0, y: 18 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.55 }}
      >
        <p className="landing-kicker">Two operating modes</p>
        <h2>Business &amp; data-science objectives</h2>
        <p>
          The same platform switches between internal <strong>simulation</strong> and external{' '}
          <strong>engagement</strong>—each mode with clear business outcomes and technical enablers.
        </p>
      </motion.div>

      <div className="objectives-tab-row">
        {tabs.map((tab, i) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(i)}
            className={`objectives-tab-btn ${activeTab === i ? 'is-active' : ''}`}
            style={{
              '--tab-color': tab.color,
            }}
          >
            {activeTab === i && (
              <motion.div
                layoutId="landingTabBg"
                style={{
                  position: 'absolute',
                  inset: 0,
                  background: `color-mix(in srgb, ${tab.color} 28%, transparent)`,
                  border: `1px solid ${tab.color}`,
                  borderRadius: 'inherit',
                  zIndex: 0,
                }}
                transition={{ type: 'spring', stiffness: 380, damping: 32 }}
              />
            )}
            <span style={{ position: 'relative', zIndex: 1, display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}>
              {tab.icon}
              <span className="tab-label-long">{tab.title}</span>
              <span className="tab-label-short">{tab.short}</span>
            </span>
          </button>
        ))}
      </div>

      <div style={{ position: 'relative', minHeight: 280 }}>
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            className="glass-panel"
            style={{
              padding: 'clamp(1.5rem, 4vw, 2.75rem)',
              borderLeft: `4px solid ${tabs[activeTab].color}`,
              borderRadius: 20,
            }}
          >
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                gap: '2.5rem',
              }}
            >
              <div>
                <h3
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    marginBottom: '1.25rem',
                    color: tabs[activeTab].color,
                    fontSize: '1.1rem',
                  }}
                >
                  <Activity size={22} /> Business objectives
                </h3>
                <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                  {tabs[activeTab].bo.map((line) => (
                    <motion.li
                      key={line}
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3 }}
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '0.85rem',
                        padding: '1rem 1.1rem',
                        borderRadius: 12,
                        background: 'color-mix(in srgb, var(--bg-color) 92%, transparent)',
                        border: '1px solid var(--glass-border)',
                      }}
                    >
                      <span
                        style={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          background: tabs[activeTab].color,
                          marginTop: 6,
                          flexShrink: 0,
                        }}
                        aria-hidden
                      />
                      {line}
                    </motion.li>
                  ))}
                </ul>
              </div>
              <div>
                <h3
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    marginBottom: '1.25rem',
                    color: tabs[activeTab].color,
                    fontSize: '1.1rem',
                  }}
                >
                  <Globe size={22} /> Features
                </h3>
                <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                  {tabs[activeTab].dso.map((line) => (
                    <motion.li
                      key={line}
                      initial={{ opacity: 0, x: 8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3 }}
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '0.85rem',
                        padding: '1rem 1.1rem',
                        borderRadius: 12,
                        background: 'color-mix(in srgb, var(--bg-color) 92%, transparent)',
                        border: '1px solid var(--glass-border)',
                      }}
                    >
                      <span
                        style={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          background: tabs[activeTab].color,
                          marginTop: 6,
                          flexShrink: 0,
                        }}
                        aria-hidden
                      />
                      {line}
                    </motion.li>
                  ))}
                </ul>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  );
}
