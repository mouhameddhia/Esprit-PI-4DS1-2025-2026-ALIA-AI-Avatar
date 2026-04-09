import React from 'react';
import { motion } from 'framer-motion';
import { Bot, Database, MessageSquare } from 'lucide-react';

const ProposedSolution = () => {
  return (
    <section style={{ padding: '6rem 2rem', backgroundColor: 'var(--bg-color-light)' }}>
      <div className="container">
        <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
          <h2>One Avatar, Two Faces</h2>
          <p style={{ color: 'var(--text-secondary)', maxWidth: '600px', margin: '1rem auto' }}>
            ALIA serves as a unified platform adapting intuitively between internal training and external commercial engagement.
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
          gap: '3rem',
          alignItems: 'center'
        }}>
          {/* Visual Side */}
          <motion.div 
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="glass-panel"
            style={{ height: '400px', position: 'relative', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          >
             <div style={{ position: 'absolute', top: 0, left: 0, width: '50%', height: '100%', background: 'var(--mode1-gradient)', zIndex: 0 }}></div>
             <div style={{ position: 'absolute', top: 0, right: 0, width: '50%', height: '100%', background: 'var(--mode2-gradient)', zIndex: 0 }}></div>
             
             <div style={{ position: 'relative', zIndex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem' }}>
                <Bot size={80} color="var(--text-primary)" />
                <div style={{ display: 'flex', gap: '1rem' }}>
                  <span style={{ background: 'rgba(52, 211, 153, 0.2)', padding: '0.5rem 1rem', borderRadius: '20px', color: '#34d399', fontSize: '0.875rem' }}>Virtual Doctor</span>
                  <span style={{ background: 'rgba(251, 146, 60, 0.2)', padding: '0.5rem 1rem', borderRadius: '20px', color: '#fb923c', fontSize: '0.875rem' }}>Medical Presenter</span>
                </div>
             </div>
          </motion.div>

          {/* Text Side */}
          <motion.div 
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}
          >
            <div style={{ display: 'flex', gap: '1.5rem' }}>
              <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(56, 189, 248, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, color: 'var(--soft-blue)' }}>
                <MessageSquare />
              </div>
              <div>
                <h3 style={{ marginBottom: '0.5rem' }}>Dynamic Role-Play</h3>
                <p style={{ color: 'var(--text-secondary)' }}>ALIA morphs into various personas (Doctor, Pharmacist, Patient) to stress-test your reps in real-life scenarios.</p>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1.5rem' }}>
              <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(251, 146, 60, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, color: 'var(--soft-orange)' }}>
                <Database />
              </div>
              <div>
                <h3 style={{ marginBottom: '0.5rem' }}>Seamless CRM Integration</h3>
                <p style={{ color: 'var(--text-secondary)' }}>After an engagement, ALIA automatically updates CRM dashboards, ensuring perfect data continuity and follow-ups.</p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default ProposedSolution;
