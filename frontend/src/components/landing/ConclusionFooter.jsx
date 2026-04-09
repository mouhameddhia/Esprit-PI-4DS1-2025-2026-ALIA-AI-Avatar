import React from 'react';
import { motion } from 'framer-motion';
import { CheckSquare, Send } from 'lucide-react';

const ConclusionFooter = () => {
  const steps = [
    "Finalize tech stack",
    "Build MVP voice pipeline",
    "Integrate avatar lip sync",
    "Implement Mode 1 then Mode 2",
    "Compliance validation",
    "CRM integration"
  ];

  return (
    <>
      <section style={{ padding: '6rem 2rem', borderTop: '1px solid var(--glass-border)', backgroundColor: 'var(--bg-color)' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
            <h2 className="text-gradient">Ready for Excellence?</h2>
            <p style={{ color: 'var(--text-secondary)', maxWidth: '600px', margin: '1rem auto' }}>
              ALIA unifies training and engagement with compliance and innovation. See our roadmap below and join the pilot today.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '4rem' }}>
            {/* Checklist */}
            <motion.div 
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="glass-panel"
              style={{ padding: '2.5rem' }}
            >
              <h3 style={{ marginBottom: '2rem' }}>Next Steps</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                {steps.map((step, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <CheckSquare color={i < 3 ? 'var(--mode-training)' : 'var(--text-secondary)'} />
                    <span style={{ color: i < 3 ? 'white' : 'var(--text-secondary)' }}>{step}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Contact / CTA */}
            <motion.div 
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}
            >
              <div style={{ background: 'var(--bg-color-light)', padding: '2.5rem', borderRadius: '16px', border: '1px solid var(--glass-border)' }}>
                <h3 style={{ marginBottom: '1.5rem' }}>Join Pilot Program</h3>
                <form style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }} onSubmit={(e) => e.preventDefault()}>
                  <input type="text" placeholder="Name" style={{ padding: '0.75rem', borderRadius: '8px', background: 'rgba(0,0,0,0.2)', border: '1px solid var(--glass-border)', color: 'white', width: '100%' }} />
                  <input type="email" placeholder="Email Address" style={{ padding: '0.75rem', borderRadius: '8px', background: 'rgba(0,0,0,0.2)', border: '1px solid var(--glass-border)', color: 'white', width: '100%' }} />
                  <textarea placeholder="Message" rows="3" style={{ padding: '0.75rem', borderRadius: '8px', background: 'rgba(0,0,0,0.2)', border: '1px solid var(--glass-border)', color: 'white', width: '100%', resize: 'none' }}></textarea>
                  <button className="btn btn-primary" style={{ width: '100%', marginTop: '0.5rem' }}>
                    Request Demo <Send size={18} />
                  </button>
                </form>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ borderTop: '1px solid var(--glass-border)', padding: '2rem 0', background: '#0a0f1e' }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ color: 'var(--text-secondary)' }}>
            &copy; {new Date().getFullYear()} ALIA. All rights reserved.
          </div>
          <div style={{ display: 'flex', gap: '2rem' }}>
            <a href="#" style={{ color: 'var(--text-secondary)', textDecoration: 'none' }}>About</a>
            <a href="#" style={{ color: 'var(--text-secondary)', textDecoration: 'none' }}>Contact</a>
            <a href="#" style={{ color: 'var(--text-secondary)', textDecoration: 'none' }}>Privacy</a>
            <a href="#" style={{ color: 'var(--text-secondary)', textDecoration: 'none' }}>Partners</a>
          </div>
        </div>
      </footer>
    </>
  );
};

export default ConclusionFooter;
