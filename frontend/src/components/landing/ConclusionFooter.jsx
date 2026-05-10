import React from 'react';
import { motion } from 'framer-motion';
import { CheckSquare, Send, ArrowRight, Mail, Phone, MapPin, ExternalLink } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const steps = [
  'Voice & text chat integrated',
  'Auth & roles (rep / physician)',
  'Conversation storage & summaries',
  'Expand simulation rubrics & analytics',
  'Compliance review with medical affairs',
  'CRM & field tooling integration',
  'Healthcare Professionals friendly',
  'Realistic, high-pressure simulations',
  'Compliant product conversations',
  'Structured interaction data for the field force',
  'Align messaging with approved claims and materials',
  'Multilingual responses (e.g. FR / EN / AR / ES)',
  'RAG over curated product knowledge',
  'Low-latency conversational UX',
  'Adaptive difficulty based on performance',
  'Sentiment & tone signals for coaching',
];

export default function ConclusionFooter() {
  const navigate = useNavigate();
  const year = new Date().getFullYear();

  return (
    <>
      <section id="get-started" className="landing-section landing-footer-cta" style={{ borderTop: '1px solid var(--glass-border)' }}>
        <div className="container">
          <motion.div
            className="landing-section-head"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.55 }}
          >
            <p className="landing-kicker">Next</p>
            <h2 className="text-gradient">Ready when you are</h2>
            <p>
              ALIA bridges simulation, evaluation, and compliant HCP engagement. Create an account to explore the portals,
              or leave a note if you are evaluating a pilot with your team.
            </p>
          </motion.div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 300px), 1fr))',
              gap: '2.5rem',
            }}
          >
            <motion.div
              initial={{ opacity: 0, x: -24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.55 }}
              className="glass-panel"
              style={{ padding: '2.25rem', borderRadius: 22 }}
            >
              <h3 style={{ marginBottom: '1.5rem', fontSize: '1.2rem' }}>Roadmap highlights</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {steps.map((step, i) => (
                  <div key={step} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.85rem' }}>
                    <CheckSquare
                      size={22}
                      color={i < 16 ? 'var(--mode-training)' : 'var(--text-secondary)'}
                      style={{ flexShrink: 0, marginTop: 2 }}
                      aria-hidden
                    />
                    <span style={{ color: i < 16 ? 'var(--text-primary)' : 'var(--text-secondary)', lineHeight: 1.55 }}>
                      {step}
                    </span>
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.55 }}
              style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: '1rem' }}
            >
              <div
                style={{
                  padding: '2.25rem',
                  borderRadius: 22,
                  border: '1px solid var(--glass-border)',
                  background: 'color-mix(in srgb, var(--bg-color-light) 90%, transparent)',
                }}
              >
                <h3 style={{ marginBottom: '1rem', fontSize: '1.2rem' }}>Open the app</h3>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '1.25rem', lineHeight: 1.6 }}>
                  Sign up as a medical representative or physician to access the portals. This demo uses secure auth and
                  saves chat sessions for history and summaries.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <motion.button
                    type="button"
                    className="btn btn-primary"
                    style={{ width: '100%', justifyContent: 'center' }}
                    onClick={() => navigate('/signup')}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    Create account <ArrowRight size={18} />
                  </motion.button>
                  <motion.button
                    type="button"
                    className="btn btn-secondary"
                    style={{ width: '100%', justifyContent: 'center' }}
                    onClick={() => navigate('/login')}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    Sign in
                  </motion.button>
                </div>
              </div>

              <div
                className="landing-cta-form"
                style={{
                  padding: '2rem',
                  borderRadius: 22,
                  border: '1px solid var(--glass-border)',
                  background: 'color-mix(in srgb, var(--glass-bg) 80%, transparent)',
                }}
              >
                <h3 style={{ marginBottom: '0.75rem', fontSize: '1.05rem' }}>Pilot interest (placeholder)</h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1rem' }}>
                  Form is visual only for now—wire it to your CRM or inbox when you are ready.
                </p>
                <form
                  className="landing-cta-form"
                  style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}
                  onSubmit={(e) => e.preventDefault()}
                >
                  <input type="text" name="name" placeholder="Name" autoComplete="name" />
                  <input type="email" name="email" placeholder="Work email" autoComplete="email" />
                  <textarea name="message" placeholder="Organization / use case" rows={3} />
                  <button type="submit" className="btn btn-primary" style={{ width: '100%', marginTop: '0.35rem' }}>
                    Request demo <Send size={18} />
                  </button>
                </form>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      <footer id="contact" className="landing-footer">
        <div className="container landing-footer-grid">
          <div>
            <div className="brand" style={{ marginBottom: '1rem' }}>
              <div className="brand-mark">AL</div>
              <div>
                <div style={{ fontWeight: 700, letterSpacing: '0.02em', color: 'var(--text-primary)' }}>ALIA</div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>AI Avatar Platform</div>
              </div>
            </div>
            <p className="landing-footer-copy">
              AI-powered simulation and engagement for pharmaceutical training, readiness, and compliant field conversations.
            </p>
            <div className="landing-footer-socials">
              <a href="https://www.linkedin.com" target="_blank" rel="noreferrer" aria-label="LinkedIn">
                <span>in</span>
              </a>
              <a href="https://www.youtube.com" target="_blank" rel="noreferrer" aria-label="YouTube">
                <span>YT</span>
              </a>
              <a href="mailto:contact@alia-platform.com" aria-label="Email">
                <Mail size={18} />
              </a>
            </div>
          </div>

          <div>
            <h4>Quick links</h4>
            <div className="landing-footer-links">
              <button type="button" className="landing-nav-link" onClick={() => navigate('/signup')}>
                Create account
              </button>
              <button type="button" className="landing-nav-link" onClick={() => navigate('/login')}>
                Sign in
              </button>
              <a href="#impact">Impact metrics</a>
              <a href="#testimonials">Testimonials</a>
              <a href="#faq">FAQ</a>
            </div>
          </div>

          <div>
            <h4>Contact</h4>
            <div className="landing-footer-contact">
              <a href="mailto:contact@alia-platform.com">
                <Mail size={16} aria-hidden /> contact@alia-platform.com
              </a>
              <a href="tel:+21600000000">
                <Phone size={16} aria-hidden /> +216 00 000 000
              </a>
              <p>
                <MapPin size={16} aria-hidden /> Tunis, Tunisia
              </p>
            </div>
            <a className="landing-footer-external" href="https://www.quantified.ai/" target="_blank" rel="noreferrer">
              Industry inspiration <ExternalLink size={14} aria-hidden />
            </a>
          </div>
        </div>

        <div className="container landing-footer-bottom">
          <span>© {year} ALIA. All rights reserved.</span>
          <span>Academic &amp; industry collaboration with Laboratoires Vital.</span>
        </div>
      </footer>
    </>
  );
}
