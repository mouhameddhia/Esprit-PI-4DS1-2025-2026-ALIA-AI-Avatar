import React from 'react';
import { motion, useMotionValue } from 'framer-motion';
import { ArrowRight, Play, ChevronDown, Sparkles } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import CursorParticleField from './CursorParticleField';
import fireflyDoctorVideo from '../../assets/Firefly doctor video.mp4';

const metricVariants = {
  hidden: { opacity: 0, y: 16 },
  show: (i) => ({
    opacity: 1,
    y: 0,
    transition: { delay: 0.45 + i * 0.08, duration: 0.5, ease: [0.22, 1, 0.36, 1] },
  }),
};

export default function HeroSection() {
  const navigate = useNavigate();
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const handleMouseMove = (event) => {
    const bounds = window.innerWidth / 2;
    const boundsY = window.innerHeight / 2;
    x.set((event.clientX - bounds) / 24);
    y.set((event.clientY - boundsY) / 24);
  };

  const goWhy = () => document.getElementById('why-alia')?.scrollIntoView({ behavior: 'smooth' });

  return (
    <section onMouseMove={handleMouseMove} className="hero-section" aria-label="Introduction">
      <CursorParticleField className="hero-cursor-particles" density={1.05} />
      <div className="hero-bg-shape" />
      <div className="hero-grid">
        <motion.div
          style={{ x, y }}
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
          className="hero-copy"
        >
          <motion.div
            className="hero-badge"
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.12, duration: 0.5 }}
          >
            <span className="hero-badge-dot" aria-hidden />
            <Sparkles size={16} style={{ opacity: 0.9 }} aria-hidden />
            Pharmaceutical training &amp; HCP engagement
          </motion.div>

          <h1 className="hero-title">
            Train reps. Engage physicians.
            <br />
            <span className="text-gradient">One intelligent avatar.</span>
          </h1>

          <p className="hero-lead">
            ALIA is an AI-powered conversational avatar platform for realistic, high-pressure simulations
            and compliant product conversations—co-created with{' '}
            <strong style={{ color: 'var(--text-primary)' }}>Laboratoires Vital</strong> as part of an
            integrated data science initiative. NLP, adaptive scenarios, and analytics in one place.
          </p>

          <div className="landing-partner-strip">
            <strong>Partners:</strong> Laboratoires Vital · SDG-aligned innovation (Health, Education, Industry)
          </div>

          <div className="hero-cta">
            <motion.button
              type="button"
              className="btn btn-primary"
              onClick={() => navigate('/signup')}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.98 }}
            >
              Get started <ArrowRight size={18} />
            </motion.button>
            <motion.button
              type="button"
              className="btn btn-secondary"
              onClick={goWhy}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.98 }}
            >
              <Play size={18} aria-hidden /> Explore the platform
            </motion.button>
          </div>

          <div className="hero-metrics">
            {[
              { k: '2 modes', s: 'Rep training & physician-facing' },
              { k: '24/7', s: 'Always-on practice & support' },
              { k: 'Evidence-led', s: 'Feedback you can measure' },
            ].map((m, i) => (
              <motion.div
                key={m.k}
                className="hero-metric"
                variants={metricVariants}
                initial="hidden"
                animate="show"
                custom={i}
                whileHover={{ y: -3, transition: { duration: 0.2 } }}
              >
                <strong>{m.k}</strong>
                <span>{m.s}</span>
              </motion.div>
            ))}
          </div>

          <button type="button" className="landing-scroll-hint" onClick={goWhy}>
            Scroll to discover
            <ChevronDown size={22} strokeWidth={2.25} aria-hidden />
          </button>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.94, y: 32 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
          className="hero-visual"
        >
          <div className="hero-visual-glow" />
          <motion.div
            className="hero-visual-card"
            animate={{
              boxShadow: [
                '0 30px 80px rgba(0,0,0,0.28)',
                '0 36px 90px rgba(124,58,237,0.18)',
                '0 30px 80px rgba(0,0,0,0.28)',
              ],
            }}
            transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }}
          >
            <video
              className="hero-visual-media"
              src={fireflyDoctorVideo}
              autoPlay
              muted
              loop
              playsInline
              preload="metadata"
              aria-label="Firefly doctor demonstration"
            />
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
