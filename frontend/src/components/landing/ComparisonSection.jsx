import React, { useRef } from 'react';
import { motion, useMotionValue, useTransform, useSpring } from 'framer-motion';
import { CheckCircle2, XCircle } from 'lucide-react';

export default function ComparisonSection() {
  const comparisonData = [
    { feature: 'Realistic, adaptive role-play', traditional: false, alia: true },
    { feature: 'Structured competency feedback', traditional: false, alia: true },
    { feature: 'Compliant, guided messaging', traditional: false, alia: true },
    { feature: '24/7 availability for practice / info', traditional: false, alia: true },
    { feature: 'Session analytics & summaries', traditional: 'Partial', alia: true },
  ];

  const cardRef = useRef(null);
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const mouseXSpring = useSpring(mx, { stiffness: 150, damping: 28 });
  const mouseYSpring = useSpring(my, { stiffness: 150, damping: 28 });
  const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], ['7deg', '-7deg']);
  const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], ['-7deg', '7deg']);

  const handleMouseMove = (e) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    mx.set((e.clientX - rect.left) / w - 0.5);
    my.set((e.clientY - rect.top) / h - 0.5);
  };

  const handleMouseLeave = () => {
    mx.set(0);
    my.set(0);
  };

  return (
    <section id="comparison" className="landing-section landing-section--alt relative">
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          overflow: 'hidden',
          pointerEvents: 'none',
          opacity: 0.25,
          transform: 'rotate(180deg)',
          zIndex: 0,
        }}
        aria-hidden
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120" preserveAspectRatio="none" style={{ width: '100%', height: '120px', display: 'block' }}>
          <path
            d="M321.39,56.44c58-10.79,114.16-30.13,172-41.86,82.39-16.72,168.19-17.73,250.45-.39C823.78,31,906.67,72,985.66,92.83c70.05,18.48,146.53,26.09,214.34,3V0H0V27.35A600.21,600.21,0,0,0,321.39,56.44Z"
            fill="url(#gradient-wave)"
          />
          <defs>
            <linearGradient id="gradient-wave" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#7c3aed" />
              <stop offset="100%" stopColor="#2dd4bf" />
            </linearGradient>
          </defs>
        </svg>
      </div>

      <div className="container" style={{ position: 'relative', zIndex: 10, perspective: '1400px' }}>
        <motion.div
          className="landing-section-head"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.55 }}
        >
          <p className="landing-kicker">Contrast</p>
          <h2>The output gap</h2>
          <p>Traditional programs versus an AI avatar that trains, assesses, and engages with consistency.</p>
        </motion.div>

        <motion.div
          ref={cardRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          style={{ rotateX, rotateY, transformStyle: 'preserve-3d' }}
        >
          <div className="comparison-table-wrap" style={{ transform: 'translateZ(24px)' }}>
            <div className="comparison-table-header">
              <div>Capability</div>
              <div style={{ textAlign: 'center' }}>Typical approach</div>
              <div style={{ textAlign: 'center', color: 'var(--soft-blue)' }}>ALIA</div>
            </div>
            {comparisonData.map((row) => (
              <motion.div
                key={row.feature}
                className="comparison-table-row"
                whileHover={{ backgroundColor: 'rgba(124, 58, 237, 0.06)' }}
                transition={{ duration: 0.2 }}
              >
                <div>{row.feature}</div>
                <div style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                  {row.traditional === true ? (
                    <CheckCircle2 style={{ margin: '0 auto', color: '#94a3b8' }} aria-label="Yes" />
                  ) : row.traditional === false ? (
                    <XCircle style={{ margin: '0 auto', color: '#ef4444' }} aria-label="No" />
                  ) : (
                    row.traditional
                  )}
                </div>
                <div style={{ textAlign: 'center', color: 'var(--mode-training)' }}>
                  {row.alia ? <CheckCircle2 style={{ margin: '0 auto' }} aria-label="Yes" /> : null}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
