import React, { useRef } from 'react';
import { motion, useMotionValue, useTransform, useSpring } from 'framer-motion';
import { CheckCircle2, XCircle } from 'lucide-react';

const ComparisonSection = () => {
  const comparisonData = [
    { feature: "Realistic Role-Play", traditional: false, alia: true },
    { feature: "Automated Scoring", traditional: false, alia: true },
    { feature: "Compliant Presentations", traditional: false, alia: true },
    { feature: "24/7 Availability", traditional: false, alia: true },
    { feature: "CRM Integration", traditional: "Partial", alia: true }
  ];

  /* 3D Tilt Logic */
  const cardRef = useRef(null);
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const mouseXSpring = useSpring(x, { stiffness: 150, damping: 25 });
  const mouseYSpring = useSpring(y, { stiffness: 150, damping: 25 });

  const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], ["8deg", "-8deg"]);
  const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], ["-8deg", "8deg"]);

  const handleMouseMove = (e) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    x.set(mouseX / width - 0.5);
    y.set(mouseY / height - 0.5);
  };

  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <section className="relative" style={{ padding: '8rem 2rem', backgroundColor: 'var(--bg-color)' }}>
      {/* Dynamic SVG Wave Background placed absolutely */}
      <div className="absolute top-0 left-0 w-full overflow-hidden leading-none opacity-20 pointer-events-none" style={{ transform: 'rotate(180deg)', zIndex: 0 }}>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120" preserveAspectRatio="none" className="w-full h-[150px]">
          <path d="M321.39,56.44c58-10.79,114.16-30.13,172-41.86,82.39-16.72,168.19-17.73,250.45-.39C823.78,31,906.67,72,985.66,92.83c70.05,18.48,146.53,26.09,214.34,3V0H0V27.35A600.21,600.21,0,0,0,321.39,56.44Z" fill="url(#gradient-wave)"></path>
          <defs>
            <linearGradient id="gradient-wave" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#7c3aed" />
              <stop offset="100%" stopColor="#2dd4bf" />
            </linearGradient>
          </defs>
        </svg>
      </div>

      <div className="container relative z-10" style={{ perspective: '1200px' }}>
        <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
          <h2>The Output Gap</h2>
          <p style={{ color: 'var(--text-secondary)' }}>Traditional vs The ALIA Way</p>
        </div>

        <motion.div 
          ref={cardRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          style={{ 
            rotateX, 
            rotateY,
            transformStyle: "preserve-3d"
          }}
          className="glass-panel"
        >
          {/* Inner Content that floats above the glass */}
          <div style={{ transform: 'translateZ(40px)', padding: '0' }}>
            <div style={{
              display: 'grid',
              gridTemplateColumns: '2fr 1fr 1fr',
              padding: '1.5rem',
              borderBottom: '1px solid var(--glass-border)',
              background: 'rgba(255,255,255,0.02)',
              fontWeight: 'bold',
              borderTopLeftRadius: '16px',
              borderTopRightRadius: '16px'
            }}>
              <div>Capabilities</div>
              <div style={{ textAlign: 'center' }}>Traditional Approach</div>
              <div style={{ textAlign: 'center', color: 'var(--soft-blue)' }}>ALIA Avatar</div>
            </div>

            {comparisonData.map((row, i) => (
              <motion.div 
                key={i} 
                whileHover={{ backgroundColor: 'rgba(255,255,255,0.05)' }}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '2fr 1fr 1fr',
                  padding: '1.5rem',
                  borderBottom: i === comparisonData.length - 1 ? 'none' : '1px solid var(--glass-border)',
                  alignItems: 'center',
                  transition: 'background-color 0.2s ease'
                }}
              >
                <div>{row.feature}</div>
                <div style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                  {row.traditional === true ? <CheckCircle2 style={{ margin: '0 auto', color: '#94a3b8' }} /> : 
                  row.traditional === false ? <XCircle style={{ margin: '0 auto', color: '#ef4444' }} /> : row.traditional}
                </div>
                <div style={{ textAlign: 'center', color: 'var(--mode-training)' }}>
                  {row.alia === true ? <CheckCircle2 style={{ margin: '0 auto' }} /> : row.alia}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default ComparisonSection;
