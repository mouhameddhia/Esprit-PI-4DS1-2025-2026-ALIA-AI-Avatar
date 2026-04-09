import React, { useState, useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { ChevronLeft, ChevronRight, BarChart2, Video, Database, LayoutDashboard } from 'lucide-react';

const chartData = [
  { month: 'Jan', score: 65 },
  { month: 'Feb', score: 68 },
  { month: 'Mar', score: 78 },
  { month: 'Apr', score: 85 },
  { month: 'May', score: 92 },
  { month: 'Jun', score: 96 },
];

export default function Mockups() {
  const [activeSlide, setActiveSlide] = useState(0);
  const slides = ['Rep training dashboard', 'Performance analytics', 'Physician / Nour chat'];
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start end', 'end start'],
  });
  const y1 = useTransform(scrollYProgress, [0, 1], [70, -70]);
  const y2 = useTransform(scrollYProgress, [0, 1], [-45, 45]);
  const y3 = useTransform(scrollYProgress, [0, 1], [0, -100]);

  return (
    <section ref={containerRef} id="product" className="landing-section container">
      <motion.div
        className="landing-section-head"
        initial={{ opacity: 0, y: 18 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.55 }}
      >
        <p className="landing-kicker">Product vision</p>
        <h2>What ALIA feels like</h2>
        <p>Progress tracking, layered data, and a focused UX for reps and HCPs—animated to suggest depth, not decoration.</p>
      </motion.div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 320px), 1fr))',
          gap: '3rem',
          alignItems: 'center',
        }}
      >
        <motion.div
          style={{ y: y1 }}
          className="glass-panel"
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.65 }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
            <BarChart2 color="var(--soft-blue)" size={22} aria-hidden />
            <h3 style={{ margin: 0, fontSize: '1.15rem' }}>Rep progression</h3>
          </div>
          <div style={{ height: 280, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--glass-border)" />
                <XAxis dataKey="month" stroke="var(--text-secondary)" tick={{ fontSize: 12 }} />
                <YAxis stroke="var(--text-secondary)" tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--bg-color-light)',
                    border: '1px solid var(--glass-border)',
                    borderRadius: 12,
                    color: 'var(--text-primary)',
                  }}
                />
                <Line type="monotone" dataKey="score" stroke="#7c3aed" strokeWidth={3} dot={{ r: 5, fill: '#7c3aed' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <div className="mockups-stack">
          <motion.div
            style={{ y: y2 }}
            className="mockups-layer-back"
            initial={{ opacity: 0, x: 40 }}
            whileInView={{ opacity: 0.55, x: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.15, duration: 0.55 }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, borderBottom: '1px solid var(--glass-border)', paddingBottom: 8 }}>
              <Database size={16} aria-hidden />
              <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Knowledge &amp; logs</span>
            </div>
            <div className="mockups-db-lines">
              <div className="mockups-db-line" style={{ width: '100%' }} />
              <div className="mockups-db-line" />
              <div className="mockups-db-line" />
            </div>
          </motion.div>

          <motion.div
            style={{ y: y3 }}
            className="mockups-layer-front"
            initial={{ opacity: 0, scale: 0.92 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.25, type: 'spring', stiffness: 260, damping: 22 }}
          >
            <div
              style={{
                padding: '1.25rem 1.35rem',
                borderBottom: '1px solid var(--glass-border)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <LayoutDashboard size={18} color="#7c3aed" aria-hidden />
                <h4 style={{ margin: 0, fontSize: '0.95rem', color: 'var(--text-primary)' }}>{slides[activeSlide]}</h4>
              </div>
              <Video size={18} color="var(--text-secondary)" aria-hidden />
            </div>
            <div style={{ padding: '1.25rem', minHeight: 240, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
              <motion.div
                style={{
                  width: '88%',
                  height: '82%',
                  background: 'color-mix(in srgb, var(--bg-color) 92%, transparent)',
                  borderRadius: 14,
                  padding: '1.25rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.85rem',
                  border: '1px solid var(--glass-border)',
                }}
                animate={{ opacity: [0.85, 1, 0.85] }}
                transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
              >
                <div style={{ width: '38%', height: 14, background: 'color-mix(in srgb, var(--text-secondary) 22%, transparent)', borderRadius: 6 }} />
                <div
                  style={{
                    width: '100%',
                    height: 72,
                    background: 'linear-gradient(100deg, rgba(124, 58, 237, 0.25), rgba(45, 212, 191, 0.2))',
                    borderRadius: 10,
                  }}
                />
                <div style={{ display: 'flex', gap: 10 }}>
                  <div style={{ flex: 1, height: 36, background: 'rgba(124,58,237,0.08)', borderRadius: 8 }} />
                  <div style={{ flex: 1, height: 36, background: 'rgba(45,212,191,0.08)', borderRadius: 8 }} />
                </div>
              </motion.div>
              <button
                type="button"
                onClick={() => setActiveSlide((p) => (p > 0 ? p - 1 : slides.length - 1))}
                style={{
                  position: 'absolute',
                  left: 12,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: '#7c3aed',
                  border: 'none',
                  color: 'white',
                  padding: 10,
                  borderRadius: '50%',
                  cursor: 'pointer',
                  display: 'flex',
                  zIndex: 3,
                }}
                aria-label="Previous slide"
              >
                <ChevronLeft size={18} />
              </button>
              <button
                type="button"
                onClick={() => setActiveSlide((p) => (p < slides.length - 1 ? p + 1 : 0))}
                style={{
                  position: 'absolute',
                  right: 12,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: '#7c3aed',
                  border: 'none',
                  color: 'white',
                  padding: 10,
                  borderRadius: '50%',
                  cursor: 'pointer',
                  display: 'flex',
                  zIndex: 3,
                }}
                aria-label="Next slide"
              >
                <ChevronRight size={18} />
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
