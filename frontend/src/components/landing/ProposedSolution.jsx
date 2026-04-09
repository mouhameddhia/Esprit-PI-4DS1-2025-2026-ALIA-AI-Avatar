import React from 'react';
import { motion } from 'framer-motion';
import { Bot, Database, MessageSquare, GitBranch } from 'lucide-react';

const features = [
  {
    icon: MessageSquare,
    color: 'var(--soft-blue)',
    bg: 'rgba(56, 189, 248, 0.12)',
    title: 'Dynamic role-play',
    body: 'ALIA becomes the physician, pharmacist, or skeptical HCP—so reps face realistic objections, evidence requests, and time pressure before they enter the field.',
  },
  {
    icon: Database,
    color: 'var(--soft-orange)',
    bg: 'rgba(251, 146, 60, 0.12)',
    title: 'From conversation to insight',
    body: 'Sessions can be summarized and stored for coaching and compliance workflows—turning every interaction into structured signal, not lost chat logs.',
  },
  {
    icon: GitBranch,
    color: '#a855f7',
    bg: 'rgba(168, 85, 247, 0.12)',
    title: 'One stack, two journeys',
    body: 'The same core powers training simulations (Mode 1) and Nour as AI pharmaceutical representative for HCPs (Mode 2), with clear separation of tone and guardrails.',
  },
];

export default function ProposedSolution() {
  return (
    <section id="platform" className="landing-section landing-section--alt">
      <div className="container">
        <motion.div
          className="landing-section-head"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.55 }}
        >
          <p className="landing-kicker">Architecture</p>
          <h2>One avatar, two faces</h2>
          <p>
            ALIA unifies simulation, evaluation, and compliant engagement—so training quality and field messaging stay
            aligned instead of living in disconnected tools.
          </p>
        </motion.div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 340px), 1fr))',
            gap: '2.5rem',
            alignItems: 'stretch',
          }}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true, margin: '-40px' }}
            transition={{ duration: 0.65, ease: [0.22, 1, 0.36, 1] }}
            className="landing-bento-visual"
          >
            <div className="landing-bento-split-left" aria-hidden />
            <div className="landing-bento-split-right" aria-hidden />
            <motion.div
              style={{ position: 'relative', zIndex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.25rem' }}
              animate={{ y: [0, -6, 0] }}
              transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }}
            >
              <Bot size={72} color="var(--text-primary)" strokeWidth={1.25} aria-hidden />
              <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '0.65rem' }}>
                <span
                  style={{
                    background: 'rgba(52, 211, 153, 0.22)',
                    padding: '0.45rem 1rem',
                    borderRadius: 9999,
                    color: '#34d399',
                    fontSize: '0.82rem',
                    fontWeight: 700,
                    border: '1px solid rgba(52, 211, 153, 0.35)',
                  }}
                >
                  Mode 1 · Virtual HCP
                </span>
                <span
                  style={{
                    background: 'rgba(251, 146, 60, 0.22)',
                    padding: '0.45rem 1rem',
                    borderRadius: 9999,
                    color: '#fb923c',
                    fontSize: '0.82rem',
                    fontWeight: 700,
                    border: '1px solid rgba(251, 146, 60, 0.35)',
                  }}
                >
                  Mode 2 · AI rep (Nour)
                </span>
              </div>
            </motion.div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 28 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}
          >
            {features.map((f, i) => {
              const Icon = f.icon;
              return (
              <motion.div
                key={f.title}
                initial={{ opacity: 0, y: 14 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08, duration: 0.45 }}
                whileHover={{ x: 4 }}
                style={{
                  display: 'flex',
                  gap: '1.25rem',
                  padding: '1.35rem',
                  borderRadius: 18,
                  border: '1px solid var(--glass-border)',
                  background: 'color-mix(in srgb, var(--bg-color) 85%, transparent)',
                }}
              >
                <div
                  style={{
                    width: 48,
                    height: 48,
                    borderRadius: 14,
                    background: f.bg,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    color: f.color,
                  }}
                >
                  <Icon size={22} />
                </div>
                <div>
                  <h3 style={{ marginBottom: '0.45rem', fontSize: '1.1rem' }}>{f.title}</h3>
                  <p style={{ color: 'var(--text-secondary)', lineHeight: 1.65, fontSize: '0.95rem' }}>{f.body}</p>
                </div>
              </motion.div>
            );
            })}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
