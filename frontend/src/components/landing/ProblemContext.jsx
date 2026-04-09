import React from 'react';
import { motion } from 'framer-motion';
import { ShieldAlert, Users, HeartPulse, GraduationCap, Target } from 'lucide-react';

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.12, delayChildren: 0.05 },
  },
};

const item = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] } },
};

export default function ProblemContext() {
  const problems = [
    {
      icon: <Users size={28} strokeWidth={1.75} />,
      title: 'Scalable rep training',
      desc: 'Role-play with static slide decks does not scale. Reps need continuous, scenario-based practice with objective feedback.',
    },
    {
      icon: <ShieldAlert size={28} strokeWidth={1.75} />,
      title: 'Compliant HCP engagement',
      desc: 'Physicians need accurate, accessible product information. Passive portals rarely drive engagement or structured insight.',
    },
    {
      icon: <HeartPulse size={28} strokeWidth={1.75} />,
      title: 'Quality under pressure',
      desc: 'Clarity, scientific accuracy, and persuasion matter in the field. Training should mirror real conversations—not generic scripts.',
    },
  ];

  return (
    <section id="why-alia" className="landing-section container" style={{ position: 'relative' }}>
      <motion.div
        className="landing-section-head"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-80px' }}
        transition={{ duration: 0.6 }}
      >
        <p className="landing-kicker">The gap</p>
        <h2 className="text-gradient">Why ALIA exists</h2>
        <p>
          ALIA targets the limits of conventional pharmaceutical learning and engagement: one-off workshops,
          generic content, and little measurable linkage between practice and performance in the field.
        </p>
      </motion.div>

      <motion.div
        className="landing-problem-grid"
        variants={container}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: '-60px' }}
      >
        {problems.map((prob) => (
          <motion.article key={prob.title} variants={item} className="landing-problem-card">
            <div className="landing-icon-ring">{prob.icon}</div>
            <h3 style={{ marginBottom: '0.85rem', fontSize: '1.2rem', position: 'relative', zIndex: 1 }}>
              {prob.title}
            </h3>
            <p style={{ color: 'var(--text-secondary)', lineHeight: 1.65, position: 'relative', zIndex: 1 }}>
              {prob.desc}
            </p>
          </motion.article>
        ))}
      </motion.div>

      <motion.div
        className="landing-sdg-row"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ delay: 0.2, duration: 0.6 }}
      >
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <HeartPulse size={20} color="#34d399" aria-hidden /> SDG 3 · Good health
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <GraduationCap size={20} color="#38bdf8" aria-hidden /> SDG 4 · Quality education
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <Target size={20} color="#fb923c" aria-hidden /> SDG 9 · Innovation
        </span>
      </motion.div>
    </section>
  );
}
