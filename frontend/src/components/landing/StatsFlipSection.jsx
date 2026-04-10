import React from 'react';
import { motion } from 'framer-motion';
import { GraduationCap, Activity, FlaskConical, HeartPulse, ShieldCheck, TrendingUp } from 'lucide-react';

const cards = [
  {
    icon: GraduationCap,
    stat: '97%',
    title: 'Mastery achieved',
    subtitle: 'Across simulation sessions',
    detail:
      'Structured role-play with scoring rubrics helps reps internalize key product narratives and objection handling faster.',
    tone: 'purple',
    featured: true,
    imageUrl: 'https://images.pexels.com/photos/7579831/pexels-photo-7579831.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=900&w=1400',
  },
  {
    icon: TrendingUp,
    stat: '4x',
    title: 'More coaching moments',
    subtitle: 'Compared to classic workflows',
    detail:
      'Session summaries and analytics equip managers with concrete moments to coach, rather than anecdotal feedback.',
    tone: 'teal',
  },
  {
    icon: Activity,
    stat: '6x',
    title: 'Practice frequency',
    subtitle: 'vs. peer-to-peer role play',
    detail:
      'Always-on avatar access removes scheduling friction and enables high-volume repetition before real HCP conversations.',
    tone: 'slate',
  },
  {
    icon: HeartPulse,
    stat: '92%',
    title: 'Realism score',
    subtitle: 'Simulations feel call-real',
    detail:
      'Medical-context prompts and adaptive dialogue create authentic pressure that mirrors field conversations.',
    tone: 'teal',
  },
  {
    icon: FlaskConical,
    stat: '5x',
    title: 'Training efficiency',
    subtitle: 'Measured in pilot rollouts',
    detail:
      'Teams ramp new launches quickly by combining compliance-safe prompts, guided paths, and role-specific feedback.',
    tone: 'purple',
  },
  {
    icon: ShieldCheck,
    stat: '24/7',
    title: 'Compliant readiness',
    subtitle: 'Anytime upskilling',
    detail:
      'A controlled simulation environment supports confidence building while reducing off-script messaging risk.',
    tone: 'slate',
    featured: true,
    imageUrl: 'https://images.pexels.com/photos/4989186/pexels-photo-4989186.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=900&w=1400',
  },
];

export default function StatsFlipSection() {
  return (
    <section id="impact" className="landing-section">
      <div className="container">
        <motion.div
          className="landing-section-head"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.55 }}
        >
          <p className="landing-kicker">Impact</p>
          <h2>Quantified outcomes, designed for Healthcare Professionals</h2>
          <p>Hover each card to reveal how ALIA drives measurable performance, readiness, and coaching depth.</p>
        </motion.div>

        <div className="impact-grid">
          {cards.map((card, i) => {
            const Icon = card.icon;
            return (
              <motion.article
                key={card.title}
                className={`impact-flip-card tone-${card.tone} ${card.featured ? 'is-featured' : ''}`}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-60px' }}
                transition={{ duration: 0.5, delay: i * 0.06 }}
              >
                <div className="impact-flip-card-inner">
                  <div
                    className="impact-card-face impact-card-front"
                    style={card.imageUrl ? { '--impact-image': `url("${card.imageUrl}")` } : undefined}
                  >
                    <Icon size={24} aria-hidden />
                    <strong>{card.stat}</strong>
                    <h3>{card.title}</h3>
                    <p>{card.subtitle}</p>
                  </div>
                  <div className="impact-card-face impact-card-back">
                    <h3>{card.title}</h3>
                    <p>{card.detail}</p>
                  </div>
                </div>
              </motion.article>
            );
          })}
        </div>
      </div>
    </section>
  );
}
