import React from 'react';
import { motion } from 'framer-motion';
import { ShieldAlert, Users, HeartPulse, GraduationCap, Target } from 'lucide-react';

const ProblemContext = () => {
  const problems = [
    { icon: <Users size={32} />, title: "Scalable Training for Reps", desc: "Traditional role-play is resource-heavy. Reps need continuous, scalable practice." },
    { icon: <ShieldAlert size={32} />, title: "Compliant Information", desc: "Doctors require accurate, compliant info. Static CRMs fail to engage actively." },
    { icon: <HeartPulse size={32} />, title: "Patient Safety at Stake", desc: "Miscommunication can lead to adverse events. Training must be perfect." }
  ];

  return (
    <section className="container" style={{ padding: '6rem 2rem', position: 'relative' }}>
      <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
        <h2 className="text-gradient">Why ALIA?</h2>
        <p style={{ color: 'var(--text-secondary)', maxWidth: '600px', margin: '1rem auto' }}>
          The pharmaceutical industry faces critical gaps in training efficiency and active client engagement.
        </p>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '2rem'
      }}>
        {problems.map((prob, i) => (
          <motion.div 
            key={i}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.2 }}
            className="glass-panel"
            style={{ padding: '2rem', textAlign: 'center' }}
          >
            <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'rgba(56, 189, 248, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.5rem', color: 'var(--soft-blue)' }}>
              {prob.icon}
            </div>
            <h3 style={{ marginBottom: '1rem', fontSize: '1.25rem' }}>{prob.title}</h3>
            <p style={{ color: 'var(--text-secondary)' }}>{prob.desc}</p>
          </motion.div>
        ))}
      </div>

      <div style={{ marginTop: '4rem', display: 'flex', justifyContent: 'center', gap: '2rem', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)' }}><HeartPulse size={20} color="#34d399" /> SDG 3: Good Health</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)' }}><GraduationCap size={20} color="#38bdf8" /> SDG 4: Quality Education</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)' }}><Target size={20} color="#fb923c" /> SDG 9: Innovation</div>
      </div>
    </section>
  );
};

export default ProblemContext;
