import React from 'react';
import { motion } from 'framer-motion';
import { Quote } from 'lucide-react';

const rowOne = [
  {
    quote:
      'ALIA helped our reps walk into physician meetings with far more clarity and confidence than legacy role-play.',
    author: 'Medical Training Lead',
    company: 'Regional Pharma Group',
  },
  {
    quote:
      'The realism of objections was impressive. Teams practiced more often because it was available exactly when needed.',
    author: 'Commercial Excellence Manager',
    company: 'Life Sciences Enterprise',
  },
  {
    quote:
      'Manager coaching quality improved immediately once we had session-level analytics and conversation highlights.',
    author: 'Sales Enablement Director',
    company: 'Global Therapeutics',
  },
  {
    quote:
      'Our onboarding cycle felt shorter, and field readiness looked stronger in practical evaluations.',
    author: 'Capability Development Partner',
    company: 'Healthcare Solutions Company',
  },
];

const rowTwo = [
  {
    quote:
      'Learners described sessions as realistic and less intimidating than face-to-face evaluation settings.',
    author: 'Clinical Education Manager',
    company: 'Biopharma Network',
  },
  {
    quote:
      'ALIA gave us repeatable, compliant practice without pulling everyone out of the field.',
    author: 'Brand Training Director',
    company: 'Specialty Pharma Team',
  },
  {
    quote:
      'The biggest win was consistency. Every rep gets the same high-quality challenge level and scoring logic.',
    author: 'Head of Sales Capability',
    company: 'International MedTech Partner',
  },
  {
    quote:
      'We now identify coaching priorities faster and spend manager time where it actually moves outcomes.',
    author: 'Commercial Operations Lead',
    company: 'Advanced Care Portfolio',
  },
];

function TestimonialTrack({ items, reverse = false }) {
  const duplicate = [...items, ...items];
  return (
    <div className="testimonials-track-wrap">
      <div className={`testimonials-track ${reverse ? 'is-reverse' : ''}`}>
        {duplicate.map((item, i) => (
          <article key={`${item.author}-${i}`} className="testimonial-card">
            <Quote size={18} aria-hidden />
            <p>{item.quote}</p>
            <div>
              <strong>{item.author}</strong>
              <span>{item.company}</span>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

export default function TestimonialsSection() {
  return (
    <section id="testimonials" className="landing-section landing-section--alt testimonials-section">
      <div className="container">
        <motion.div
          className="landing-section-head"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.55 }}
        >
          <p className="landing-kicker">Testimonials</p>
          <h2>What HCPs say after switching to ALIA</h2>
          <p>Don't take our word for it. Hear from the ones who made the switch</p>
        </motion.div>
      </div>

      <div className="testimonials-marquee">
        <TestimonialTrack items={rowOne} />
        <TestimonialTrack items={rowTwo} reverse />
      </div>
    </section>
  );
}
