import React, { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ChevronDown } from 'lucide-react';

const faqItems = [
  {
    q: 'How does ALIA help pharmaceutical reps improve faster?',
    a: 'ALIA creates realistic, repeatable simulation sessions with targeted feedback so reps can practice objections, messaging, and confidence-building before live physician interactions.',
  },
  {
    q: 'Can ALIA support compliant and guided conversations?',
    a: 'Yes. ALIA is designed for structured, guided messaging with guardrails that support compliant communication and safer rehearsal in regulated environments.',
  },
  {
    q: 'What can managers measure after sessions?',
    a: 'Managers can review session summaries, competency signals, and progression trends to provide more precise coaching and prioritize the highest-impact interventions.',
  },
  {
    q: 'Does ALIA replace trainers or coaching teams?',
    a: 'No. It amplifies trainers and managers by handling scalable practice and diagnostics, while human experts focus on strategic coaching and nuanced field readiness.',
  },
  {
    q: 'Is ALIA useful for physicians as well as reps?',
    a: 'Yes. The platform supports multiple modes, including physician-facing flows for guided education and product information experiences.',
  },
];

export default function FaqSection() {
  const [openIndex, setOpenIndex] = useState(0);

  return (
    <section id="faq" className="landing-section faq-section">
      <div className="container">
        <motion.div
          className="landing-section-head"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.55 }}
        >
          <p className="landing-kicker">FAQ</p>
          <h2>Answers for product, training, and rollout teams</h2>
          <p>Clear answers to the questions teams ask most before piloting ALIA.</p>
        </motion.div>

        <div className="faq-list">
          {faqItems.map((item, i) => {
            const isOpen = openIndex === i;
            return (
              <motion.article
                key={item.q}
                className={`faq-item ${isOpen ? 'is-open' : ''}`}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-70px' }}
                transition={{ duration: 0.35, delay: i * 0.04 }}
              >
                <button
                  type="button"
                  className="faq-trigger"
                  onClick={() => setOpenIndex(isOpen ? -1 : i)}
                  aria-expanded={isOpen}
                >
                  <span>{item.q}</span>
                  <ChevronDown size={18} className={isOpen ? 'is-open' : ''} aria-hidden />
                </button>
                <AnimatePresence initial={false}>
                  {isOpen ? (
                    <motion.div
                      key="faq-content"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3, ease: 'easeOut' }}
                      className="faq-content-wrap"
                    >
                      <p className="faq-content">{item.a}</p>
                    </motion.div>
                  ) : null}
                </AnimatePresence>
              </motion.article>
            );
          })}
        </div>
      </div>
    </section>
  );
}
