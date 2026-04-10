import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth0 } from '@auth0/auth0-react';
import HeroSection from './HeroSection';
import ProblemContext from './ProblemContext';
import ComparisonSection from './ComparisonSection';
import ObjectivesTabs from './ObjectivesTabs';
import ProposedSolution from './ProposedSolution';
import Mockups from './Mockups';
import StatsFlipSection from './StatsFlipSection';
import TestimonialsSection from './TestimonialsSection';
import FaqSection from './FaqSection';
import ConclusionFooter from './ConclusionFooter';
import { LogoutButton } from '../shared/AuthComponent';

const navTargets = [
  { id: 'why-alia', label: 'Why ALIA' },
  { id: 'comparison', label: 'Compare' },
  { id: 'modes', label: 'Modes' },
  { id: 'platform', label: 'Platform' },
  { id: 'product', label: 'Product' },
  { id: 'impact', label: 'Impact' },
  { id: 'testimonials', label: 'Testimonials' },
  { id: 'faq', label: 'FAQ' },
  { id: 'get-started', label: 'Get started' },
];

function scrollToId(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

export default function LandingPage() {
  const { isAuthenticated, isLoading } = useAuth0();
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      <div className="landing-header-wrap">
        <header className="landing-header-inner">
          <button
            type="button"
            className="brand"
            onClick={() => scrollToId('top')}
            style={{ background: 'none', border: 'none', cursor: 'pointer', textAlign: 'left' }}
          >
            <div className="brand-mark">AL</div>
            <div>
              <div style={{ fontWeight: 700, letterSpacing: '0.02em', color: 'var(--text-primary)' }}>
                ALIA
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                AI Avatar Platform
              </div>
            </div>
          </button>

          <nav className="landing-header-nav" aria-label="Page sections">
            {navTargets.map(({ id, label }) => (
              <button key={id} type="button" className="landing-nav-link" onClick={() => scrollToId(id)}>
                {label}
              </button>
            ))}
          </nav>

          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexShrink: 0 }}>
            {!isLoading &&
              (isAuthenticated ? (
                <LogoutButton />
              ) : (
                <>
                  <button type="button" onClick={() => navigate('/login')} className="btn btn-secondary" style={{ padding: '0.55rem 1.1rem', fontSize: '0.9rem' }}>
                    Sign In
                  </button>
                  <button type="button" onClick={() => navigate('/signup')} className="btn btn-primary" style={{ padding: '0.55rem 1.1rem', fontSize: '0.9rem' }}>
                    Sign Up
                  </button>
                </>
              ))}
          </div>
        </header>
      </div>

      <div id="top" />
      <HeroSection />
      <ProblemContext />
      <ComparisonSection />
      <ObjectivesTabs />
      <ProposedSolution />
      <Mockups />
      <StatsFlipSection />
      <TestimonialsSection />
      <FaqSection />
      <ConclusionFooter />
    </div>
  );
}
