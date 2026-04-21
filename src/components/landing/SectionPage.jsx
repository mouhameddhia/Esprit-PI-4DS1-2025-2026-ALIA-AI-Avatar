import React, { useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth0 } from '@auth0/auth0-react';
import { ArrowLeft, ArrowRight, LayoutGrid } from 'lucide-react';
import { LogoutButton } from '../shared/AuthComponent';
import '../../styles/section-page.css';

export const SECTION_META = [
  {
    id: 'why-alia',
    path: '/why-alia',
    label: 'Why ALIA',
    headline: 'The Problem We Solve',
    sub: 'Traditional pharma training is broken — costly, unscalable, and impossible to measure. ALIA replaces it with AI-powered, on-demand simulation.',
    icon: '🧠',
    glow: '#7c3aed',
    orb1: 'rgba(124,58,237,0.55)',
    orb2: 'rgba(45,212,191,0.35)',
    orb3: 'rgba(124,58,237,0.2)',
  },
  {
    id: 'compare',
    path: '/compare',
    label: 'Compare',
    headline: 'ALIA vs. The Status Quo',
    sub: 'See how ALIA outperforms every dimension of traditional pharma training and engagement — speed, cost, compliance, and scale.',
    icon: '⚖️',
    glow: '#0ea5e9',
    orb1: 'rgba(14,165,233,0.5)',
    orb2: 'rgba(124,58,237,0.3)',
    orb3: 'rgba(14,165,233,0.18)',
  },
  {
    id: 'modes',
    path: '/modes',
    label: 'Modes',
    headline: 'Two Modes, One AI Engine',
    sub: 'Training mode sharpens rep expertise with realistic physician avatars. Engagement mode drives physician adoption with personalized AI outreach.',
    icon: '🔄',
    glow: '#10b981',
    orb1: 'rgba(16,185,129,0.5)',
    orb2: 'rgba(234,88,12,0.38)',
    orb3: 'rgba(16,185,129,0.18)',
  },
  {
    id: 'platform',
    path: '/platform',
    label: 'Platform',
    headline: 'The Intelligence Layer',
    sub: 'NLP pipelines, hybrid RAG, real-time persona simulation, and clinical knowledge indexing — all composable in one enterprise-grade AI platform.',
    icon: '⚡',
    glow: '#6366f1',
    orb1: 'rgba(99,102,241,0.55)',
    orb2: 'rgba(67,56,202,0.32)',
    orb3: 'rgba(99,102,241,0.18)',
  },
  {
    id: 'product',
    path: '/product',
    label: 'Product',
    headline: 'Built for the Field',
    sub: 'Intuitive dashboards for reps, real-time AI coaching, and physician-side analytics — a product designed by practitioners, for practitioners.',
    icon: '📱',
    glow: '#a78bfa',
    orb1: 'rgba(167,139,250,0.5)',
    orb2: 'rgba(124,58,237,0.38)',
    orb3: 'rgba(167,139,250,0.2)',
  },
  {
    id: 'impact',
    path: '/impact',
    label: 'Impact',
    headline: 'Numbers That Matter',
    sub: 'From training accuracy to physician engagement — measurable outcomes that redefine how pharmaceutical companies operate at scale.',
    icon: '📈',
    glow: '#2dd4bf',
    orb1: 'rgba(45,212,191,0.5)',
    orb2: 'rgba(14,165,233,0.32)',
    orb3: 'rgba(45,212,191,0.18)',
  },
  {
    id: 'testimonials',
    path: '/testimonials',
    label: 'Testimonials',
    headline: 'What Practitioners Say',
    sub: 'Real feedback from medical representatives and physicians who rely on ALIA every day to train smarter and engage better.',
    icon: '💬',
    glow: '#f59e0b',
    orb1: 'rgba(245,158,11,0.45)',
    orb2: 'rgba(124,58,237,0.28)',
    orb3: 'rgba(245,158,11,0.15)',
  },
  {
    id: 'faq',
    path: '/faq',
    label: 'FAQ',
    headline: 'Questions Answered',
    sub: 'Everything you need to know about ALIA — from integration and compliance to ROI and deployment timelines.',
    icon: '❓',
    glow: '#94a3b8',
    orb1: 'rgba(71,85,105,0.45)',
    orb2: 'rgba(124,58,237,0.22)',
    orb3: 'rgba(71,85,105,0.15)',
  },
  {
    id: 'get-started',
    path: '/get-started',
    label: 'Get Started',
    headline: 'Ready to Transform?',
    sub: 'Join leading pharmaceutical companies using ALIA to train smarter, engage physicians at scale, and measure what matters.',
    icon: '🚀',
    glow: '#7c3aed',
    orb1: 'rgba(124,58,237,0.55)',
    orb2: 'rgba(45,212,191,0.35)',
    orb3: 'rgba(124,58,237,0.2)',
  },
];

export default function SectionPage({ sectionId, children }) {
  const meta = SECTION_META.find(s => s.id === sectionId);
  const navigate = useNavigate();
  const { isAuthenticated, isLoading } = useAuth0();

  const currentIdx = SECTION_META.findIndex(s => s.id === sectionId);
  const prev = SECTION_META[currentIdx - 1] ?? null;
  const next = SECTION_META[currentIdx + 1] ?? null;

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' });
  }, [sectionId]);

  if (!meta) return null;

  return (
    <motion.div
      className="sp-root"
      key={sectionId}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
    >
      {/* ── Animated background ── */}
      <div className="sp-bg">
        <div className="sp-bg-orb sp-bg-orb--1" style={{ background: meta.orb1 }} />
        <div className="sp-bg-orb sp-bg-orb--2" style={{ background: meta.orb2 }} />
        <div className="sp-bg-orb sp-bg-orb--3" style={{ background: meta.orb3 }} />
        <div className="sp-bg-grid" />
        <div className="sp-bg-noise" />
      </div>

      {/* ── Fixed header ── */}
      <header className="sp-header">
        <div className="sp-header-inner">
          <Link to="/" className="sp-brand">
            <div className="sp-brand-mark">AL</div>
            <div>
              <div className="sp-brand-name">ALIA</div>
              <div className="sp-brand-sub">AI Avatar Platform</div>
            </div>
          </Link>

          <nav className="sp-header-nav" aria-label="Section navigation">
            {SECTION_META.map(s => (
              <Link
                key={s.id}
                to={s.path}
                className={`sp-header-nav-link${s.id === sectionId ? ' sp-header-nav-link--active' : ''}`}
              >
                {s.label}
              </Link>
            ))}
          </nav>

          <div className="sp-header-actions">
            {!isLoading && (
              isAuthenticated ? (
                <LogoutButton />
              ) : (
                <>
                  <Link to="/login" className="btn btn-secondary" style={{ padding: '0.52rem 1rem', fontSize: '0.88rem' }}>
                    Sign In
                  </Link>
                  <Link to="/signup" className="btn btn-primary" style={{ padding: '0.52rem 1rem', fontSize: '0.88rem' }}>
                    Sign Up
                  </Link>
                </>
              )
            )}
          </div>
        </div>
      </header>

      {/* ── Hero banner ── */}
      <section className="sp-hero">
        <div className="sp-hero-content">
          <span className="sp-hero-icon">{meta.icon}</span>

          <div className="sp-hero-badge">
            <span className="sp-hero-badge-dot" style={{ background: meta.glow, boxShadow: `0 0 8px ${meta.glow}` }} />
            ALIA · {meta.label}
          </div>

          <h1 className="sp-hero-title">{meta.headline}</h1>
          <p className="sp-hero-sub">{meta.sub}</p>

          <div className="sp-hero-divider">
            <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              Explore sections
            </span>
          </div>

          <div className="sp-pills">
            {SECTION_META.map(s => (
              <Link
                key={s.id}
                to={s.path}
                className={`sp-pill${s.id === sectionId ? ' sp-pill--active' : ''}`}
                style={s.id === sectionId ? { '--sp-glow': meta.glow } : {}}
              >
                {s.label}
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* ── Section content ── */}
      <div className="sp-content">
        {children}
      </div>

      {/* ── Bottom navigation ── */}
      <footer className="sp-foot">
        <div className="sp-foot-inner">
          {/* Prev */}
          {prev ? (
            <button
              className="sp-foot-arrow"
              onClick={() => navigate(prev.path)}
              aria-label={`Go to ${prev.label}`}
            >
              <ArrowLeft size={18} strokeWidth={2.2} />
              <span className="sp-foot-arrow-label">
                <span className="sp-foot-arrow-small">Previous</span>
                {prev.label}
              </span>
            </button>
          ) : (
            <Link to="/" className="sp-foot-arrow">
              <ArrowLeft size={18} strokeWidth={2.2} />
              <span className="sp-foot-arrow-label">
                <span className="sp-foot-arrow-small">Back to</span>
                Home
              </span>
            </Link>
          )}

          {/* Center */}
          <div className="sp-foot-center">
            <Link to="/" className="sp-foot-home">
              <LayoutGrid size={15} strokeWidth={2} />
              View Full Landing Page
            </Link>
            <span className="sp-foot-count">
              {currentIdx + 1} / {SECTION_META.length}
            </span>
          </div>

          {/* Next */}
          {next ? (
            <button
              className="sp-foot-arrow sp-foot-arrow--next"
              onClick={() => navigate(next.path)}
              aria-label={`Go to ${next.label}`}
            >
              <span className="sp-foot-arrow-label">
                <span className="sp-foot-arrow-small">Next</span>
                {next.label}
              </span>
              <ArrowRight size={18} strokeWidth={2.2} />
            </button>
          ) : (
            <div />
          )}
        </div>
      </footer>
    </motion.div>
  );
}
