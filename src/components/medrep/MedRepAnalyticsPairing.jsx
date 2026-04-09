import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft, Target, Users, TrendingUp, ArrowRight,
  MessageSquare, Zap, ShieldCheck, Activity, Sparkles,
  ChevronRight, Award, BarChart2, Brain,
  Clock, Star, MapPin, Briefcase, Eye, UserCheck,
  Layers, GitBranch, Filter, Search, PieChart
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  AreaChart, Area, LineChart, Line
} from 'recharts';
import './MedRepAnalyticsPairing.css';

/* ─── Data ──────────────────────────────────────────────── */
const PRODUCTS = [
  { id: 'pr1', name: 'CardioGuard', category: 'Cardiovascular', recs: 2, efficiency: 88, trend: '+5%', icon: '❤️' },
  { id: 'pr2', name: 'NeuroShield', category: 'Neurology', recs: 2, efficiency: 92, trend: '+8%', icon: '🧠' },
  { id: 'pr3', name: 'DiabetoCare Plus', category: 'Endocrinology', recs: 3, efficiency: 90, trend: '+3%', icon: '💊' },
  { id: 'pr4', name: 'RespiClear', category: 'Respiratory', recs: 2, efficiency: 89, trend: '+6%', icon: '🫁' },
  { id: 'pr5', name: 'OncoPro', category: 'Oncology', recs: 2, efficiency: 95, trend: '+12%', icon: '🔬' },
  { id: 'pr6', name: 'PainRelief XR', category: 'Pain Mgmt.', recs: 1, efficiency: 87, trend: '+2%', icon: '💉' },
];

const BAR_COLORS = ['#7c3aed', '#8b5cf6', '#a78bfa', '#7c3aed', '#6d28d9', '#8b5cf6'];

const PAIRINGS = {
  pr1: [
    { initial: 'A', name: 'Ahmed Hassan', rank: 1, score: 95, specialty: 'Cardiovascular', reasoning: 'Cardiovascular specialty expertise, consistently high performance in similar product categories. 8+ years in therapeutic area.', subScores: { relationship: 92, territory: 88, experience: 97, knowledge: 94 }, territory: 'Tunis North', visits: 142, conversion: '89%' },
    { initial: 'F', name: 'Fatima Ali', rank: 2, score: 82, specialty: 'General Practice', reasoning: 'Strong presentation skills and deep regional market knowledge. Excellent rapport with KOLs.', subScores: { relationship: 85, territory: 90, experience: 73, knowledge: 80 }, territory: 'Tunis South', visits: 118, conversion: '76%' },
  ],
  pr2: [
    { initial: 'S', name: 'Sara Khalid', rank: 1, score: 91, specialty: 'Neurology', reasoning: 'Neurology background and excellent physician relationship management. Published in field journals.', subScores: { relationship: 94, territory: 86, experience: 90, knowledge: 95 }, territory: 'Sfax Region', visits: 156, conversion: '84%' },
    { initial: 'M', name: 'Mohamed Radi', rank: 2, score: 79, specialty: 'Internal Medicine', reasoning: 'Proven track record with specialist physicians. Cross-functional experience.', subScores: { relationship: 82, territory: 78, experience: 76, knowledge: 80 }, territory: 'Sousse', visits: 98, conversion: '71%' },
  ],
  pr3: [
    { initial: 'L', name: 'Layla Nasser', rank: 1, score: 93, specialty: 'Endocrinology', reasoning: 'Endocrinology specialty focus with high HCP engagement scores. Top performer in metabolic portfolio.', subScores: { relationship: 90, territory: 92, experience: 95, knowledge: 96 }, territory: 'Grand Tunis', visits: 167, conversion: '91%' },
    { initial: 'A', name: 'Ahmed Hassan', rank: 2, score: 85, specialty: 'Cardiovascular', reasoning: 'Cross-therapeutic strength across metabolic conditions. Strong clinical discussion capability.', subScores: { relationship: 88, territory: 84, experience: 86, knowledge: 82 }, territory: 'Tunis North', visits: 142, conversion: '89%' },
  ],
  pr4: [
    { initial: 'F', name: 'Fatima Ali', rank: 1, score: 88, specialty: 'General Practice', reasoning: 'Respiratory background and strong pulmonologist network. Territory leadership experience.', subScores: { relationship: 91, territory: 88, experience: 84, knowledge: 89 }, territory: 'Tunis South', visits: 118, conversion: '76%' },
    { initial: 'S', name: 'Sara Khalid', rank: 2, score: 76, specialty: 'Neurology', reasoning: 'High patient interaction volume in territory. Adaptable to new therapeutic areas.', subScores: { relationship: 79, territory: 74, experience: 72, knowledge: 78 }, territory: 'Sfax Region', visits: 156, conversion: '84%' },
  ],
  pr5: [
    { initial: 'M', name: 'Mohamed Radi', rank: 1, score: 90, specialty: 'Internal Medicine', reasoning: 'Oncology access specialist with top-tier KOL relationships. Experience with hospital formulary.', subScores: { relationship: 95, territory: 88, experience: 86, knowledge: 91 }, territory: 'Sousse', visits: 98, conversion: '71%' },
    { initial: 'L', name: 'Layla Nasser', rank: 2, score: 81, specialty: 'Endocrinology', reasoning: 'Strong scientific communication and data-driven discussions. Detail-oriented presentations.', subScores: { relationship: 83, territory: 80, experience: 78, knowledge: 84 }, territory: 'Grand Tunis', visits: 167, conversion: '91%' },
  ],
  pr6: [
    { initial: 'A', name: 'Ahmed Hassan', rank: 1, score: 87, specialty: 'Cardiovascular', reasoning: 'Broad analgesic portfolio experience and high territory coverage. Strong hospital access.', subScores: { relationship: 89, territory: 90, experience: 85, knowledge: 84 }, territory: 'Tunis North', visits: 142, conversion: '89%' },
    { initial: 'F', name: 'Fatima Ali', rank: 2, score: 80, specialty: 'General Practice', reasoning: 'Established pain management specialist relationships. Good follow-up cadence.', subScores: { relationship: 83, territory: 82, experience: 76, knowledge: 79 }, territory: 'Tunis South', visits: 118, conversion: '76%' },
  ],
};

const SCATTER_DATA = [
  { x: 3.1, y: 85, z: 200, name: 'Ahmed H.' },
  { x: 3.4, y: 88, z: 260, name: 'Fatima A.' },
  { x: 4.2, y: 92, z: 300, name: 'Sara K.' },
  { x: 4.0, y: 90, z: 280, name: 'Mohamed R.' },
  { x: 6.1, y: 87, z: 240, name: 'Layla N.' },
];
const SCATTER_COLORS = ['#7c3aed', '#2dd4bf', '#f59e0b', '#ec4899', '#10b981'];

const TREND_DATA = [
  { month: 'Jan', score: 78 },
  { month: 'Feb', score: 82 },
  { month: 'Mar', score: 85 },
  { month: 'Apr', score: 88 },
  { month: 'May', score: 91 },
  { month: 'Jun', score: 94 },
];

const STRATEGIES = [
  {
    icon: <Zap size={22} />,
    title: 'Value-Based Outcomes',
    alignment: 96,
    desc: 'Focus on long-term patient health improvements and cost-effectiveness versus standard of care.',
    color: '#7c3aed',
    metrics: { adoption: '78%', impact: 'High', sessions: 24 },
    tags: ['ROI', 'Long-term'],
  },
  {
    icon: <ShieldCheck size={22} />,
    title: 'Clinical Peer-Review',
    alignment: 94,
    desc: 'Leverage data from peer-reviewed journals including NEJM, The Lancet and JAMA.',
    color: '#2dd4bf',
    metrics: { adoption: '84%', impact: 'Very High', sessions: 31 },
    tags: ['Evidence', 'Clinical'],
  },
  {
    icon: <BarChart2 size={22} />,
    title: 'Cost-Benefit Analysis',
    alignment: 89,
    desc: 'Highlight total cost of treatment savings and productivity impact for the healthcare system.',
    color: '#f59e0b',
    metrics: { adoption: '65%', impact: 'Medium', sessions: 18 },
    tags: ['Economics', 'Savings'],
  },
  {
    icon: <Award size={22} />,
    title: 'Guideline Alignment',
    alignment: 92,
    desc: 'Demonstrate how the product fits within current national and international treatment guidelines.',
    color: '#10b981',
    metrics: { adoption: '72%', impact: 'High', sessions: 22 },
    tags: ['Guidelines', 'Compliance'],
  },
];

/* ─── SVG Circle Score ────────────────────────────────────── */
const CircleScore = ({ score, size = 72, strokeWidth = 5, color = '#7c3aed' }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (score / 100) * circumference;
  return (
    <div className="ap-circle-score" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="var(--glass-border)" strokeWidth={strokeWidth} />
        <motion.circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke={color} strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.4, ease: [0.22, 1, 0.36, 1] }}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <span className="ap-circle-score__val">{score}%</span>
    </div>
  );
};

/* ─── Mini Bar ───────────────────────────────────────────── */
const MiniBar = ({ label, value, color = '#7c3aed', delay = 0 }) => (
  <div className="ap-mini-bar">
    <div className="ap-mini-bar__header">
      <span className="ap-mini-bar__label">{label}</span>
      <span className="ap-mini-bar__value" style={{ color }}>{value}%</span>
    </div>
    <div className="ap-mini-bar__track">
      <motion.div
        className="ap-mini-bar__fill"
        style={{ background: color }}
        initial={{ width: 0 }}
        animate={{ width: `${value}%` }}
        transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1], delay }}
      />
    </div>
  </div>
);

/* ─── Custom Tooltip ─────────────────────────────────────── */
const CustomBarTooltip = ({ active, payload, label }) => {
  if (active && payload?.length) {
    return (
      <div className="ap-tooltip">
        <p className="ap-tooltip-label">{label}</p>
        <p className="ap-tooltip-value">{payload[0].value}% efficiency</p>
      </div>
    );
  }
  return null;
};

const CustomScatterTooltip = ({ active, payload }) => {
  if (active && payload?.length) {
    const data = payload[0]?.payload;
    return (
      <div className="ap-tooltip">
        <p className="ap-tooltip-label">{data?.name}</p>
        <p className="ap-tooltip-value">{data?.y}% score</p>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
          {data?.x} yrs experience
        </p>
      </div>
    );
  }
  return null;
};

/* ─── Animations ─────────────────────────────────────────── */
const stagger = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.07, delayChildren: 0.1 } } };
const fadeUp = { hidden: { opacity: 0, y: 28 }, show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 260, damping: 24 } } };

/* ─── Component ──────────────────────────────────────────── */
const MedRepAnalyticsPairing = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('pairing');
  const [selectedProduct, setSelectedProduct] = useState('pr1');

  const currentProduct = PRODUCTS.find(p => p.id === selectedProduct);
  const currentPairings = PAIRINGS[selectedProduct] || [];

  return (
    <div className="ap-root">
      {/* Ambient background */}
      <div className="ap-aura ap-aura--purple" />
      <div className="ap-aura ap-aura--teal" />
      <div className="ap-aura ap-aura--glow" />

      {/* ── Navbar ── */}
      <nav className="ap-nav">
        <div className="ap-nav__left">
          <button className="ap-nav__back" onClick={() => navigate('/portal')}>
            <ArrowLeft size={16} />
            <span>Back</span>
          </button>
          <div className="ap-nav__brand">
            <div className="ap-nav__brand-icon">
              <Brain size={18} />
            </div>
            <div>
              <h1 className="ap-nav__title">Rep Intelligence Hub</h1>
              <p className="ap-nav__subtitle">AI-Powered Analytics & Pairing Engine</p>
            </div>
          </div>
        </div>
        <div className="ap-nav__right">
          <div className="ap-nav__status">
            <span className="ap-nav__status-dot" />
            <span>Live Data</span>
          </div>
          <div className="ap-nav__date">
            {new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
          </div>
        </div>
      </nav>

      {/* ── Page Content ── */}
      <motion.div className="ap-page" variants={stagger} initial="hidden" animate="show">

        {/* ── Tab Switcher ── */}
        <motion.div variants={fadeUp} className="ap-tabs__wrap">
          <div className="ap-tabs">
            {[
              { id: 'pairing', label: 'Product-Rep Pairing', icon: <GitBranch size={16} /> },
              { id: 'messaging', label: 'Messaging Strategies', icon: <MessageSquare size={16} /> },
            ].map(tab => (
              <button
                key={tab.id}
                className={`ap-tabs__btn ${activeTab === tab.id ? 'is-active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {activeTab === tab.id && (
                  <motion.div layoutId="activeTabPill" className="ap-tabs__pill" transition={{ type: 'spring', stiffness: 380, damping: 30 }} />
                )}
                <span className="ap-tabs__text">
                  {tab.icon}
                  {tab.label}
                </span>
              </button>
            ))}
          </div>
        </motion.div>

        {/* ══════════════════════════════════════════════
            DYNAMIC CONTENT
        ══════════════════════════════════════════════ */}
        <AnimatePresence mode="wait">

          {/* ── TAB 1: Product-Rep Pairing ── */}
          {activeTab === 'pairing' && (
            <motion.div key="t-pairing" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.35 }}>

              {/* KPI Row */}
              <motion.div className="ap-kpi-row" variants={stagger} initial="hidden" animate="show">
                {[
                  { icon: <Target size={20} />, val: '94%', label: 'Match Quality', badge: '+12%', accent: '#7c3aed', sub: 'vs last quarter' },
                  { icon: <Users size={20} />, val: '5', label: 'Active Reps', badge: 'Active', accent: '#2dd4bf', sub: 'field deployed' },
                  { icon: <TrendingUp size={20} />, val: '6', label: 'Products', badge: '+8%', accent: '#f59e0b', sub: 'in portfolio' },
                  { icon: <PieChart size={20} />, val: '91%', label: 'Coverage', badge: '+4%', accent: '#ec4899', sub: 'territory reach' },
                ].map((k, i) => (
                  <motion.div key={i} variants={fadeUp} className="ap-kpi">
                    <div className="ap-kpi__head">
                      <div className="ap-kpi__icon" style={{ '--kpi-color': k.accent }}>{k.icon}</div>
                      <span className="ap-kpi__badge">{k.badge}</span>
                    </div>
                    <span className="ap-kpi__val">{k.val}</span>
                    <span className="ap-kpi__label">{k.label}</span>
                    <span className="ap-kpi__sub">{k.sub}</span>
                  </motion.div>
                ))}
              </motion.div>

              {/* Split: Product List + Pairings */}
              <div className="ap-split">

                {/* Left — Product Selector */}
                <motion.aside variants={fadeUp} initial="hidden" animate="show" className="ap-products">
                  <div className="ap-products__header">
                    <h3 className="ap-section-title">
                      <Layers size={16} />
                      Product Portfolio
                    </h3>
                    <span className="ap-products__count">{PRODUCTS.length} products</span>
                  </div>
                  <ul className="ap-products__list">
                    {PRODUCTS.map(p => (
                      <li key={p.id}>
                        <button
                          className={`ap-products__item ${selectedProduct === p.id ? 'is-active' : ''}`}
                          onClick={() => setSelectedProduct(p.id)}
                        >
                          <div className="ap-products__item-left">
                            <span className="ap-products__emoji">{p.icon}</span>
                            <div>
                              <div className="ap-products__name">{p.name}</div>
                              <div className="ap-products__meta">
                                <span className="ap-products__cat">{p.category}</span>
                                <span className="ap-products__dot">·</span>
                                <span>{p.recs} recs</span>
                              </div>
                            </div>
                          </div>
                          <div className="ap-products__item-right">
                            <div className="ap-products__eff">
                              <div className="ap-products__eff-bar">
                                <div className="ap-products__eff-fill" style={{ width: `${p.efficiency}%` }} />
                              </div>
                              <span className="ap-products__eff-num">{p.efficiency}%</span>
                            </div>
                            <ChevronRight size={14} className="ap-products__arrow" />
                          </div>
                        </button>
                      </li>
                    ))}
                  </ul>

                  {/* Trend Sparkline */}
                  <div className="ap-products__trend">
                    <div className="ap-products__trend-header">
                      <span className="ap-products__trend-title">Match Score Trend</span>
                      <span className="ap-products__trend-badge">6mo</span>
                    </div>
                    <ResponsiveContainer width="100%" height={80}>
                      <AreaChart data={TREND_DATA}>
                        <defs>
                          <linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#7c3aed" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#7c3aed" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <Area type="monotone" dataKey="score" stroke="#7c3aed" strokeWidth={2} fill="url(#trendGrad)" dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </motion.aside>

                {/* Right — AI Pairings */}
                <motion.section variants={fadeUp} initial="hidden" animate="show" className="ap-pairings">
                  <div className="ap-pairings__head">
                    <div>
                      <h2 className="ap-pairings__title">AI-Recommended Pairings</h2>
                      <p className="ap-pairings__sub">
                        Optimized matches for <strong>{currentProduct?.name}</strong>
                        <span className="ap-pairings__cat-tag">{currentProduct?.category}</span>
                      </p>
                    </div>
                    <span className="ap-badge-ai"><Sparkles size={13} /> AI-Powered</span>
                  </div>

                  {/* Rep Cards */}
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={selectedProduct}
                      initial={{ opacity: 0, x: 16 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -16 }}
                      transition={{ duration: 0.25 }}
                      className="ap-reps"
                    >
                      {currentPairings.map((rep, idx) => (
                        <div key={idx} className={`ap-rep ${idx === 0 ? 'ap-rep--top' : ''}`}>
                          {idx === 0 && <div className="ap-rep__best-badge"><Star size={12} /> Best Match</div>}
                          <div className="ap-rep__content">
                            {/* Left: Avatar + Info */}
                            <div className="ap-rep__info">
                              <div className="ap-rep__avatar-wrap">
                                <div className="ap-rep__avatar">{rep.initial}</div>
                                <span className="ap-rep__rank-badge">#{rep.rank}</span>
                              </div>
                              <div className="ap-rep__details">
                                <div className="ap-rep__name">{rep.name}</div>
                                <div className="ap-rep__spec">{rep.specialty}</div>
                                <div className="ap-rep__meta-row">
                                  <span><MapPin size={12} /> {rep.territory}</span>
                                  <span><Eye size={12} /> {rep.visits} visits</span>
                                  <span><TrendingUp size={12} /> {rep.conversion}</span>
                                </div>
                              </div>
                            </div>

                            {/* Right: Score Circle */}
                            <div className="ap-rep__score-area">
                              <CircleScore score={rep.score} size={80} strokeWidth={6} color={idx === 0 ? '#7c3aed' : '#2dd4bf'} />
                              <span className="ap-rep__score-label">Match Score</span>
                            </div>
                          </div>

                          {/* Sub-scores */}
                          <div className="ap-rep__subscores">
                            <MiniBar label="Relationship" value={rep.subScores.relationship} color="#7c3aed" delay={0.1} />
                            <MiniBar label="Territory" value={rep.subScores.territory} color="#2dd4bf" delay={0.2} />
                            <MiniBar label="Experience" value={rep.subScores.experience} color="#f59e0b" delay={0.3} />
                            <MiniBar label="Knowledge" value={rep.subScores.knowledge} color="#ec4899" delay={0.4} />
                          </div>

                          {/* AI Reasoning */}
                          <div className="ap-rep__reason">
                            <div className="ap-rep__reason-header">
                              <Brain size={14} />
                              <span className="ap-rep__reason-tag">AI Reasoning</span>
                            </div>
                            <p>{rep.reasoning}</p>
                          </div>

                          {/* Actions */}
                          <div className="ap-rep__actions">
                            <button className="ap-btn ap-btn--primary">
                              <UserCheck size={16} />
                              Assign Rep
                            </button>
                            <button className="ap-btn ap-btn--ghost">View Profile</button>
                          </div>
                        </div>
                      ))}
                    </motion.div>
                  </AnimatePresence>

                  {/* Pairing Flow */}
                  <div className="ap-flow">
                    <h3 className="ap-section-title">
                      <GitBranch size={16} />
                      Pairing Flow
                    </h3>
                    <div className="ap-flow__diagram">
                      <div className="ap-flow__node ap-flow__node--product">
                        <span className="ap-flow__node-emoji">{currentProduct?.icon}</span>
                        <span className="ap-flow__node-label">Product</span>
                        <span className="ap-flow__node-name">{currentProduct?.name}</span>
                      </div>
                      <div className="ap-flow__connector">
                        <div className="ap-flow__line" />
                        <Sparkles size={16} className="ap-flow__spark" />
                        <div className="ap-flow__line" />
                      </div>
                      <div className="ap-flow__targets">
                        {currentPairings.map((rep, i) => (
                          <div key={i} className="ap-flow__node ap-flow__node--rep">
                            <div className="ap-flow__rep-avatar">{rep.initial}</div>
                            <span className="ap-flow__node-name">{rep.name}</span>
                            <span className="ap-flow__match-tag">{rep.score}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </motion.section>
              </div>

              {/* Charts */}
              <div className="ap-charts">
                <motion.div variants={fadeUp} initial="hidden" animate="show" className="ap-chart-card">
                  <div className="ap-chart-card__header">
                    <div>
                      <h3 className="ap-chart-card__title">Pairing Efficiency by Product</h3>
                      <p className="ap-chart-card__sub">Performance across portfolio</p>
                    </div>
                    <span className="ap-chart-card__badge"><BarChart2 size={13} /> Analytics</span>
                  </div>
                  <div className="ap-chart-card__body">
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={PRODUCTS} barSize={36} barGap={8}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(150,150,180,0.1)" vertical={false} />
                        <XAxis dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false} />
                        <YAxis domain={[0, 100]} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false} />
                        <Tooltip content={<CustomBarTooltip />} cursor={{ fill: 'rgba(124,58,237,0.05)' }} />
                        <Bar dataKey="efficiency" radius={[8, 8, 0, 0]}>
                          {PRODUCTS.map((_, i) => <Cell key={i} fill={BAR_COLORS[i]} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </motion.div>

                <motion.div variants={fadeUp} initial="hidden" animate="show" className="ap-chart-card">
                  <div className="ap-chart-card__header">
                    <div>
                      <h3 className="ap-chart-card__title">Rep Performance Distribution</h3>
                      <p className="ap-chart-card__sub">Experience vs. Score mapping</p>
                    </div>
                    <span className="ap-chart-card__badge"><Activity size={13} /> Live</span>
                  </div>
                  <div className="ap-chart-card__body">
                    <ResponsiveContainer width="100%" height={280}>
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(150,150,180,0.1)" />
                        <XAxis type="number" dataKey="x" name="Experience"
                          tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false}
                          label={{ value: 'Experience (yrs)', position: 'insideBottom', offset: -4, fill: 'var(--text-secondary)', fontSize: 11 }}
                        />
                        <YAxis type="number" dataKey="y" name="Score" domain={[70, 100]}
                          tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false}
                        />
                        <Tooltip content={<CustomScatterTooltip />} />
                        <Scatter data={SCATTER_DATA}>
                          {SCATTER_DATA.map((_, i) => <Cell key={i} fill={SCATTER_COLORS[i]} r={10} />)}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </motion.div>
              </div>

            </motion.div>
          )}

          {/* ── TAB 2: Messaging Strategies ── */}
          {activeTab === 'messaging' && (
            <motion.div key="t-messaging" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.35 }}>

              {/* KPI Row */}
              <motion.div className="ap-kpi-row ap-kpi-row--three" variants={stagger} initial="hidden" animate="show">
                {[
                  { icon: <Activity size={20} />, val: '92%', label: 'Avg. Alignment', badge: '+15%', accent: '#2dd4bf', sub: 'across frameworks' },
                  { icon: <MessageSquare size={20} />, val: '4', label: 'Frameworks', badge: 'Active', accent: '#7c3aed', sub: 'in strategy library' },
                  { icon: <Briefcase size={20} />, val: '95', label: 'Sessions', badge: '+22%', accent: '#f59e0b', sub: 'this quarter' },
                ].map((k, i) => (
                  <motion.div key={i} variants={fadeUp} className="ap-kpi">
                    <div className="ap-kpi__head">
                      <div className="ap-kpi__icon" style={{ '--kpi-color': k.accent }}>{k.icon}</div>
                      <span className="ap-kpi__badge">{k.badge}</span>
                    </div>
                    <span className="ap-kpi__val">{k.val}</span>
                    <span className="ap-kpi__label">{k.label}</span>
                    <span className="ap-kpi__sub">{k.sub}</span>
                  </motion.div>
                ))}
              </motion.div>

              {/* Strategy Section Header */}
              <div className="ap-msg-header">
                <div>
                  <h2 className="ap-msg-header__title">High Impact Strategies</h2>
                  <p className="ap-msg-header__sub">Select a framework to explore alignment data & evidence</p>
                </div>
                <span className="ap-badge-ai"><Sparkles size={13} /> Strategy Library</span>
              </div>

              {/* Strategy Cards Grid */}
              <motion.div className="ap-strategies" variants={stagger} initial="hidden" animate="show">
                {STRATEGIES.map((s, i) => (
                  <motion.div key={i} variants={fadeUp} className="ap-strat" whileHover={{ y: -6, transition: { duration: 0.25 } }}>
                    <div className="ap-strat__header">
                      <div className="ap-strat__icon" style={{ '--strat-color': s.color }}>{s.icon}</div>
                      <CircleScore score={s.alignment} size={52} strokeWidth={4} color={s.color} />
                    </div>
                    <h3 className="ap-strat__title">{s.title}</h3>
                    <div className="ap-strat__tags">
                      {s.tags.map((tag, ti) => (
                        <span key={ti} className="ap-strat__tag" style={{ '--strat-color': s.color }}>{tag}</span>
                      ))}
                    </div>
                    <p className="ap-strat__desc">{s.desc}</p>

                    {/* Metrics mini row */}
                    <div className="ap-strat__metrics">
                      <div className="ap-strat__metric">
                        <span className="ap-strat__metric-label">Adoption</span>
                        <span className="ap-strat__metric-val">{s.metrics.adoption}</span>
                      </div>
                      <div className="ap-strat__metric">
                        <span className="ap-strat__metric-label">Impact</span>
                        <span className="ap-strat__metric-val">{s.metrics.impact}</span>
                      </div>
                      <div className="ap-strat__metric">
                        <span className="ap-strat__metric-label">Sessions</span>
                        <span className="ap-strat__metric-val">{s.metrics.sessions}</span>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="ap-strat__bar">
                      <motion.div
                        className="ap-strat__bar-fill"
                        style={{ background: s.color }}
                        initial={{ width: 0 }}
                        animate={{ width: `${s.alignment}%` }}
                        transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1], delay: i * 0.12 }}
                      />
                    </div>

                    <button className="ap-strat__cta" style={{ '--strat-color': s.color }}>
                      Review Evidence <ArrowRight size={14} />
                    </button>
                  </motion.div>
                ))}
              </motion.div>

            </motion.div>
          )}

        </AnimatePresence>
      </motion.div>
    </div>
  );
};

export default MedRepAnalyticsPairing;
