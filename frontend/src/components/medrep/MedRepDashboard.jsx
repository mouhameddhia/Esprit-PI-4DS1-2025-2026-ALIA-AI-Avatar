import React from 'react';
import { 
  LogOut, 
  ArrowLeft, 
  Target, 
  TrendingUp, 
  Activity, 
  CheckCircle,
  AlertTriangle,
  Brain
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
  BarChart, Bar,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import './MedRepPortal.css'; // Re-use ambient portal styles
import './MedRepDashboard.css';

// Mock Data for Charts based on the Subagent Report
const performanceData = [
  { date: '2026-01-15', score: 82 },
  { date: '2026-01-20', score: 86 },
  { date: '2026-01-25', score: 89 },
  { date: '2026-01-28', score: 91 },
  { date: '2026-02-01', score: 94 },
];

const skillsData = [
  { subject: 'Product Knowledge', A: 90, fullMark: 100 },
  { subject: 'Empathy', A: 94, fullMark: 100 },
  { subject: 'Objection Handling', A: 75, fullMark: 100 },
  { subject: 'Persuasion', A: 82, fullMark: 100 },
  { subject: 'Clarity', A: 88, fullMark: 100 },
  { subject: 'Rapport', A: 92, fullMark: 100 },
];

const productData = [
  { name: 'CardioGuard', score: 94 },
  { name: 'NeuroShield', score: 88 },
  { name: 'DiabetoCare', score: 91 },
  { name: 'RespiClear', score: 85 },
  { name: 'OncoPro', score: 89 },
];

const recentSessions = [
  {
    product: 'CardioGuard',
    date: '2026-02-01',
    doctor: 'Dr. Conservative',
    scores: { clarity: 93, accuracy: 95, persuasion: 94 },
    overall: 94
  },
  {
    product: 'DiabetoCare Plus',
    date: '2026-01-28',
    doctor: 'Dr. Friendly',
    scores: { clarity: 95, accuracy: 90, persuasion: 91 },
    overall: 92
  },
  {
    product: 'RespiClear',
    date: '2026-01-22',
    doctor: 'Dr. Skeptic',
    scores: { clarity: 88, accuracy: 92, persuasion: 85 },
    overall: 88
  }
];

const gapAnalysisData = [
  {
    title: 'Objection Handling',
    priority: 'high',
    current: 75,
    target: 90,
    scenarios: ['Dr. Skeptical - CardioGuard', 'Dr. Conservative - OncoPro']
  },
  {
    title: 'Clinical Data Presentation',
    priority: 'medium',
    current: 82,
    target: 95,
    scenarios: ['Dr. Academic - NeuroShield']
  },
  {
    title: 'Time-Efficient Communication',
    priority: 'low',
    current: 88,
    target: 95,
    scenarios: ['Dr. Busy - RespiClear']
  }
];

const MedRepDashboard = () => {
  const navigate = useNavigate();

  const handleSignOut = () => {
    navigate('/login');
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } }
  };

  return (
    <div className="portal-container">
      {/* Re-using the beautiful aura backgrounds */}
      <div className="portal-bg-aura"></div>
      <div className="portal-bg-aura-2"></div>

      {/* Navbar directly adapted from MedRepPortal */}
      <nav className="portal-navbar">
        <div className="portal-nav-left">
          <div className="portal-nav-logo">A</div>
          <div className="portal-nav-text">
            <span className="portal-nav-title">ALIA Platform</span>
            <span className="portal-nav-subtitle">Medical Representative Portal</span>
          </div>
        </div>
        <div className="portal-nav-right">
          <button className="portal-signout-btn" onClick={handleSignOut}>
            <LogOut size={18} />
            <span>Sign Out</span>
          </button>
        </div>
      </nav>

      <motion.main 
        className="dashboard-main"
        variants={containerVariants}
        initial="hidden"
        animate="show"
      >
        <motion.div variants={itemVariants} className="dashboard-header">
          <div>
            <h1>Performance Dashboard</h1>
            <p>Your comprehensive analytics and AI-powered interaction metrics.</p>
          </div>
          <button className="dashboard-back-btn" onClick={() => navigate('/portal')}>
            <ArrowLeft size={16} /> Back to Portal
          </button>
        </motion.div>

        {/* Top KPIs Row */}
        <motion.div variants={itemVariants} className="dashboard-stats-grid">
          <div className="dashboard-stat-card border-t-4 border-purple-500">
            <span className="dashboard-stat-title"><Target size={16} style={{ color: '#a855f7' }}/> Overall Score</span>
            <span className="dashboard-stat-value">92%</span>
            <span className="dashboard-stat-trend trend-up"><TrendingUp size={14}/> +3% from last month</span>
          </div>
          <div className="dashboard-stat-card border-t-4 border-teal-500">
            <span className="dashboard-stat-title"><CheckCircle size={16} style={{ color: '#2dd4bf' }}/> Sessions Completed</span>
            <span className="dashboard-stat-value">47</span>
            <span className="dashboard-stat-trend trend-up"><TrendingUp size={14}/> +12 this week</span>
          </div>
          <div className="dashboard-stat-card border-t-4 border-blue-500">
            <span className="dashboard-stat-title"><Activity size={16} style={{ color: '#3b82f6' }}/> Average Score</span>
            <span className="dashboard-stat-value">89%</span>
            <span className="dashboard-stat-trend trend-up"><TrendingUp size={14}/> Top 10% of Reps</span>
          </div>
          <div className="dashboard-stat-card border-t-4 border-orange-500">
            <span className="dashboard-stat-title"><AlertTriangle size={16} style={{ color: '#f97316' }}/> Priority Focus Area</span>
            <span className="dashboard-stat-value" style={{ fontSize: '1.5rem', marginTop: 'auto' }}>Objection Handling</span>
          </div>
        </motion.div>

        {/* AI-Powered Gap Analysis Section */}
        <motion.div variants={itemVariants} className="gap-analysis-wrapper">
          <div className="gap-analysis-header">
            <div className="gap-header-icon">
              <Brain size={32} />
            </div>
            <div className="gap-header-text">
              <h2>AI-Powered Gap Analysis</h2>
              <p>Personalized training recommendations based on your performance data</p>
            </div>
          </div>
          
          <div className="gap-cards-grid">
            {gapAnalysisData.map((gap, index) => (
              <div key={index} className={`gap-card ${gap.priority === 'high' ? 'high-priority' : ''}`}>
                <div className="gap-card-title">
                  <h3>{gap.title}</h3>
                  <span className={`gap-badge ${gap.priority}`}>
                    {gap.priority} priority
                  </span>
                </div>
                
                <div className="gap-progress-container">
                  <div className="gap-progress-row">
                    <span className="gap-progress-label">Current</span>
                    <span className="gap-progress-value">{gap.current}%</span>
                  </div>
                  <div className="gap-progress-bar-bg">
                    <div 
                      className="gap-progress-bar-fill" 
                      style={{ width: `${gap.current}%` }}
                    ></div>
                  </div>
                  <div className="gap-progress-row">
                    <span className="gap-progress-label">Target</span>
                    <span className="gap-progress-value" style={{ color: '#7c3aed' }}>{gap.target}%</span>
                  </div>
                </div>

                <div className="gap-scenarios">
                  <div className="gap-scenarios-title">Recommended Scenarios:</div>
                  {gap.scenarios.map((scenario, i) => (
                    <div key={i} className="gap-scenario-item">
                      {scenario}
                    </div>
                  ))}
                </div>

                <button className="gap-start-btn">Start Training</button>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Detailed Performance Metrics - Full Width */}
        <motion.div variants={itemVariants} className="dashboard-charts-grid">
          <div className="dashboard-chart-card">
            <div className="dashboard-chart-header">
              <span className="dashboard-chart-title">Detailed Performance Metrics</span>
            </div>
            <div className="dashboard-chart-content">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={performanceData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="date" stroke="var(--text-secondary)" fontSize={12} />
                  <YAxis stroke="var(--text-secondary)" fontSize={12} domain={[0, 100]} />
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: 'var(--bg-color)', borderColor: 'var(--glass-border)', borderRadius: '8px' }}
                    itemStyle={{ color: 'var(--text-primary)' }}
                  />
                  <Line type="monotone" dataKey="score" stroke="#7c3aed" strokeWidth={3} dot={{ r: 4, fill: '#7c3aed' }} activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </motion.div>

        {/* Bar & Radar Charts - 2 Columns */}
        <motion.div variants={itemVariants} className="dashboard-charts-grid two-cols">
          <div className="dashboard-chart-card">
            <div className="dashboard-chart-header">
              <span className="dashboard-chart-title">Product Specific Performance</span>
            </div>
            <div className="dashboard-chart-content">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={productData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} vertical={false} />
                  <XAxis dataKey="name" stroke="var(--text-secondary)" fontSize={12} />
                  <YAxis stroke="var(--text-secondary)" fontSize={12} domain={[0, 100]} />
                  <RechartsTooltip 
                    cursor={{ fill: 'transparent' }} 
                    contentStyle={{ backgroundColor: 'var(--bg-color)', borderColor: 'var(--glass-border)', borderRadius: '8px' }}
                  />
                  <Bar dataKey="score" fill="#7c3aed" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="dashboard-chart-card">
            <div className="dashboard-chart-header">
              <span className="dashboard-chart-title">Skills Assessment</span>
            </div>
            <div className="dashboard-chart-content">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={skillsData}>
                  <PolarGrid stroke="var(--glass-border)" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: 'transparent', border: 'none' }} axisLine={false} />
                  <Radar name="Score" dataKey="A" stroke="#2dd4bf" fill="#2dd4bf" fillOpacity={0.4} />
                  <RechartsTooltip contentStyle={{ backgroundColor: 'var(--bg-color)', borderColor: 'var(--glass-border)', borderRadius: '8px' }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </motion.div>

        {/* Recent Sessions List */}
        <motion.div variants={itemVariants} className="dashboard-chart-card">
          <div className="dashboard-chart-header">
            <span className="dashboard-chart-title">Recent Training Sessions</span>
          </div>
          <div className="dashboard-sessions-list">
            {recentSessions.map((session, index) => (
              <div key={index} className="dashboard-session-item">
                <div className={`session-score-box ${session.overall >= 90 ? 'score-excellent' : 'score-good'}`}>
                  {session.overall}
                </div>
                <div className="session-details">
                  <span className="session-title">{session.product}</span>
                  <span className="session-meta">
                    {session.date} • vs {session.doctor}
                  </span>
                  <div className="session-subscores">
                    <span>Clarity: {session.scores.clarity}</span>
                    <span>Accuracy: {session.scores.accuracy}</span>
                    <span>Persuasion: {session.scores.persuasion}</span>
                  </div>
                </div>
                <div className="session-overall">
                  <span className="session-overall-label">Overall Score</span>
                  <span className="session-overall-value">{session.overall}%</span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

      </motion.main>
    </div>
  );
};

export default MedRepDashboard;
