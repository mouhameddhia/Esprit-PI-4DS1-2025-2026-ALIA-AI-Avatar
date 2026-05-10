import React from 'react';
import { BarChart2, Users, BookOpen, TrendingUp, ExternalLink } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line,
} from 'recharts';

const SCORE_DATA = [
  { month: 'Jan', score: 72 }, { month: 'Feb', score: 76 }, { month: 'Mar', score: 74 },
  { month: 'Apr', score: 81 }, { month: 'May', score: 85 }, { month: 'Jun', score: 88 },
];

const INTERACTIONS_DATA = [
  { week: 'W1', count: 142 }, { week: 'W2', count: 198 }, { week: 'W3', count: 175 },
  { week: 'W4', count: 230 }, { week: 'W5', count: 261 }, { week: 'W6', count: 244 },
];

const KPI = [
  { icon: TrendingUp, label: 'Avg Rep Score', value: '88%', trend: '+4%', color: '#10b981' },
  { icon: Users, label: 'Interactions', value: '1,250', trend: '+12%', color: '#818cf8' },
  { icon: BookOpen, label: 'Training Completion', value: '73%', trend: '+8%', color: '#f59e0b' },
  { icon: BarChart2, label: 'Active Reps', value: '34', trend: '+2', color: '#7c3aed' },
];

export default function AnalyticsWidget() {
  return (
    <div className="widget">
      <div className="widget-header">
        <div>
          <h2 className="widget-title">Performance Analytics</h2>
          <p className="widget-subtitle">PowerBI embed placeholder — replace iframe below with real embed URL</p>
        </div>
        <button className="btn btn-ghost btn-sm">
          <ExternalLink size={14} /> Open PowerBI
        </button>
      </div>

      {/* KPI Cards */}
      <div className="kpi-grid">
        {KPI.map(({ icon: Icon, label, value, trend, color }) => (
          <div className="kpi-card" key={label}>
            <div className="kpi-icon" style={{ backgroundColor: `${color}22`, color }}>
              <Icon size={20} />
            </div>
            <div className="kpi-body">
              <span className="kpi-value">{value}</span>
              <span className="kpi-label">{label}</span>
            </div>
            <span className="kpi-trend" style={{ color }}>{trend}</span>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="chart-grid">
        <div className="chart-card">
          <h4 className="chart-title">Rep Performance Score (6-month)</h4>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={SCORE_DATA}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="month" tick={{ fill: '#9ca3af', fontSize: 12 }} />
              <YAxis domain={[60, 100]} tick={{ fill: '#9ca3af', fontSize: 12 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                labelStyle={{ color: '#f9fafb' }}
              />
              <Line type="monotone" dataKey="score" stroke="#7c3aed" strokeWidth={2} dot={{ r: 4, fill: '#7c3aed' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h4 className="chart-title">Weekly Interactions</h4>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={INTERACTIONS_DATA}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="week" tick={{ fill: '#9ca3af', fontSize: 12 }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 12 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                labelStyle={{ color: '#f9fafb' }}
              />
              <Bar dataKey="count" fill="#818cf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* PowerBI placeholder iframe */}
      <div className="powerbi-placeholder">
        <BarChart2 size={40} style={{ opacity: 0.3 }} />
        <p>PowerBI Report Embed</p>
        <span>Replace this area with a real PowerBI iframe:</span>
        <code>{'<iframe src="https://app.powerbi.com/reportEmbed?reportId=..." />'}</code>
      </div>
    </div>
  );
}
