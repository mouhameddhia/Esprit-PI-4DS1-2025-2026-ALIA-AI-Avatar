import React from 'react';
import {
  BarChart2, Package, Users, BellRing, Brain, LogOut,
  LayoutDashboard, ChevronRight,
} from 'lucide-react';

const NAV_ITEMS = [
  { key: 'analytics', label: 'Analytics', icon: BarChart2 },
  { key: 'products', label: 'Products', icon: Package },
  { key: 'users', label: 'Users', icon: Users },
  { key: 'alerts', label: 'Alerts', icon: BellRing },
];

export default function AdminSidebar({ active, onChange, alertCount = 0 }) {
  return (
    <aside className="admin-sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">
          <Brain size={22} />
        </div>
        <div>
          <span className="sidebar-logo-name">ALIA</span>
          <span className="sidebar-logo-sub">Admin Panel</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <p className="sidebar-section-label">Dashboard</p>
        {NAV_ITEMS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            className={`sidebar-item ${active === key ? 'active' : ''}`}
            onClick={() => onChange(key)}
          >
            <Icon size={18} />
            <span>{label}</span>
            {key === 'alerts' && alertCount > 0 && (
              <span className="sidebar-badge">{alertCount}</span>
            )}
            {active === key && <ChevronRight size={14} className="sidebar-chevron" />}
          </button>
        ))}
      </nav>

      {/* Bottom */}
      <div className="sidebar-bottom">
        <div className="sidebar-divider" />
        <button
          className="sidebar-item sidebar-logout"
          onClick={() => {
            localStorage.clear();
            window.location.href = '/login';
          }}
        >
          <LogOut size={18} />
          <span>Sign Out</span>
        </button>
      </div>
    </aside>
  );
}
