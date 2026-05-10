import React from 'react';
import { Bell, Menu, ShieldCheck } from 'lucide-react';

const SECTION_TITLES = {
  analytics: 'Performance Analytics',
  products:  'Product Management',
  users:     'User Management',
  alerts:    'Knowledge Gap Alerts',
};

export default function AdminNavbar({ section, user, alertCount = 0, onMenuToggle }) {
  return (
    <header className="admin-navbar">
      <div className="navbar-left">
        <button className="menu-toggle" onClick={onMenuToggle} aria-label="Toggle menu">
          <Menu size={20} />
        </button>
        <div>
          <h1 className="navbar-title">{SECTION_TITLES[section] || 'Dashboard'}</h1>
          <p className="navbar-breadcrumb">Admin / {SECTION_TITLES[section]}</p>
        </div>
      </div>

      <div className="navbar-right">
        {alertCount > 0 && (
          <div className="navbar-alert-btn" title={`${alertCount} open alerts`}>
            <Bell size={18} />
            <span className="navbar-badge">{alertCount}</span>
          </div>
        )}

        <div className="navbar-user">
          <div className="navbar-user-info">
            <span className="navbar-user-name">{user?.name || 'Admin'}</span>
            <span className="navbar-user-role">
              <ShieldCheck size={11} /> {user?.role || 'admin'}
            </span>
          </div>
          <div className="navbar-avatar">
            {(user?.name || 'A')[0].toUpperCase()}
          </div>
        </div>
      </div>
    </header>
  );
}
