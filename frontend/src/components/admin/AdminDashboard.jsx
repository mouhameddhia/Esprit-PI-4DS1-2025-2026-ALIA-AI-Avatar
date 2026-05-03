import React, { useState, useEffect } from 'react';
import AdminSidebar from './AdminSidebar';
import AdminNavbar from './AdminNavbar';
import AnalyticsWidget from './widgets/AnalyticsWidget';
import ProductsWidget from './widgets/ProductsWidget';
import UsersWidget from './widgets/UsersWidget';
import AlertsWidget from './widgets/AlertsWidget';
import { useCurrentUser } from '../../hooks/useCurrentUser';
import { alertsApi } from '../../utils/api';
import './AdminDashboard.css';

export default function AdminDashboard() {
  const { user } = useCurrentUser();
  const [section, setSection] = useState('analytics');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [openAlertCount, setOpenAlertCount] = useState(0);

  // Fetch open alert count for sidebar badge
  useEffect(() => {
    alertsApi.list({ status: 'open', limit: 200 })
      .then((data) => setOpenAlertCount(data?.length ?? 0))
      .catch(() => {});
  }, []);

  const SECTIONS = {
    analytics: <AnalyticsWidget />,
    products:  <ProductsWidget />,
    users:     <UsersWidget />,
    alerts:    <AlertsWidget />,
  };

  return (
    <div className={`admin-layout ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
      <AdminSidebar
        active={section}
        onChange={(s) => { setSection(s); setSidebarOpen(true); }}
        alertCount={openAlertCount}
      />

      <div className="admin-main">
        <AdminNavbar
          section={section}
          user={user}
          alertCount={openAlertCount}
          onMenuToggle={() => setSidebarOpen((v) => !v)}
        />

        <main className="admin-content">
          {SECTIONS[section]}
        </main>
      </div>
    </div>
  );
}
