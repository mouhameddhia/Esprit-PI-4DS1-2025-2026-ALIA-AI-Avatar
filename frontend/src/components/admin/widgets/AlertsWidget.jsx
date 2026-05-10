import React, { useState, useEffect, useCallback } from 'react';
import { BellRing, CheckCircle, Trash2, Filter } from 'lucide-react';
import { alertsApi } from '../../../utils/api';
import ConfirmDialog from '../shared/ConfirmDialog';

const SEVERITY_STYLES = {
  HIGH:   { bg: '#ef444422', color: '#ef4444', label: 'HIGH' },
  MEDIUM: { bg: '#f59e0b22', color: '#f59e0b', label: 'MED' },
  LOW:    { bg: '#10b98122', color: '#10b981', label: 'LOW' },
};

function AlertCard({ alert, onResolve, onDelete }) {
  const sev = SEVERITY_STYLES[alert.severity] || SEVERITY_STYLES.LOW;
  const isResolved = alert.status === 'resolved';

  return (
    <div className={`alert-card ${isResolved ? 'alert-resolved' : ''}`}>
      <div className="alert-card-header">
        <span className="severity-badge" style={{ backgroundColor: sev.bg, color: sev.color }}>
          {sev.label}
        </span>
        <span className="alert-timestamp">{new Date(alert.timestamp).toLocaleString()}</span>
        {isResolved && <span className="resolved-tag"><CheckCircle size={12} /> Resolved</span>}
      </div>

      <p className="alert-message">"{alert.user_message}"</p>

      <div className="alert-meta">
        <div className="alert-meta-item">
          <span className="meta-label">Intent</span>
          <span className="meta-value">{alert.detected_intent}</span>
        </div>
        {alert.missing_entity && (
          <div className="alert-meta-item">
            <span className="meta-label">Missing Entity</span>
            <span className="meta-value">{alert.missing_entity}</span>
          </div>
        )}
      </div>

      {alert.missing_info && (
        <p className="alert-info-text">{alert.missing_info}</p>
      )}

      <div className="alert-card-actions">
        {!isResolved && (
          <button className="btn btn-success btn-sm" onClick={() => onResolve(alert)}>
            <CheckCircle size={14} /> Mark Resolved
          </button>
        )}
        <button className="btn btn-danger-ghost btn-sm" onClick={() => onDelete(alert)}>
          <Trash2 size={14} />
        </button>
      </div>
    </div>
  );
}

export default function AlertsWidget() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [severityFilter, setSeverityFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('open');

  const [deleteTarget, setDeleteTarget] = useState(null);
  const [resolveTarget, setResolveTarget] = useState(null);

  const fetchAlerts = useCallback(async () => {
    setLoading(true);
    try {
      const data = await alertsApi.list({
        severity: severityFilter || undefined,
        status: statusFilter || undefined,
      });
      setAlerts(data || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [severityFilter, statusFilter]);

  useEffect(() => { fetchAlerts(); }, [fetchAlerts]);

  const handleResolve = async () => {
    if (!resolveTarget) return;
    try {
      await alertsApi.resolve(resolveTarget.id);
      setResolveTarget(null);
      fetchAlerts();
    } catch (err) {
      alert(err.message);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await alertsApi.delete(deleteTarget.id);
      setDeleteTarget(null);
      fetchAlerts();
    } catch (err) {
      alert(err.message);
    }
  };

  const openCount = alerts.filter((a) => a.status === 'open').length;
  const highCount = alerts.filter((a) => a.severity === 'HIGH' && a.status === 'open').length;

  return (
    <div className="widget">
      <div className="widget-header">
        <div className="widget-title-row">
          <BellRing size={20} className="widget-icon" />
          <div>
            <h2 className="widget-title">Knowledge Gap Alerts</h2>
            <p className="widget-subtitle">Detected when ALIA lacks answers to user questions</p>
          </div>
        </div>
        <div className="alert-counts">
          {highCount > 0 && (
            <span className="count-badge" style={{ backgroundColor: '#ef444422', color: '#ef4444' }}>
              {highCount} critical
            </span>
          )}
          <span className="count-badge" style={{ backgroundColor: '#f59e0b22', color: '#f59e0b' }}>
            {openCount} open
          </span>
        </div>
      </div>

      <div className="table-toolbar">
        <div className="filter-group">
          <Filter size={16} style={{ color: '#9ca3af' }} />
          <select className="filter-select" value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}>
            <option value="">All Severities</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>
          <select className="filter-select" value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="">All Status</option>
            <option value="open">Open</option>
            <option value="resolved">Resolved</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div className="loading-state"><span className="spinner" /> Loading alerts…</div>
      ) : alerts.length === 0 ? (
        <div className="empty-state">
          <BellRing size={36} style={{ opacity: 0.3 }} />
          <p>No alerts found for selected filters.</p>
        </div>
      ) : (
        <div className="alerts-list">
          {alerts.map((alert) => (
            <AlertCard
              key={alert.id}
              alert={alert}
              onResolve={setResolveTarget}
              onDelete={setDeleteTarget}
            />
          ))}
        </div>
      )}

      <ConfirmDialog
        open={!!resolveTarget}
        title="Mark as Resolved"
        message="Mark this alert as resolved? It will be archived and removed from the open queue."
        onConfirm={handleResolve}
        onCancel={() => setResolveTarget(null)}
        danger={false}
      />

      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete Alert"
        message="Permanently delete this alert? This action cannot be undone."
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
