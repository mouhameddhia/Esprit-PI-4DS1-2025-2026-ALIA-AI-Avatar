import React, { useState, useEffect, useCallback } from 'react';
import { Search, Trash2, ShieldCheck, Users } from 'lucide-react';
import { usersApi } from '../../../utils/api';
import DataTable from '../shared/DataTable';
import Modal from '../shared/Modal';
import ConfirmDialog from '../shared/ConfirmDialog';

const ROLES = ['admin', 'medrep', 'physician'];
const ROLE_COLORS = { admin: '#7c3aed', medrep: '#10b981', physician: '#3b82f6' };

export default function UsersWidget() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState('');
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 10;

  const [roleTarget, setRoleTarget] = useState(null);
  const [selectedRole, setSelectedRole] = useState('');
  const [saving, setSaving] = useState(false);

  const [deleteTarget, setDeleteTarget] = useState(null);

  const currentUserEmail = localStorage.getItem('userEmail');

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    try {
      const data = await usersApi.list({
        search: search || undefined,
        role: roleFilter || undefined,
        skip: (page - 1) * PAGE_SIZE,
        limit: PAGE_SIZE,
      });
      setUsers(data || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [search, roleFilter, page]);

  useEffect(() => { fetchUsers(); }, [fetchUsers]);

  const openRoleModal = (user) => {
    setRoleTarget(user);
    setSelectedRole(user.role);
  };

  const handleRoleChange = async () => {
    if (!roleTarget || !selectedRole) return;
    setSaving(true);
    try {
      await usersApi.updateRole(roleTarget.id, selectedRole);
      setRoleTarget(null);
      fetchUsers();
    } catch (err) {
      alert(err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await usersApi.delete(deleteTarget.id);
      setDeleteTarget(null);
      fetchUsers();
    } catch (err) {
      alert(err.message);
    }
  };

  const columns = [
    {
      key: 'name', label: 'Name',
      render: (v, row) => (
        <div className="user-cell">
          <div className="user-avatar" style={{ backgroundColor: ROLE_COLORS[row.role] + '33', color: ROLE_COLORS[row.role] }}>
            {(v || 'U')[0].toUpperCase()}
          </div>
          <div>
            <div className="user-name">{v}</div>
            <div className="user-email">{row.email}</div>
          </div>
        </div>
      ),
    },
    {
      key: 'role', label: 'Role',
      render: (v) => (
        <span className="badge" style={{ backgroundColor: (ROLE_COLORS[v] || '#9ca3af') + '22', color: ROLE_COLORS[v] || '#9ca3af' }}>
          {v}
        </span>
      ),
    },
    {
      key: 'created_at', label: 'Joined',
      render: (v) => v ? new Date(v).toLocaleDateString() : '—',
    },
    {
      key: '_actions', label: 'Actions',
      render: (_, row) => {
        const isSelf = row.email === currentUserEmail;
        return (
          <div className="row-actions">
            <button className="btn btn-ghost btn-sm" onClick={() => openRoleModal(row)} title="Change Role">
              <ShieldCheck size={14} /> Role
            </button>
            <button
              className="btn btn-danger-ghost btn-sm"
              onClick={() => !isSelf && setDeleteTarget(row)}
              disabled={isSelf}
              title={isSelf ? 'Cannot delete your own account' : 'Delete user'}
            >
              <Trash2 size={14} />
            </button>
          </div>
        );
      },
    },
  ];

  return (
    <div className="widget">
      <div className="widget-header">
        <div className="widget-title-row">
          <Users size={20} className="widget-icon" />
          <h2 className="widget-title">User Management</h2>
        </div>
        <span className="widget-count">{users.length} users</span>
      </div>

      <div className="table-toolbar">
        <div className="search-box">
          <Search size={16} />
          <input
            placeholder="Search by name or email…"
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1); }}
          />
        </div>
        <select
          className="filter-select"
          value={roleFilter}
          onChange={(e) => { setRoleFilter(e.target.value); setPage(1); }}
        >
          <option value="">All Roles</option>
          {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
      </div>

      <DataTable
        columns={columns}
        rows={users}
        loading={loading}
        emptyText="No users found."
        page={page}
        pageSize={PAGE_SIZE}
        total={users.length < PAGE_SIZE ? (page - 1) * PAGE_SIZE + users.length : page * PAGE_SIZE + 1}
        onPageChange={setPage}
      />

      {/* Change Role Modal */}
      <Modal open={!!roleTarget} title="Change User Role" onClose={() => setRoleTarget(null)} width="380px">
        {roleTarget && (
          <div className="admin-form">
            <p style={{ color: '#9ca3af', marginBottom: '1rem' }}>
              Update role for <strong style={{ color: '#f9fafb' }}>{roleTarget.name}</strong>
            </p>
            <div className="form-group">
              <label>New Role</label>
              <select value={selectedRole} onChange={(e) => setSelectedRole(e.target.value)}>
                {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
            <div className="form-actions">
              <button className="btn btn-ghost" onClick={() => setRoleTarget(null)}>Cancel</button>
              <button className="btn btn-primary" onClick={handleRoleChange} disabled={saving}>
                {saving ? 'Saving…' : 'Update Role'}
              </button>
            </div>
          </div>
        )}
      </Modal>

      {/* Delete Confirm */}
      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete User"
        message={`Permanently delete "${deleteTarget?.name}" (${deleteTarget?.email})? This cannot be undone.`}
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
