import React, { useState, useEffect, useCallback } from 'react';
import { Plus, Search, Pencil, Trash2, Package } from 'lucide-react';
import { productsApi } from '../../../utils/api';
import DataTable from '../shared/DataTable';
import Modal from '../shared/Modal';
import ConfirmDialog from '../shared/ConfirmDialog';

const EMPTY_FORM = {
  name: '', description: '', category: '',
  indications: '', contraindications: '', dosage: '',
};

const CATEGORIES = ['Phytotherapy', 'Supplements', 'Vitamins', 'Homeopathy', 'Dermatology', 'Other'];

export default function ProductsWidget() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('');
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 10;

  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState('');

  const [deleteTarget, setDeleteTarget] = useState(null);

  const fetchProducts = useCallback(async () => {
    setLoading(true);
    try {
      const data = await productsApi.list({
        search: search || undefined,
        category: categoryFilter || undefined,
        skip: (page - 1) * PAGE_SIZE,
        limit: PAGE_SIZE,
      });
      setProducts(data || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [search, categoryFilter, page]);

  useEffect(() => { fetchProducts(); }, [fetchProducts]);

  const openCreate = () => {
    setEditing(null);
    setForm(EMPTY_FORM);
    setFormError('');
    setModalOpen(true);
  };

  const openEdit = (product) => {
    setEditing(product);
    setForm({
      name: product.name,
      description: product.description,
      category: product.category,
      indications: product.indications.join(', '),
      contraindications: product.contraindications.join(', '),
      dosage: product.dosage,
    });
    setFormError('');
    setModalOpen(true);
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setFormError('');
    setSaving(true);
    try {
      const payload = {
        ...form,
        indications: form.indications.split(',').map((s) => s.trim()).filter(Boolean),
        contraindications: form.contraindications.split(',').map((s) => s.trim()).filter(Boolean),
      };
      if (editing) {
        await productsApi.update(editing.id, payload);
      } else {
        await productsApi.create(payload);
      }
      setModalOpen(false);
      fetchProducts();
    } catch (err) {
      setFormError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await productsApi.delete(deleteTarget.id);
      setDeleteTarget(null);
      fetchProducts();
    } catch (err) {
      alert(err.message);
    }
  };

  const columns = [
    { key: 'name', label: 'Product Name' },
    { key: 'category', label: 'Category', render: (v) => <span className="badge badge-purple">{v}</span> },
    {
      key: 'indications', label: 'Indications',
      render: (v) => (
        <span className="text-truncate">{Array.isArray(v) ? v.slice(0, 2).join(', ') : v}{Array.isArray(v) && v.length > 2 ? '…' : ''}</span>
      ),
    },
    { key: 'dosage', label: 'Dosage' },
    {
      key: 'created_at', label: 'Created',
      render: (v) => v ? new Date(v).toLocaleDateString() : '—',
    },
    {
      key: '_actions', label: 'Actions',
      render: (_, row) => (
        <div className="row-actions">
          <button className="btn btn-ghost btn-sm" onClick={() => openEdit(row)} title="Edit">
            <Pencil size={14} />
          </button>
          <button className="btn btn-danger-ghost btn-sm" onClick={() => setDeleteTarget(row)} title="Delete">
            <Trash2 size={14} />
          </button>
        </div>
      ),
    },
  ];

  return (
    <div className="widget">
      <div className="widget-header">
        <div className="widget-title-row">
          <Package size={20} className="widget-icon" />
          <h2 className="widget-title">Product Management</h2>
        </div>
        <button className="btn btn-primary" onClick={openCreate}>
          <Plus size={16} /> Add Product
        </button>
      </div>

      <div className="table-toolbar">
        <div className="search-box">
          <Search size={16} />
          <input
            placeholder="Search products…"
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1); }}
          />
        </div>
        <select
          className="filter-select"
          value={categoryFilter}
          onChange={(e) => { setCategoryFilter(e.target.value); setPage(1); }}
        >
          <option value="">All Categories</option>
          {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      <DataTable
        columns={columns}
        rows={products}
        loading={loading}
        emptyText="No products found. Add your first product."
        page={page}
        pageSize={PAGE_SIZE}
        total={products.length < PAGE_SIZE ? (page - 1) * PAGE_SIZE + products.length : page * PAGE_SIZE + 1}
        onPageChange={setPage}
      />

      {/* Create / Edit Modal */}
      <Modal
        open={modalOpen}
        title={editing ? 'Edit Product' : 'Add Product'}
        onClose={() => setModalOpen(false)}
        width="640px"
      >
        <form onSubmit={handleSave} className="admin-form">
          {formError && <div className="form-error">{formError}</div>}

          <div className="form-row">
            <div className="form-group">
              <label>Product Name *</label>
              <input required value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
            </div>
            <div className="form-group">
              <label>Category *</label>
              <select required value={form.category} onChange={(e) => setForm({ ...form, category: e.target.value })}>
                <option value="">Select category</option>
                {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>

          <div className="form-group">
            <label>Description *</label>
            <textarea rows={3} required value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })} />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Indications <span className="form-hint">(comma-separated)</span></label>
              <input value={form.indications} onChange={(e) => setForm({ ...form, indications: e.target.value })}
                placeholder="e.g. Anxiety, Sleep disorders" />
            </div>
            <div className="form-group">
              <label>Contraindications <span className="form-hint">(comma-separated)</span></label>
              <input value={form.contraindications}
                onChange={(e) => setForm({ ...form, contraindications: e.target.value })}
                placeholder="e.g. Pregnancy, Children under 12" />
            </div>
          </div>

          <div className="form-group">
            <label>Dosage *</label>
            <input required value={form.dosage} onChange={(e) => setForm({ ...form, dosage: e.target.value })}
              placeholder="e.g. 2 capsules twice daily with meals" />
          </div>

          <div className="form-actions">
            <button type="button" className="btn btn-ghost" onClick={() => setModalOpen(false)}>Cancel</button>
            <button type="submit" className="btn btn-primary" disabled={saving}>
              {saving ? 'Saving…' : editing ? 'Save Changes' : 'Create Product'}
            </button>
          </div>
        </form>
      </Modal>

      {/* Delete Confirm */}
      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete Product"
        message={`Are you sure you want to delete "${deleteTarget?.name}"? This action cannot be undone.`}
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
