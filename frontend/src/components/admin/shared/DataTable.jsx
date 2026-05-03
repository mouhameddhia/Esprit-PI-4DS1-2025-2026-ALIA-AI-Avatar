import React from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

/**
 * Generic data table.
 * columns: [{ key, label, render? }]
 * rows: array of objects
 */
export default function DataTable({
  columns,
  rows,
  loading,
  emptyText = 'No records found.',
  page,
  pageSize,
  total,
  onPageChange,
}) {
  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="table-wrapper">
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col.key}>{col.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={columns.length} className="table-state">
                  <span className="spinner" /> Loading…
                </td>
              </tr>
            ) : rows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="table-state">
                  {emptyText}
                </td>
              </tr>
            ) : (
              rows.map((row, i) => (
                <tr key={row.id ?? i}>
                  {columns.map((col) => (
                    <td key={col.key}>
                      {col.render ? col.render(row[col.key], row) : (row[col.key] ?? '—')}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <span className="pagination-info">
            Page {page} of {totalPages} · {total} records
          </span>
          <div className="pagination-controls">
            <button
              className="btn btn-ghost btn-sm"
              disabled={page <= 1}
              onClick={() => onPageChange(page - 1)}
            >
              <ChevronLeft size={16} />
            </button>
            <button
              className="btn btn-ghost btn-sm"
              disabled={page >= totalPages}
              onClick={() => onPageChange(page + 1)}
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
