import React from 'react';
import { AlertTriangle } from 'lucide-react';
import Modal from './Modal';

export default function ConfirmDialog({ open, title, message, onConfirm, onCancel, danger = true }) {
  return (
    <Modal open={open} title={title} onClose={onCancel} width="420px">
      <div className="confirm-body">
        {danger && (
          <div className="confirm-icon">
            <AlertTriangle size={28} />
          </div>
        )}
        <p className="confirm-message">{message}</p>
        <div className="confirm-actions">
          <button className="btn btn-ghost" onClick={onCancel}>Cancel</button>
          <button className={`btn ${danger ? 'btn-danger' : 'btn-primary'}`} onClick={onConfirm}>
            Confirm
          </button>
        </div>
      </div>
    </Modal>
  );
}
