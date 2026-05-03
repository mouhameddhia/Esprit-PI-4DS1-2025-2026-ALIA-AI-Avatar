import { useAuth0 } from '@auth0/auth0-react';
import { Navigate } from 'react-router-dom';

export function ProtectedRoute({ children }) {
  const { isAuthenticated, isLoading } = useAuth0();
  const jwtToken = localStorage.getItem('token');

  if (isLoading) {
    return (
      <div style={{ textAlign: 'center', padding: '2rem' }}>
        <p>Loading authentication…</p>
      </div>
    );
  }

  if (!isAuthenticated && !jwtToken) {
    return <Navigate to="/" replace />;
  }

  return children;
}

/**
 * Guards admin-only routes.
 * Checks JWT token + role stored in localStorage after login.
 */
export function AdminRoute({ children }) {
  const jwtToken = localStorage.getItem('token');
  const userRole = localStorage.getItem('userRole');

  if (!jwtToken) {
    return <Navigate to="/login" replace />;
  }

  if (userRole && userRole !== 'admin') {
    return <Navigate to="/" replace />;
  }

  return children;
}
