import { useAuth0 } from '@auth0/auth0-react';
import { Navigate } from 'react-router-dom';

export function ProtectedRoute({ children }) {
  const { isAuthenticated, isLoading } = useAuth0();

  // Check for JWT token in localStorage (for traditional login)
  const jwtToken = localStorage.getItem('token');

  if (isLoading) {
    return (
      <div style={{ textAlign: 'center', padding: '2rem' }}>
        <p>Loading authentication...</p>
      </div>
    );
  }

  // Allow access if either Auth0 is authenticated or JWT token exists
  if (!isAuthenticated && !jwtToken) {
    return <Navigate to="/" replace />;
  }

  return children;
}
