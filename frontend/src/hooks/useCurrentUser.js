import { useState, useEffect } from 'react';
import { authApi } from '../utils/api';

export function useCurrentUser() {
  const [user, setUser] = useState(() => {
    // Hydrate from localStorage for instant render
    const role = localStorage.getItem('userRole');
    const name = localStorage.getItem('userName');
    const email = localStorage.getItem('userEmail');
    if (role && email) return { role, name, email };
    return null;
  });
  const [loading, setLoading] = useState(!user);
  const [error, setError] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      setLoading(false);
      return;
    }

    authApi.me()
      .then((data) => {
        if (data) {
          setUser(data);
          localStorage.setItem('userRole', data.role);
          localStorage.setItem('userName', data.name);
          localStorage.setItem('userEmail', data.email);
        }
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  return { user, loading, error };
}
