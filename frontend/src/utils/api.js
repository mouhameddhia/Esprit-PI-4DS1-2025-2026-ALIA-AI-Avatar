const BASE_URL = 'http://localhost:8000';

function getToken() {
  return localStorage.getItem('token');
}

async function request(path, options = {}) {
  const token = getToken();
  const headers = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers || {}),
  };

  const res = await fetch(`${BASE_URL}${path}`, { ...options, headers });

  if (res.status === 401) {
    localStorage.removeItem('token');
    localStorage.removeItem('userRole');
    localStorage.removeItem('userName');
    localStorage.removeItem('userEmail');
    window.location.href = '/login';
    return;
  }

  if (res.status === 204) return null;

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `Request failed: ${res.status}`);
  }
  return data;
}

// ─── Auth ─────────────────────────────────────────────────────────────────────
export const authApi = {
  me: () => request('/auth/me'),
};

// ─── Products ─────────────────────────────────────────────────────────────────
export const productsApi = {
  list: (params = {}) => {
    const qs = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== '')
    ).toString();
    return request(`/products${qs ? `?${qs}` : ''}`);
  },
  create: (body) => request('/products', { method: 'POST', body: JSON.stringify(body) }),
  update: (id, body) => request(`/products/${id}`, { method: 'PUT', body: JSON.stringify(body) }),
  delete: (id) => request(`/products/${id}`, { method: 'DELETE' }),
};

// ─── Users ────────────────────────────────────────────────────────────────────
export const usersApi = {
  list: (params = {}) => {
    const qs = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== '')
    ).toString();
    return request(`/users${qs ? `?${qs}` : ''}`);
  },
  updateRole: (id, role) => request(`/users/${id}`, { method: 'PUT', body: JSON.stringify({ role }) }),
  delete: (id) => request(`/users/${id}`, { method: 'DELETE' }),
};

// ─── Alerts ───────────────────────────────────────────────────────────────────
export const alertsApi = {
  list: (params = {}) => {
    const qs = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== '')
    ).toString();
    return request(`/alerts${qs ? `?${qs}` : ''}`);
  },
  resolve: (id) => request(`/alerts/${id}/resolve`, { method: 'PATCH' }),
  delete: (id) => request(`/alerts/${id}`, { method: 'DELETE' }),
};
