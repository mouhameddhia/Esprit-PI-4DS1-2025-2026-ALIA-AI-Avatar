import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, UserPlus, X, Shield, Eye, EyeOff } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth0 } from '@auth0/auth0-react';
import AntigravitySwarm from './AntigravitySwarm';
import './LoginPage.css';

async function getIdTokenClaimsWithRetry(getIdTokenClaims, attempts = 5, delayMs = 250) {
  let lastClaims = null;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      lastClaims = await getIdTokenClaims();
      if (lastClaims?.__raw) {
        return lastClaims;
      }
    } catch {
      // retry below
    }
    if (attempt < attempts - 1) {
      await new Promise((resolve) => window.setTimeout(resolve, delayMs));
    }
  }
  return lastClaims;
}

async function waitForAuthState(isAuthenticated, attempts = 20, delayMs = 250) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (isAuthenticated()) {
      return true;
    }
    if (attempt < attempts - 1) {
      await new Promise((resolve) => window.setTimeout(resolve, delayMs));
    }
  }
  return false;
}

const PENDING_AUTH0_PROFILE_KEY = 'alia_pending_auth0_profile';

const SignupPage = ({ onClose }) => {
  const [role, setRole] = useState('Medical Rep');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isTraditionalLoading, setIsTraditionalLoading] = useState(false);
  const navigate = useNavigate();
  const { loginWithRedirect, getIdTokenClaims, getAccessTokenSilently, user, isLoading, error, isAuthenticated } = useAuth0();
  const [isAuthLoading, setIsAuthLoading] = useState(false);
  const authFlowStartedRef = useRef(false);

  const syncAuth0Profile = async (syncRole, syncName) => {
    await waitForAuthState(() => isAuthenticated);
    const tokenClaims = await getIdTokenClaimsWithRetry(getIdTokenClaims);
    let authToken = tokenClaims?.__raw;
    if (!authToken) {
      try {
        authToken = await getAccessTokenSilently();
      } catch {
        // handled by fallback below
      }
    }
    if (!authToken) {
      throw new Error('Unable to retrieve Auth0 token');
    }

    const response = await fetch('http://localhost:8000/auth/auth0-sync', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${authToken}`,
      },
      body: JSON.stringify({
        role: syncRole.toLowerCase().replace(' ', ''),
        name: syncName || user?.name || '',
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Unable to sync Auth0 profile');
    }

    return await response.json();
  };

  useEffect(() => {
    if (!isAuthenticated || isLoading || authFlowStartedRef.current) {
      return;
    }

    const pendingRaw = sessionStorage.getItem(PENDING_AUTH0_PROFILE_KEY);
    if (!pendingRaw) {
      return;
    }

    authFlowStartedRef.current = true;
    (async () => {
      try {
        const pending = JSON.parse(pendingRaw);
        const syncResult = await syncAuth0Profile(pending.role, pending.name || '');
        if (syncResult?.access_token) {
          localStorage.setItem('token', syncResult.access_token);
        }
        sessionStorage.removeItem(PENDING_AUTH0_PROFILE_KEY);
        if (pending.role === 'Medical Rep') {
          navigate('/portal');
        } else if (pending.role === 'Physician') {
          navigate('/physician/portal');
        }
      } catch (err) {
        console.error('Auth0 redirect signup sync failed:', err);
        alert(err.message || 'Auth0 signup failed');
      } finally {
        authFlowStartedRef.current = false;
        setIsAuthLoading(false);
      }
    })();
  }, [isAuthenticated, isLoading, navigate, syncAuth0Profile]);

  const handleAuth0Signup = async () => {
    setIsAuthLoading(true);
    try {
      sessionStorage.setItem(
        PENDING_AUTH0_PROFILE_KEY,
        JSON.stringify({ role, name }),
      );
      await loginWithRedirect({
        authorizationParams: { screen_hint: 'signup' },
        appState: { returnTo: window.location.pathname },
      });
    } catch (err) {
      console.error('Auth0 signup error:', err);
      alert(err.message || 'Auth0 signup failed');
      sessionStorage.removeItem(PENDING_AUTH0_PROFILE_KEY);
      setIsAuthLoading(false);
    } finally {
      // keep loading state while redirect is in progress
    }
  };

  const handleTraditionalSignup = async (e) => {
    e.preventDefault();
    setIsTraditionalLoading(true);
    try {
      const response = await fetch('http://localhost:8000/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: email,
          password: password,
          name: name,
          role: role.toLowerCase().replace(' ', ''),
        }),
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('token', data.access_token);
        // Navigate based on role
        if (role === 'Medical Rep') {
          navigate('/portal');
        } else if (role === 'Physician') {
          navigate('/physician/portal');
        }
      } else {
        const errorData = await response.json();
        alert(errorData.detail || 'Signup failed');
      }
    } catch (err) {
      console.error('Traditional signup error:', err);
      alert('Signup failed');
    } finally {
      setIsTraditionalLoading(false);
    }
  };

  return (
    <AnimatePresence>
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="login-page-container"
      >
        <AntigravitySwarm />
        <button className="login-close-btn" onClick={onClose} aria-label="Close">
          <X size={24} />
        </button>

        <div className="login-left-panel">
          <div className="login-left-content">
            <motion.div 
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.1 }}
              className="login-logo-container"
            >
              <div className="login-logo-icon">
                <Brain size={36} />
              </div>
              <div className="login-logo-text">
                <h1>ALIA</h1>
                <span>AI Avatar Platform</span>
              </div>
            </motion.div>

            <motion.h2 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="login-title"
            >
              Welcome to the Future of<br />
              <span>Pharmaceutical Training</span>
            </motion.h2>
            
            <motion.p 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="login-subtitle"
            >
              Interact with ALIA, your intelligent AI representative, for training simulations and instant product knowledge.
            </motion.p>

            <motion.div 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="login-badges"
            >
              <span className="login-badge">
                <span className="login-badge-dot" style={{ backgroundColor: '#7c3aed' }}></span> Interactive AI Training
              </span>
              <span className="login-badge">
                <span className="login-badge-dot" style={{ backgroundColor: '#10b981' }}></span> Real-time Analytics
              </span>
              <span className="login-badge">
                <span className="login-badge-dot" style={{ backgroundColor: '#ea580c' }}></span> Smart Pairing System
              </span>
            </motion.div>
          </div>
        </div>

        <div className="login-right-panel">
           <motion.div 
             initial={{ opacity: 0, scale: 0.95 }}
             animate={{ opacity: 1, scale: 1 }}
             transition={{ delay: 0.2, type: 'spring', stiffness: 100 }}
             className="login-card"
           >
            <h2 className="login-card-title">Sign Up</h2>

            {error && !isPopupClosedError(error) && (
              <div style={{ color: 'red', marginBottom: '1rem' }}>Error: {error.message}</div>
            )}

            <form onSubmit={handleTraditionalSignup}>
              <div className="login-form-group">
                <label className="login-form-label">I am a:</label>
                <div className="login-role-toggles">
                  <button
                    type="button"
                    onClick={() => setRole('Medical Rep')}
                    className={`login-role-btn ${role === 'Medical Rep' ? 'active' : ''}`}
                  >
                    Medical Rep
                  </button>
                  <button
                    type="button"
                    onClick={() => setRole('Physician')}
                    className={`login-role-btn ${role === 'Physician' ? 'active' : ''}`}
                  >
                    Physician
                  </button>
                </div>
              </div>

              <div className="login-form-group">
                <label className="login-form-label">Full Name</label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="login-input"
                  required
                />
              </div>

              <div className="login-form-group">
                <label className="login-form-label">Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="login-input"
                  required
                />
              </div>

              <div className="login-form-group">
                <label className="login-form-label">Password</label>
                <div className="login-password-input-container">
                  <input
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="login-input"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="login-password-toggle"
                    aria-label={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                  </button>
                </div>
              </div>

              <button 
                type="submit" 
                className="login-submit-btn"
                disabled={isTraditionalLoading}
              >
                <UserPlus size={18} /> {isTraditionalLoading ? 'Creating Account...' : 'Sign Up'}
              </button>
            </form>

            <div className="login-divider">
              <span>or</span>
            </div>

            <button 
              onClick={handleAuth0Signup}
              className="login-auth0-btn"
              disabled={isAuthLoading || isLoading}
            >
              <Shield size={18} /> {isAuthLoading ? 'Creating Account...' : 'Continue with Auth0'}
            </button>

            <p className="login-footer-text">
              Already have an account? <Link to="/login">Sign In</Link>
            </p>
          </motion.div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default SignupPage;
