import React from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import { useAuth0 } from '@auth0/auth0-react';
import ThemeToggle from './components/shared/ThemeToggle';
import CustomCursor from './components/shared/CustomCursor';
import HeroSection from './components/landing/HeroSection';
import ProblemContext from './components/landing/ProblemContext';
import ComparisonSection from './components/landing/ComparisonSection';
import ObjectivesTabs from './components/landing/ObjectivesTabs';
import ProposedSolution from './components/landing/ProposedSolution';
import Mockups from './components/landing/Mockups';
import ConclusionFooter from './components/landing/ConclusionFooter';
import LoginPage from './components/shared/LoginPage';
import SignupPage from './components/shared/SignupPage';
import MedRepPortal from './components/medrep/MedRepPortal';
import MedRepDashboard from './components/medrep/MedRepDashboard';
import MedRepTraining from './components/medrep/MedRepTraining';
import MedRepAnalyticsPairing from './components/medrep/MedRepAnalyticsPairing';
import MedRepSimulation from './components/medrep/MedRepSimulation';
import PhysicianPortal from './components/physician/PhysicianPortal';
import { LogoutButton, UserProfile } from './components/shared/AuthComponent';
import { ProtectedRoute } from './components/shared/ProtectedRoute';

function LandingPage() {
  const { isAuthenticated, isLoading } = useAuth0();
  const navigate = useNavigate();

  return (
    <>
      <header className="landing-header">
        <div className="brand">
          <div className="brand-mark">ALIA</div>
          <div>
            <div style={{ fontWeight: 700, letterSpacing: '0.02em' }}>ALIA</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>AI Pharma Experience</div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          {!isLoading && (
            isAuthenticated ? (
              <>
                <LogoutButton />
              </>
            ) : (
              <div style={{ display: 'flex', gap: '1rem' }}>
                <button onClick={() => navigate('/login')} className="btn btn-primary">Sign In</button>
                <button onClick={() => navigate('/signup')} className="btn btn-secondary">Sign Up</button>
              </div>
            )
          )}
        </div>
      </header>

      <HeroSection />
      <ProblemContext />
      <ComparisonSection />
      <ObjectivesTabs />
      <ProposedSolution />
      <Mockups />
      <ConclusionFooter />
    </>
  );
}

function ProfilePage() {
  return (
    <div style={{ padding: '2rem' }}>
      <UserProfile />
    </div>
  );
}

function App() {
  return (
    <div className="App relative">
      <CustomCursor />
      <ThemeToggle />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LoginPage onClose={() => window.history.back()} />} />
        <Route path="/signup" element={<SignupPage onClose={() => window.history.back()} />} />
        <Route path="/profile" element={<ProtectedRoute><ProfilePage /></ProtectedRoute>} />
        <Route path="/portal" element={<ProtectedRoute><MedRepPortal /></ProtectedRoute>} />
        <Route path="/rep/dashboard" element={<ProtectedRoute><MedRepDashboard /></ProtectedRoute>} />
        <Route path="/rep/training" element={<ProtectedRoute><MedRepTraining /></ProtectedRoute>} />
        <Route path="/rep/simulation" element={<ProtectedRoute><MedRepSimulation /></ProtectedRoute>} />
        <Route path="/analytics/pairing" element={<ProtectedRoute><MedRepAnalyticsPairing /></ProtectedRoute>} />
        <Route path="/physician/portal" element={<ProtectedRoute><PhysicianPortal /></ProtectedRoute>} />
      </Routes>
    </div>
  );
}

export default App;
