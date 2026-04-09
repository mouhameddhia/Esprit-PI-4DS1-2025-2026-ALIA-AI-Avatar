import React from 'react';
import { Routes, Route } from 'react-router-dom';
import ThemeToggle from './components/shared/ThemeToggle';
import CustomCursor from './components/shared/CustomCursor';
import LandingPage from './components/landing/LandingPage';
import LoginPage from './components/shared/LoginPage';
import SignupPage from './components/shared/SignupPage';
import MedRepPortal from './components/medrep/MedRepPortal';
import MedRepDashboard from './components/medrep/MedRepDashboard';
import MedRepTraining from './components/medrep/MedRepTraining';
import MedRepAnalyticsPairing from './components/medrep/MedRepAnalyticsPairing';
import MedRepSimulation from './components/medrep/MedRepSimulation';
import PhysicianPortal from './components/physician/PhysicianPortal';
import { UserProfile } from './components/shared/AuthComponent';
import { ProtectedRoute } from './components/shared/ProtectedRoute';

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
