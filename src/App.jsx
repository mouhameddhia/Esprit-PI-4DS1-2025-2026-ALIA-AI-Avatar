import React from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';
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
import { LogIn } from 'lucide-react';

function LandingPage() {
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

        <button 
          onClick={() => navigate('/login')}
          className="btn btn-primary"
        >
          <LogIn size={18} /> Sign Up / Login
        </button>
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

function App() {
  const navigate = useNavigate();

  return (
    <div className="App relative">
      <CustomCursor />
      <ThemeToggle />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LoginPage onClose={() => navigate('/')} />} />
        <Route path="/signup" element={<SignupPage onClose={() => navigate('/')} />} />
        <Route path="/portal" element={<MedRepPortal />} />
        <Route path="/rep/dashboard" element={<MedRepDashboard />} />
        <Route path="/rep/training" element={<MedRepTraining />} />
        <Route path="/rep/simulation" element={<MedRepSimulation />} />
        <Route path="/analytics/pairing" element={<MedRepAnalyticsPairing />} />
        <Route path="/physician/portal" element={<PhysicianPortal />} />
      </Routes>
    </div>
  );
}

export default App;
