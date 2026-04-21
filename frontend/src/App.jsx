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
import SectionPage from './components/landing/SectionPage';
import ProblemContext from './components/landing/ProblemContext';
import ComparisonSection from './components/landing/ComparisonSection';
import ObjectivesTabs from './components/landing/ObjectivesTabs';
import ProposedSolution from './components/landing/ProposedSolution';
import Mockups from './components/landing/Mockups';
import StatsFlipSection from './components/landing/StatsFlipSection';
import TestimonialsSection from './components/landing/TestimonialsSection';
import FaqSection from './components/landing/FaqSection';
import ConclusionFooter from './components/landing/ConclusionFooter';

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
        {/* Landing page — all sections together */}
        <Route path="/" element={<LandingPage />} />

        {/* Individual section routes */}
        <Route path="/why-alia" element={
          <SectionPage sectionId="why-alia"><ProblemContext /></SectionPage>
        } />
        <Route path="/compare" element={
          <SectionPage sectionId="compare"><ComparisonSection /></SectionPage>
        } />
        <Route path="/modes" element={
          <SectionPage sectionId="modes"><ObjectivesTabs /></SectionPage>
        } />
        <Route path="/platform" element={
          <SectionPage sectionId="platform"><ProposedSolution /></SectionPage>
        } />
        <Route path="/product" element={
          <SectionPage sectionId="product"><Mockups /></SectionPage>
        } />
        <Route path="/impact" element={
          <SectionPage sectionId="impact"><StatsFlipSection /></SectionPage>
        } />
        <Route path="/testimonials" element={
          <SectionPage sectionId="testimonials"><TestimonialsSection /></SectionPage>
        } />
        <Route path="/faq" element={
          <SectionPage sectionId="faq"><FaqSection /></SectionPage>
        } />
        <Route path="/get-started" element={
          <SectionPage sectionId="get-started"><ConclusionFooter /></SectionPage>
        } />

        {/* Auth routes */}
        <Route path="/login" element={<LoginPage onClose={() => window.history.back()} />} />
        <Route path="/signup" element={<SignupPage onClose={() => window.history.back()} />} />

        {/* Protected routes */}
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
