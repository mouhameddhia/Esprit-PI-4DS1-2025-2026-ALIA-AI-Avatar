import React, { useRef } from 'react';
import { motion, useMotionValue, useTransform } from 'framer-motion';
import { ArrowRight, Play } from 'lucide-react';
import { Canvas, useFrame } from '@react-three/fiber';
import { MeshDistortMaterial, Sphere, Environment, Float } from '@react-three/drei';

const AiBlob = () => {
  const meshRef = useRef();
  
  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    if (meshRef.current) {
      meshRef.current.rotation.x = t * 0.1;
      meshRef.current.rotation.y = t * 0.2;
      // Slight 3D parallax reaction to mouse
      meshRef.current.position.x = (state.mouse.x * 0.8) + Math.sin(t * 0.5) * 0.2;
      meshRef.current.position.y = (state.mouse.y * 0.8) + Math.cos(t * 0.5) * 0.2;
    }
  });

  return (
    <Float speed={2} rotationIntensity={1} floatIntensity={2}>
      <Sphere ref={meshRef} args={[1, 128, 128]} scale={2.2}>
        <MeshDistortMaterial 
          color="#a855f7" 
          attach="material" 
          distort={0.4} 
          speed={1.5} 
          roughness={0.1}
          metalness={0.8}
        />
      </Sphere>
    </Float>
  );
};

const HeroSection = () => {
  // Title Parallax Tracking
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const handleMouseMove = (event) => {
    const bounds = window.innerWidth / 2;
    const boundsY = window.innerHeight / 2;
    x.set((event.clientX - bounds) / 20);
    y.set((event.clientY - boundsY) / 20);
  };

  return (
    <section 
      onMouseMove={handleMouseMove}
      className="hero-section"
    >
      <div className="hero-bg-shape"></div>
      <div className="hero-grid">
        <motion.div 
          style={{ x, y }}
          initial={{ opacity: 0, scale: 0.94 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, ease: 'easeOut' }}
          className="hero-copy"
        >
          <div className="hero-badge">
            <span className="hero-badge-dot"></span>
            ALIA 2.0 Web Engine Online
          </div>

          <h1 className="hero-title">
            The AI Avatar for <br />
            <span className="text-gradient">Medical & Pharma</span> Excellence
          </h1>

          <p className="hero-lead">
            Empower your team with a highly realistic, responsive AI avatar built to transform training, compliance, and physician engagement into measurable outcomes.
          </p>

          <div className="hero-cta">
            <button className="btn btn-primary">
              Deploy AI <ArrowRight size={18} />
            </button>
            <button className="btn btn-secondary">
              <Play size={18} /> Interact Now
            </button>
          </div>

          <div className="hero-metrics">
            <div className="hero-metric">
              <strong>98%</strong>
              <span>Faster rep onboarding</span>
            </div>
            <div className="hero-metric">
              <strong>24/7</strong>
              <span>AI-led physician support</span>
            </div>
            <div className="hero-metric">
              <strong>100%</strong>
              <span>Compliance assurance</span>
            </div>
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.5, delay: 0.25 }}
          className="hero-visual"
        >
          <div className="hero-visual-glow"></div>
          <div className="hero-visual-card">
            <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
              <ambientLight intensity={0.55} />
              <directionalLight position={[10, 10, 5]} intensity={1.6} color="#ffffff" />
              <directionalLight position={[-8, -8, -4]} intensity={0.65} color="#2dd4bf" />
              <AiBlob />
              <Environment preset="city" />
            </Canvas>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default HeroSection;
