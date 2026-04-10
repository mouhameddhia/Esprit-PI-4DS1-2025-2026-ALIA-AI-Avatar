import React, { useEffect, useRef } from 'react';

const AntigravitySwarm = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    // Google colors + ALIA purple
    const colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#7C3AED', '#A78BFA'];

    const numParticles = 0; // Reduced amount for a lighter, cleaner swarm
    const particles = [];

    // Mouse tracking
    let targetX = width / 2;
    let targetY = height / 2;

    const handleMouseMove = (e) => {
      targetX = e.clientX;
      targetY = e.clientY;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('resize', () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    });

    for (let i = 0; i < numParticles; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 4,
        vy: (Math.random() - 0.5) * 4,
        width: Math.random() * 2 + 1.5,
        color: colors[Math.floor(Math.random() * colors.length)],
        // Random attributes to give each "bee/bird" a unique flight pattern
        wanderTheta: Math.random() * Math.PI * 2,
        wanderSpeed: Math.random() * 0.1 + 0.05,
        attractionFactor: Math.random() * 0.003 + 0.001,
        maxSpeed: Math.random() * 4 + 4,
      });
    }

    let animationFrameId;

    const animate = () => {
      // Create a slight trailing effect to make the movement smoother
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)'; // Assumes light mode by default, wait we should just clear if we want clean streaks
      // Actually clearing looks sharper with the drawn lines
      ctx.clearRect(0, 0, width, height);

      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        // 1. Mouse Attraction (Flock towards cursor)
        const dx = targetX - p.x;
        const dy = targetY - p.y;

        p.vx += dx * p.attractionFactor;
        p.vy += dy * p.attractionFactor;

        // 2. Wandering Behavior (Organic bee/bird jitter)
        p.wanderTheta += (Math.random() - 0.5) * p.wanderSpeed;
        p.vx += Math.cos(p.wanderTheta) * 0.5;
        p.vy += Math.sin(p.wanderTheta) * 0.5;

        // 3. Separation (Avoid grouping exactly in one single point)
        // Simplified fast separation rule against center of mass roughly
        const distToMouse = Math.sqrt(dx * dx + dy * dy);
        if (distToMouse < 40) {
          // Push outwards if too close to exact cursor point to create a buzzing cloud
          p.vx -= (dx / distToMouse) * 0.5;
          p.vy -= (dy / distToMouse) * 0.5;
        }

        // 4. Limit Speed
        const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
        if (speed > p.maxSpeed) {
          p.vx = (p.vx / speed) * p.maxSpeed;
          p.vy = (p.vy / speed) * p.maxSpeed;
        }

        // Apply velocity
        p.x += p.vx;
        p.y += p.vy;

        // Draw as a streak based on its velocity to match the "dash" aesthetic
        // The faster it moves, the longer the streak
        const trailX = p.x - p.vx * 3;
        const trailY = p.y - p.vy * 3;

        ctx.beginPath();
        ctx.moveTo(trailX, trailY);
        ctx.lineTo(p.x, p.y);

        ctx.strokeStyle = p.color;
        ctx.lineWidth = p.width;
        ctx.lineCap = 'round';

        // Optional: slight glow
        ctx.shadowBlur = isDark ? 8 : 4;
        ctx.shadowColor = p.color;

        ctx.stroke();

        // Reset shadow for performance
        ctx.shadowBlur = 0;
      }

      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 0,
      }}
    />
  );
};

export default AntigravitySwarm;
