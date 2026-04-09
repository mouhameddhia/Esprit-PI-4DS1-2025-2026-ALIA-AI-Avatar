import { useRef, useEffect, useCallback } from 'react';

function hash(i) {
  const x = Math.sin(i * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

/**
 * Full-bleed canvas layer: dots and thin rings drift toward the cursor (Antigravity-style).
 * Uses window mousemove + bounds check so pointer-events stay none (clicks pass through).
 */
export default function CursorParticleField({ className = '', density = 1 }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const mouseRef = useRef({ x: 0, y: 0, active: false });
  const particlesRef = useRef([]);
  const dimsRef = useRef({ w: 0, h: 0 });
  const rafRef = useRef(0);
  const timeRef = useRef(0);

  const initParticles = useCallback(
    (w, h) => {
      const area = w * h;
      const base = Math.floor((area / 1650) * density);
      const n = Math.max(96, Math.min(560, base));
      const particles = [];
      for (let i = 0; i < n; i++) {
        const hr = hash(i);
        const ring = hr < 0.14;
        const hx = hash(i + 1) * w;
        const hy = hash(i + 2) * h;
        particles.push({
          hx,
          hy,
          x: hx,
          y: hy,
          vx: 0,
          vy: 0,
          r: ring ? 0 : 0.55 + hash(i + 3) * 1.35,
          ring,
          phase: hash(i + 4) * Math.PI * 2,
          hue: hr < 0.52 ? 'a' : 'b',
        });
      }
      particlesRef.current = particles;
      dimsRef.current = { w, h };
      mouseRef.current = { x: w * 0.5, y: h * 0.45, active: false };
    },
    [density],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return undefined;

    const ctx = canvas.getContext('2d');
    if (!ctx) return undefined;

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let cancelled = false;

    const resizeCore = () => {
      const rect = container.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.floor(rect.width));
      const h = Math.max(1, Math.floor(rect.height));
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      initParticles(w, h);
    };

    const drawStatic = () => {
      const { w, h } = dimsRef.current;
      ctx.clearRect(0, 0, w, h);
      particlesRef.current.forEach((p) => {
        if (p.ring) {
          ctx.beginPath();
          ctx.arc(p.hx, p.hy, 3.2, 0, Math.PI * 2);
          ctx.strokeStyle =
            p.hue === 'a' ? 'rgba(96, 165, 250, 0.28)' : 'rgba(167, 139, 250, 0.26)';
          ctx.lineWidth = 1;
          ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.arc(p.hx, p.hy, p.r, 0, Math.PI * 2);
          ctx.fillStyle =
            p.hue === 'a' ? 'rgba(96, 165, 250, 0.38)' : 'rgba(192, 167, 255, 0.34)';
          ctx.fill();
        }
      });
    };

    const onResize = () => {
      resizeCore();
      if (reduced) drawStatic();
    };

    const ro = new ResizeObserver(onResize);
    ro.observe(container);
    onResize();

    if (reduced) {
      return () => {
        ro.disconnect();
      };
    }

    const onMove = (e) => {
      const rect = container.getBoundingClientRect();
      const pad = 24;
      if (
        e.clientX < rect.left - pad ||
        e.clientX > rect.right + pad ||
        e.clientY < rect.top - pad ||
        e.clientY > rect.bottom + pad
      ) {
        mouseRef.current.active = false;
        return;
      }
      mouseRef.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        active: true,
      };
    };

    window.addEventListener('mousemove', onMove, { passive: true });

    const attractBase = 0.00072;
    const spring = 0.031;
    const friction = 0.87;

    const tick = (t) => {
      if (cancelled) return;
      timeRef.current = t * 0.001;
      const { w, h } = dimsRef.current;
      const m = mouseRef.current;
      const tx = m.active ? m.x : w * 0.52;
      const ty = m.active ? m.y : h * 0.42;

      particlesRef.current.forEach((p) => {
        let fx = (p.hx - p.x) * spring;
        let fy = (p.hy - p.y) * spring;
        const dx = tx - p.x;
        const dy = ty - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy) + 72;
        const pull = (attractBase * 72000) / (dist * dist);
        fx += dx * pull;
        fy += dy * pull;
        p.vx = (p.vx + fx) * friction;
        p.vy = (p.vy + fy) * friction;
        p.x += p.vx;
        p.y += p.vy;
      });

      ctx.clearRect(0, 0, w, h);
      const pulse = timeRef.current;
      particlesRef.current.forEach((p) => {
        if (p.ring) {
          const rr = 3.4 + Math.sin(pulse * 1.1 + p.phase) * 0.45;
          ctx.beginPath();
          ctx.arc(p.x, p.y, rr, 0, Math.PI * 2);
          ctx.strokeStyle =
            p.hue === 'a' ? 'rgba(96, 165, 250, 0.32)' : 'rgba(167, 139, 250, 0.3)';
          ctx.lineWidth = 1;
          ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
          ctx.fillStyle =
            p.hue === 'a' ? 'rgba(96, 165, 250, 0.48)' : 'rgba(192, 167, 255, 0.42)';
          ctx.fill();
        }
      });

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      window.removeEventListener('mousemove', onMove);
    };
  }, [initParticles]);

  return (
    <div ref={containerRef} className={`cursor-particle-field ${className}`.trim()} aria-hidden>
      <canvas ref={canvasRef} />
    </div>
  );
}
