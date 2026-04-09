import React, { useState, useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { ChevronLeft, ChevronRight, BarChart2, Video, Database, LayoutDashboard } from 'lucide-react';

const data = [
  { month: 'Jan', score: 65 },
  { month: 'Feb', score: 68 },
  { month: 'Mar', score: 78 },
  { month: 'Apr', score: 85 },
  { month: 'May', score: 92 },
  { month: 'Jun', score: 96 },
];

const Mockups = () => {
  const [activeSlide, setActiveSlide] = useState(0);
  const slides = ["Rep Training Dashboard", "Performance Analytics", "Doctor Presentation View"];

  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"]
  });

  const y1 = useTransform(scrollYProgress, [0, 1], [80, -80]);
  const y2 = useTransform(scrollYProgress, [0, 1], [-50, 50]);
  const y3 = useTransform(scrollYProgress, [0, 1], [0, -120]);

  return (
    <section ref={containerRef} className="container relative" style={{ padding: '8rem 2rem' }}>
      <div style={{ textAlign: 'center', marginBottom: '6rem' }}>
        <h2>System Mockups & Analytics</h2>
        <p style={{ color: 'var(--text-secondary)' }}>Visualizing the ALIA experience</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '4rem', alignItems: 'center' }}>
        {/* Interactive Chart with Scroll Parallax */}
        <motion.div 
          style={{ y: y1 }}
          className="glass-panel"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1 }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '2rem' }}>
            <BarChart2 color="var(--soft-blue)" />
            <h3>Rep Progression Tracking</h3>
          </div>
          <div style={{ height: '300px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="month" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                  itemStyle={{ color: '#38bdf8' }}
                />
                <Line type="monotone" dataKey="score" stroke="#7c3aed" strokeWidth={4} dot={{ r: 6, fill: '#7c3aed' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Layered Floating UI Mockup */}
        <div style={{ position: 'relative', height: '450px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          
          {/* Back Layer Floating */}
          <motion.div
            style={{ y: y2 }}
            className="absolute right-0 top-0 w-3/4 h-64 glass-panel opacity-50"
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 0.5, x: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
          >
             <div className="flex items-center gap-2 mb-4 border-b border-gray-700 pb-2"><Database size={16}/> Database Engine</div>
             <div className="space-y-3">
               <div className="h-4 bg-gray-700/50 rounded w-full"></div>
               <div className="h-4 bg-gray-700/50 rounded w-5/6"></div>
               <div className="h-4 bg-gray-700/50 rounded w-4/6"></div>
             </div>
          </motion.div>

          {/* Front Layer App Window */}
          <motion.div 
            style={{ y: y3 }}
            className="relative z-10 w-full glass-panel shadow-2xl backdrop-blur-2xl border border-purple-500/30"
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.4, type: 'spring' }}
          >
             {/* Header */}
             <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--glass-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
               <div className="flex items-center gap-2">
                 <LayoutDashboard size={18} color="#7c3aed" />
                 <h4 style={{ color: 'white', margin: 0 }}>{slides[activeSlide]}</h4>
               </div>
               <Video size={18} color="var(--text-secondary)" />
             </div>
             
             {/* Interactive Screen Content */}
             <div style={{ height: '250px', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden' }}>
                <div style={{ width: '85%', height: '80%', background: 'rgba(0,0,0,0.4)', borderRadius: '12px', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ width: '40%', height: '16px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px' }}></div>
                  <div style={{ width: '100%', height: '80px', background: 'linear-gradient(90deg, rgba(124, 58, 237, 0.2), rgba(45, 212, 191, 0.2))', borderRadius: '8px' }}></div>
                  <div style={{ display: 'flex', gap: '1rem' }}>
                     <div style={{ flex: 1, height: '40px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px' }}></div>
                     <div style={{ flex: 1, height: '40px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px' }}></div>
                  </div>
                </div>
                
                {/* Carousel controls */}
                <button 
                  onClick={() => setActiveSlide((prev) => (prev > 0 ? prev - 1 : slides.length - 1))}
                  style={{ position: 'absolute', left: '1rem', background: '#7c3aed', border: 'none', color: 'white', padding: '0.5rem', borderRadius: '50%', cursor: 'none', zIndex: 20 }}
                  className="hover:scale-110 transition-transform"
                >
                  <ChevronLeft />
                </button>
                <button 
                  onClick={() => setActiveSlide((prev) => (prev < slides.length - 1 ? prev + 1 : 0))}
                  style={{ position: 'absolute', right: '1rem', background: '#7c3aed', border: 'none', color: 'white', padding: '0.5rem', borderRadius: '50%', cursor: 'none', zIndex: 20 }}
                  className="hover:scale-110 transition-transform"
                >
                  <ChevronRight />
                </button>
             </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default Mockups;
