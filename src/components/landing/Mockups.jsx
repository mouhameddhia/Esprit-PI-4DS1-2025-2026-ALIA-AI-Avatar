import React, { useState, useRef, useEffect, useCallback, useId } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ChevronLeft, ChevronRight, Video, Database, LayoutDashboard, TrendingUp } from 'lucide-react';

const chartData = [
  { month: 'Jan', score: 65 },
  { month: 'Feb', score: 68 },
  { month: 'Mar', score: 78 },
  { month: 'Apr', score: 85 },
  { month: 'May', score: 92 },
  { month: 'Jun', score: 96 },
];

const VIDEO_SLIDES = [
  {
    title: 'Performance analytics',
    caption: 'Audio-to-video synthesis · left-side view',
    src: new URL(
      '../../assets/67ff9faac266bb379ddc0ea2_6895294eb0f8b15bbe95cbc0_Audio to Video - Left Side U 30 MB-transcode.mp4',
      import.meta.url,
    ).href,
  },
  {
    title: 'Nour in conversation',
    caption: 'Nelly · avatar from audio (preview)',
    src: new URL(
      '../../assets/67ff9faac266bb379ddc0ea2_68952a70df76b4e5734ba5ee_Nelly - Audio to Video U 30 MB-transcode.mp4',
      import.meta.url,
    ).href,
  },
];

const AUTO_ADVANCE_MS = 14000;

const springDeck = { type: 'spring', stiffness: 68, damping: 19, mass: 0.82 };

export default function Mockups() {
  const [activeSlide, setActiveSlide] = useState(0);
  const [isHoveringCarousel, setIsHoveringCarousel] = useState(false);
  const videoRefs = useRef([]);
  const containerRef = useRef(null);
  const gradId = useId().replace(/:/g, '');
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start end', 'end start'],
  });
  const y2 = useTransform(scrollYProgress, [0, 1], [-45, 45]);

  const goPrev = useCallback(() => {
    setActiveSlide((p) => (p > 0 ? p - 1 : VIDEO_SLIDES.length - 1));
  }, []);

  const goNext = useCallback(() => {
    setActiveSlide((p) => (p < VIDEO_SLIDES.length - 1 ? p + 1 : 0));
  }, []);

  useEffect(() => {
    if (isHoveringCarousel) return undefined;
    const id = window.setInterval(goNext, AUTO_ADVANCE_MS);
    return () => window.clearInterval(id);
  }, [isHoveringCarousel, goNext]);

  useEffect(() => {
    videoRefs.current.forEach((el, i) => {
      if (!el) return;
      if (i === activeSlide) {
        el.play().catch(() => {});
      } else {
        el.pause();
      }
    });
  }, [activeSlide]);

  const current = VIDEO_SLIDES[activeSlide];
  const latestScore = chartData[chartData.length - 1].score;
  const delta = latestScore - chartData[0].score;

  return (
    <section ref={containerRef} id="product" className="landing-section container mockups-section">
      <motion.div
        className="landing-section-head mockups-section-head"
        initial={{ opacity: 0, y: 18 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.55 }}
      >
        <p className="landing-kicker">Product vision</p>
        <h2>What ALIA feels like</h2>
        <p>
          ALIA gives reps structured practice with an AI physician: every interaction is captured as analytics your
          team can review—so coaching stays grounded in real behaviour, not guesswork.
        </p>
      </motion.div>

      <div className="mockups-product-grid">
        <motion.div
          className="mockup-rep-card"
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.65 }}
        >
          <div className="mockup-rep-card__accent" aria-hidden />
          <div className="mockup-rep-card__top">
            <span className="mockup-rep-card__eyebrow">Supervised sessions</span>
            <TrendingUp className="mockup-rep-card__trend-icon" size={20} strokeWidth={2} aria-hidden />
          </div>
          <h3 className="mockup-rep-card__title">Competency trajectory</h3>
          <p className="mockup-rep-card__lede">
            Rolling readiness score from visits debriefed in the portal—so managers see momentum, not a single snapshot.
          </p>
          <div className="mockup-rep-card__hero">
            <span className="mockup-rep-card__score">{latestScore}</span>
            <span className="mockup-rep-card__score-max">/ 100</span>
            <span className="mockup-rep-card__delta">+{delta} vs Jan</span>
          </div>
          <div className="mockup-rep-card__chart">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
                <defs>
                  <linearGradient id={`mockup-rep-grad-${gradId}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#a78bfa" stopOpacity={0.45} />
                    <stop offset="55%" stopColor="#7c3aed" stopOpacity={0.12} />
                    <stop offset="100%" stopColor="#7c3aed" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="month"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                  dy={6}
                />
                <YAxis
                  domain={[50, 100]}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                  width={36}
                />
                <Tooltip
                  cursor={{ stroke: 'color-mix(in srgb, #7c3aed 40%, transparent)', strokeWidth: 1 }}
                  contentStyle={{
                    backgroundColor: 'var(--bg-color-light)',
                    border: '1px solid var(--glass-border)',
                    borderRadius: 12,
                    color: 'var(--text-primary)',
                    fontSize: 13,
                  }}
                  formatter={(v) => [`${v}`, 'Score']}
                />
                <Area
                  type="monotone"
                  dataKey="score"
                  stroke="#c4b5fd"
                  strokeWidth={2.5}
                  fill={`url(#mockup-rep-grad-${gradId})`}
                  dot={false}
                  activeDot={{ r: 5, fill: '#7c3aed', stroke: '#fff', strokeWidth: 2 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <ul className="mockup-rep-card__chips">
            <li>Visit transcripts &amp; rubrics</li>
            <li>Manager annotations</li>
            <li>Export for QBRs</li>
          </ul>
        </motion.div>

        <div className="mockups-stack">
          <motion.div
            style={{ y: y2 }}
            className="mockups-layer-back"
            initial={{ opacity: 0, x: 40 }}
            whileInView={{ opacity: 0.45, x: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.15, duration: 0.55 }}
          >
            <div className="mockups-layer-back__header">
              <Database size={16} aria-hidden />
              <span>Knowledge &amp; logs</span>
            </div>
            <div className="mockups-db-lines">
              <div className="mockups-db-line" style={{ width: '100%' }} />
              <div className="mockups-db-line" />
              <div className="mockups-db-line" />
            </div>
          </motion.div>

          <div
            className="mockups-card-deck"
            onMouseEnter={() => setIsHoveringCarousel(true)}
            onMouseLeave={() => setIsHoveringCarousel(false)}
          >
            <div className="mockups-deck-viewport">
              {VIDEO_SLIDES.map((slide, i) => {
                const isFront = i === activeSlide;
                return (
                  <motion.div
                    key={slide.src}
                    className={`mockups-rotating-card ${isFront ? '' : 'mockups-rotating-card--back'}`}
                    initial={false}
                    animate={
                      isFront
                        ? {
                            zIndex: 3,
                            scale: 1,
                            x: 0,
                            y: 0,
                            rotateY: 0,
                            rotateX: 0,
                            z: 32,
                            opacity: 1,
                            filter: 'brightness(1)',
                          }
                        : {
                            zIndex: 0,
                            scale: 0.86,
                            x: 48,
                            y: 24,
                            rotateY: -48,
                            rotateX: 5,
                            z: -24,
                            opacity: 0,
                            filter: 'brightness(0.9)',
                          }
                    }
                    transition={{
                      ...springDeck,
                      opacity: isFront
                        ? { duration: 0.26, delay: 0.08, ease: [0.22, 1, 0.36, 1] }
                        : { duration: 0.09, ease: [0.4, 0, 1, 1] },
                    }}
                    style={{
                      transformOrigin: '50% 50%',
                      pointerEvents: isFront ? 'auto' : 'none',
                    }}
                    aria-hidden={!isFront}
                  >
                    <div className="mockups-rotating-card__toolbar">
                      <div className="mockups-rotating-card__toolbar-left">
                        <LayoutDashboard size={18} color="#7c3aed" aria-hidden />
                        <h4>{slide.title}</h4>
                      </div>
                      <Video size={18} color="var(--text-secondary)" aria-hidden />
                    </div>

                    <div className="mockups-carousel-body">
                      <div className="mockups-carousel-stage">
                        <video
                          ref={(el) => {
                            videoRefs.current[i] = el;
                          }}
                          src={slide.src}
                          controls={isFront}
                          controlsList={isFront ? undefined : 'nodownload'}
                          playsInline
                          muted
                          loop
                          preload="metadata"
                          aria-label={slide.caption}
                          tabIndex={isFront ? 0 : -1}
                        />
                      </div>
                    </div>
                  </motion.div>
                );
              })}

              <div className="mockups-deck-nav-layer">
                <button type="button" className="mockups-nav-btn mockups-nav-btn--prev" onClick={goPrev} aria-label="Previous video">
                  <ChevronLeft size={18} />
                </button>
                <button type="button" className="mockups-nav-btn mockups-nav-btn--next" onClick={goNext} aria-label="Next video">
                  <ChevronRight size={18} />
                </button>
              </div>
            </div>

            <div className="mockups-carousel-dots" role="tablist" aria-label="Choose preview clip">
              {VIDEO_SLIDES.map((_, i) => (
                <button
                  key={i}
                  type="button"
                  role="tab"
                  aria-selected={i === activeSlide}
                  className={`mockups-carousel-dot ${i === activeSlide ? 'is-active' : ''}`}
                  onClick={() => setActiveSlide(i)}
                  aria-label={`Show clip ${i + 1}`}
                />
              ))}
            </div>
            <p className="mockups-carousel-caption">{current.caption}</p>
          </div>
        </div>
      </div>
    </section>
  );
}
