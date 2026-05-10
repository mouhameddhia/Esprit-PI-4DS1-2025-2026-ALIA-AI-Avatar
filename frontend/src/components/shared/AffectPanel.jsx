import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity, AlertTriangle, Zap, CheckCircle2,
  Brain, Mic, FlaskConical, ChevronDown, ChevronUp,
} from 'lucide-react';
import './AffectPanel.css';

const CONF_META = {
  low:    { color: '#ef4444', label: 'Low' },
  medium: { color: '#f59e0b', label: 'Medium' },
  high:   { color: '#10b981', label: 'High' },
};

const URG_META = {
  routine:  { color: '#64748b', bg: 'rgba(100,116,139,0.12)', label: 'Routine' },
  elevated: { color: '#f59e0b', bg: 'rgba(245,158,11,0.12)',  label: 'Elevated' },
  urgent:   { color: '#ef4444', bg: 'rgba(239,68,68,0.12)',   label: '⚡ Urgent' },
};

const EMOTION_META = {
  neu: { color: '#64748b', bg: 'rgba(100,116,139,0.12)', emoji: '😐', label: 'Neutral' },
  hap: { color: '#10b981', bg: 'rgba(16,185,129,0.12)',  emoji: '😊', label: 'Happy'   },
  sad: { color: '#3b82f6', bg: 'rgba(59,130,246,0.12)',  emoji: '😔', label: 'Sad'     },
  ang: { color: '#ef4444', bg: 'rgba(239,68,68,0.12)',   emoji: '😤', label: 'Angry'   },
};

// ─── LIME / SHAP token bar ────────────────────────────────────────────────────

function TokenBar({ token, importance, maxAbs }) {
  const pct    = maxAbs > 0 ? Math.abs(importance) / maxAbs : 0;
  const isPos  = importance >= 0;
  const color  = isPos ? '#10b981' : '#ef4444';
  const width  = `${Math.round(pct * 100)}%`;
  return (
    <div className="xai-token-row">
      <span className="xai-token-label">{token}</span>
      <div className="xai-bar-track">
        <motion.div
          className="xai-bar-fill"
          style={{ width, background: color, marginLeft: isPos ? 0 : 'auto' }}
          initial={{ width: 0 }}
          animate={{ width }}
          transition={{ duration: 0.4 }}
        />
      </div>
      <span className="xai-token-score" style={{ color }}>
        {importance > 0 ? '+' : ''}{importance.toFixed(3)}
      </span>
    </div>
  );
}

// ─── Audio segment bar ────────────────────────────────────────────────────────

function SegmentBar({ feature, importance, maxAbs }) {
  const pct   = maxAbs > 0 ? Math.abs(importance) / maxAbs : 0;
  const isPos = importance >= 0;
  const color = isPos ? '#6366f1' : '#94a3b8';
  const width = `${Math.round(pct * 100)}%`;
  return (
    <div className="xai-token-row">
      <span className="xai-token-label xai-seg-label">{feature}</span>
      <div className="xai-bar-track">
        <motion.div
          className="xai-bar-fill"
          style={{ width, background: color }}
          initial={{ width: 0 }}
          animate={{ width }}
          transition={{ duration: 0.4 }}
        />
      </div>
      <span className="xai-token-score" style={{ color }}>
        {importance > 0 ? '+' : ''}{importance.toFixed(3)}
      </span>
    </div>
  );
}

// ─── Explanation section (shared) ─────────────────────────────────────────────

function ExplainSection({ title, tokens, isSegments = false }) {
  if (!tokens || tokens.length === 0) return null;
  const maxAbs = Math.max(...tokens.map(t => Math.abs(t.importance)), 0.001);
  return (
    <div className="xai-section">
      <div className="xai-section-title">{title}</div>
      {tokens.map((t, i) =>
        isSegments
          ? <SegmentBar key={i} feature={t.feature ?? t.token} importance={t.importance} maxAbs={maxAbs} />
          : <TokenBar   key={i} token={t.token}                 importance={t.importance} maxAbs={maxAbs} />
      )}
    </div>
  );
}

// ─── Explanation panel (collapsible) ─────────────────────────────────────────

function ExplainPanel({ explanation }) {
  const [open, setOpen] = useState(false);
  if (!explanation) return null;

  const shap = explanation.shap;
  const lime = explanation.lime;

  const hasShap = shap && shap.dimensions;
  const hasLime = lime && lime.dimensions;
  if (!hasShap && !hasLime) return null;

  return (
    <div className="xai-panel">
      <button className="xai-toggle" onClick={() => setOpen(o => !o)}>
        <FlaskConical size={12} />
        <span>LIME / SHAP Explanation</span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            className="xai-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <p className="xai-legend">
              <span style={{ color: '#10b981' }}>■ pushes signal up</span>
              {'  '}
              <span style={{ color: '#ef4444' }}>■ pushes signal down</span>
            </p>

            {/* SHAP dimensions */}
            {hasShap && Object.entries(shap.dimensions).map(([dim, data]) =>
              data.tokens && (
                <ExplainSection
                  key={`shap-${dim}`}
                  title={`SHAP — ${dim.replace('_', ' ')}`}
                  tokens={data.tokens}
                />
              )
            )}

            {/* LIME dimensions */}
            {hasLime && Object.entries(lime.dimensions).map(([dim, data]) =>
              data.tokens && (
                <ExplainSection
                  key={`lime-${dim}`}
                  title={`LIME — ${dim.replace('_', ' ')}`}
                  tokens={data.tokens}
                />
              )
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Audio explanation panel ──────────────────────────────────────────────────

function AudioExplainPanel({ audioExplanation }) {
  const [open, setOpen] = useState(false);
  if (!audioExplanation) return null;

  const lime = audioExplanation.lime;
  const shap = audioExplanation.shap;
  const textExp = audioExplanation.text_explanation;

  return (
    <div className="xai-panel">
      <button className="xai-toggle" onClick={() => setOpen(o => !o)}>
        <FlaskConical size={12} />
        <span>Audio LIME / SHAP</span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            className="xai-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <p className="xai-legend">
              <span style={{ color: '#6366f1' }}>■ drives emotion up</span>
              {'  '}
              <span style={{ color: '#94a3b8' }}>■ drives emotion down</span>
            </p>

            {lime?.segments && (
              <ExplainSection title="LIME — audio segments" tokens={lime.segments} isSegments />
            )}
            {shap?.segments && (
              <ExplainSection title="SHAP — audio segments" tokens={shap.segments} isSegments />
            )}

            {/* Text explanation on transcription */}
            {textExp?.shap?.dimensions && Object.entries(textExp.shap.dimensions).map(([dim, data]) =>
              data.tokens && (
                <ExplainSection
                  key={`txt-shap-${dim}`}
                  title={`SHAP (transcript) — ${dim.replace('_', ' ')}`}
                  tokens={data.tokens}
                />
              )
            )}
            {textExp?.lime?.dimensions && Object.entries(textExp.lime.dimensions).map(([dim, data]) =>
              data.tokens && (
                <ExplainSection
                  key={`txt-lime-${dim}`}
                  title={`LIME (transcript) — ${dim.replace('_', ' ')}`}
                  tokens={data.tokens}
                />
              )
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Main AffectPanel ────────────────────────────────────────────────────────

export default function AffectPanel({ affect, audioAffect, mode, explanation, audioExplanation }) {
  if (!affect || Object.keys(affect).length === 0) return null;

  const {
    rep_confidence     = 'medium',
    engagement_level   = 'engaged',
    frustration_signal = false,
    stress_signal      = false,
    query_urgency      = 'routine',
    affect_source      = 'rules',
  } = affect;

  const isPhysician     = mode === 'physician_portal';
  const hasFlaggedSignal = frustration_signal || stress_signal;

  const hasAudio    = audioAffect && audioAffect.emotion;
  const emotion     = hasAudio ? audioAffect.emotion : null;
  const emotionMeta = emotion ? (EMOTION_META[emotion] ?? EMOTION_META.neu) : null;
  const audioPct    = hasAudio ? Math.round((audioAffect.confidence ?? 0) * 100) : 0;

  const textConfident = rep_confidence === 'high' || rep_confidence === 'medium';
  const voiceNegative = emotion === 'sad' || emotion === 'ang';
  const divergence    = hasAudio && !isPhysician && textConfident && voiceNegative;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={JSON.stringify(affect) + JSON.stringify(audioAffect)}
        className="affect-panel"
        initial={{ opacity: 0, y: 8, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ duration: 0.25 }}
      >
        {/* Header */}
        <div className="affect-header">
          <Brain size={13} />
          <span>Affect Analysis</span>
          <span className="affect-source-chip">{affect_source}</span>
        </div>

        {/* ── MEDREP DIMENSIONS ── */}
        {!isPhysician && (
          <>
            <div className="affect-row">
              <span className="affect-label">Confidence</span>
              <div className="conf-segments">
                {['low', 'medium', 'high'].map((lvl) => (
                  <div
                    key={lvl}
                    className={`conf-seg ${rep_confidence === lvl ? 'active' : ''}`}
                    style={rep_confidence === lvl
                      ? { background: CONF_META[lvl].color, color: '#fff' }
                      : {}}
                  >
                    {CONF_META[lvl].label}
                  </div>
                ))}
              </div>
            </div>

            <div className="affect-row">
              <span className="affect-label">Engagement</span>
              <span
                className="affect-badge"
                style={engagement_level === 'engaged'
                  ? { color: '#10b981', background: 'rgba(16,185,129,0.1)' }
                  : { color: '#94a3b8', background: 'rgba(148,163,184,0.1)' }}
              >
                {engagement_level === 'engaged' ? '● Engaged' : '○ Passive'}
              </span>
            </div>
          </>
        )}

        {/* ── PHYSICIAN URGENCY ── */}
        {isPhysician && (
          <div className="affect-row">
            <span className="affect-label">Urgency</span>
            <span
              className="affect-badge"
              style={{
                color: URG_META[query_urgency]?.color,
                background: URG_META[query_urgency]?.bg,
                fontWeight: query_urgency === 'urgent' ? 700 : 600,
              }}
            >
              {URG_META[query_urgency]?.label ?? query_urgency}
            </span>
          </div>
        )}

        {/* ── FLAGS ── */}
        <div className="affect-flags">
          {frustration_signal && (
            <motion.span
              className="affect-flag"
              style={{ color: '#f59e0b', background: 'rgba(245,158,11,0.1)' }}
              initial={{ scale: 0.8 }} animate={{ scale: 1 }}
            >
              <AlertTriangle size={12} /> Frustration
            </motion.span>
          )}
          {stress_signal && (
            <motion.span
              className="affect-flag"
              style={{ color: '#ef4444', background: 'rgba(239,68,68,0.1)' }}
              initial={{ scale: 0.8 }} animate={{ scale: 1 }}
            >
              <Zap size={12} /> Stress
            </motion.span>
          )}
          {!hasFlaggedSignal && (
            <span className="affect-flag" style={{ color: '#10b981', background: 'rgba(16,185,129,0.08)' }}>
              <CheckCircle2 size={12} /> Clear
            </span>
          )}
        </div>

        {/* ── TEXT LIME / SHAP EXPLANATION ── */}
        <ExplainPanel explanation={explanation} />

        {/* ── VOICE EMOTION ── */}
        {hasAudio && (
          <motion.div
            className="audio-affect-row"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            transition={{ duration: 0.2 }}
          >
            <div className="audio-affect-header">
              <Mic size={11} />
              <span>Voice</span>
              <span className="affect-source-chip">speechbrain</span>
            </div>

            <div className="affect-row" style={{ marginTop: '0.25rem' }}>
              <span className="affect-label">Emotion</span>
              <span
                className="affect-badge"
                style={{ color: emotionMeta.color, background: emotionMeta.bg, fontWeight: 700 }}
              >
                {emotionMeta.emoji} {emotionMeta.label}
              </span>
              <span className="audio-confidence">{audioPct}%</span>
            </div>

            <div className="audio-bar-wrap">
              <div
                className="audio-bar"
                style={{ width: `${audioPct}%`, background: emotionMeta.color }}
              />
            </div>

            {divergence && (
              <motion.div className="divergence-alert" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                ⚠ Voice contradicts text — rep sounds {emotionMeta.label.toLowerCase()} despite positive words
              </motion.div>
            )}

            {/* ── AUDIO LIME / SHAP EXPLANATION ── */}
            <AudioExplainPanel audioExplanation={audioExplanation} />
          </motion.div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
