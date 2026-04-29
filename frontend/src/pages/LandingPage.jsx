import { useNavigate } from 'react-router-dom'
import { useEffect, useRef, useState } from 'react'

const FEATURES = [
  {
    icon: '⬆',
    title: 'Upload Any Dataset',
    desc: 'CSV, Excel, TSV, JSON — we parse it instantly and tell you what\'s inside.',
    accent: '#6c63ff',
  },
  {
    icon: '🔍',
    title: 'Auto Model Detection',
    desc: 'We analyse your data and suggest the best ML approach — classification, regression, clustering, or neural network.',
    accent: '#0ea5e9',
  },
  {
    icon: '⚙',
    title: 'Smart Preprocessing',
    desc: 'Missing values, outliers, duplicates, scaling — handled automatically before training.',
    accent: '#10b981',
  },
  {
    icon: '🚀',
    title: 'Visible ML Pipeline',
    desc: 'Watch every stage: preprocess → train → evaluate. No black boxes.',
    accent: '#f59e0b',
  },
  {
    icon: '📊',
    title: 'Full Metrics Suite',
    desc: 'Accuracy, F1, ROC-AUC, confusion matrix, cross-validation, feature importance and more.',
    accent: '#a78bfa',
  },
  {
    icon: '💾',
    title: 'Export Python Code',
    desc: 'Get a clean, commented, ready-to-run Python script for your exact model config.',
    accent: '#34d399',
  },
]

const MODELS = [
  { label: 'Classification', desc: 'Predict categories', icon: '◈', accent: '#6c63ff', example: 'spam / churn / disease' },
  { label: 'Regression',     desc: 'Predict numbers',   icon: '◉', accent: '#0ea5e9', example: 'price / sales / score' },
  { label: 'Clustering',     desc: 'Group similar rows', icon: '◎', accent: '#10b981', example: 'segments / anomalies' },
  { label: 'Neural Network', desc: 'Deep learning MLP',  icon: '◌', accent: '#f59e0b', example: 'complex patterns' },
]

const STEPS = [
  { n: '01', label: 'Upload Dataset',   desc: 'CSV, Excel, TSV or JSON' },
  { n: '02', label: 'Auto-Analysis',    desc: 'Health check + model suggestion' },
  { n: '03', label: 'Preprocess',       desc: 'Clean and prepare your data' },
  { n: '04', label: 'Configure Model',  desc: 'Pick algorithm and features' },
  { n: '05', label: 'Train Pipeline',   desc: 'Watch the stages run live' },
  { n: '06', label: 'Results + Export', desc: 'Full metrics and Python code' },
]

// Animated counter
function Counter({ target, suffix = '', duration = 1500 }) {
  const [val, setVal] = useState(0)
  const ref = useRef()
  useEffect(() => {
    const start = performance.now()
    function tick(now) {
      const p = Math.min((now - start) / duration, 1)
      const ease = 1 - Math.pow(1 - p, 3)
      setVal(Math.floor(ease * target))
      if (p < 1) ref.current = requestAnimationFrame(tick)
    }
    ref.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(ref.current)
  }, [target, duration])
  return <>{val}{suffix}</>
}

export default function LandingPage() {
  const navigate = useNavigate()
  const [hovered, setHovered] = useState(null)

  return (
    <div style={{ minHeight: '100vh', fontFamily: "'DM Sans', sans-serif" }}>

      {/* ── NAVBAR ── */}
      <nav style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100,
        padding: '0 40px', height: 60,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: 'rgba(2,2,8,0.85)', backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: 'linear-gradient(135deg,#6c63ff,#4f46e5)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 15, fontWeight: 800, color: '#fff',
            boxShadow: '0 0 16px rgba(108,99,255,0.45)',
          }}>M</div>
          <span style={{ fontWeight: 700, fontSize: 16, letterSpacing: '-0.02em' }}>ML Platform</span>
          <span style={{ fontSize: 10, color: '#555', fontFamily: 'monospace', background: 'rgba(108,99,255,0.12)', border: '1px solid rgba(108,99,255,0.25)', borderRadius: 99, padding: '2px 8px' }}>v2.0</span>
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <div style={{ width: 7, height: 7, borderRadius: '50%', background: '#10b981', boxShadow: '0 0 8px #10b981' }} />
          <span style={{ fontSize: 12, color: '#555', fontFamily: 'monospace' }}>local</span>
          <button
            onClick={() => navigate('/upload')}
            style={{
              padding: '8px 20px', borderRadius: 8,
              background: 'linear-gradient(135deg,#6c63ff,#4f46e5)',
              border: 'none', color: '#fff', fontSize: 13, fontWeight: 600,
              cursor: 'pointer', boxShadow: '0 0 16px rgba(108,99,255,0.4)',
            }}
          >
            Get Started →
          </button>
        </div>
      </nav>

      {/* ── HERO ── */}
      <section style={{ paddingTop: 140, paddingBottom: 100, textAlign: 'center', position: 'relative', overflow: 'hidden' }}>
        {/* Background decoration */}
        <div style={{
          position: 'absolute', top: '10%', left: '50%', transform: 'translateX(-50%)',
          width: 600, height: 600,
          background: 'radial-gradient(circle, rgba(108,99,255,0.12) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />

        <div style={{ position: 'relative', maxWidth: 800, margin: '0 auto', padding: '0 24px' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            background: 'rgba(108,99,255,0.1)', border: '1px solid rgba(108,99,255,0.25)',
            borderRadius: 99, padding: '6px 18px', marginBottom: 28,
            fontSize: 12, color: '#a09af0', letterSpacing: '0.1em', fontFamily: 'monospace',
          }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#6c63ff', boxShadow: '0 0 6px #6c63ff' }} />
            NO-CODE · END-TO-END · ML PIPELINE
          </div>

          <h1 style={{
            fontSize: 'clamp(40px,7vw,76px)', fontWeight: 800,
            lineHeight: 1.05, letterSpacing: '-0.04em', margin: '0 0 24px',
          }}>
            Upload a dataset.<br />
            <span style={{
              background: 'linear-gradient(135deg, #6c63ff, #0ea5e9)',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
            }}>Build ML models.</span><br />
            Get the code.
          </h1>

          <p style={{ fontSize: 18, color: '#666', lineHeight: 1.7, marginBottom: 40, maxWidth: 560, margin: '0 auto 40px' }}>
            Drop your CSV, Excel or JSON file. We detect the right ML approach, preprocess your data, train the model, and hand you back production-ready Python code.
          </p>

          <div style={{ display: 'flex', gap: 14, justifyContent: 'center', flexWrap: 'wrap' }}>
            <button
              onClick={() => navigate('/upload')}
              style={{
                padding: '14px 36px', borderRadius: 12,
                background: 'linear-gradient(135deg,#6c63ff,#4f46e5)',
                border: 'none', color: '#fff', fontSize: 16, fontWeight: 700,
                cursor: 'pointer', letterSpacing: '-0.01em',
                boxShadow: '0 0 32px rgba(108,99,255,0.4)',
                transition: 'all 0.2s',
              }}
              onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
              onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
            >
              🚀 Start with your dataset
            </button>
            <button
              onClick={() => document.getElementById('how').scrollIntoView({ behavior: 'smooth' })}
              style={{
                padding: '14px 36px', borderRadius: 12,
                background: 'transparent',
                border: '1px solid rgba(255,255,255,0.1)',
                color: '#888', fontSize: 16, fontWeight: 500,
                cursor: 'pointer', transition: 'all 0.2s',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.3)'; e.currentTarget.style.color = '#ccc' }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'; e.currentTarget.style.color = '#888' }}
            >
              See how it works ↓
            </button>
          </div>

          {/* Supported formats pill */}
          <div style={{ marginTop: 24, display: 'flex', justifyContent: 'center', gap: 8, flexWrap: 'wrap' }}>
            {['CSV', 'Excel (.xlsx)', 'TSV', 'JSON'].map(f => (
              <span key={f} style={{
                fontSize: 11, color: '#555', fontFamily: 'monospace',
                background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: 6, padding: '3px 10px',
              }}>{f}</span>
            ))}
          </div>
        </div>
      </section>

      {/* ── STATS ── */}
      <section style={{ padding: '40px 24px 80px', maxWidth: 900, margin: '0 auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16 }}>
          {[
            { val: 4,    suffix: '',   label: 'ML model types' },
            { val: 18,   suffix: '+',  label: 'Algorithms' },
            { val: 20,   suffix: '+',  label: 'Metrics tracked' },
            { val: 100,  suffix: '%',  label: 'No-code workflow' },
          ].map((s, i) => (
            <div key={i} style={{
              background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: 16, padding: '28px 24px', textAlign: 'center',
            }}>
              <div style={{ fontSize: 42, fontWeight: 800, color: '#6c63ff', fontFamily: 'monospace', letterSpacing: '-0.04em', marginBottom: 6 }}>
                <Counter target={s.val} suffix={s.suffix} duration={1200 + i * 200} />
              </div>
              <div style={{ fontSize: 13, color: '#555' }}>{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── HOW IT WORKS ── */}
      <section id="how" style={{ padding: '80px 24px', maxWidth: 900, margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: 56 }}>
          <div style={{ fontSize: 11, color: '#6c63ff', textTransform: 'uppercase', letterSpacing: '0.12em', fontFamily: 'monospace', marginBottom: 12 }}>THE PIPELINE</div>
          <h2 style={{ fontSize: 'clamp(28px,4vw,44px)', fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 12 }}>Six steps. Zero guesswork.</h2>
          <p style={{ fontSize: 15, color: '#555', maxWidth: 480, margin: '0 auto' }}>The full ML pipeline runs automatically. You configure, we execute.</p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 16 }}>
          {STEPS.map((s, i) => (
            <div key={i} style={{
              background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: 14, padding: '24px',
              transition: 'all 0.2s',
              cursor: 'default',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = 'rgba(108,99,255,0.4)'
              e.currentTarget.style.background = 'rgba(108,99,255,0.04)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'
              e.currentTarget.style.background = 'rgba(255,255,255,0.02)'
            }}
            >
              <div style={{ fontSize: 11, color: '#6c63ff', fontFamily: 'monospace', marginBottom: 10 }}>{s.n}</div>
              <div style={{ fontSize: 15, fontWeight: 700, color: '#e8e8f0', marginBottom: 6, letterSpacing: '-0.01em' }}>{s.label}</div>
              <div style={{ fontSize: 12, color: '#555', lineHeight: 1.5 }}>{s.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── MODEL TYPES ── */}
      <section style={{ padding: '80px 24px', maxWidth: 900, margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: 48 }}>
          <div style={{ fontSize: 11, color: '#0ea5e9', textTransform: 'uppercase', letterSpacing: '0.12em', fontFamily: 'monospace', marginBottom: 12 }}>MODEL TYPES</div>
          <h2 style={{ fontSize: 'clamp(28px,4vw,44px)', fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 12 }}>Auto-detected from your data</h2>
          <p style={{ fontSize: 15, color: '#555', maxWidth: 480, margin: '0 auto' }}>Upload your file and we tell you which model type fits best — with reasons.</p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 14 }}>
          {MODELS.map((m, i) => (
            <div key={i} style={{
              background: hovered === i ? `rgba(${hexToRgb(m.accent)},0.07)` : 'rgba(255,255,255,0.02)',
              border: `1px solid ${hovered === i ? m.accent + '55' : 'rgba(255,255,255,0.06)'}`,
              borderRadius: 16, padding: '28px 24px',
              cursor: 'pointer', transition: 'all 0.25s', position: 'relative', overflow: 'hidden',
            }}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            onClick={() => navigate('/upload')}
            >
              {hovered === i && (
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${m.accent}, transparent)` }} />
              )}
              <div style={{ fontSize: 28, marginBottom: 14, color: m.accent }}>{m.icon}</div>
              <div style={{ fontSize: 17, fontWeight: 700, color: hovered === i ? m.accent : '#e8e8f0', marginBottom: 6, letterSpacing: '-0.01em' }}>{m.label}</div>
              <div style={{ fontSize: 13, color: '#555', marginBottom: 10 }}>{m.desc}</div>
              <div style={{ fontSize: 11, color: '#3a3a4a', fontFamily: 'monospace' }}>{m.example}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── FEATURES ── */}
      <section style={{ padding: '80px 24px', maxWidth: 900, margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: 48 }}>
          <div style={{ fontSize: 11, color: '#10b981', textTransform: 'uppercase', letterSpacing: '0.12em', fontFamily: 'monospace', marginBottom: 12 }}>FEATURES</div>
          <h2 style={{ fontSize: 'clamp(28px,4vw,44px)', fontWeight: 800, letterSpacing: '-0.03em' }}>Everything you need. Nothing you don't.</h2>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 14 }}>
          {FEATURES.map((f, i) => (
            <div key={i} style={{
              background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: 14, padding: '22px',
              transition: 'all 0.2s',
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = f.accent + '44'; e.currentTarget.style.background = `rgba(${hexToRgb(f.accent)},0.04)` }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'; e.currentTarget.style.background = 'rgba(255,255,255,0.02)' }}
            >
              <div style={{ fontSize: 22, marginBottom: 12, color: f.accent }}>{f.icon}</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#e8e8f0', marginBottom: 8, letterSpacing: '-0.01em' }}>{f.title}</div>
              <div style={{ fontSize: 12, color: '#555', lineHeight: 1.6 }}>{f.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA ── */}
      <section style={{ padding: '80px 24px 120px', textAlign: 'center' }}>
        <div style={{
          maxWidth: 600, margin: '0 auto',
          background: 'rgba(108,99,255,0.06)', border: '1px solid rgba(108,99,255,0.2)',
          borderRadius: 24, padding: '56px 40px',
          position: 'relative', overflow: 'hidden',
        }}>
          <div style={{ position: 'absolute', top: -60, left: '50%', transform: 'translateX(-50%)', width: 300, height: 300, background: 'radial-gradient(circle,rgba(108,99,255,0.15) 0%,transparent 70%)', pointerEvents: 'none' }} />
          <h2 style={{ fontSize: 36, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 14 }}>Ready to build?</h2>
          <p style={{ fontSize: 15, color: '#666', marginBottom: 32 }}>Upload your dataset and have a trained model with results in under 2 minutes.</p>
          <button
            onClick={() => navigate('/upload')}
            style={{
              padding: '15px 40px', borderRadius: 12,
              background: 'linear-gradient(135deg,#6c63ff,#4f46e5)',
              border: 'none', color: '#fff', fontSize: 16, fontWeight: 700,
              cursor: 'pointer', boxShadow: '0 0 32px rgba(108,99,255,0.5)',
              transition: 'all 0.2s',
            }}
            onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
            onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
          >
            Upload your dataset →
          </button>
          <p style={{ fontSize: 11, color: '#444', marginTop: 16, fontFamily: 'monospace' }}>
            CSV · Excel · TSV · JSON · No account required
          </p>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer style={{ borderTop: '1px solid rgba(255,255,255,0.04)', padding: '24px 40px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 24, height: 24, borderRadius: 6, background: 'linear-gradient(135deg,#6c63ff,#4f46e5)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, fontWeight: 800, color: '#fff' }}>M</div>
          <span style={{ fontSize: 13, color: '#444', fontWeight: 600 }}>ML Platform</span>
        </div>
        <span style={{ fontSize: 11, color: '#333', fontFamily: 'monospace' }}>No data stored · runs locally</span>
      </footer>
    </div>
  )
}

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1,3),16)
  const g = parseInt(hex.slice(3,5),16)
  const b = parseInt(hex.slice(5,7),16)
  return `${r},${g},${b}`
}