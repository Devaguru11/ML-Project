import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { preprocessDataset } from '../api/client'

const CLUSTER_COLORS = ['#6c63ff','#0ea5e9','#10b981','#f59e0b','#f87171','#a78bfa']

const MODEL_META = {
  classification:  { accent: '#6c63ff', icon: '◈', page: '/classification' },
  regression:      { accent: '#0ea5e9', icon: '◉', page: '/regression' },
  clustering:      { accent: '#10b981', icon: '◎', page: '/clustering' },
  'neural-network':{ accent: '#f59e0b', icon: '◌', page: '/neural-network' },
}

const CONF_COLOR = { high: '#10b981', medium: '#f59e0b', low: '#f87171' }

export default function DatasetReport() {
  const [analysis, setAnalysis]       = useState(null)
  const [ds, setDs]                   = useState(null)
  const [selected, setSelected]       = useState(null)
  const [preprocessing, setPreprocessing] = useState(false)
  const [preprocessDone, setPreprocessDone] = useState(false)
  const [preprocessResult, setPreprocessResult] = useState(null)
  const [error, setError]             = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    const a = sessionStorage.getItem('dsAnalysis')
    const d = sessionStorage.getItem('dataset')
    if (!a || !d) { navigate('/upload'); return }
    const parsed = JSON.parse(a)
    setAnalysis(parsed)
    setDs(JSON.parse(d))
    // Auto-select the top suggestion
    if (parsed.suggestions && parsed.suggestions.length > 0) {
      setSelected(parsed.suggestions[0].type)
    }
  }, [navigate])

  async function handlePreprocessAndContinue() {
    if (!selected) { setError('Select a model type first.'); return }
    setError('')
    setPreprocessing(true)
    try {
      const csvRaw = sessionStorage.getItem('csvRaw')
      const result = await preprocessDataset(csvRaw, {
        drop_duplicates: true,
        drop_high_missing: true,
        high_missing_threshold: 0.3,
      })
      setPreprocessResult(result)
      // Store the cleaned CSV for training
      sessionStorage.setItem('csvRaw', result.clean_csv_data)
      sessionStorage.setItem('modelType', selected)
      // Update dataset columns if cols were dropped
      const currentDs = JSON.parse(sessionStorage.getItem('dataset') || '{}')
      sessionStorage.setItem('dataset', JSON.stringify({ ...currentDs, columns: result.columns }))
      setPreprocessDone(true)
    } catch (e) {
      setError(e.message || 'Preprocessing failed.')
    } finally {
      setPreprocessing(false)
    }
  }

  function handleContinue() {
    const meta = MODEL_META[selected]
    if (meta) navigate(meta.page)
  }

  if (!analysis) return null

  const topSuggestion = analysis.suggestions?.[0]
  const health = analysis.health_score ?? 0
  const healthColor = health >= 80 ? '#10b981' : health >= 50 ? '#f59e0b' : '#f87171'
  const healthLabel = health >= 80 ? 'Healthy' : health >= 50 ? 'Fair' : 'Needs attention'

  return (
    <div style={{ minHeight: '100vh' }}>
      {/* Header */}
      <header style={{
        borderBottom: '1px solid rgba(255,255,255,0.05)', padding: '16px 40px',
        display: 'flex', alignItems: 'center', gap: 14,
        background: 'rgba(2,2,8,0.85)', backdropFilter: 'blur(20px)',
        position: 'sticky', top: 0, zIndex: 50,
      }}>
        <Link to='/upload' style={{ color: '#555', fontSize: 13, textDecoration: 'none' }}
          onMouseEnter={e => e.currentTarget.style.color = '#aaa'}
          onMouseLeave={e => e.currentTarget.style.color = '#555'}
        >← Re-upload</Link>
        <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)' }} />
        <span style={{ fontSize: 14, fontWeight: 600, color: '#e8e8f0' }}>Dataset Report</span>
        <span style={{ marginLeft: 'auto', fontSize: 12, color: '#555', fontFamily: 'monospace' }}>
          {sessionStorage.getItem('csvFile') || 'dataset'}
        </span>
        {/* Step indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginLeft: 16 }}>
          {['Upload','Report','Configure','Train','Results'].map((s,i) => (
            <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <div style={{
                width: 20, height: 20, borderRadius: '50%',
                background: i <= 1 ? '#6c63ff' : 'rgba(255,255,255,0.05)',
                border: `1px solid ${i <= 1 ? '#6c63ff' : 'rgba(255,255,255,0.08)'}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 9, fontWeight: 700, color: i <= 1 ? '#fff' : '#555',
              }}>{i <= 1 ? '✓' : i+1}</div>
              {i < 4 && <div style={{ width: 12, height: 1, background: 'rgba(255,255,255,0.05)' }} />}
            </div>
          ))}
        </div>
      </header>

      <main style={{ maxWidth: 900, margin: '0 auto', padding: '40px 24px 80px' }}>

        {/* ── Title ── */}
        <div style={{ marginBottom: 32 }}>
          <div style={{ fontSize: 11, color: '#6c63ff', textTransform: 'uppercase', letterSpacing: '0.12em', fontFamily: 'monospace', marginBottom: 10 }}>STEP 2 OF 5</div>
          <h1 style={{ fontSize: 28, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 8 }}>Dataset Analysis Report</h1>
          <p style={{ fontSize: 14, color: '#555' }}>We've analysed your file. Review the findings and select an ML approach to continue.</p>
        </div>

        {/* ── Overview strip ── */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 10, marginBottom: 24 }}>
          {[
            { label: 'Total rows',      value: analysis.n_rows?.toLocaleString() },
            { label: 'Columns',         value: analysis.n_cols },
            { label: 'Numeric cols',    value: analysis.n_numeric },
            { label: 'Categorical cols',value: analysis.n_categorical },
            { label: 'Missing data',    value: analysis.total_missing_pct + '%' },
          ].map(m => (
            <div key={m.label} style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 12, padding: '14px 14px', textAlign: 'center' }}>
              <div style={{ fontSize: 22, fontWeight: 800, color: '#e8e8f0', fontFamily: 'monospace', marginBottom: 4 }}>{m.value}</div>
              <div style={{ fontSize: 10, color: '#555' }}>{m.label}</div>
            </div>
          ))}
        </div>

        {/* ── Health score + Issues ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 16, marginBottom: 24 }}>
          {/* Health dial */}
          <div style={{ background: 'rgba(255,255,255,0.02)', border: `1px solid ${healthColor}33`, borderRadius: 16, padding: '28px 24px', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ position: 'relative', width: 120, height: 120, margin: '0 auto 16px' }}>
              <svg viewBox='0 0 120 120' style={{ transform: 'rotate(-90deg)' }}>
                <circle cx='60' cy='60' r='50' fill='none' stroke='rgba(255,255,255,0.05)' strokeWidth='10' />
                <circle cx='60' cy='60' r='50' fill='none' stroke={healthColor}
                  strokeWidth='10' strokeLinecap='round'
                  strokeDasharray={`${health * 3.14} 314`}
                  style={{ transition: 'stroke-dasharray 1s ease', filter: `drop-shadow(0 0 6px ${healthColor})` }}
                />
              </svg>
              <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 800, color: healthColor, fontFamily: 'monospace', lineHeight: 1 }}>{health}</div>
                <div style={{ fontSize: 10, color: '#555' }}>/ 100</div>
              </div>
            </div>
            <div style={{ fontSize: 15, fontWeight: 700, color: healthColor, marginBottom: 4 }}>{healthLabel}</div>
            <div style={{ fontSize: 11, color: '#555' }}>Dataset health score</div>
          </div>

          {/* Issues / preprocessing steps */}
          <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 16, padding: '20px 22px' }}>
            <div style={{ fontSize: 11, color: '#555', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 14, fontWeight: 600 }}>Issues & Preprocessing Plan</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {(analysis.preprocess_steps || []).map((step, i) => {
                const sevColor = step.severity === 'high' ? '#f87171' : step.severity === 'medium' ? '#fbbf24' : step.severity === 'ok' ? '#10b981' : '#0ea5e9'
                return (
                  <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 10, padding: '10px 12px', background: `rgba(${sevColor === '#f87171' ? '239,68,68' : sevColor === '#fbbf24' ? '245,158,11' : '16,185,129'},0.05)`, border: `1px solid ${sevColor}22`, borderRadius: 8 }}>
                    <div style={{ fontSize: 14, flexShrink: 0, color: sevColor, marginTop: 1 }}>
                      {step.severity === 'ok' ? '✓' : step.severity === 'high' ? '⚠' : step.auto ? '⚙' : '→'}
                    </div>
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 600, color: '#e0e0e0', marginBottom: 2 }}>{step.step}</div>
                      <div style={{ fontSize: 11, color: '#666', lineHeight: 1.4 }}>{step.detail}</div>
                    </div>
                    {step.auto && (
                      <div style={{ marginLeft: 'auto', flexShrink: 0, fontSize: 10, color: '#10b981', background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.2)', borderRadius: 4, padding: '2px 8px', whiteSpace: 'nowrap' }}>Auto</div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* ── Outliers / Skew / Corr (compact) ── */}
        {(analysis.outlier_cols?.length > 0 || analysis.skewed_cols?.length > 0 || analysis.high_corr_pairs?.length > 0) && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12, marginBottom: 24 }}>
            {analysis.outlier_cols?.length > 0 && (
              <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 14, padding: '16px' }}>
                <div style={{ fontSize: 11, color: '#f87171', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>Outlier Columns</div>
                {analysis.outlier_cols.slice(0, 4).map(o => (
                  <div key={o.column} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 12 }}>
                    <span style={{ color: '#aaa', fontFamily: 'monospace' }}>{o.column}</span>
                    <span style={{ color: '#f87171' }}>{o.pct}%</span>
                  </div>
                ))}
                <div style={{ fontSize: 10, color: '#444', marginTop: 8 }}>Handled by RobustScaler</div>
              </div>
            )}
            {analysis.skewed_cols?.length > 0 && (
              <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 14, padding: '16px' }}>
                <div style={{ fontSize: 11, color: '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>Skewed Columns</div>
                {analysis.skewed_cols.slice(0, 4).map(s => (
                  <div key={s.column} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 12 }}>
                    <span style={{ color: '#aaa', fontFamily: 'monospace' }}>{s.column}</span>
                    <span style={{ color: '#f59e0b' }}>skew {s.skew}</span>
                  </div>
                ))}
                <div style={{ fontSize: 10, color: '#444', marginTop: 8 }}>Scaling reduces impact</div>
              </div>
            )}
            {analysis.high_corr_pairs?.length > 0 && (
              <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 14, padding: '16px' }}>
                <div style={{ fontSize: 11, color: '#a78bfa', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>High Correlation</div>
                {analysis.high_corr_pairs.slice(0, 4).map((p, i) => (
                  <div key={i} style={{ marginBottom: 6, fontSize: 11 }}>
                    <span style={{ color: '#aaa', fontFamily: 'monospace' }}>{p.col_a} ↔ {p.col_b}</span>
                    <span style={{ color: '#a78bfa', marginLeft: 8 }}>r={p.r}</span>
                  </div>
                ))}
                <div style={{ fontSize: 10, color: '#444', marginTop: 8 }}>Consider dropping one column</div>
              </div>
            )}
          </div>
        )}

        {/* ── Model Suggestions ── */}
        <div style={{ marginBottom: 28 }}>
          <div style={{ fontSize: 11, color: '#555', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 16, fontWeight: 600 }}>
            🤖 Suggested ML Approach — pick one to continue
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {(analysis.suggestions || []).map((s, i) => {
              const meta  = MODEL_META[s.type] || {}
              const isSel = selected === s.type
              const isTop = i === 0
              return (
                <button
                  key={s.type}
                  onClick={() => setSelected(s.type)}
                  style={{
                    background: isSel ? `rgba(${hexToRgb(meta.accent)},0.08)` : 'rgba(255,255,255,0.02)',
                    border: `1.5px solid ${isSel ? meta.accent : 'rgba(255,255,255,0.06)'}`,
                    borderRadius: 14, padding: '18px 20px', cursor: 'pointer', textAlign: 'left',
                    transition: 'all 0.2s',
                    boxShadow: isSel ? `0 0 20px ${meta.accent}25` : 'none',
                    position: 'relative', overflow: 'hidden',
                  }}
                >
                  {isSel && <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg,transparent,${meta.accent},transparent)` }} />}
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
                    {/* Icon + badge */}
                    <div style={{ flexShrink: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
                      <div style={{ width: 44, height: 44, borderRadius: 12, background: `rgba(${hexToRgb(meta.accent)},0.12)`, border: `1px solid ${meta.accent}44`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22, color: meta.accent }}>
                        {meta.icon}
                      </div>
                      {isTop && (
                        <div style={{ fontSize: 9, color: '#10b981', background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.25)', borderRadius: 99, padding: '1px 8px', whiteSpace: 'nowrap' }}>Best match</div>
                      )}
                    </div>

                    {/* Text */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6, flexWrap: 'wrap' }}>
                        <span style={{ fontSize: 15, fontWeight: 700, color: isSel ? meta.accent : '#e8e8f0', letterSpacing: '-0.01em' }}>{s.label}</span>
                        <span style={{ fontSize: 11, color: CONF_COLOR[s.confidence], background: `rgba(${hexToRgb(CONF_COLOR[s.confidence])},0.1)`, border: `1px solid ${CONF_COLOR[s.confidence]}44`, borderRadius: 99, padding: '1px 10px' }}>
                          {s.confidence} confidence
                        </span>
                        {s.target_col && (
                          <span style={{ fontSize: 11, color: '#555', fontFamily: 'monospace', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 6, padding: '1px 8px' }}>
                            target: {s.target_col}
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: 13, color: '#666', lineHeight: 1.55, marginBottom: 8 }}>{s.reason}</div>
                      <div style={{ fontSize: 11, color: '#555' }}>
                        Recommended: <span style={{ color: meta.accent, fontFamily: 'monospace' }}>{s.recommended_model}</span>
                      </div>
                    </div>

                    {/* Radio */}
                    <div style={{
                      flexShrink: 0, width: 20, height: 20, borderRadius: '50%', marginTop: 4,
                      border: `2px solid ${isSel ? meta.accent : 'rgba(255,255,255,0.15)'}`,
                      background: isSel ? meta.accent : 'transparent',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      transition: 'all 0.2s',
                    }}>
                      {isSel && <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#fff' }} />}
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        </div>

        {/* ── Preprocessing result ── */}
        {preprocessDone && preprocessResult && (
          <div style={{ background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.25)', borderRadius: 14, padding: '20px 22px', marginBottom: 24 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: '#10b981', marginBottom: 12 }}>✓ Preprocessing complete</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 14 }}>
              {[
                { label: 'Rows before', value: preprocessResult.original_rows },
                { label: 'Rows after',  value: preprocessResult.final_rows },
                { label: 'Cols after',  value: preprocessResult.final_cols },
                { label: 'Missing left',value: preprocessResult.remaining_missing },
              ].map(m => (
                <div key={m.label} style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: '#34d399', fontFamily: 'monospace' }}>{m.value}</div>
                  <div style={{ fontSize: 10, color: '#555' }}>{m.label}</div>
                </div>
              ))}
            </div>
            {preprocessResult.steps_applied.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                {preprocessResult.steps_applied.map((step, i) => (
                  <div key={i} style={{ fontSize: 12, color: '#555', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ color: '#10b981' }}>✓</span>{step}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {error && (
          <div style={{ padding: '12px 16px', background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 10, fontSize: 13, color: '#f87171', marginBottom: 20, display: 'flex', gap: 10 }}>
            <span>⚠</span>{error}
          </div>
        )}

        {/* ── Action buttons ── */}
        <div style={{ display: 'flex', gap: 12 }}>
          {!preprocessDone ? (
            <button
              onClick={handlePreprocessAndContinue}
              disabled={!selected || preprocessing}
              style={{
                flex: 1, padding: '15px', borderRadius: 12,
                background: !selected || preprocessing ? '#222' : `linear-gradient(135deg,${MODEL_META[selected]?.accent || '#6c63ff'},${MODEL_META[selected]?.accent || '#6c63ff'}bb)`,
                border: 'none', color: '#fff', fontSize: 15, fontWeight: 700,
                cursor: !selected || preprocessing ? 'not-allowed' : 'pointer',
                opacity: !selected ? 0.5 : 1,
                boxShadow: selected && !preprocessing ? `0 0 24px ${MODEL_META[selected]?.accent || '#6c63ff'}40` : 'none',
                transition: 'all 0.2s',
              }}
            >
              {preprocessing ? '⚙ Preprocessing data...' : `⚙ Preprocess & continue with ${selected ? (selected.charAt(0).toUpperCase() + selected.slice(1)) : '...'}`}
            </button>
          ) : (
            <>
              <button
                onClick={() => navigate('/visualise')}
                style={{
                  padding: '15px 28px', borderRadius: 12,
                  background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)',
                  color: '#888', fontSize: 14, fontWeight: 600, cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
              >
                📊 Explore data first
              </button>
              <button
                onClick={handleContinue}
                style={{
                  flex: 1, padding: '15px', borderRadius: 12,
                  background: `linear-gradient(135deg,${MODEL_META[selected]?.accent || '#6c63ff'},${MODEL_META[selected]?.accent || '#6c63ff'}bb)`,
                  border: 'none', color: '#fff', fontSize: 15, fontWeight: 700,
                  cursor: 'pointer',
                  boxShadow: `0 0 24px ${MODEL_META[selected]?.accent || '#6c63ff'}40`,
                  transition: 'all 0.2s',
                }}
                onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-1px)'}
                onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
              >
                Configure {selected?.charAt(0).toUpperCase() + selected?.slice(1)} model →
              </button>
            </>
          )}
        </div>

        {!preprocessDone && selected && (
          <p style={{ fontSize: 11, color: '#444', textAlign: 'center', marginTop: 12 }}>
            Preprocessing will remove duplicates, fill missing values, and clean the dataset before training.
          </p>
        )}
      </main>
    </div>
  )
}

function hexToRgb(hex) {
  if (!hex || !hex.startsWith('#')) return '108,99,255'
  const r = parseInt(hex.slice(1,3),16)
  const g = parseInt(hex.slice(3,5),16)
  const b = parseInt(hex.slice(5,7),16)
  return `${r},${g},${b}`
}