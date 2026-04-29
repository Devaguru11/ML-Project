import { useState, useRef } from 'react'

const STAGES = [
  { id: 'preprocess', label: 'Pre-processing',  desc: 'Impute → Scale → Encode', color: '#0ea5e9' },
  { id: 'train',      label: 'Training',         desc: 'Fitting the model',       color: '#6c63ff' },
  { id: 'evaluate',   label: 'Evaluation',       desc: 'Metrics + Cross-val',     color: '#10b981' },
  { id: 'done',       label: 'Complete',         desc: 'Results ready',           color: '#f59e0b' },
]

export default function PipelineTrainer({ onTrain, onResult, accent = '#6c63ff', children }) {
  const [phase, setPhase]     = useState('idle')   // idle | running | done | error
  const [pct, setPct]         = useState(0)
  const [stageId, setStageId] = useState('')
  const [msg, setMsg]         = useState('')
  const [log, setLog]         = useState([])
  const abortRef = useRef(null)

  async function handleTrain() {
    setPhase('running')
    setPct(0)
    setStageId('preprocess')
    setMsg('Starting pipeline...')
    setLog([])

    try {
      await onTrain((event) => {
        if (event.stage === 'error') {
          setPhase('error')
          setMsg(event.msg)
          return
        }
        setStageId(event.stage)
        setPct(event.pct)
        setMsg(event.msg)
        setLog(prev => [...prev.slice(-6), { stage: event.stage, msg: event.msg, pct: event.pct }])
        if (event.stage === 'done' && event.data) {
          setPhase('done')
          onResult(event.data)
        }
      })
    } catch(e) {
      setPhase('error')
      setMsg(e.message)
    }
  }

  function reset() {
    setPhase('idle')
    setPct(0)
    setStageId('')
    setMsg('')
    setLog([])
  }

  const currentStageIdx = STAGES.findIndex(s => s.id === stageId)

  return (
    <div>
      {/* Config form — shown when idle */}
      {phase === 'idle' && (
        <div>
          {children}
          <TrainButton onClick={handleTrain} accent={accent} label="Start Training" />
        </div>
      )}

      {/* Pipeline progress — shown when running */}
      {phase === 'running' && (
        <div>
          <div style={{ marginBottom: 28 }}>
            <h3 style={{ fontSize: 16, fontWeight: 600, color: '#f0f0f0', marginBottom: 4 }}>Pipeline running</h3>
            <p style={{ fontSize: 13, color: '#555' }}>Do not close this page. Training in progress...</p>
          </div>

          {/* Stage indicators */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 28 }}>
            {STAGES.map((s, i) => {
              const isActive   = s.id === stageId
              const isComplete = i < currentStageIdx
              return (
                <div key={s.id} style={{
                  padding: '14px 12px',
                  background: isActive ? `rgba(${hexRgb(s.color)},0.1)` : isComplete ? 'rgba(16,185,129,0.06)' : 'rgba(255,255,255,0.01)',
                  border: `1px solid ${isActive ? s.color : isComplete ? 'rgba(16,185,129,0.3)' : 'rgba(255,255,255,0.06)'}`,
                  borderRadius: 12, textAlign: 'center', transition: 'all 0.3s',
                  boxShadow: isActive ? `0 0 20px ${s.color}30` : 'none',
                }}>
                  <div style={{ fontSize: 18, marginBottom: 6 }}>
                    {isComplete ? '✓' : isActive ? <Spinner color={s.color} /> : '○'}
                  </div>
                  <div style={{ fontSize: 12, fontWeight: 600, color: isActive ? s.color : isComplete ? '#10b981' : '#555' }}>{s.label}</div>
                  <div style={{ fontSize: 10, color: '#444', marginTop: 2 }}>{s.desc}</div>
                </div>
              )
            })}
          </div>

          {/* Progress bar */}
          <div style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: 12 }}>
              <span style={{ color: '#888' }}>{msg}</span>
              <span style={{ color: accent, fontFamily: 'monospace', fontWeight: 600 }}>{pct}%</span>
            </div>
            <div style={{ height: 6, background: 'rgba(255,255,255,0.06)', borderRadius: 99, overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${pct}%`,
                background: `linear-gradient(90deg, ${accent}, ${accent}cc)`,
                borderRadius: 99,
                transition: 'width 0.4s ease',
                boxShadow: `0 0 12px ${accent}60`,
              }} />
            </div>
          </div>

          {/* Log */}
          <div style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: 10, padding: '12px 14px', fontFamily: 'monospace', fontSize: 11 }}>
            {log.map((l, i) => (
              <div key={i} style={{ color: i === log.length - 1 ? '#a09af0' : '#444', marginBottom: 3, display: 'flex', gap: 10 }}>
                <span style={{ color: '#333', minWidth: 36 }}>{l.pct}%</span>
                <span>{l.msg}</span>
              </div>
            ))}
            {log.length === 0 && <span style={{ color: '#333' }}>Initialising...</span>}
          </div>
        </div>
      )}

      {/* Error state */}
      {phase === 'error' && (
        <div>
          <div style={{ padding: '20px', background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 12, marginBottom: 16, textAlign: 'center' }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>✕</div>
            <div style={{ fontSize: 15, fontWeight: 600, color: '#f87171', marginBottom: 8 }}>Training failed</div>
            <div style={{ fontSize: 13, color: '#888' }}>{msg}</div>
          </div>
          <TrainButton onClick={reset} accent="#f87171" label="← Try again" />
        </div>
      )}
    </div>
  )
}

function Spinner({ color }) {
  return (
    <div style={{
      width: 18, height: 18,
      border: `2px solid ${color}30`,
      borderTop: `2px solid ${color}`,
      borderRadius: '50%',
      animation: 'spin 0.7s linear infinite',
      display: 'inline-block',
    }} />
  )
}

function TrainButton({ onClick, accent, label }) {
  return (
    <button onClick={onClick} style={{
      width: '100%', padding: '14px',
      background: `linear-gradient(135deg, ${accent}, ${accent}bb)`,
      border: 'none', borderRadius: 12,
      color: '#fff', fontSize: 15, fontWeight: 700,
      cursor: 'pointer', letterSpacing: '-0.01em',
      boxShadow: `0 0 24px ${accent}40`,
      transition: 'all 0.2s',
    }}
    onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-1px)'}
    onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
    >
      {label}
    </button>
  )
}

function hexRgb(hex) {
  const r = parseInt(hex.slice(1,3),16)
  const g = parseInt(hex.slice(3,5),16)
  const b = parseInt(hex.slice(5,7),16)
  return `${r},${g},${b}`
}