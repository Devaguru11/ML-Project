import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { trainNeuralStream } from '../api/client'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip as RTooltip, ResponsiveContainer,
  ScatterChart, Scatter, Legend
} from 'recharts'
import CodeExport from '../components/CodeExport.jsx'
import PipelineTrainer from '../components/PipelineTrainer.jsx'

const ACCENT = '#f59e0b'

const ACTIVATIONS = [
  { id: 'relu',     label: 'ReLU',    tip: 'Best default. Fast, avoids vanishing gradients.',           rec: 'Use for most problems.' },
  { id: 'tanh',     label: 'Tanh',    tip: 'Outputs −1 to 1. Better gradient flow for shallow nets.',   rec: 'Use for shallow networks.' },
  { id: 'logistic', label: 'Sigmoid', tip: 'Classic activation. Can suffer vanishing gradients.',       rec: 'Use for binary output layers.' },
]

export default function NeuralNetworkPage() {
  const [ds, setDs]                   = useState(null)
  const [problemType, setProblemType] = useState('classification')
  const [target, setTarget]           = useState('')
  const [features, setFeatures]       = useState([])
  const [testSize, setTestSize]       = useState(0.2)
  const [layers, setLayers]           = useState([64, 32])
  const [activation, setActivation]   = useState('relu')
  const [maxIter, setMaxIter]         = useState(200)
  const [results, setResults]         = useState(null)
  const [csvRaw, setCsvRaw]           = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    const d = sessionStorage.getItem('dataset')
    const c = sessionStorage.getItem('csvRaw')
    if (!d) { navigate('/'); return }
    const parsed = JSON.parse(d)
    setDs(parsed) // eslint-disable-line react-hooks/set-state-in-effect
    setCsvRaw(c || '')
    const numCols = parsed.columns
      .filter(col => col.dtype.includes('int') || col.dtype.includes('float'))
      .map(c => c.name)
    const catCols = parsed.columns
      .filter(col => !col.dtype.includes('int') && !col.dtype.includes('float'))
      .map(c => c.name)
    setTarget(catCols[0] || numCols[numCols.length - 1] || '')
    setFeatures(numCols)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const toggleFeature = col =>
    setFeatures(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col])
  const addLayer    = ()      => setLayers(prev => [...prev, 32])
  const removeLayer = idx     => setLayers(prev => prev.filter((_, i) => i !== idx))
  const updateLayer = (idx, v)=> setLayers(prev => prev.map((x, i) => i === idx ? Number(v) : x))

  async function doTrain(onEvent) {
    if (!target)               throw new Error('Select a target column.')
    if (features.length === 0) throw new Error('Select at least one feature.')
    if (features.includes(target)) throw new Error('Target cannot also be a feature.')
    if (!csvRaw)               throw new Error('CSV data missing. Please re-upload.')
    await trainNeuralStream(
      { problem_type: problemType, target, features, test_size: testSize,
        csv_data: csvRaw, hidden_layers: layers, activation, max_iter: maxIter,
        filename: sessionStorage.getItem('csvFile') || 'dataset.csv' },
      onEvent
    )
  }

  if (!ds) return null
  const allCols = ds.columns.map(c => c.name)
  const numCols = ds.columns
    .filter(c => c.dtype.includes('int') || c.dtype.includes('float'))
    .map(c => c.name)

  return (
    <div style={{ minHeight: '100vh' }}>
      <header style={headerStyle}>
        <Link to='/dataset-report' style={backLinkStyle}
          onMouseEnter={e => e.currentTarget.style.color = '#aaa'}
          onMouseLeave={e => e.currentTarget.style.color = '#555'}>← Back</Link>
        <div style={divStyle} />
        <span style={{ color: ACCENT, fontSize: 14, fontWeight: 600 }}>Neural Network</span>
        {results && <span style={doneStyle}>✓ Complete</span>}
      </header>

      <main style={{ maxWidth: 900, margin: '0 auto', padding: '40px 24px' }}>
        {!results ? (
          <PipelineTrainer
            onTrain={doTrain}
            onResult={r => setResults({
              ...r, target, features, test_size: testSize,
              hidden_layers: layers, activation, max_iter: maxIter,
            })}
            accent={ACCENT}
          >
            <div style={{ marginBottom: 28 }}>
              <h2 style={h2}>Configure Neural Network</h2>
              <p style={{ color: '#555', fontSize: 14 }}>
                Build a multilayer perceptron. Features are auto-scaled before training.
              </p>
            </div>

            {/* Problem type */}
            <Section label='1. Problem type'>
              <div style={{ display: 'flex', gap: 10 }}>
                {['classification', 'regression'].map(pt => (
                  <button key={pt} onClick={() => setProblemType(pt)} style={{
                    flex: 1, padding: '12px', borderRadius: 10, cursor: 'pointer',
                    background: problemType === pt ? 'rgba(245,158,11,0.1)' : 'rgba(255,255,255,0.01)',
                    border: `1px solid ${problemType === pt ? ACCENT : 'rgba(255,255,255,0.06)'}`,
                    color: problemType === pt ? ACCENT : '#777',
                    fontWeight: 600, fontSize: 13, textTransform: 'capitalize',
                    transition: 'all 0.2s',
                  }}>{pt}</button>
                ))}
              </div>
            </Section>

            {/* Target */}
            <Section label='2. Target column'>
              <select value={target} onChange={e => setTarget(e.target.value)} style={selectStyle}>
                {allCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
              <p style={{ fontSize: 11, color: '#444', marginTop: 6 }}>
                {problemType === 'classification'
                  ? 'Pick a categorical column (e.g. "species").'
                  : 'Pick a numeric column (e.g. "price").'}
              </p>
            </Section>

            {/* Features */}
            <Section label='3. Feature columns'>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {numCols.map(c => (
                  <button key={c} onClick={() => toggleFeature(c)} style={{
                    padding: '6px 12px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
                    background: features.includes(c) ? 'rgba(245,158,11,0.12)' : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${features.includes(c) ? ACCENT : 'rgba(255,255,255,0.07)'}`,
                    color: features.includes(c) ? ACCENT : '#666', transition: 'all 0.15s',
                  }}>{c}</button>
                ))}
              </div>
              <p style={{ fontSize: 11, color: '#444', marginTop: 8 }}>
                {features.length} selected · All features are StandardScaled before training
              </p>
            </Section>

            {/* Architecture */}
            <Section label='4. Network architecture'>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 14 }}>
                {layers.map((neurons, idx) => (
                  <div key={idx} style={{
                    display: 'flex', alignItems: 'center', gap: 12,
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    borderRadius: 10, padding: '10px 14px',
                  }}>
                    <span style={{ fontSize: 11, color: '#555', minWidth: 56 }}>Layer {idx + 1}</span>
                    <input type='range' min={4} max={256} step={4} value={neurons}
                      onChange={e => updateLayer(idx, e.target.value)}
                      style={{ flex: 1, accentColor: ACCENT }} />
                    <span style={{ fontSize: 15, fontWeight: 700, color: ACCENT,
                      fontFamily: 'monospace', minWidth: 40, textAlign: 'right' }}>{neurons}</span>
                    <span style={{ fontSize: 11, color: '#444' }}>neurons</span>
                    {layers.length > 1 && (
                      <button onClick={() => removeLayer(idx)} style={{
                        background: 'rgba(239,68,68,0.08)',
                        border: '1px solid rgba(239,68,68,0.2)',
                        borderRadius: 6, color: '#f87171', fontSize: 12,
                        padding: '3px 8px', cursor: 'pointer',
                      }}>✕</button>
                    )}
                  </div>
                ))}
              </div>

              {/* Architecture diagram */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6,
                overflowX: 'auto', padding: '10px 0', marginBottom: 12 }}>
                <LayerPill label={`In\n${features.length}`} color='#555' />
                {layers.map((n, i) => (
                  <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Arrow />
                    <LayerPill label={`H${i+1}\n${n}`} color={ACCENT} />
                  </span>
                ))}
                <Arrow />
                <LayerPill label='Out' color='#6c63ff' />
              </div>

              {layers.length < 6 && (
                <button onClick={addLayer} style={{
                  padding: '7px 14px', background: 'transparent',
                  border: '1px dashed rgba(245,158,11,0.3)',
                  borderRadius: 8, color: '#777', fontSize: 12, cursor: 'pointer',
                }}>+ Add hidden layer</button>
              )}
            </Section>

            {/* Activation */}
            <Section label='5. Activation function'>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {ACTIVATIONS.map(a => (
                  <button key={a.id} onClick={() => setActivation(a.id)} style={{
                    background: activation === a.id ? 'rgba(245,158,11,0.08)' : 'rgba(255,255,255,0.01)',
                    border: `1px solid ${activation === a.id ? ACCENT : 'rgba(255,255,255,0.06)'}`,
                    borderRadius: 10, padding: '10px 14px', cursor: 'pointer', textAlign: 'left',
                    transition: 'all 0.2s',
                  }}>
                    <div style={{ fontWeight: 600, fontSize: 13,
                      color: activation === a.id ? ACCENT : '#e0e0e0', marginBottom: 2 }}>{a.label}</div>
                    <div style={{ fontSize: 11, color: '#555', marginBottom: 2 }}>{a.tip}</div>
                    <div style={{ fontSize: 11,
                      color: activation === a.id ? '#b45309' : '#2a2a3a' }}>{a.rec}</div>
                  </button>
                ))}
              </div>
            </Section>

            {/* Max iterations */}
            <Section label={`6. Max iterations — ${maxIter}`}>
              <input type='range' min={50} max={1000} step={50} value={maxIter}
                onChange={e => setMaxIter(Number(e.target.value))}
                style={{ width: '100%', accentColor: ACCENT }} />
              <div style={{ display: 'flex', justifyContent: 'space-between',
                fontSize: 11, color: '#444', marginTop: 4 }}>
                <span>50 — fast, may not converge</span>
                <span>1000 — thorough, slower</span>
              </div>
              <p style={{ fontSize: 11, color: '#444', marginTop: 6 }}>
                Early stopping enabled — training halts when validation stops improving.
              </p>
            </Section>

            {/* Test split */}
            <Section label={`7. Test split — ${Math.round(testSize * 100)}%`}>
              <input type='range' min={10} max={40} value={testSize * 100}
                onChange={e => setTestSize(Number(e.target.value) / 100)}
                style={{ width: '100%', accentColor: ACCENT }} />
              <div style={{ display: 'flex', justifyContent: 'space-between',
                fontSize: 11, color: '#444', marginTop: 4 }}>
                <span>10% — more training data</span>
                <span>40% — more reliable test</span>
              </div>
            </Section>
          </PipelineTrainer>
        ) : (
          <NeuralResults results={results} onReset={() => setResults(null)} />
        )}
      </main>
    </div>
  )
}

function NeuralResults({ results, onReset }) {
  const [tab, setTab] = useState('metrics')
  const isClf = results.problem_type === 'classification'

  const classMetrics = [
    { label: 'Test Accuracy',  value: results.accuracy + '%',       color: '#f59e0b', desc: 'Correct predictions on test set' },
    { label: 'Train Accuracy', value: results.train_accuracy + '%', color: '#b45309', desc: 'Correct predictions on training set' },
    { label: 'F1 Score',       value: results.f1 + '%',             color: '#6c63ff', desc: 'Harmonic mean of precision and recall' },
    { label: 'Precision',      value: results.precision + '%',      color: '#10b981', desc: 'Of predicted positives, how many correct' },
    { label: 'Recall',         value: results.recall + '%',         color: '#0ea5e9', desc: 'Of actual positives, how many caught' },
    { label: 'ROC-AUC',        value: results.roc_auc ?? 'N/A',    color: '#a78bfa', desc: 'Area under ROC curve' },
    { label: 'Iterations',     value: results.n_iter,               color: '#fb923c', desc: 'Epochs until early stopping' },
    { label: 'Fit Status',     value: results.fit_status || 'good', color: results.fit_status === 'overfitting' ? '#f59e0b' : '#10b981', desc: 'Overfitting check' },
  ]

  const regMetrics = [
    { label: 'MAE',         value: results.mae,           color: '#f59e0b', desc: 'Average prediction error' },
    { label: 'RMSE',        value: results.rmse,          color: '#0ea5e9', desc: 'Root mean squared error' },
    { label: 'R² Score',    value: results.r2,            color: '#10b981', desc: '1.0 = perfect, 0 = no better than mean' },
    { label: 'Train R²',    value: results.train_r2,      color: '#059669', desc: 'R² on training data' },
    { label: 'MAPE',        value: results.mape + '%',    color: '#fb923c', desc: 'Mean Absolute Percentage Error' },
    { label: 'Median AE',   value: results.median_ae,     color: '#a78bfa', desc: 'Median absolute error (robust to outliers)' },
    { label: 'Iterations',  value: results.n_iter,        color: '#38bdf8', desc: 'Epochs until early stopping' },
    { label: 'Fit Status',  value: results.fit_status || 'good', color: results.fit_status === 'overfitting' ? '#f59e0b' : '#10b981', desc: 'Overfitting check' },
  ]

  const metrics = isClf ? classMetrics : regMetrics

  const lossCurveData = (results.loss_curve || []).map((v, i) => ({
    epoch: i + 1, loss: v,
    val: results.val_curve?.[i] ?? null,
  }))

  const tabs = [
    ['metrics', 'All Metrics'],
    ['loss',    'Loss Curve'],
    isClf ? ['matrix', 'Confusion Matrix'] : ['scatter', 'Predictions'],
  ]

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
        marginBottom: 24, gap: 16, flexWrap: 'wrap' }}>
        <div>
          <h2 style={h2}>Neural Network Results</h2>
          <p style={{ color: '#555', fontSize: 13 }}>
            {results.n_train} train · {results.n_test} test · {results.n_iter} iterations
          </p>
        </div>
        <button onClick={onReset} style={reconfigBtn}>← Reconfigure</button>
      </div>

      {results.fit_warning && (
        <div style={{ padding: '10px 14px', background: 'rgba(245,158,11,0.07)',
          border: '1px solid rgba(245,158,11,0.25)', borderRadius: 10, fontSize: 13,
          color: '#fbbf24', marginBottom: 16, display: 'flex', gap: 10 }}>
          <span>⚠</span><span>{results.fit_warning}</span>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20,
        borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: 12 }}>
        {tabs.map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)} style={{
            padding: '6px 14px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
            background: tab === id ? 'rgba(245,158,11,0.15)' : 'transparent',
            border: `1px solid ${tab === id ? ACCENT : 'transparent'}`,
            color: tab === id ? ACCENT : '#666', transition: 'all 0.2s',
          }}>{label}</button>
        ))}
      </div>

      {/* All Metrics */}
      {tab === 'metrics' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 24 }}>
          {metrics.map(m => (
            <div key={m.label} title={m.desc} style={{
              background: 'rgba(255,255,255,0.02)', border: `1px solid ${m.color}20`,
              borderRadius: 12, padding: '14px 12px', textAlign: 'center', cursor: 'help',
            }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: m.color,
                fontFamily: 'monospace', marginBottom: 4 }}>{m.value}</div>
              <div style={{ fontSize: 11, color: '#777', marginBottom: 2 }}>{m.label}</div>
              <div style={{ fontSize: 9, color: '#444', lineHeight: 1.4 }}>{m.desc}</div>
            </div>
          ))}
        </div>
      )}

      {/* Loss Curve */}
      {tab === 'loss' && lossCurveData.length > 0 && (
        <div style={card}>
          <p style={cardLabel}>Training & validation loss curve</p>
          <p style={{ fontSize: 11, color: '#444', marginBottom: 16 }}>
            Loss should decrease and flatten.
            Rising validation loss while training loss falls = overfitting.
          </p>
          <ResponsiveContainer width='100%' height={280}>
            <LineChart data={lossCurveData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray='3 3' stroke='rgba(255,255,255,0.04)' />
              <XAxis dataKey='epoch' tick={{ fill: '#555', fontSize: 11 }}
                label={{ value: 'Epoch', position: 'insideBottom', offset: -10, fill: '#555', fontSize: 11 }} />
              <YAxis tick={{ fill: '#555', fontSize: 11 }} />
              <RTooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12, color: '#888', paddingTop: 8 }} />
              <Line type='monotone' dataKey='loss' name='Train loss'
                stroke={ACCENT} strokeWidth={2} dot={false} />
              {lossCurveData.some(d => d.val !== null) && (
                <Line type='monotone' dataKey='val' name='Val accuracy'
                  stroke='#10b981' strokeWidth={2} dot={false} strokeDasharray='4 4' />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Confusion Matrix */}
      {tab === 'matrix' && isClf && results.confusion_matrix && (
        <div style={card}>
          <p style={cardLabel}>Confusion matrix</p>
          <p style={{ fontSize: 11, color: '#444', marginBottom: 16 }}>
            Rows = actual · Columns = predicted.
            <strong style={{ color: '#fbbf24' }}> Yellow diagonal</strong> = correct.
            <strong style={{ color: '#f87171' }}> Red</strong> = mistakes.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', fontSize: 13, fontFamily: 'monospace' }}>
              <thead>
                <tr>
                  <th style={{ padding: '8px 14px', color: '#444', fontSize: 11, textAlign: 'left' }}>
                    Act ↓ / Pred →
                  </th>
                  {results.labels.map(l => (
                    <th key={l} style={{ padding: '8px 14px', color: '#777', fontSize: 11 }}>{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.confusion_matrix.map((row, i) => (
                  <tr key={i}>
                    <td style={{ padding: '8px 14px', color: '#888', fontWeight: 600 }}>
                      {results.labels[i]}
                    </td>
                    {row.map((val, j) => (
                      <td key={j} style={{
                        padding: '12px 24px', textAlign: 'center', borderRadius: 6,
                        background: i === j ? 'rgba(245,158,11,0.2)'
                          : val > 0 ? 'rgba(239,68,68,0.12)' : 'transparent',
                        color: i === j ? '#fbbf24' : val > 0 ? '#f87171' : '#444',
                        fontWeight: i === j ? 700 : 400, fontSize: 16,
                      }}>{val}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Predictions scatter (regression) */}
      {tab === 'scatter' && !isClf && results.scatter && (
        <div style={card}>
          <p style={cardLabel}>Predicted vs actual</p>
          <p style={{ fontSize: 11, color: '#444', marginBottom: 16 }}>
            Each dot = one test row. Dots on the diagonal = perfect predictions.
          </p>
          <ResponsiveContainer width='100%' height={300}>
            <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray='3 3' stroke='rgba(255,255,255,0.04)' />
              <XAxis dataKey='actual' name='Actual' tick={{ fill: '#666', fontSize: 11 }}
                label={{ value: 'Actual', position: 'insideBottom', offset: -10, fill: '#555', fontSize: 11 }} />
              <YAxis dataKey='predicted' name='Predicted' tick={{ fill: '#666', fontSize: 11 }} />
              <RTooltip contentStyle={tooltipStyle} />
              <Scatter data={results.scatter} fill={ACCENT} opacity={0.7} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      <CodeExport payload={{
        model_type:    'neural',
        model_name:    'mlp',
        problem_type:  results.problem_type,
        target:        results.target        || '',
        features:      results.features      || [],
        test_size:     results.test_size      || 0.2,
        hidden_layers: results.hidden_layers  || [64, 32],
        activation:    results.activation     || 'relu',
        max_iter:      results.max_iter        || 200,
      }} />
    </div>
  )
}

/* ── small reusable components ── */
function LayerPill({ label, color }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.02)', border: `1px solid ${color}`,
      borderRadius: 8, padding: '6px 10px', textAlign: 'center',
      minWidth: 48, flexShrink: 0,
    }}>
      {label.split('\n').map((line, i) => (
        <div key={i} style={{
          fontSize: i === 0 ? 9 : 12,
          color: i === 0 ? '#555' : color,
          fontFamily: 'monospace', fontWeight: i === 1 ? 700 : 400,
        }}>{line}</div>
      ))}
    </div>
  )
}

function Arrow() {
  return <span style={{ color: '#2a2a3a', fontSize: 16 }}>→</span>
}

function Section({ label, children }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <p style={{ fontSize: 11, color: '#666', textTransform: 'uppercase',
        letterSpacing: '0.08em', fontWeight: 600, marginBottom: 12 }}>{label}</p>
      {children}
    </div>
  )
}

/* ── shared styles ── */
const headerStyle  = { borderBottom: '1px solid rgba(255,255,255,0.05)', padding: '16px 40px', display: 'flex', alignItems: 'center', gap: 12, background: 'rgba(2,2,8,0.85)', backdropFilter: 'blur(20px)', position: 'sticky', top: 0, zIndex: 50 }
const backLinkStyle= { color: '#555', fontSize: 13, textDecoration: 'none', transition: 'color 0.2s' }
const divStyle     = { width: 1, height: 16, background: 'rgba(255,255,255,0.08)' }
const doneStyle    = { marginLeft: 'auto', fontSize: 12, color: '#10b981', fontFamily: 'monospace' }
const h2           = { fontSize: 22, fontWeight: 700, letterSpacing: '-0.02em', marginBottom: 4 }
const reconfigBtn  = { padding: '8px 16px', background: 'transparent', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#888', fontSize: 13, cursor: 'pointer' }
const selectStyle  = { background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, color: '#e0e0e0', padding: '8px 12px', fontSize: 13, cursor: 'pointer', width: '100%', outline: 'none' }
const card         = { background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 14, padding: '20px', marginBottom: 20 }
const cardLabel    = { fontSize: 11, color: '#555', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6, fontWeight: 600 }
const tooltipStyle = { background: '#0d0d18', border: '1px solid rgba(245,158,11,0.3)', borderRadius: 8, color: '#f0f0f0', fontSize: 12 }