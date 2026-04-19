import CodeExport from '../components/CodeExport.jsx'
import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { trainNeural } from '../api/client'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell,
} from 'recharts'

const ACCENT = '#f59e0b'

const ACTIVATIONS = [
  { id: 'relu',     label: 'ReLU',    tip: 'Best default. Fast and avoids vanishing gradients.' },
  { id: 'tanh',     label: 'Tanh',    tip: 'Outputs between −1 and 1. Good for shallow nets.' },
  { id: 'logistic', label: 'Sigmoid', tip: 'Classic activation. Can suffer vanishing gradients.' },
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
  const [loading, setLoading]         = useState(false)
  const [results, setResults]         = useState(null)
  const [error, setError]             = useState('')
  const [csvRaw, setCsvRaw]           = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    const d = sessionStorage.getItem('dataset')
    const c = sessionStorage.getItem('csvRaw')
    if (!d) { navigate('/'); return }
    const parsed = JSON.parse(d)
    setDs(parsed)
    setCsvRaw(c || '')
    const numCols = parsed.columns.filter(col => col.dtype.includes('int') || col.dtype.includes('float')).map(c => c.name)
    const catCols = parsed.columns.filter(col => !col.dtype.includes('int') && !col.dtype.includes('float')).map(c => c.name)
    setTarget(catCols[0] || numCols[numCols.length - 1] || '')
    setFeatures(numCols)
  }, [navigate])

  function toggleFeature(col) {
    setFeatures(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col])
  }
  function addLayer()            { setLayers(prev => [...prev, 32]) }
  function removeLayer(idx)      { setLayers(prev => prev.filter((_, i) => i !== idx)) }
  function updateLayer(idx, val) { setLayers(prev => prev.map((v, i) => i === idx ? Number(val) : v)) }

  async function handleTrain() {
    if (!target)               { setError('Select a target column.'); return }
    if (features.length === 0) { setError('Select at least one feature.'); return }
    if (features.includes(target)) { setError('Target cannot also be a feature.'); return }
    if (!csvRaw)               { setError('CSV data missing. Please re-upload.'); return }
    setError('')
    setLoading(true)
    try {
      const res = await trainNeural({ problem_type: problemType, target, features, test_size: testSize, csv_data: csvRaw, hidden_layers: layers, activation, max_iter: maxIter })
      // ── DAY 5: store model config alongside results ──
      setResults({ ...res, target, features, test_size: testSize, hidden_layers: layers, activation, max_iter: maxIter })
    } catch(e) { setError(e.message) }
    finally { setLoading(false) }
  }

  if (!ds) return null
  const allCols = ds.columns.map(c => c.name)
  const numCols = ds.columns.filter(c => c.dtype.includes('int') || c.dtype.includes('float')).map(c => c.name)

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0f' }}>
      <header style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', padding: '16px 40px', display: 'flex', alignItems: 'center', gap: 12 }}>
        <Link to='/visualise' style={{ color: '#555', fontSize: 13, textDecoration: 'none' }}>← Back</Link>
        <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)' }} />
        <span style={{ color: ACCENT, fontSize: 14, fontWeight: 500 }}>Neural Network</span>
      </header>

      <main style={{ maxWidth: 860, margin: '0 auto', padding: '40px 24px' }}>
        {!results ? (
          <div>
            <h2 style={{ fontSize: 22, fontWeight: 600, color: '#f0f0f0', marginBottom: 4 }}>Configure MLP</h2>
            <p style={{ color: '#555', fontSize: 14, marginBottom: 28 }}>Build a multilayer perceptron for classification or regression.</p>

            <Section label='1. Problem type'>
              <div style={{ display: 'flex', gap: 10 }}>
                {['classification', 'regression'].map(pt => (
                  <button key={pt} onClick={() => setProblemType(pt)} style={{
                    flex: 1, padding: '12px', borderRadius: 10, cursor: 'pointer',
                    background: problemType === pt ? 'rgba(245,158,11,0.12)' : '#13131a',
                    border: `1px solid ${problemType === pt ? ACCENT : 'rgba(255,255,255,0.07)'}`,
                    color: problemType === pt ? ACCENT : '#777',
                    fontWeight: 600, fontSize: 13, textTransform: 'capitalize', transition: 'all 0.2s',
                  }}>{pt}</button>
                ))}
              </div>
            </Section>

            <Section label='2. Target column'>
              <select value={target} onChange={e => setTarget(e.target.value)} style={selectStyle}>
                {allCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
              <p style={{ fontSize: 11, color: '#444', marginTop: 6 }}>
                {problemType === 'classification' ? 'Should be a categorical column.' : 'Should be a numeric column.'}
              </p>
            </Section>

            <Section label='3. Feature columns (inputs)'>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {numCols.map(c => (
                  <button key={c} onClick={() => toggleFeature(c)} style={{
                    padding: '6px 12px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
                    background: features.includes(c) ? 'rgba(245,158,11,0.15)' : '#13131a',
                    border: `1px solid ${features.includes(c) ? ACCENT : 'rgba(255,255,255,0.08)'}`,
                    color: features.includes(c) ? ACCENT : '#666', transition: 'all 0.15s',
                  }}>{c}</button>
                ))}
              </div>
              <p style={{ fontSize: 11, color: '#444', marginTop: 8 }}>{features.length} selected</p>
            </Section>

            <Section label='4. Hidden layers'>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
                {layers.map((neurons, idx) => (
                  <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 12, background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10, padding: '10px 14px' }}>
                    <span style={{ fontSize: 11, color: '#555', minWidth: 56 }}>Layer {idx + 1}</span>
                    <input type='range' min={4} max={256} step={4} value={neurons} onChange={e => updateLayer(idx, e.target.value)} style={{ flex: 1, accentColor: ACCENT }} />
                    <span style={{ fontSize: 15, fontWeight: 600, color: ACCENT, fontFamily: 'monospace', minWidth: 40, textAlign: 'right' }}>{neurons}</span>
                    <span style={{ fontSize: 11, color: '#555', minWidth: 50 }}>neurons</span>
                    {layers.length > 1 && (
                      <button onClick={() => removeLayer(idx)} style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 6, color: '#f87171', fontSize: 12, padding: '4px 8px', cursor: 'pointer' }}>✕</button>
                    )}
                  </div>
                ))}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12, overflowX: 'auto', padding: '8px 0' }}>
                <LayerPill label={`In\n(${features.length})`} color='#555' />
                {layers.map((n, i) => (
                  <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Arrow />
                    <LayerPill label={`H${i+1}\n(${n})`} color={ACCENT} />
                  </span>
                ))}
                <Arrow />
                <LayerPill label='Out' color='#6c63ff' />
              </div>
              {layers.length < 6 && (
                <button onClick={addLayer} style={{ padding: '8px 16px', background: 'transparent', border: '1px dashed rgba(245,158,11,0.3)', borderRadius: 8, color: '#888', fontSize: 12, cursor: 'pointer' }}>+ Add layer</button>
              )}
            </Section>

            <Section label='5. Activation function'>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {ACTIVATIONS.map(a => (
                  <button key={a.id} onClick={() => setActivation(a.id)} style={{
                    background: activation === a.id ? 'rgba(245,158,11,0.1)' : '#13131a',
                    border: `1px solid ${activation === a.id ? ACCENT : 'rgba(255,255,255,0.07)'}`,
                    borderRadius: 10, padding: '10px 16px', cursor: 'pointer', textAlign: 'left',
                  }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: activation === a.id ? ACCENT : '#e0e0e0', marginBottom: 2 }}>{a.label}</div>
                    <div style={{ fontSize: 11, color: '#555' }}>{a.tip}</div>
                  </button>
                ))}
              </div>
            </Section>

            <Section label={`6. Max iterations — ${maxIter}`}>
              <input type='range' min={50} max={1000} step={50} value={maxIter} onChange={e => setMaxIter(Number(e.target.value))} style={{ width: '100%', accentColor: ACCENT }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#555', marginTop: 4 }}>
                <span>50 (fast)</span><span>1000 (thorough)</span>
              </div>
              <p style={{ fontSize: 11, color: '#444', marginTop: 6 }}>Early stopping is enabled — training may finish before this limit.</p>
            </Section>

            <Section label={`7. Test split — ${Math.round(testSize * 100)}% held back`}>
              <input type='range' min={10} max={40} value={testSize * 100} onChange={e => setTestSize(Number(e.target.value) / 100)} style={{ width: '100%', accentColor: ACCENT }} />
            </Section>

            {error && <div style={{ padding: '10px 14px', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 8, fontSize: 13, color: '#f87171', marginBottom: 16 }}>{error}</div>}

            <button onClick={handleTrain} disabled={loading} style={{
              width: '100%', padding: '14px', background: loading ? '#333' : ACCENT,
              border: 'none', borderRadius: 10, color: '#fff', fontSize: 15, fontWeight: 600, cursor: loading ? 'default' : 'pointer', transition: 'all 0.2s',
            }}>
              {loading ? 'Training...' : 'Train neural network'}
            </button>
          </div>
        ) : (
          <NeuralResults results={results} onReset={() => setResults(null)} />
        )}
      </main>
    </div>
  )
}

function NeuralResults({ results, onReset }) {
  const isClassification = results.problem_type === 'classification'
  const metrics = isClassification ? [
    { label: 'Accuracy',  value: results.accuracy  + '%', color: '#f59e0b' },
    { label: 'F1 Score',  value: results.f1        + '%', color: '#6c63ff' },
    { label: 'Precision', value: results.precision + '%', color: '#10b981' },
    { label: 'Recall',    value: results.recall    + '%', color: '#0ea5e9' },
  ] : [
    { label: 'MAE',  value: results.mae,  color: '#f59e0b' },
    { label: 'RMSE', value: results.rmse, color: '#0ea5e9' },
    { label: 'R²',   value: results.r2,   color: '#10b981' },
    { label: 'MSE',  value: results.mse,  color: '#6c63ff' },
  ]
  const lossCurveData = (results.loss_curve || []).map((v, i) => ({ epoch: i + 1, loss: v }))

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 22, fontWeight: 600, color: '#f0f0f0', marginBottom: 4 }}>Results</h2>
          <p style={{ color: '#555', fontSize: 13 }}>{results.n_train} training rows · {results.n_test} test rows · {results.n_iter} iterations</p>
        </div>
        <button onClick={onReset} style={{ padding: '8px 16px', background: 'transparent', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#888', fontSize: 13, cursor: 'pointer' }}>← Reconfigure</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 24 }}>
        {metrics.map(m => (
          <div key={m.label} style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10, padding: '16px 14px', textAlign: 'center' }}>
            <div style={{ fontSize: 22, fontWeight: 600, color: m.color, fontFamily: 'monospace', marginBottom: 4 }}>{m.value}</div>
            <div style={{ fontSize: 12, color: '#555' }}>{m.label}</div>
          </div>
        ))}
      </div>

      {lossCurveData.length > 0 && (
        <div style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: '20px', marginBottom: 20 }}>
          <p style={{ fontSize: 12, color: '#555', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>Training loss curve</p>
          <ResponsiveContainer width='100%' height={220}>
            <LineChart data={lossCurveData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray='3 3' stroke='rgba(255,255,255,0.04)' />
              <XAxis dataKey='epoch' tick={{ fill: '#555', fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#555', fontSize: 11 }} />
              <YAxis tick={{ fill: '#555', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#13131a', border: '1px solid #333', borderRadius: 8, color: '#f0f0f0' }} />
              <Line type='monotone' dataKey='loss' stroke='#f59e0b' strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <p style={{ fontSize: 11, color: '#444', marginTop: 10 }}>Loss should decrease and flatten as the network learns.</p>
        </div>
      )}

      {isClassification && results.confusion_matrix && (
        <div style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: '20px', marginBottom: 20 }}>
          <p style={{ fontSize: 12, color: '#555', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>Confusion matrix</p>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', fontSize: 13, fontFamily: 'monospace' }}>
              <thead><tr>
                <th style={{ padding: '8px 12px', color: '#444', fontSize: 11 }}></th>
                {results.labels.map(l => <th key={l} style={{ padding: '8px 12px', color: '#888', fontSize: 11 }}>Pred: {l}</th>)}
              </tr></thead>
              <tbody>
                {results.confusion_matrix.map((row, i) => (
                  <tr key={i}>
                    <td style={{ padding: '8px 12px', color: '#888', fontSize: 11 }}>Act: {results.labels[i]}</td>
                    {row.map((val, j) => (
                      <td key={j} style={{
                        padding: '10px 20px', textAlign: 'center', borderRadius: 4,
                        background: i === j ? 'rgba(245,158,11,0.2)' : val > 0 ? 'rgba(239,68,68,0.12)' : 'transparent',
                        color: i === j ? '#fbbf24' : val > 0 ? '#f87171' : '#555',
                        fontWeight: i === j ? 600 : 400,
                      }}>{val}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!isClassification && results.scatter && results.scatter.length > 0 && (
        <div style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: '20px', marginBottom: 20 }}>
          <p style={{ fontSize: 12, color: '#555', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>Predicted vs actual</p>
          <ResponsiveContainer width='100%' height={280}>
            <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray='3 3' stroke='rgba(255,255,255,0.05)' />
              <XAxis dataKey='actual'    name='Actual'    tick={{ fill: '#666', fontSize: 11 }} />
              <YAxis dataKey='predicted' name='Predicted' tick={{ fill: '#666', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#13131a', border: '1px solid #333', borderRadius: 8, color: '#f0f0f0' }} />
              <Scatter data={results.scatter} fill='#f59e0b' opacity={0.7} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── DAY 5: Code export ── */}
      <CodeExport payload={{
        model_type: 'neural',
        model_name: 'mlp',
        problem_type: results.problem_type,
        target: results.target || '',
        features: results.features || [],
        test_size: results.test_size || 0.2,
        hidden_layers: results.hidden_layers || [64, 32],
        activation: results.activation || 'relu',
        max_iter: results.max_iter || 200,
      }} />
    </div>
  )
}

function LayerPill({ label, color }) {
  return (
    <div style={{ background: '#13131a', border: `1px solid ${color}`, borderRadius: 8, padding: '6px 10px', textAlign: 'center', minWidth: 48, flexShrink: 0 }}>
      {label.split('\n').map((line, i) => (
        <div key={i} style={{ fontSize: i === 0 ? 10 : 12, color: i === 0 ? '#666' : color, fontFamily: 'monospace', fontWeight: i === 1 ? 600 : 400 }}>{line}</div>
      ))}
    </div>
  )
}

function Arrow() {
  return <span style={{ color: '#333', fontSize: 16 }}>→</span>
}

function Section({ label, children }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <p style={{ fontSize: 12, color: '#666', textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 500, marginBottom: 12 }}>{label}</p>
      {children}
    </div>
  )
}

const selectStyle = {
  background: '#13131a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8,
  color: '#e0e0e0', padding: '8px 12px', fontSize: 13, cursor: 'pointer', width: '100%',
}