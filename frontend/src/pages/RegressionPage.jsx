import CodeExport from '../components/CodeExport.jsx'
import Tooltip from '../components/Tooltip.jsx'
import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { trainRegressor } from '../api/client'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip as RechartTooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'

const MODELS = [
  { id: 'linear_regression', label: 'Linear Regression',      tip: 'Fast baseline. Works well when relationship is linear.',            rec: 'Best for: simple problems, good starting point.' },
  { id: 'ridge',             label: 'Ridge Regression',        tip: 'Linear with regularisation. Prevents overfitting.',                 rec: 'Best for: when features are correlated with each other.' },
  { id: 'lasso',             label: 'Lasso Regression',        tip: 'Shrinks unimportant features to zero.',                            rec: 'Best for: when you want automatic feature selection.' },
  { id: 'decision_tree',     label: 'Decision Tree Regressor', tip: 'Non-linear. Can overfit — use with caution.',                      rec: 'Best for: non-linear data, but watch for overfitting.' },
  { id: 'random_forest',     label: 'Random Forest Regressor', tip: 'Most robust option. Good default for regression.',                  rec: 'Best for: most regression problems. Good default.' },
]

const ACCENT = '#0ea5e9'

const HYPERPARAM_TIPS = {
  target:    'The numeric column your model will predict (e.g. "price", "tip", "temperature").',
  features:  'The input columns used to make predictions. All must be numeric.',
  test_size: 'Percentage of rows held back for testing. 20% is a standard choice.',
}

export default function RegressionPage() {
  const [ds, setDs]             = useState(null)
  const [model, setModel]       = useState('random_forest')
  const [target, setTarget]     = useState('')
  const [features, setFeatures] = useState([])
  const [testSize, setTestSize] = useState(0.2)
  const [loading, setLoading]   = useState(false)
  const [results, setResults]   = useState(null)
  const [error, setError]       = useState('')
  const [csvRaw, setCsvRaw]     = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    const d = sessionStorage.getItem('dataset')
    const c = sessionStorage.getItem('csvRaw')
    if (!d) { navigate('/'); return }
    const parsed = JSON.parse(d)
    setDs(parsed)
    setCsvRaw(c || '')
    const numCols = parsed.columns.filter(col => col.dtype.includes('int') || col.dtype.includes('float')).map(c => c.name)
    setTarget(numCols[numCols.length - 1] || '')
    setFeatures(numCols.slice(0, -1))
  }, [navigate])

  function toggleFeature(col) {
    setFeatures(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col])
  }

  async function handleTrain() {
    if (!target)               { setError('Select a target column.'); return }
    if (features.length === 0) { setError('Select at least one feature column.'); return }
    if (features.includes(target)) { setError('Target column cannot also be a feature.'); return }
    if (!csvRaw)               { setError('CSV data missing. Please go back and re-upload your file.'); return }
    setError('')
    setLoading(true)
    try {
      const res = await trainRegressor({ model_name: model, target, features, test_size: testSize, csv_data: csvRaw })
      setResults({ ...res, model_name: model, features, test_size: testSize, target })
    } catch(e) { setError(e.message) }
    finally { setLoading(false) }
  }

  if (!ds) return null
  const numCols = ds.columns.filter(c => c.dtype.includes('int') || c.dtype.includes('float')).map(c => c.name)

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0f' }}>
      <header style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', padding: '16px 40px', display: 'flex', alignItems: 'center', gap: 12 }}>
        <Link to='/visualise' style={{ color: '#555', fontSize: 13, textDecoration: 'none' }}>← Back</Link>
        <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)' }} />
        <span style={{ color: ACCENT, fontSize: 14, fontWeight: 500 }}>Regression</span>
      </header>

      <main style={{ maxWidth: 860, margin: '0 auto', padding: '40px 24px' }}>
        {!results ? (
          <div>
            <h2 style={{ fontSize: 22, fontWeight: 600, color: '#f0f0f0', marginBottom: 4 }}>Configure model</h2>
            <p style={{ color: '#555', fontSize: 14, marginBottom: 28 }}>Set up your regression model.</p>

            <Section label='1. Pick algorithm'>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {MODELS.map(m => (
                  <button key={m.id} onClick={() => setModel(m.id)} style={{
                    background: model === m.id ? 'rgba(14,165,233,0.1)' : '#13131a',
                    border: `1px solid ${model === m.id ? ACCENT : 'rgba(255,255,255,0.07)'}`,
                    borderRadius: 10, padding: '12px 16px', cursor: 'pointer', textAlign: 'left',
                  }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: model === m.id ? ACCENT : '#e0e0e0', marginBottom: 3 }}>{m.label}</div>
                    <div style={{ fontSize: 11, color: '#555', marginBottom: 4 }}>{m.tip}</div>
                    <div style={{ fontSize: 11, color: model === m.id ? '#3b8fb5' : '#3a3a4a' }}>{m.rec}</div>
                  </button>
                ))}
              </div>
            </Section>

            <Section label='2. Target column' tooltip={HYPERPARAM_TIPS.target}>
              <select value={target} onChange={e => setTarget(e.target.value)} style={selectStyle}>
                {numCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </Section>

            <Section label='3. Feature columns' tooltip={HYPERPARAM_TIPS.features}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {numCols.map(c => (
                  <button key={c} onClick={() => toggleFeature(c)} style={{
                    padding: '6px 12px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
                    background: features.includes(c) ? 'rgba(14,165,233,0.15)' : '#13131a',
                    border: `1px solid ${features.includes(c) ? ACCENT : 'rgba(255,255,255,0.08)'}`,
                    color: features.includes(c) ? ACCENT : '#666',
                  }}>{c}</button>
                ))}
              </div>
            </Section>

            <Section label={`4. Test split — ${Math.round(testSize * 100)}%`} tooltip={HYPERPARAM_TIPS.test_size}>
              <input type='range' min={10} max={40} value={testSize * 100}
                onChange={e => setTestSize(Number(e.target.value) / 100)}
                style={{ width: '100%', accentColor: ACCENT }} />
            </Section>

            {error && <ErrorBox message={error} />}

            <button onClick={handleTrain} disabled={loading} style={{
              width: '100%', padding: '14px', background: loading ? '#333' : ACCENT,
              border: 'none', borderRadius: 10, color: '#fff', fontSize: 15, fontWeight: 600, cursor: 'pointer',
            }}>
              {loading ? 'Training...' : 'Train model'}
            </button>
          </div>
        ) : (
          <RegressionResults results={results} onReset={() => setResults(null)} />
        )}
      </main>
    </div>
  )
}

function RegressionResults({ results, onReset }) {
  const metrics = [
    { label: 'MAE',  value: results.mae,  tip: 'Mean Absolute Error' },
    { label: 'RMSE', value: results.rmse, tip: 'Root Mean Squared Error' },
    { label: 'R²',   value: results.r2,   tip: '1.0 is perfect' },
    { label: 'MSE',  value: results.mse,  tip: 'Mean Squared Error' },
  ]
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 22, fontWeight: 600, color: '#f0f0f0', marginBottom: 4 }}>Results</h2>
          <p style={{ color: '#555', fontSize: 13 }}>{results.n_train} training rows · {results.n_test} test rows</p>
        </div>
        <button onClick={onReset} style={{ padding: '8px 16px', background: 'transparent', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#888', fontSize: 13, cursor: 'pointer' }}>← Reconfigure</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 24 }}>
        {metrics.map(m => (
          <div key={m.label} title={m.tip} style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10, padding: '16px 14px', textAlign: 'center', cursor: 'help' }}>
            <div style={{ fontSize: 22, fontWeight: 600, color: '#0ea5e9', fontFamily: 'monospace', marginBottom: 4 }}>{m.value}</div>
            <div style={{ fontSize: 12, color: '#555' }}>{m.label}</div>
          </div>
        ))}
      </div>

      {results.scatter && results.scatter.length > 0 && (
        <div style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: '20px', marginBottom: 20 }}>
          <p style={{ fontSize: 12, color: '#555', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>Predicted vs actual</p>
          <ResponsiveContainer width='100%' height={280}>
            <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray='3 3' stroke='rgba(255,255,255,0.05)' />
              <XAxis dataKey='actual'    name='Actual'    tick={{ fill: '#666', fontSize: 11 }} label={{ value: 'Actual', position: 'insideBottom', offset: -5, fill: '#555', fontSize: 11 }} />
              <YAxis dataKey='predicted' name='Predicted' tick={{ fill: '#666', fontSize: 11 }} />
              <RechartTooltip contentStyle={{ background: '#13131a', border: '1px solid #333', borderRadius: 8, color: '#f0f0f0' }} />
              <Scatter data={results.scatter} fill='#0ea5e9' opacity={0.7} />
            </ScatterChart>
          </ResponsiveContainer>
          <p style={{ fontSize: 11, color: '#444', marginTop: 10, borderLeft: '3px solid rgba(14,165,233,0.4)', paddingLeft: 10 }}>Dots on the diagonal = perfect predictions. Spread = error.</p>
        </div>
      )}

      {results.feature_importance && results.feature_importance.length > 0 && (
        <div style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: '20px', marginBottom: 20 }}>
          <p style={{ fontSize: 12, color: '#555', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>Feature importance</p>
          <ResponsiveContainer width='100%' height={Math.max(160, results.feature_importance.length * 36)}>
            <BarChart data={results.feature_importance} layout='vertical' margin={{ top: 0, right: 40, left: 10, bottom: 0 }}>
              <XAxis type='number' tick={{ fill: '#555', fontSize: 11 }} />
              <YAxis type='category' dataKey='feature' tick={{ fill: '#aaa', fontSize: 11 }} width={110} />
              <RechartTooltip contentStyle={{ background: '#13131a', border: '1px solid #333', borderRadius: 8, color: '#f0f0f0' }} />
              <Bar dataKey='importance' radius={[0, 4, 4, 0]} fill='#0ea5e9' />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <CodeExport payload={{
        model_type: 'regression',
        model_name: results.model_name || 'random_forest',
        target: results.target || '',
        features: results.features || [],
        test_size: results.test_size || 0.2,
      }} />
    </div>
  )
}

function Section({ label, tooltip, children }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
        <p style={{ fontSize: 12, color: '#666', textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 500, margin: 0 }}>{label}</p>
        {tooltip && <Tooltip text={tooltip} />}
      </div>
      {children}
    </div>
  )
}

function ErrorBox({ message }) {
  return <div style={{ padding: '10px 14px', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 8, fontSize: 13, color: '#f87171', marginBottom: 16 }}>{message}</div>
}

const selectStyle = {
  background: '#13131a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8,
  color: '#e0e0e0', padding: '8px 12px', fontSize: 13, cursor: 'pointer', width: '100%',
}