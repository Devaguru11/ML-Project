import CodeExport from '../components/CodeExport.jsx'
import Tooltip from '../components/Tooltip.jsx'
import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { trainClassifier } from '../api/client'
import { BarChart, Bar, XAxis, YAxis, Tooltip as RechartTooltip, ResponsiveContainer } from 'recharts'

const MODELS = [
  { id: 'logistic_regression', label: 'Logistic Regression', tip: 'Fast and simple. Best for linearly separable data.',     rec: 'Best for: binary problems, small datasets, when you need speed.' },
  { id: 'decision_tree',       label: 'Decision Tree',       tip: 'Easy to understand. Can overfit on small data.',          rec: 'Best for: when you need to explain decisions clearly.' },
  { id: 'random_forest',       label: 'Random Forest',       tip: 'Robust and accurate. Good default choice.',              rec: 'Best for: most classification problems. Good default.' },
  { id: 'svm',                 label: 'SVM',                 tip: 'Great for high-dimensional data.',                       rec: 'Best for: text data, high-dimensional features.' },
  { id: 'knn',                 label: 'KNN',                 tip: 'Simple nearest-neighbour method.',                       rec: 'Best for: small datasets where similar rows = similar labels.' },
]

const ACCENT = '#6c63ff'

const HYPERPARAM_TIPS = {
  target:    'The column your model will try to predict. Must be categorical (e.g. "species", "survived").',
  features:  'The input columns the model uses to make predictions. Only numeric columns work here.',
  test_size: 'How much data to hold back for testing. 20% is a safe default. More = more reliable test, less training data.',
}

export default function ClassificationPage() {
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
    const catCols = parsed.columns.filter(col => !col.dtype.includes('int') && !col.dtype.includes('float')).map(c => c.name)
    setTarget(catCols[0] || parsed.columns[parsed.columns.length - 1].name)
    setFeatures(numCols)
  }, [navigate])

  function toggleFeature(col) {
    setFeatures(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col])
  }

  async function handleTrain() {
    if (!target)                  { setError('Select a target column.'); return }
    if (features.length === 0)    { setError('Select at least one feature column.'); return }
    if (features.includes(target)) { setError('Target column cannot also be a feature.'); return }
    if (!csvRaw)                  { setError('CSV data missing. Please go back and re-upload your file.'); return }
    setError('')
    setLoading(true)
    try {
      const res = await trainClassifier({ model_name: model, target, features, test_size: testSize, csv_data: csvRaw })
      // Merge config into results so ClassificationResults can pass it to CodeExport
      setResults({ ...res, model_name: model, features, test_size: testSize, target })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  if (!ds) return null
  const allCols = ds.columns.map(c => c.name)
  const numCols = ds.columns.filter(c => c.dtype.includes('int') || c.dtype.includes('float')).map(c => c.name)

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0f' }}>
      <header style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', padding: '16px 40px', display: 'flex', alignItems: 'center', gap: 12 }}>
        <Link to='/visualise' style={{ color: '#555', fontSize: 13, textDecoration: 'none' }}>← Back</Link>
        <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)' }} />
        <span style={{ color: ACCENT, fontSize: 14, fontWeight: 500 }}>Classification</span>
      </header>

      <main style={{ maxWidth: 860, margin: '0 auto', padding: '40px 24px' }}>
        {!results ? (
          <div>
            <h2 style={{ fontSize: 22, fontWeight: 600, color: '#f0f0f0', marginBottom: 4 }}>Configure model</h2>
            <p style={{ color: '#555', fontSize: 14, marginBottom: 28 }}>Set up your classification model.</p>

            <Section label='1. Pick algorithm'>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {MODELS.map(m => (
                  <button key={m.id} onClick={() => setModel(m.id)} style={{
                    background: model === m.id ? 'rgba(108,99,255,0.1)' : '#13131a',
                    border: `1px solid ${model === m.id ? ACCENT : 'rgba(255,255,255,0.07)'}`,
                    borderRadius: 10, padding: '12px 16px', cursor: 'pointer', textAlign: 'left', transition: 'all 0.2s',
                  }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: model === m.id ? ACCENT : '#e0e0e0', marginBottom: 3 }}>{m.label}</div>
                    <div style={{ fontSize: 11, color: '#555', marginBottom: 4 }}>{m.tip}</div>
                    <div style={{ fontSize: 11, color: model === m.id ? '#7c75cc' : '#3a3a4a' }}>{m.rec}</div>
                  </button>
                ))}
              </div>
            </Section>

            <Section label='2. Target column' tooltip={HYPERPARAM_TIPS.target}>
              <select value={target} onChange={e => setTarget(e.target.value)} style={selectStyle}>
                {allCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </Section>

            <Section label='3. Feature columns' tooltip={HYPERPARAM_TIPS.features}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {numCols.map(c => (
                  <button key={c} onClick={() => toggleFeature(c)} style={{
                    padding: '6px 12px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
                    background: features.includes(c) ? 'rgba(108,99,255,0.15)' : '#13131a',
                    border: `1px solid ${features.includes(c) ? ACCENT : 'rgba(255,255,255,0.08)'}`,
                    color: features.includes(c) ? '#a09af0' : '#666', transition: 'all 0.15s',
                  }}>{c}</button>
                ))}
              </div>
              <p style={{ fontSize: 11, color: '#444', marginTop: 8 }}>Only numeric columns can be features. {features.length} selected.</p>
            </Section>

            <Section label={`4. Test split — ${Math.round(testSize * 100)}%`} tooltip={HYPERPARAM_TIPS.test_size}>
              <input type='range' min={10} max={40} value={testSize * 100}
                onChange={e => setTestSize(Number(e.target.value) / 100)}
                style={{ width: '100%', accentColor: ACCENT }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#555', marginTop: 4 }}>
                <span>10% test</span><span>40% test</span>
              </div>
            </Section>

            {error && <ErrorBox message={error} />}

            <button onClick={handleTrain} disabled={loading} style={{
              width: '100%', padding: '14px', background: loading ? '#333' : ACCENT,
              border: 'none', borderRadius: 10, color: '#fff', fontSize: 15,
              fontWeight: 600, cursor: loading ? 'default' : 'pointer', transition: 'all 0.2s',
            }}>
              {loading ? 'Training...' : 'Train model'}
            </button>
          </div>
        ) : (
          <ClassificationResults results={results} onReset={() => setResults(null)} />
        )}
      </main>
    </div>
  )
}

function ClassificationResults({ results, onReset }) {
  const metrics = [
    { label: 'Accuracy',  value: results.accuracy  + '%', color: '#6c63ff', sub: results.train_accuracy ? `Train: ${results.train_accuracy}%` : 'Test accuracy' },
    { label: 'F1 Score',  value: results.f1        + '%', color: '#0ea5e9', sub: 'Balance of precision/recall' },
    { label: 'Precision', value: results.precision + '%', color: '#10b981', sub: 'Of predicted positives' },
    { label: 'Recall',    value: results.recall    + '%', color: '#f59e0b', sub: 'Of actual positives' },
  ]

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 22, fontWeight: 700, color: '#f0f0f0', marginBottom: 4 }}>Results</h2>
          <p style={{ color: '#555', fontSize: 13 }}>
            {results.n_train} train · {results.n_test} test
            {results.cv_accuracy && ` · CV: ${results.cv_accuracy}% ±${results.cv_std}%`}
          </p>
        </div>
        <button onClick={onReset} style={{ padding: '8px 16px', background: 'transparent', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#888', fontSize: 13, cursor: 'pointer' }}>
          ← Reconfigure
        </button>
      </div>

      {/* Fit warning */}
      {results.fit_warning && (
        <div style={{ padding: '10px 14px', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.25)', borderRadius: 8, fontSize: 13, color: '#fbbf24', marginBottom: 16, display: 'flex', gap: 8 }}>
          <span>⚠</span><span>{results.fit_warning}</span>
        </div>
      )}

      {/* CV score banner */}
      {results.cv_accuracy && (
        <div style={{ padding: '10px 14px', background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.2)', borderRadius: 8, fontSize: 12, color: '#34d399', marginBottom: 16 }}>
          Cross-validation accuracy: <strong>{results.cv_accuracy}% ± {results.cv_std}%</strong> — more reliable than a single train/test split.
        </div>
      )}

      {/* Metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 24 }}>
        {metrics.map((m, i) => (
          <div key={m.label} style={{
            animationDelay: `${i * 0.1}s`,
            background: 'rgba(255,255,255,0.02)',
            border: `1px solid ${m.color}22`,
            borderRadius: 12, padding: '16px 14px', textAlign: 'center',
          }}>
            <div style={{ fontSize: 26, fontWeight: 700, color: m.color, fontFamily: 'monospace', marginBottom: 4 }}>{m.value}</div>
            <div style={{ fontSize: 12, color: '#888', marginBottom: 4 }}>{m.label}</div>
            <div style={{ fontSize: 10, color: '#444' }}>{m.sub}</div>
          </div>
        ))}
      </div>

      {/* Feature importance */}
      {results.feature_importance && results.feature_importance.length > 0 && (
        <div style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: '20px', marginBottom: 20 }}>
          <p style={{ fontSize: 12, color: '#555', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>Feature importance</p>
          <ResponsiveContainer width='100%' height={Math.max(160, results.feature_importance.length * 36)}>
            <BarChart data={results.feature_importance} layout='vertical' margin={{ top: 0, right: 40, left: 10, bottom: 0 }}>
              <XAxis type='number' tick={{ fill: '#555', fontSize: 11 }} />
              <YAxis type='category' dataKey='feature' tick={{ fill: '#aaa', fontSize: 11 }} width={110} />
              <RechartTooltip contentStyle={{ background: '#13131a', border: '1px solid #333', borderRadius: 8, color: '#f0f0f0' }} />
              <Bar dataKey='importance' radius={[0, 4, 4, 0]} fill='#6c63ff' />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Confusion matrix */}
      {results.confusion_matrix && (
        <div style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: '20px', marginBottom: 20 }}>
          <p style={{ fontSize: 12, color: '#555', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>Confusion matrix</p>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', fontSize: 13, fontFamily: 'monospace' }}>
              <thead>
                <tr>
                  <th style={{ padding: '8px 12px', color: '#444', fontSize: 11 }}></th>
                  {results.labels.map(l => (
                    <th key={l} style={{ padding: '8px 12px', color: '#888', fontSize: 11 }}>Pred: {l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.confusion_matrix.map((row, i) => (
                  <tr key={i}>
                    <td style={{ padding: '8px 12px', color: '#888', fontSize: 11 }}>Act: {results.labels[i]}</td>
                    {row.map((val, j) => (
                      <td key={j} style={{
                        padding: '10px 20px', textAlign: 'center', borderRadius: 4,
                        background: i === j ? 'rgba(108,99,255,0.25)' : val > 0 ? 'rgba(226,75,74,0.15)' : 'transparent',
                        color: i === j ? '#a09af0' : val > 0 ? '#f87171' : '#555',
                        fontWeight: i === j ? 600 : 400,
                      }}>{val}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: 11, color: '#444', marginTop: 10 }}>Purple diagonal = correct predictions. Red = mistakes.</p>
        </div>
      )}

      {/* Code export — always at the bottom of results */}
      <CodeExport payload={{
        model_type: 'classification',
        model_name: results.model_name || 'random_forest',
        target:     results.target     || '',
        features:   results.features   || [],
        test_size:  results.test_size  || 0.2,
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
  return (
    <div style={{ padding: '10px 14px', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 8, fontSize: 13, color: '#f87171', marginBottom: 16 }}>
      {message}
    </div>
  )
}

const selectStyle = {
  background: '#13131a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8,
  color: '#e0e0e0', padding: '8px 12px', fontSize: 13, cursor: 'pointer', width: '100%',
}