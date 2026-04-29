import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { trainRegressorStream } from '../api/client'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip as RTooltip, ResponsiveContainer, BarChart, Bar, Cell, ReferenceLine
} from 'recharts'
import CodeExport from '../components/CodeExport.jsx'
import Tooltip from '../components/Tooltip.jsx'
import PipelineTrainer from '../components/PipelineTrainer.jsx'

const MODELS = [
  { id: 'linear_regression', label: 'Linear Regression',      tip: 'Fast baseline. Assumes linear relationship.',        rec: 'Best for: simple, interpretable problems.' },
  { id: 'ridge',             label: 'Ridge Regression',        tip: 'Regularised linear. Prevents overfitting.',          rec: 'Best for: correlated features.' },
  { id: 'lasso',             label: 'Lasso Regression',        tip: 'Shrinks weak features to zero.',                     rec: 'Best for: automatic feature selection.' },
  { id: 'decision_tree',     label: 'Decision Tree Regressor', tip: 'Non-linear. Can overfit without depth limit.',        rec: 'Best for: non-linear patterns.' },
  { id: 'random_forest',     label: 'Random Forest Regressor', tip: 'Ensemble of trees. Robust, accurate.',               rec: 'Best for: most problems. Best default.' },
]

const ACCENT = '#0ea5e9'

export default function RegressionPage() {
  const [ds, setDs]             = useState(null)
  const [model, setModel]       = useState('random_forest')
  const [target, setTarget]     = useState('')
  const [features, setFeatures] = useState([])
  const [testSize, setTestSize] = useState(0.2)
  const [results, setResults]   = useState(null)
  const [csvRaw, setCsvRaw]     = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    const d = sessionStorage.getItem('dataset')
    const c = sessionStorage.getItem('csvRaw')
    if (!d) { navigate('/'); return }
    const parsed = JSON.parse(d)
    setDs(parsed)
    setCsvRaw(c || '')
    const numCols = parsed.columns
      .filter(col => col.dtype.includes('int') || col.dtype.includes('float'))
      .map(c => c.name)
    setTarget(numCols[numCols.length - 1] || '')
    setFeatures(numCols.slice(0, -1))
  }, [navigate])

  function toggleFeature(col) {
    setFeatures(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col])
  }

  async function doTrain(onEvent) {
    if (!target)               throw new Error('Select a target column.')
    if (features.length === 0) throw new Error('Select at least one feature column.')
    if (features.includes(target)) throw new Error('Target column cannot also be a feature.')
    if (!csvRaw)               throw new Error('CSV data missing. Please go back and re-upload.')
    await trainRegressorStream(
      { model_name: model, target, features, test_size: testSize, csv_data: csvRaw,
        filename: sessionStorage.getItem('csvFile') || 'dataset.csv' },
      onEvent
    )
  }

  if (!ds) return null
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
        <span style={{ color: ACCENT, fontSize: 14, fontWeight: 600 }}>Regression</span>
        {results && <span style={doneStyle}>✓ Training complete</span>}
      </header>

      <main style={{ maxWidth: 900, margin: '0 auto', padding: '40px 24px' }}>
        {!results ? (
          <PipelineTrainer
            onTrain={doTrain}
            onResult={r => setResults({ ...r, model_name: model, features, test_size: testSize, target })}
            accent={ACCENT}
          >
            <div style={{ marginBottom: 28 }}>
              <h2 style={h2}>Configure Regression</h2>
              <p style={{ color: '#555', fontSize: 14 }}>
                Set up your model then click Train — the full pipeline runs automatically.
              </p>
            </div>

            <Section label='1. Pick algorithm'>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {MODELS.map(m => (
                  <button key={m.id} onClick={() => setModel(m.id)} style={{
                    background: model === m.id ? 'rgba(14,165,233,0.08)' : 'rgba(255,255,255,0.01)',
                    border: `1px solid ${model === m.id ? ACCENT : 'rgba(255,255,255,0.06)'}`,
                    borderRadius: 10, padding: '12px 16px', cursor: 'pointer', textAlign: 'left',
                    boxShadow: model === m.id ? '0 0 16px rgba(14,165,233,0.15)' : 'none',
                    transition: 'all 0.2s',
                  }}>
                    <div style={{ fontWeight: 600, fontSize: 13,
                      color: model === m.id ? ACCENT : '#e0e0e0', marginBottom: 3 }}>{m.label}</div>
                    <div style={{ fontSize: 11, color: '#555', marginBottom: 3 }}>{m.tip}</div>
                    <div style={{ fontSize: 11,
                      color: model === m.id ? '#3b8fb5' : '#2a2a3a' }}>{m.rec}</div>
                  </button>
                ))}
              </div>
            </Section>

            <Section label='2. Target column'
              tooltip='The numeric column to predict (e.g. "price", "tip", "score").'>
              <select value={target} onChange={e => setTarget(e.target.value)} style={selectStyle}>
                {numCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </Section>

            <Section label='3. Feature columns'
              tooltip='Numeric input columns the model uses to make predictions.'>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {numCols.map(c => (
                  <button key={c} onClick={() => toggleFeature(c)} style={{
                    padding: '6px 12px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
                    background: features.includes(c) ? 'rgba(14,165,233,0.12)' : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${features.includes(c) ? ACCENT : 'rgba(255,255,255,0.07)'}`,
                    color: features.includes(c) ? ACCENT : '#666', transition: 'all 0.15s',
                  }}>{c}</button>
                ))}
              </div>
            </Section>

            <Section label={`4. Test split — ${Math.round(testSize * 100)}%`}
              tooltip='Percentage of rows held back for evaluation. 20% is standard.'>
              <input type='range' min={10} max={40} value={testSize * 100}
                onChange={e => setTestSize(Number(e.target.value) / 100)}
                style={{ width: '100%', accentColor: ACCENT }} />
              <div style={{ display: 'flex', justifyContent: 'space-between',
                fontSize: 11, color: '#444', marginTop: 4 }}>
                <span>10% — more training data</span><span>40% — more reliable test</span>
              </div>
            </Section>
          </PipelineTrainer>
        ) : (
          <RegressionResults results={results} onReset={() => setResults(null)} />
        )}
      </main>
    </div>
  )
}

function PreprocessPanel({ stats }) {
  if (!stats) return null
  const items = [
    { label: 'Total rows',     value: stats.total_rows },
    { label: 'Train rows',     value: stats.train_rows },
    { label: 'Test rows',      value: stats.test_rows },
    { label: 'Features',       value: stats.n_features },
    { label: 'Missing filled', value: stats.missing_filled },
    { label: 'Scaler',         value: stats.scaler },
    { label: 'Target mean',    value: stats.target_mean },
    { label: 'Target std',     value: stats.target_std },
  ]
  return (
    <div style={{ background: 'rgba(14,165,233,0.04)', border: '1px solid rgba(14,165,233,0.15)',
      borderRadius: 14, padding: '16px 20px', marginBottom: 20 }}>
      <p style={{ fontSize: 11, color: '#0ea5e9', textTransform: 'uppercase',
        letterSpacing: '0.08em', marginBottom: 12, fontWeight: 600 }}>⚙ Preprocessing applied</p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8 }}>
        {items.map(item => (
          <div key={item.label} style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8',
              fontFamily: 'monospace', marginBottom: 2 }}>{item.value}</div>
            <div style={{ fontSize: 10, color: '#555' }}>{item.label}</div>
          </div>
        ))}
      </div>
      <p style={{ fontSize: 10, color: '#444', marginTop: 10,
        borderTop: '1px solid rgba(255,255,255,0.04)', paddingTop: 8 }}>
        Pipeline: SimpleImputer (median) → RobustScaler → model · Random train/test split
      </p>
    </div>
  )
}

function RegressionResults({ results, onReset }) {
  const [tab, setTab] = useState('overview')

  const overviewMetrics = [
    { label: 'MAE',           value: results.mae,          color: '#0ea5e9', desc: 'Average absolute error (same units as target)' },
    { label: 'RMSE',          value: results.rmse,         color: '#6c63ff', desc: 'Penalises large errors more than MAE' },
    { label: 'R² Score',      value: results.r2,           color: '#10b981', desc: '1.0 = perfect · 0 = no better than mean' },
    { label: 'Adjusted R²',   value: results.adj_r2,       color: '#34d399', desc: 'R² penalised for unnecessary features' },
    { label: 'Train R²',      value: results.train_r2,     color: '#4338ca', desc: 'R² on training set — compare to test R²' },
    { label: 'MSE',           value: results.mse,          color: '#818cf8', desc: 'Mean squared error' },
    { label: 'Median AE',     value: results.median_ae,    color: '#f59e0b', desc: 'Median absolute error — robust to outliers' },
    { label: 'MAPE',          value: (results.mape ?? '—') + (results.mape != null ? '%' : ''), color: '#fb923c', desc: 'Mean Absolute Percentage Error' },
    { label: 'Explained Var', value: results.explained_var,color: '#a78bfa', desc: 'How much target variance the model explains' },
    { label: 'CV R²',         value: results.cv_r2 != null ? results.cv_r2 : 'N/A', color: '#34d399', desc: `Cross-val R² ± ${results.cv_std ?? '—'}` },
    { label: 'Max Error',     value: results.max_error,    color: '#f87171', desc: 'Worst single prediction error' },
    { label: 'Test rows',     value: results.n_test,       color: '#555',    desc: 'Rows used for evaluation' },
  ]

  const residualData = (results.residuals || []).map((r, i) => ({
    predicted: results.scatter?.[i]?.predicted ?? 0, residual: r,
  }))

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
        marginBottom: 20, gap: 16, flexWrap: 'wrap' }}>
        <div>
          <h2 style={h2}>Training Results</h2>
          <p style={{ color: '#555', fontSize: 13 }}>
            {results.n_train} train · {results.n_test} test
            {results.cv_r2 != null && ` · CV R²: ${results.cv_r2} ± ${results.cv_std}`}
          </p>
        </div>
        <button onClick={onReset} style={reconfigBtn}>← Reconfigure</button>
      </div>

      <PreprocessPanel stats={results.preprocess_stats} />

      {results.fit_warning && (
        <div style={{ padding: '10px 14px', background: 'rgba(245,158,11,0.07)',
          border: '1px solid rgba(245,158,11,0.25)', borderRadius: 10, fontSize: 13,
          color: '#fbbf24', marginBottom: 16, display: 'flex', gap: 10 }}>
          <span>⚠</span><span>{results.fit_warning}</span>
        </div>
      )}

      {results.cv_r2 != null && (
        <div style={{ padding: '10px 14px', background: 'rgba(16,185,129,0.05)',
          border: '1px solid rgba(16,185,129,0.2)', borderRadius: 10, fontSize: 12,
          color: '#34d399', marginBottom: 20 }}>
          <strong>Cross-validation R²: {results.cv_r2} ± {results.cv_std}</strong>
          {' '}— averaged over multiple folds.
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20,
        borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: 12, flexWrap: 'wrap' }}>
        {[
          ['overview',   'All Metrics'],
          ['scatter',    'Predicted vs Actual'],
          ['residuals',  'Residuals'],
          ['importance', 'Feature Importance'],
        ].map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)} style={{
            padding: '6px 14px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
            background: tab === id ? 'rgba(14,165,233,0.15)' : 'transparent',
            border: `1px solid ${tab === id ? ACCENT : 'transparent'}`,
            color: tab === id ? ACCENT : '#666', transition: 'all 0.2s',
          }}>{label}</button>
        ))}
      </div>

      {/* All Metrics */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 24 }}>
          {overviewMetrics.map(m => (
            <div key={m.label} title={m.desc} style={{
              background: 'rgba(255,255,255,0.02)', border: `1px solid ${m.color}20`,
              borderRadius: 12, padding: '14px 12px', textAlign: 'center', cursor: 'help',
            }}>
              <div style={{ fontSize: 17, fontWeight: 700, color: m.color,
                fontFamily: 'monospace', marginBottom: 4 }}>{m.value}</div>
              <div style={{ fontSize: 11, color: '#777', marginBottom: 2 }}>{m.label}</div>
              <div style={{ fontSize: 9, color: '#444', lineHeight: 1.4 }}>{m.desc}</div>
            </div>
          ))}
        </div>
      )}

      {/* Scatter */}
      {tab === 'scatter' && results.scatter && results.scatter.length > 0 && (
        <div style={card}>
          <p style={cardLabel}>Predicted vs actual</p>
          <p style={{ fontSize: 11, color: '#444', marginBottom: 16 }}>
            Each dot = one test row. Perfect model = all dots on the diagonal.
          </p>
          <ResponsiveContainer width='100%' height={340}>
            <ScatterChart margin={{ top: 20, right: 20, left: 0, bottom: 30 }}>
              <CartesianGrid strokeDasharray='3 3' stroke='rgba(255,255,255,0.04)' />
              <XAxis dataKey='actual' name='Actual' tick={{ fill: '#666', fontSize: 11 }}
                label={{ value: 'Actual Value', position: 'insideBottom', offset: -15, fill: '#555', fontSize: 11 }} />
              <YAxis dataKey='predicted' name='Predicted' tick={{ fill: '#666', fontSize: 11 }}
                label={{ value: 'Predicted', angle: -90, position: 'insideLeft', fill: '#555', fontSize: 11 }} />
              <RTooltip contentStyle={tooltipStyle} formatter={v => [v.toFixed(4)]} />
              <Scatter data={results.scatter} fill={ACCENT} opacity={0.65} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Residuals */}
      {tab === 'residuals' && residualData.length > 0 && (
        <div style={card}>
          <p style={cardLabel}>Residuals plot</p>
          <p style={{ fontSize: 11, color: '#444', marginBottom: 16 }}>
            Residual = actual − predicted. Random scatter around zero = good model.
            Patterns = the model is missing something.
          </p>
          <ResponsiveContainer width='100%' height={300}>
            <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 30 }}>
              <CartesianGrid strokeDasharray='3 3' stroke='rgba(255,255,255,0.04)' />
              <XAxis dataKey='predicted' name='Predicted' tick={{ fill: '#666', fontSize: 11 }}
                label={{ value: 'Predicted Value', position: 'insideBottom', offset: -15, fill: '#555', fontSize: 11 }} />
              <YAxis dataKey='residual' name='Residual' tick={{ fill: '#666', fontSize: 11 }}
                label={{ value: 'Residual', angle: -90, position: 'insideLeft', fill: '#555', fontSize: 11 }} />
              <ReferenceLine y={0} stroke='rgba(255,255,255,0.2)' strokeDasharray='4 4' />
              <RTooltip contentStyle={tooltipStyle} formatter={(v, name) => [v.toFixed(4), name]} />
              <Scatter data={residualData} fill='#f59e0b' opacity={0.65} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Feature Importance */}
      {tab === 'importance' && (
        <div style={card}>
          {results.feature_importance && results.feature_importance.length > 0 ? (
            <>
              <p style={cardLabel}>Feature importance</p>
              <p style={{ fontSize: 11, color: '#444', marginBottom: 16 }}>
                Longer bar = stronger influence on the predicted value.
              </p>
              <ResponsiveContainer width='100%'
                height={Math.max(200, results.feature_importance.length * 44)}>
                <BarChart data={results.feature_importance} layout='vertical'
                  margin={{ top: 0, right: 70, left: 10, bottom: 0 }}>
                  <XAxis type='number' tick={{ fill: '#555', fontSize: 11 }} />
                  <YAxis type='category' dataKey='feature'
                    tick={{ fill: '#aaa', fontSize: 11 }} width={120} />
                  <RTooltip contentStyle={tooltipStyle}
                    formatter={v => [v.toFixed(4), 'Importance']} />
                  <Bar dataKey='importance' radius={[0, 6, 6, 0]}
                    label={{ position: 'right', formatter: v => v.toFixed(3), fill: '#666', fontSize: 10 }}>
                    {results.feature_importance.map((_, i) => (
                      <Cell key={i}
                        fill={i === 0 ? ACCENT : i === 1 ? '#38bdf8' : 'rgba(14,165,233,0.35)'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </>
          ) : (
            <div style={{ padding: '40px 0', textAlign: 'center', color: '#555' }}>
              Feature importance not available for this model type.
            </div>
          )}
        </div>
      )}

      <CodeExport payload={{
        model_type: 'regression',
        model_name: results.model_name || 'random_forest',
        target:     results.target    || '',
        features:   results.features  || [],
        test_size:  results.test_size || 0.2,
      }} />
    </div>
  )
}

function Section({ label, tooltip, children }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
        <p style={{ fontSize: 11, color: '#666', textTransform: 'uppercase',
          letterSpacing: '0.08em', fontWeight: 600, margin: 0 }}>{label}</p>
        {tooltip && <Tooltip text={tooltip} />}
      </div>
      {children}
    </div>
  )
}

const headerStyle  = { borderBottom: '1px solid rgba(255,255,255,0.05)', padding: '16px 40px', display: 'flex', alignItems: 'center', gap: 12, background: 'rgba(2,2,8,0.85)', backdropFilter: 'blur(20px)', position: 'sticky', top: 0, zIndex: 50 }
const backLinkStyle= { color: '#555', fontSize: 13, textDecoration: 'none', transition: 'color 0.2s' }
const divStyle     = { width: 1, height: 16, background: 'rgba(255,255,255,0.08)' }
const doneStyle    = { marginLeft: 'auto', fontSize: 12, color: '#10b981', fontFamily: 'monospace' }
const h2           = { fontSize: 22, fontWeight: 700, letterSpacing: '-0.02em', marginBottom: 4 }
const reconfigBtn  = { padding: '8px 16px', background: 'transparent', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#888', fontSize: 13, cursor: 'pointer' }
const selectStyle  = { background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, color: '#e0e0e0', padding: '8px 12px', fontSize: 13, cursor: 'pointer', width: '100%', outline: 'none' }
const card         = { background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 14, padding: '20px', marginBottom: 24 }
const cardLabel    = { fontSize: 11, color: '#555', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6, fontWeight: 600 }
const tooltipStyle = { background: '#0d0d18', border: '1px solid rgba(14,165,233,0.3)', borderRadius: 8, color: '#f0f0f0', fontSize: 12 }