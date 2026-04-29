import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { trainClassifierStream } from '../api/client'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip as RTooltip,
  ResponsiveContainer, Cell, RadarChart, PolarGrid,
  PolarAngleAxis, Radar
} from 'recharts'
import CodeExport from '../components/CodeExport.jsx'
import Tooltip from '../components/Tooltip.jsx'
import PipelineTrainer from '../components/PipelineTrainer.jsx'

const MODELS = [
  { id: 'logistic_regression', label: 'Logistic Regression', tip: 'Fast, interpretable. Best for linearly separable data.',  rec: 'Best for: binary problems, small datasets.' },
  { id: 'decision_tree',       label: 'Decision Tree',       tip: 'Transparent decisions. Can overfit on small data.',        rec: 'Best for: explainability, quick insights.' },
  { id: 'random_forest',       label: 'Random Forest',       tip: '200 trees averaged. Robust, handles noise well.',          rec: 'Best for: most problems. Best default.' },
  { id: 'svm',                 label: 'SVM',                 tip: 'Finds optimal decision boundary. Great for many features.', rec: 'Best for: high-dimensional data.' },
  { id: 'knn',                 label: 'KNN',                 tip: 'Classifies by nearest neighbours. Simple but slow.',       rec: 'Best for: small datasets, spatial data.' },
]

const ACCENT = '#6c63ff'

export default function ClassificationPage() {
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
    const numCols = parsed.columns.filter(col =>
      col.dtype.includes('int') || col.dtype.includes('float')).map(c => c.name)
    const catCols = parsed.columns.filter(col =>
      !col.dtype.includes('int') && !col.dtype.includes('float')).map(c => c.name)
    setTarget(catCols[0] || parsed.columns[parsed.columns.length - 1].name)
    setFeatures(numCols)
  }, [navigate])

  function toggleFeature(col) {
    setFeatures(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col])
  }

  async function doTrain(onEvent) {
    if (!target)                   throw new Error('Select a target column.')
    if (features.length === 0)     throw new Error('Select at least one feature column.')
    if (features.includes(target)) throw new Error('Target column cannot also be a feature.')
    if (!csvRaw)                   throw new Error('CSV data missing. Please go back and re-upload.')
    await trainClassifierStream(
      { model_name: model, target, features, test_size: testSize, csv_data: csvRaw,
        filename: sessionStorage.getItem('csvFile') || 'dataset.csv' },
      onEvent
    )
  }

  if (!ds) return null
  const allCols = ds.columns.map(c => c.name)
  const numCols = ds.columns.filter(c =>
    c.dtype.includes('int') || c.dtype.includes('float')).map(c => c.name)

  return (
    <div style={{ minHeight: '100vh' }}>
      <header style={headerStyle}>
        <Link to='/dataset-report' style={backLinkStyle}
          onMouseEnter={e => e.currentTarget.style.color = '#aaa'}
          onMouseLeave={e => e.currentTarget.style.color = '#555'}>
          ← Back
        </Link>
        <div style={dividerStyle} />
        <span style={{ color: ACCENT, fontSize: 14, fontWeight: 600 }}>Classification</span>
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
              <h2 style={h2}>Configure Classification</h2>
              <p style={{ color: '#555', fontSize: 14 }}>
                Set up your model then click Train — the full pipeline runs automatically.
              </p>
            </div>

            <Section label='1. Pick algorithm'>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {MODELS.map(m => (
                  <button key={m.id} onClick={() => setModel(m.id)} style={{
                    background: model === m.id ? 'rgba(108,99,255,0.08)' : 'rgba(255,255,255,0.01)',
                    border: `1px solid ${model === m.id ? ACCENT : 'rgba(255,255,255,0.06)'}`,
                    borderRadius: 10, padding: '12px 16px', cursor: 'pointer', textAlign: 'left',
                    boxShadow: model === m.id ? '0 0 16px rgba(108,99,255,0.15)' : 'none',
                    transition: 'all 0.2s',
                  }}>
                    <div style={{ fontWeight: 600, fontSize: 13,
                      color: model === m.id ? ACCENT : '#e0e0e0', marginBottom: 3 }}>{m.label}</div>
                    <div style={{ fontSize: 11, color: '#555', marginBottom: 3 }}>{m.tip}</div>
                    <div style={{ fontSize: 11, color: model === m.id ? '#7c75cc' : '#2a2a3a' }}>{m.rec}</div>
                  </button>
                ))}
              </div>
            </Section>

            <Section label='2. Target column'
              tooltip='The column to predict. Should be categorical (e.g. "species", "survived").'>
              <select value={target} onChange={e => setTarget(e.target.value)} style={selectStyle}>
                {allCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </Section>

            <Section label='3. Feature columns'
              tooltip='Numeric input columns the model uses to make predictions.'>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {numCols.map(c => (
                  <button key={c} onClick={() => toggleFeature(c)} style={{
                    padding: '6px 12px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
                    background: features.includes(c) ? 'rgba(108,99,255,0.12)' : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${features.includes(c) ? ACCENT : 'rgba(255,255,255,0.07)'}`,
                    color: features.includes(c) ? '#a09af0' : '#666', transition: 'all 0.15s',
                  }}>{c}</button>
                ))}
              </div>
              <p style={{ fontSize: 11, color: '#444', marginTop: 8 }}>
                {features.length} selected · Only numeric columns can be features
              </p>
            </Section>

            <Section label={`4. Test split — ${Math.round(testSize * 100)}%`}
              tooltip='Rows withheld for evaluation. 20% is the standard default.'>
              <input type='range' min={10} max={40} value={testSize * 100}
                onChange={e => setTestSize(Number(e.target.value) / 100)}
                style={{ width: '100%', accentColor: ACCENT }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#444', marginTop: 4 }}>
                <span>10% — more training data</span><span>40% — more reliable test</span>
              </div>
            </Section>
          </PipelineTrainer>
        ) : (
          <ClassificationResults results={results} onReset={() => setResults(null)} />
        )}
      </main>
    </div>
  )
}

function PreprocessPanel({ stats }) {
  if (!stats) return null
  const items = [
    { label: 'Total rows',    value: stats.total_rows },
    { label: 'Train rows',    value: stats.train_rows },
    { label: 'Test rows',     value: stats.test_rows },
    { label: 'Features used', value: stats.n_features },
    { label: 'Missing filled',value: stats.missing_filled },
    { label: 'Scaler',        value: stats.scaler },
    { label: 'Classes',       value: stats.n_classes },
    { label: 'Label encoded', value: stats.label_encoded ? 'Yes' : 'No' },
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
        Pipeline: SimpleImputer (median) → RobustScaler → model · Stratified train/test split
      </p>
    </div>
  )
}

function ClassificationResults({ results, onReset }) {
  const [tab, setTab] = useState('overview')

  const overviewMetrics = [
    { label: 'Test Accuracy',  value: results.accuracy + '%',       color: '#6c63ff', desc: 'Correct predictions on unseen data' },
    { label: 'Train Accuracy', value: results.train_accuracy + '%', color: '#4338ca', desc: 'Correct predictions on training data' },
    { label: 'F1 Score',       value: results.f1 + '%',             color: '#0ea5e9', desc: 'Harmonic mean of precision & recall' },
    { label: 'Precision',      value: results.precision + '%',      color: '#10b981', desc: 'Of predicted positives, how many correct' },
    { label: 'Recall',         value: results.recall + '%',         color: '#f59e0b', desc: 'Of actual positives, how many caught' },
    { label: 'ROC-AUC',        value: results.roc_auc != null ? results.roc_auc : 'N/A', color: '#a78bfa', desc: 'Area under ROC curve (1.0 = perfect)' },
    { label: 'CV Accuracy',    value: results.cv_accuracy != null ? results.cv_accuracy + '%' : 'N/A', color: '#34d399', desc: `Cross-val ± ${results.cv_std ?? '—'}%` },
    { label: 'Classes',        value: results.n_classes, color: '#fb923c', desc: 'Number of target classes' },
  ]

  const radarData = [
    { metric: 'Accuracy',  value: results.accuracy },
    { metric: 'F1',        value: results.f1 },
    { metric: 'Precision', value: results.precision },
    { metric: 'Recall',    value: results.recall },
    { metric: 'CV Score',  value: results.cv_accuracy || 0 },
    { metric: 'ROC-AUC',   value: results.roc_auc ? results.roc_auc * 100 : 0 },
  ]

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
        marginBottom: 20, gap: 16, flexWrap: 'wrap' }}>
        <div>
          <h2 style={h2}>Training Results</h2>
          <p style={{ color: '#555', fontSize: 13 }}>
            {results.n_train} train · {results.n_test} test · {results.n_classes} classes
            {results.cv_accuracy != null && ` · CV: ${results.cv_accuracy}% ± ${results.cv_std}%`}
          </p>
        </div>
        <button onClick={onReset} style={reconfigBtn}>← Reconfigure</button>
      </div>

      <PreprocessPanel stats={results.preprocess_stats} />

      {results.fit_warning && (
        <div style={{ padding: '10px 14px',
          background: results.fit_status === 'overfitting' ? 'rgba(245,158,11,0.07)' : 'rgba(239,68,68,0.07)',
          border: `1px solid ${results.fit_status === 'overfitting' ? 'rgba(245,158,11,0.3)' : 'rgba(239,68,68,0.3)'}`,
          borderRadius: 10, fontSize: 13,
          color: results.fit_status === 'overfitting' ? '#fbbf24' : '#f87171',
          marginBottom: 16, display: 'flex', gap: 10, alignItems: 'flex-start' }}>
          <span>⚠</span><span>{results.fit_warning}</span>
        </div>
      )}

      {results.cv_accuracy != null && (
        <div style={{ padding: '10px 14px', background: 'rgba(16,185,129,0.05)',
          border: '1px solid rgba(16,185,129,0.2)', borderRadius: 10, fontSize: 12,
          color: '#34d399', marginBottom: 20 }}>
          <strong>Cross-validation: {results.cv_accuracy}% ± {results.cv_std}%</strong>
          {' '}— averaged over multiple folds. More reliable than a single split.
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20,
        borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: 12, flexWrap: 'wrap' }}>
        {[
          ['overview',   'All Metrics'],
          ['radar',      'Radar Chart'],
          ['importance', 'Feature Importance'],
          ['matrix',     'Confusion Matrix'],
          ['perclass',   'Per-class Report'],
        ].map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)} style={{
            padding: '6px 14px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
            background: tab === id ? 'rgba(108,99,255,0.15)' : 'transparent',
            border: `1px solid ${tab === id ? ACCENT : 'transparent'}`,
            color: tab === id ? '#a09af0' : '#666', transition: 'all 0.2s',
          }}>{label}</button>
        ))}
      </div>

      {/* All Metrics */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 24 }}>
          {overviewMetrics.map(m => (
            <div key={m.label} title={m.desc} style={{
              background: 'rgba(255,255,255,0.02)', border: `1px solid ${m.color}22`,
              borderRadius: 12, padding: '14px 12px', textAlign: 'center', cursor: 'help',
            }}>
              <div style={{ fontSize: 22, fontWeight: 700, color: m.color,
                fontFamily: 'monospace', marginBottom: 4 }}>{m.value}</div>
              <div style={{ fontSize: 11, color: '#777', marginBottom: 2 }}>{m.label}</div>
              <div style={{ fontSize: 9, color: '#444', lineHeight: 1.4 }}>{m.desc}</div>
            </div>
          ))}
        </div>
      )}

      {/* Radar */}
      {tab === 'radar' && (
        <div style={card}>
          <p style={cardLabel}>Model performance radar</p>
          <p style={{ fontSize: 11, color: '#444', marginBottom: 16 }}>
            Outer edge = 100%. Larger filled area = better overall model.
          </p>
          <ResponsiveContainer width='100%' height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke='rgba(255,255,255,0.06)' />
              <PolarAngleAxis dataKey='metric' tick={{ fill: '#666', fontSize: 11 }} />
              <Radar name='Score' dataKey='value' stroke={ACCENT} fill={ACCENT}
                fillOpacity={0.18} strokeWidth={2} />
              <RTooltip contentStyle={tooltipStyle} formatter={v => [v.toFixed(1) + '%']} />
            </RadarChart>
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
                Longer bar = the model relies more on that column. Top features have most predictive power.
              </p>
              <ResponsiveContainer width='100%' height={Math.max(200, results.feature_importance.length * 44)}>
                <BarChart data={results.feature_importance} layout='vertical'
                  margin={{ top: 0, right: 70, left: 10, bottom: 0 }}>
                  <XAxis type='number' tick={{ fill: '#555', fontSize: 11 }} tickFormatter={v => v.toFixed(2)} />
                  <YAxis type='category' dataKey='feature' tick={{ fill: '#aaa', fontSize: 11 }} width={120} />
                  <RTooltip contentStyle={tooltipStyle} formatter={v => [v.toFixed(4), 'Importance']} />
                  <Bar dataKey='importance' radius={[0, 6, 6, 0]}
                    label={{ position: 'right', formatter: v => v.toFixed(3), fill: '#666', fontSize: 10 }}>
                    {results.feature_importance.map((_, i) => (
                      <Cell key={i} fill={i === 0 ? ACCENT : i === 1 ? '#7c75ff' : 'rgba(108,99,255,0.35)'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </>
          ) : (
            <div style={{ padding: '40px 0', textAlign: 'center', color: '#555' }}>
              Feature importance not available for SVM / KNN.
            </div>
          )}
        </div>
      )}

      {/* Confusion Matrix */}
      {tab === 'matrix' && results.confusion_matrix && (
        <div style={card}>
          <p style={cardLabel}>Confusion matrix</p>
          <p style={{ fontSize: 11, color: '#444', marginBottom: 20 }}>
            Rows = actual · Columns = predicted.{' '}
            <strong style={{ color: '#a09af0' }}>Purple diagonal</strong> = correct.{' '}
            <strong style={{ color: '#f87171' }}>Red</strong> = mistakes.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'separate', borderSpacing: 4,
              fontSize: 13, fontFamily: 'monospace' }}>
              <thead>
                <tr>
                  <th style={{ padding: '8px 14px', color: '#444', fontSize: 11, textAlign: 'left' }}>
                    Actual ↓ / Predicted →
                  </th>
                  {results.labels.map(l => (
                    <th key={l} style={{ padding: '8px 14px', color: '#777', fontSize: 11, textAlign: 'center' }}>
                      {l}
                    </th>
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
                        padding: '12px 22px', textAlign: 'center', borderRadius: 8,
                        background: i === j ? 'rgba(108,99,255,0.22)'
                          : val > 0 ? 'rgba(226,75,74,0.12)' : 'rgba(255,255,255,0.02)',
                        color: i === j ? '#a09af0' : val > 0 ? '#f87171' : '#444',
                        fontWeight: i === j ? 700 : 400, fontSize: 18,
                        border: i === j ? '1px solid rgba(108,99,255,0.35)'
                          : '1px solid rgba(255,255,255,0.03)',
                      }}>{val}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ display: 'flex', gap: 20, marginTop: 14, fontSize: 11, color: '#555' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ width: 14, height: 14, borderRadius: 3,
                background: 'rgba(108,99,255,0.25)', border: '1px solid rgba(108,99,255,0.4)' }} />
              Correct
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ width: 14, height: 14, borderRadius: 3, background: 'rgba(226,75,74,0.15)' }} />
              Mistake
            </div>
          </div>
        </div>
      )}

      {/* Per-class report */}
      {tab === 'perclass' && results.class_report && (
        <div style={card}>
          <p style={cardLabel}>Per-class breakdown</p>
          <p style={{ fontSize: 11, color: '#444', marginBottom: 16 }}>
            Precision, recall and F1 for each class. Support = actual rows in test set.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                  {['Class', 'Precision', 'Recall', 'F1 Score', 'Support'].map(h => (
                    <th key={h} style={{
                      padding: '8px 14px',
                      textAlign: h === 'Class' ? 'left' : 'center',
                      color: '#555', fontSize: 11,
                      textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 500,
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {/* Per-label rows — only labels returned by backend */}
                {Object.entries(results.class_report)
                  .filter(([k]) => !['accuracy','macro avg','weighted avg'].includes(k))
                  .map(([label, r]) => (
                    <tr key={label} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                      <td style={{ padding: '10px 14px', color: '#e0e0e0',
                        fontWeight: 600, fontFamily: 'monospace' }}>{label}</td>
                      <td style={{ padding: '10px 14px', textAlign: 'center',
                        color: '#10b981', fontFamily: 'monospace' }}>
                        {r.precision != null ? (r.precision*100).toFixed(1)+'%' : '—'}
                      </td>
                      <td style={{ padding: '10px 14px', textAlign: 'center',
                        color: '#f59e0b', fontFamily: 'monospace' }}>
                        {r.recall != null ? (r.recall*100).toFixed(1)+'%' : '—'}
                      </td>
                      <td style={{ padding: '10px 14px', textAlign: 'center',
                        color: '#6c63ff', fontFamily: 'monospace' }}>
                        {r['f1-score'] != null ? (r['f1-score']*100).toFixed(1)+'%' : '—'}
                      </td>
                      <td style={{ padding: '10px 14px', textAlign: 'center',
                        color: '#555', fontFamily: 'monospace' }}>{r.support ?? '—'}</td>
                    </tr>
                  ))
                }
                {/* Weighted avg */}
                {results.class_report['weighted avg'] && (() => {
                  const wa = results.class_report['weighted avg']
                  return (
                    <tr style={{ borderTop: '1px solid rgba(255,255,255,0.08)',
                      background: 'rgba(255,255,255,0.02)' }}>
                      <td style={{ padding: '10px 14px', color: '#888',
                        fontFamily: 'monospace', fontStyle: 'italic' }}>weighted avg</td>
                      <td style={{ padding: '10px 14px', textAlign: 'center',
                        color: '#10b981', fontFamily: 'monospace', fontWeight: 600 }}>
                        {(wa.precision*100).toFixed(1)}%</td>
                      <td style={{ padding: '10px 14px', textAlign: 'center',
                        color: '#f59e0b', fontFamily: 'monospace', fontWeight: 600 }}>
                        {(wa.recall*100).toFixed(1)}%</td>
                      <td style={{ padding: '10px 14px', textAlign: 'center',
                        color: '#6c63ff', fontFamily: 'monospace', fontWeight: 600 }}>
                        {(wa['f1-score']*100).toFixed(1)}%</td>
                      <td style={{ padding: '10px 14px', textAlign: 'center',
                        color: '#555', fontFamily: 'monospace' }}>{wa.support}</td>
                    </tr>
                  )
                })()}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <CodeExport payload={{
        model_type: 'classification',
        model_name: results.model_name || 'random_forest',
        target:     results.target    || '',
        features:   results.features  || [],
        test_size:  results.test_size || 0.2,
      }} />
    </div>
  )
}

/* ── shared small components ── */
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

/* ── shared styles ── */
const headerStyle = {
  borderBottom: '1px solid rgba(255,255,255,0.05)', padding: '16px 40px',
  display: 'flex', alignItems: 'center', gap: 12,
  background: 'rgba(2,2,8,0.85)', backdropFilter: 'blur(20px)',
  position: 'sticky', top: 0, zIndex: 50,
}
const backLinkStyle = { color: '#555', fontSize: 13, textDecoration: 'none', transition: 'color 0.2s' }
const dividerStyle  = { width: 1, height: 16, background: 'rgba(255,255,255,0.08)' }
const doneStyle     = { marginLeft: 'auto', fontSize: 12, color: '#10b981', fontFamily: 'monospace' }
const h2            = { fontSize: 22, fontWeight: 700, letterSpacing: '-0.02em', marginBottom: 4 }
const reconfigBtn   = {
  padding: '8px 16px', background: 'transparent',
  border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8,
  color: '#888', fontSize: 13, cursor: 'pointer',
}
const selectStyle   = {
  background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 8, color: '#e0e0e0', padding: '8px 12px',
  fontSize: 13, cursor: 'pointer', width: '100%', outline: 'none',
}
const card          = {
  background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
  borderRadius: 14, padding: '20px', marginBottom: 24,
}
const cardLabel     = {
  fontSize: 11, color: '#555', textTransform: 'uppercase',
  letterSpacing: '0.08em', marginBottom: 6, fontWeight: 600,
}
const tooltipStyle  = {
  background: '#0d0d18', border: '1px solid rgba(108,99,255,0.3)',
  borderRadius: 8, color: '#f0f0f0', fontSize: 12,
}