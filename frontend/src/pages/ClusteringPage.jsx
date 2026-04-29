import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { trainClusterStream } from '../api/client'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip as RTooltip, ResponsiveContainer, Cell, BarChart, Bar
} from 'recharts'
import CodeExport from '../components/CodeExport.jsx'
import PipelineTrainer from '../components/PipelineTrainer.jsx'

const MODELS = [
  { id: 'kmeans',        label: 'K-Means',      tip: 'Partitions data into exactly K clusters. Fast, widely used.', rec: 'Best for: most tasks. Best starting point.' },
  { id: 'dbscan',        label: 'DBSCAN',        tip: 'Finds clusters of any shape. Marks outliers as noise.',       rec: 'Best for: irregular shapes, anomaly detection.' },
  { id: 'agglomerative', label: 'Agglomerative', tip: 'Hierarchical bottom-up clustering.',                          rec: 'Best for: when you want a hierarchy of clusters.' },
]

const ACCENT        = '#10b981'
const CLUSTER_COLORS= ['#10b981','#6c63ff','#f59e0b','#0ea5e9','#f87171','#a78bfa','#34d399','#fb923c','#38bdf8','#e879f9','#facc15','#4ade80']
const noiseColor    = '#3a3a3a'

export default function ClusteringPage() {
  const [ds, setDs]               = useState(null)
  const [model, setModel]         = useState('kmeans')
  const [features, setFeatures]   = useState([])
  const [nClusters, setNClusters] = useState(3)
  const [eps, setEps]             = useState(0.5)
  const [minSamples, setMinSamples] = useState(5)
  const [results, setResults]     = useState(null)
  const [csvRaw, setCsvRaw]       = useState('')
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
    setFeatures(numCols)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  function toggleFeature(col) {
    setFeatures(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col])
  }

  async function doTrain(onEvent) {
    if (features.length < 2) throw new Error('Select at least 2 feature columns.')
    if (!csvRaw)             throw new Error('CSV data missing. Please re-upload.')
    await trainClusterStream(
      { model_name: model, features, csv_data: csvRaw,
        n_clusters: nClusters, eps, min_samples: minSamples,
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
        <span style={{ color: ACCENT, fontSize: 14, fontWeight: 600 }}>Clustering</span>
        {results && <span style={doneStyle}>✓ Complete</span>}
      </header>

      <main style={{ maxWidth: 900, margin: '0 auto', padding: '40px 24px' }}>
        {!results ? (
          <PipelineTrainer
            onTrain={doTrain}
            onResult={r => setResults({
              ...r, model_name: model, features,
              n_clusters: nClusters, eps, min_samples: minSamples,
            })}
            accent={ACCENT}
          >
            <div style={{ marginBottom: 28 }}>
              <h2 style={h2}>Configure Clustering</h2>
              <p style={{ color: '#555', fontSize: 14 }}>
                Clustering finds natural groups in your data without needing labels.
              </p>
            </div>

            {/* Algorithm */}
            <Section label='1. Pick algorithm'>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {MODELS.map(m => (
                  <button key={m.id} onClick={() => setModel(m.id)} style={{
                    background: model === m.id ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.01)',
                    border: `1px solid ${model === m.id ? ACCENT : 'rgba(255,255,255,0.06)'}`,
                    borderRadius: 10, padding: '12px 16px', cursor: 'pointer', textAlign: 'left',
                    transition: 'all 0.2s',
                  }}>
                    <div style={{ fontWeight: 600, fontSize: 13,
                      color: model === m.id ? ACCENT : '#e0e0e0', marginBottom: 3 }}>{m.label}</div>
                    <div style={{ fontSize: 11, color: '#555', marginBottom: 3 }}>{m.tip}</div>
                    <div style={{ fontSize: 11,
                      color: model === m.id ? '#0d9268' : '#2a2a3a' }}>{m.rec}</div>
                  </button>
                ))}
              </div>
            </Section>

            {/* K-Means / Agglomerative params */}
            {(model === 'kmeans' || model === 'agglomerative') && (
              <Section label='2. Number of clusters'>
                <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                  <input type='range' min={2} max={12} value={nClusters}
                    onChange={e => setNClusters(Number(e.target.value))}
                    style={{ flex: 1, accentColor: ACCENT }} />
                  <span style={{ fontSize: 28, fontWeight: 700, color: ACCENT,
                    fontFamily: 'monospace', minWidth: 32,
                    textShadow: '0 0 12px rgba(16,185,129,0.4)' }}>{nClusters}</span>
                </div>
                <p style={{ fontSize: 11, color: '#444', marginTop: 6 }}>
                  K = how many groups to find. Try 2–5 first, then check silhouette score.
                </p>
              </Section>
            )}

            {/* DBSCAN params */}
            {model === 'dbscan' && (
              <Section label='2. DBSCAN parameters'>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                  <div>
                    <p style={{ fontSize: 11, color: '#666', marginBottom: 8 }}>Epsilon — neighbourhood radius</p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <input type='range' min={0.1} max={3} step={0.1} value={eps}
                        onChange={e => setEps(parseFloat(e.target.value))}
                        style={{ flex: 1, accentColor: ACCENT }} />
                      <span style={{ fontSize: 16, fontWeight: 700, color: ACCENT,
                        fontFamily: 'monospace', minWidth: 36 }}>{eps}</span>
                    </div>
                    <p style={{ fontSize: 10, color: '#444', marginTop: 4 }}>Smaller = tighter clusters</p>
                  </div>
                  <div>
                    <p style={{ fontSize: 11, color: '#666', marginBottom: 8 }}>Min samples per cluster</p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <input type='range' min={2} max={20} value={minSamples}
                        onChange={e => setMinSamples(Number(e.target.value))}
                        style={{ flex: 1, accentColor: ACCENT }} />
                      <span style={{ fontSize: 16, fontWeight: 700, color: ACCENT,
                        fontFamily: 'monospace', minWidth: 36 }}>{minSamples}</span>
                    </div>
                    <p style={{ fontSize: 10, color: '#444', marginTop: 4 }}>Higher = fewer, denser clusters</p>
                  </div>
                </div>
              </Section>
            )}

            {/* Features */}
            <Section label='3. Feature columns (select ≥ 2)'>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {numCols.map(c => (
                  <button key={c} onClick={() => toggleFeature(c)} style={{
                    padding: '6px 12px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
                    background: features.includes(c) ? 'rgba(16,185,129,0.12)' : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${features.includes(c) ? ACCENT : 'rgba(255,255,255,0.07)'}`,
                    color: features.includes(c) ? ACCENT : '#666', transition: 'all 0.15s',
                  }}>{c}</button>
                ))}
              </div>
              <p style={{ fontSize: 11, color: '#444', marginTop: 8 }}>
                {features.length} selected · Features are scaled and reduced to 2D for visualisation
              </p>
            </Section>
          </PipelineTrainer>
        ) : (
          <ClusteringResults results={results} onReset={() => setResults(null)} />
        )}
      </main>
    </div>
  )
}

function ClusteringResults({ results, onReset }) {
  const allMetrics = [
    { label: 'Clusters Found',      value: results.n_clusters_found,          color: '#10b981', desc: 'Valid clusters (noise excluded)' },
    { label: 'Silhouette Score',    value: results.silhouette   ?? 'N/A',      color: '#6c63ff', desc: '+1 = well separated · 0 = overlap · −1 = wrong' },
    { label: 'Davies-Bouldin',      value: results.davies_bouldin ?? 'N/A',    color: '#0ea5e9', desc: 'Lower is better' },
    { label: 'Calinski-Harabasz',   value: results.calinski_harabasz ?? 'N/A', color: '#f59e0b', desc: 'Higher is better' },
    { label: 'Inertia (K-Means)',   value: results.inertia ?? 'N/A',           color: '#a78bfa', desc: 'Sum of squared distances to centres' },
    { label: 'Noise Points',        value: results.n_noise,                    color: '#f87171', desc: 'Rows that fit no cluster (DBSCAN only)' },
    { label: 'Total Points',        value: results.total_points,               color: '#555',    desc: 'Total rows processed' },
    { label: 'PCA Variance PC1',    value: results.pca_variance ? results.pca_variance[0] + '%' : 'N/A', color: '#38bdf8', desc: 'Variance explained by first principal component' },
  ]

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
        marginBottom: 24, gap: 16, flexWrap: 'wrap' }}>
        <div>
          <h2 style={h2}>Clustering Results</h2>
          <p style={{ color: '#555', fontSize: 13 }}>
            {results.n_clusters_found} cluster{results.n_clusters_found !== 1 ? 's' : ''} found
            · {results.total_points} total points
            {results.n_noise > 0 ? ` · ${results.n_noise} noise points` : ''}
          </p>
        </div>
        <button onClick={onReset} style={reconfigBtn}>← Reconfigure</button>
      </div>

      {/* Metrics grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 24 }}>
        {allMetrics.map(m => (
          <div key={m.label} title={m.desc} style={{
            background: 'rgba(255,255,255,0.02)', border: `1px solid ${m.color}20`,
            borderRadius: 12, padding: '14px 12px', textAlign: 'center', cursor: 'help',
          }}>
            <div style={{ fontSize: 18, fontWeight: 700, color: m.color,
              fontFamily: 'monospace', marginBottom: 4 }}>{m.value}</div>
            <div style={{ fontSize: 10, color: '#777', marginBottom: 2 }}>{m.label}</div>
            <div style={{ fontSize: 9, color: '#444', lineHeight: 1.4 }}>{m.desc}</div>
          </div>
        ))}
      </div>

      {/* Scatter */}
      <div style={card}>
        <p style={cardLabel}>2D cluster scatter (PCA)</p>
        <p style={{ fontSize: 11, color: '#444', marginBottom: 16 }}>
          Each colour = one cluster. PCA reduces your features to 2D.
          PC1 explains {results.pca_variance?.[0]}% of variance,
          PC2 explains {results.pca_variance?.[1]}%.
        </p>
        <ResponsiveContainer width='100%' height={340}>
          <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray='3 3' stroke='rgba(255,255,255,0.03)' />
            <XAxis dataKey='x' name='PC1' tick={{ fill: '#555', fontSize: 11 }}
              label={{ value: `PC1 (${results.pca_variance?.[0]}%)`,
                position: 'insideBottom', offset: -8, fill: '#555', fontSize: 11 }} />
            <YAxis dataKey='y' name='PC2' tick={{ fill: '#555', fontSize: 11 }} />
            <RTooltip contentStyle={tooltipStyle}
              formatter={(val, name, props) => [
                val.toFixed(3),
                props.payload.cluster === -1 ? 'Noise' : `Cluster ${props.payload.cluster}`,
              ]} />
            <Scatter data={results.scatter} fill={ACCENT}>
              {results.scatter.map((entry, i) => (
                <Cell key={i}
                  fill={entry.cluster === -1
                    ? noiseColor
                    : CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length]}
                  opacity={entry.cluster === -1 ? 0.25 : 0.8} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Cluster sizes */}
      <div style={card}>
        <p style={cardLabel}>Cluster composition</p>
        <ResponsiveContainer width='100%'
          height={Math.max(120, results.cluster_sizes.length * 48)}>
          <BarChart data={results.cluster_sizes} layout='vertical'
            margin={{ top: 0, right: 80, left: 10, bottom: 0 }}>
            <XAxis type='number' tick={{ fill: '#555', fontSize: 11 }} />
            <YAxis type='category' dataKey='label'
              tick={{ fill: '#aaa', fontSize: 11 }} width={90} />
            <RTooltip contentStyle={tooltipStyle}
              formatter={(v, n, p) => [`${v} rows (${p.payload.pct}%)`, 'Size']} />
            <Bar dataKey='size' radius={[0, 6, 6, 0]}
              label={{ position: 'right',
                formatter: (v, entry) => `${entry}%`,
                fill: '#666', fontSize: 10 }}>
              {results.cluster_sizes.map((entry, i) => (
                <Cell key={i}
                  fill={entry.cluster === -1
                    ? noiseColor
                    : CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <CodeExport payload={{
        model_type:  'clustering',
        model_name:  results.model_name  || 'kmeans',
        features:    results.features    || [],
        n_clusters:  results.n_clusters  || 3,
        eps:         results.eps         || 0.5,
        min_samples: results.min_samples || 5,
      }} />
    </div>
  )
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

const headerStyle  = { borderBottom: '1px solid rgba(255,255,255,0.05)', padding: '16px 40px', display: 'flex', alignItems: 'center', gap: 12, background: 'rgba(2,2,8,0.85)', backdropFilter: 'blur(20px)', position: 'sticky', top: 0, zIndex: 50 }
const backLinkStyle= { color: '#555', fontSize: 13, textDecoration: 'none', transition: 'color 0.2s' }
const divStyle     = { width: 1, height: 16, background: 'rgba(255,255,255,0.08)' }
const doneStyle    = { marginLeft: 'auto', fontSize: 12, color: '#10b981', fontFamily: 'monospace' }
const h2           = { fontSize: 22, fontWeight: 700, letterSpacing: '-0.02em', marginBottom: 4 }
const reconfigBtn  = { padding: '8px 16px', background: 'transparent', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#888', fontSize: 13, cursor: 'pointer' }
const card         = { background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 14, padding: '20px', marginBottom: 16 }
const cardLabel    = { fontSize: 11, color: '#555', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6, fontWeight: 600 }
const tooltipStyle = { background: '#0d0d18', border: '1px solid rgba(16,185,129,0.3)', borderRadius: 8, color: '#f0f0f0', fontSize: 12 }