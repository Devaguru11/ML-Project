import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { trainCluster } from '../api/client'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
  BarChart, Bar
} from 'recharts'

const MODELS = [
  { id: 'kmeans',        label: 'K-Means',         tip: 'Partitions data into K clusters. Fast and widely used.' },
  { id: 'dbscan',        label: 'DBSCAN',           tip: 'Finds clusters of arbitrary shape. Handles noise/outliers.' },
  { id: 'agglomerative', label: 'Agglomerative',    tip: 'Builds a hierarchy of clusters bottom-up.' },
]

const ACCENT = '#10b981'

// Distinct colours per cluster (up to 12)
const CLUSTER_COLORS = [
  '#10b981','#6c63ff','#f59e0b','#0ea5e9','#f87171',
  '#a78bfa','#34d399','#fb923c','#38bdf8','#e879f9',
  '#facc15','#4ade80',
]
const noiseColor = '#444'

export default function ClusteringPage() {
  const [ds, setDs]           = useState(null)
  const [model, setModel]     = useState('kmeans')
  const [features, setFeatures] = useState([])
  const [nClusters, setNClusters] = useState(3)
  const [eps, setEps]         = useState(0.5)
  const [minSamples, setMinSamples] = useState(5)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError]     = useState('')
  const [csvRaw, setCsvRaw]   = useState('')
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
    setFeatures(numCols)
  }, [navigate])

  function toggleFeature(col) {
    setFeatures(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col])
  }

  async function handleTrain() {
    if (features.length < 2) { setError('Select at least 2 feature columns.'); return }
    if (!csvRaw)              { setError('CSV data missing. Please re-upload.'); return }
    setError('')
    setLoading(true)
    try {
      const payload = {
        model_name: model,
        features,
        csv_data: csvRaw,
        n_clusters: nClusters,
        eps,
        min_samples: minSamples,
      }
      const res = await trainCluster(payload)
      setResults(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  if (!ds) return null
  const numCols = ds.columns
    .filter(c => c.dtype.includes('int') || c.dtype.includes('float'))
    .map(c => c.name)

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0f' }}>
      <header style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', padding: '16px 40px', display: 'flex', alignItems: 'center', gap: 12 }}>
        <Link to='/visualise' style={{ color: '#555', fontSize: 13, textDecoration: 'none' }}>← Back</Link>
        <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)' }} />
        <span style={{ color: ACCENT, fontSize: 14, fontWeight: 500 }}>Clustering</span>
      </header>

      <main style={{ maxWidth: 860, margin: '0 auto', padding: '40px 24px' }}>
        {!results ? (
          <div>
            <h2 style={{ fontSize: 22, fontWeight: 600, color: '#f0f0f0', marginBottom: 4 }}>Configure model</h2>
            <p style={{ color: '#555', fontSize: 14, marginBottom: 28 }}>Set up your clustering model.</p>

            {/* Algorithm */}
            <Section label='1. Pick algorithm'>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {MODELS.map(m => (
                  <button key={m.id} onClick={() => setModel(m.id)} style={{
                    background: model === m.id ? 'rgba(16,185,129,0.1)' : '#13131a',
                    border: `1px solid ${model === m.id ? ACCENT : 'rgba(255,255,255,0.07)'}`,
                    borderRadius: 10, padding: '12px 16px', cursor: 'pointer', textAlign: 'left', transition: 'all 0.2s',
                  }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: model === m.id ? ACCENT : '#e0e0e0', marginBottom: 3 }}>{m.label}</div>
                    <div style={{ fontSize: 11, color: '#555' }}>{m.tip}</div>
                  </button>
                ))}
              </div>
            </Section>

            {/* Dynamic params */}
            {(model === 'kmeans' || model === 'agglomerative') && (
              <Section label='2. Number of clusters'>
                <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                  <input
                    type='range' min={2} max={12} value={nClusters}
                    onChange={e => setNClusters(Number(e.target.value))}
                    style={{ flex: 1, accentColor: ACCENT }}
                  />
                  <span style={{ fontSize: 24, fontWeight: 600, color: ACCENT, fontFamily: 'monospace', minWidth: 32 }}>{nClusters}</span>
                </div>
                <p style={{ fontSize: 11, color: '#444', marginTop: 6 }}>K = number of cluster groups to find</p>
              </Section>
            )}

            {model === 'dbscan' && (
              <Section label='2. DBSCAN parameters'>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                  <div>
                    <p style={{ fontSize: 11, color: '#666', marginBottom: 8 }}>Epsilon (neighbourhood radius)</p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <input
                        type='range' min={0.1} max={3} step={0.1} value={eps}
                        onChange={e => setEps(parseFloat(e.target.value))}
                        style={{ flex: 1, accentColor: ACCENT }}
                      />
                      <span style={{ fontSize: 16, fontWeight: 600, color: ACCENT, fontFamily: 'monospace', minWidth: 36 }}>{eps}</span>
                    </div>
                  </div>
                  <div>
                    <p style={{ fontSize: 11, color: '#666', marginBottom: 8 }}>Min samples per cluster</p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <input
                        type='range' min={2} max={20} value={minSamples}
                        onChange={e => setMinSamples(Number(e.target.value))}
                        style={{ flex: 1, accentColor: ACCENT }}
                      />
                      <span style={{ fontSize: 16, fontWeight: 600, color: ACCENT, fontFamily: 'monospace', minWidth: 36 }}>{minSamples}</span>
                    </div>
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
                    background: features.includes(c) ? 'rgba(16,185,129,0.15)' : '#13131a',
                    border: `1px solid ${features.includes(c) ? ACCENT : 'rgba(255,255,255,0.08)'}`,
                    color: features.includes(c) ? ACCENT : '#666', transition: 'all 0.15s',
                  }}>{c}</button>
                ))}
              </div>
              <p style={{ fontSize: 11, color: '#444', marginTop: 8 }}>{features.length} selected · PCA will reduce to 2D for visualisation</p>
            </Section>

            {error && (
              <div style={{ padding: '10px 14px', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 8, fontSize: 13, color: '#f87171', marginBottom: 16 }}>
                {error}
              </div>
            )}

            <button onClick={handleTrain} disabled={loading} style={{
              width: '100%', padding: '14px', background: loading ? '#333' : ACCENT,
              border: 'none', borderRadius: 10, color: '#fff', fontSize: 15,
              fontWeight: 600, cursor: loading ? 'default' : 'pointer', transition: 'all 0.2s',
            }}>
              {loading ? 'Clustering...' : 'Run clustering'}
            </button>
          </div>
        ) : (
          <ClusteringResults results={results} onReset={() => setResults(null)} />
        )}
      </main>
    </div>
  )
}

function ClusteringResults({ results, onReset }) {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 22, fontWeight: 600, color: '#f0f0f0', marginBottom: 4 }}>Results</h2>
          <p style={{ color: '#555', fontSize: 13 }}>
            {results.n_clusters_found} cluster{results.n_clusters_found !== 1 ? 's' : ''} found
            {results.n_noise > 0 ? ` · ${results.n_noise} noise points` : ''}
          </p>
        </div>
        <button onClick={onReset} style={{ padding: '8px 16px', background: 'transparent', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#888', fontSize: 13, cursor: 'pointer' }}>← Reconfigure</button>
      </div>

      {/* Silhouette score card */}
      <div style={{ display: 'grid', gridTemplateColumns: results.silhouette !== null ? '1fr 1fr' : '1fr', gap: 10, marginBottom: 24 }}>
        <div style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10, padding: '20px', textAlign: 'center' }}>
          <div style={{ fontSize: 32, fontWeight: 600, color: '#10b981', fontFamily: 'monospace', marginBottom: 4 }}>
            {results.n_clusters_found}
          </div>
          <div style={{ fontSize: 12, color: '#555' }}>Clusters found</div>
        </div>
        {results.silhouette !== null && (
          <div style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10, padding: '20px', textAlign: 'center' }}>
            <div style={{ fontSize: 32, fontWeight: 600, color: '#6c63ff', fontFamily: 'monospace', marginBottom: 4 }}>
              {results.silhouette}
            </div>
            <div style={{ fontSize: 12, color: '#555' }}>Silhouette score</div>
            <div style={{ fontSize: 11, color: '#444', marginTop: 4 }}>−1 worst · 0 overlap · +1 best</div>
          </div>
        )}
      </div>

      {/* PCA Scatter plot */}
      <div style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: '20px', marginBottom: 20 }}>
        <p style={{ fontSize: 12, color: '#555', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>
          2D cluster scatter (PCA)
        </p>
        <ResponsiveContainer width='100%' height={320}>
          <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray='3 3' stroke='rgba(255,255,255,0.04)' />
            <XAxis dataKey='x' name='PC1' tick={{ fill: '#555', fontSize: 11 }} label={{ value: 'PC 1', position: 'insideBottom', offset: -5, fill: '#555', fontSize: 11 }} />
            <YAxis dataKey='y' name='PC2' tick={{ fill: '#555', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#13131a', border: '1px solid #333', borderRadius: 8, color: '#f0f0f0', fontSize: 12 }}
              formatter={(val, name) => [val.toFixed(3), name]}
            />
            <Scatter
              data={results.scatter}
              fill='#10b981'
            >
              {results.scatter.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.cluster === -1 ? noiseColor : CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length]}
                  opacity={entry.cluster === -1 ? 0.3 : 0.8}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
        <p style={{ fontSize: 11, color: '#444', marginTop: 10 }}>Each colour = one cluster · Grey = noise (DBSCAN only)</p>
      </div>

      {/* Cluster size breakdown */}
      <div style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: '20px', marginBottom: 20 }}>
        <p style={{ fontSize: 12, color: '#555', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>Cluster sizes</p>
        <ResponsiveContainer width='100%' height={Math.max(120, results.cluster_sizes.length * 44)}>
          <BarChart data={results.cluster_sizes} layout='vertical' margin={{ top: 0, right: 40, left: 10, bottom: 0 }}>
            <XAxis type='number' tick={{ fill: '#555', fontSize: 11 }} />
            <YAxis type='category' dataKey='label' tick={{ fill: '#aaa', fontSize: 11 }} width={90} />
            <Tooltip contentStyle={{ background: '#13131a', border: '1px solid #333', borderRadius: 8, color: '#f0f0f0' }} />
            <Bar dataKey='size' radius={[0, 4, 4, 0]}>
              {results.cluster_sizes.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.cluster === -1 ? noiseColor : CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length]}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function Section({ label, children }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <p style={{ fontSize: 12, color: '#666', textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 500, marginBottom: 12 }}>{label}</p>
      {children}
    </div>
  )
}