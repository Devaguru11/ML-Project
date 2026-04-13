import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'

const META = {
  'classification': { label: 'Classification', accent: '#6c63ff' },
  'regression':     { label: 'Regression',     accent: '#0ea5e9' },
  'clustering':     { label: 'Clustering',     accent: '#10b981' },
  'neural-network': { label: 'Neural Network', accent: '#f59e0b' },
}

export default function ModelPage({ type }) {
  const [ds, setDs]  = useState(null)
  const navigate     = useNavigate()
  const meta         = META[type]

  useEffect(() => {
    const raw = sessionStorage.getItem('dataset')
    if (!raw) { navigate('/'); return }
    setDs(JSON.parse(raw))
  }, [navigate])

  if (!ds) return null

  const numCols = ds.columns.filter(c => c.dtype.includes('int') || c.dtype.includes('float'))
  const missing = ds.columns.filter(c => c.missing > 0)

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0f' }}>
      <header style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', padding: '16px 40px', display: 'flex', alignItems: 'center', gap: 12 }}>
        <Link to="/" style={{ color: '#555', fontSize: 13, textDecoration: 'none' }}>← Back</Link>
        <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)' }} />
        <span style={{ color: meta.accent, fontSize: 14, fontWeight: 500 }}>{meta.label}</span>
        <span style={{ marginLeft: 'auto', fontSize: 12, color: '#444', fontFamily: 'monospace' }}>{ds.filename}</span>
      </header>

      <main style={{ maxWidth: 820, margin: '0 auto', padding: '48px 24px' }}>
        <h2 className="fade-up" style={{ fontSize: 24, fontWeight: 600, color: '#f0f0f0', marginBottom: 6 }}>Dataset loaded</h2>
        <p className="fade-up" style={{ color: '#555', fontSize: 14, marginBottom: 28 }}>Here's a summary of your data.</p>

        <div className="fade-up delay-1" style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 28 }}>
          {[
            { label: 'Rows',         value: ds.rows.toLocaleString() },
            { label: 'Columns',      value: ds.columns.length },
            { label: 'Numeric cols', value: numCols.length },
            { label: 'Missing cols', value: missing.length || 'None' },
          ].map(s => (
            <div key={s.label} style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10, padding: '16px 14px' }}>
              <div style={{ fontSize: 22, fontWeight: 600, color: '#f0f0f0', fontFamily: 'monospace', marginBottom: 4 }}>{s.value}</div>
              <div style={{ fontSize: 12, color: '#555' }}>{s.label}</div>
            </div>
          ))}
        </div>

        <div className="fade-up delay-2" style={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, overflow: 'hidden', marginBottom: 24 }}>
          <div style={{ padding: '12px 16px', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'grid', gridTemplateColumns: '1fr 120px 70px 70px' }}>
            {['Column','Type','Unique','Missing'].map(h => (
              <span key={h} style={{ fontSize: 11, color: '#444', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 500 }}>{h}</span>
            ))}
          </div>
          {ds.columns.map((col, i) => (
            <div key={col.name} style={{ padding: '10px 16px', display: 'grid', gridTemplateColumns: '1fr 120px 70px 70px', borderBottom: i < ds.columns.length-1 ? '1px solid rgba(255,255,255,0.04)' : 'none', alignItems: 'center' }}>
              <span style={{ fontSize: 13, color: '#ddd', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{col.name}</span>
              <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 99, background: col.dtype.includes('int') || col.dtype.includes('float') ? 'rgba(14,165,233,0.1)' : 'rgba(108,99,255,0.1)', color: col.dtype.includes('int') || col.dtype.includes('float') ? '#0ea5e9' : '#a09af0', fontFamily: 'monospace', display: 'inline-block' }}>{col.dtype}</span>
              <span style={{ fontSize: 13, color: '#555', fontFamily: 'monospace' }}>{col.unique}</span>
              <span style={{ fontSize: 13, fontFamily: 'monospace', color: col.missing > 0 ? '#f87171' : '#555' }}>{col.missing > 0 ? col.missing : '—'}</span>
            </div>
          ))}
        </div>

        <div className="fade-up delay-3">
          <p style={{ fontSize: 11, color: '#444', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 500, marginBottom: 10 }}>First 5 rows</p>
          <div style={{ overflowX: 'auto', borderRadius: 10, border: '1px solid rgba(255,255,255,0.07)' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ background: '#13131a', borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
                  {ds.columns.map(c => <th key={c.name} style={{ padding: '10px 14px', textAlign: 'left', color: '#555', fontWeight: 500, whiteSpace: 'nowrap' }}>{c.name}</th>)}
                </tr>
              </thead>
              <tbody>
                {ds.preview.map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    {ds.columns.map(c => <td key={c.name} style={{ padding: '8px 14px', color: '#777', whiteSpace: 'nowrap' }}>{row[c.name]}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="fade-up delay-4" style={{ marginTop: 32, background: 'rgba(108,99,255,0.06)', border: '1px solid rgba(108,99,255,0.2)', borderRadius: 12, padding: '20px 24px' }}>
          <div style={{ fontWeight: 600, fontSize: 14, color: '#f0f0f0', marginBottom: 4 }}>🚧 Day 2 coming next</div>
          <div style={{ fontSize: 13, color: '#666' }}>Data visualisation — histogram, correlation heatmap, scatter plot, missing values chart.</div>
        </div>
      </main>
    </div>
  )
}
