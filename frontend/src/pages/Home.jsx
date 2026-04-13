import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadCSV } from '../api/client'

const MODELS = [
  { type: 'classification', label: 'Classification', desc: 'Predict categories — spam, churn, disease.', accent: '#6c63ff' },
  { type: 'regression',     label: 'Regression',     desc: 'Predict numbers — prices, sales, scores.',  accent: '#0ea5e9' },
  { type: 'clustering',     label: 'Clustering',     desc: 'Group similar data — segments, anomalies.', accent: '#10b981' },
  { type: 'neural-network', label: 'Neural Network', desc: 'Deep learning MLP — flexible, powerful.',   accent: '#f59e0b' },
]

export default function Home() {
  const [selected, setSelected] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError]   = useState('')
  const fileRef  = useRef()
  const navigate = useNavigate()

  async function handleFile(file) {
    if (!selected) { setError('Pick a model type first.'); return }
    if (!file) return
    if (!file.name.endsWith('.csv')) { setError('Only CSV files supported.'); return }
    setError('')
    setUploading(true)
    try {
      const data = await uploadCSV(file)
      sessionStorage.setItem('dataset', JSON.stringify(data))
      sessionStorage.setItem('modelType', selected)
      navigate(`/${selected}`)
    } catch(e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0f' }}>
      <header style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', padding: '20px 40px', display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ width: 28, height: 28, background: '#6c63ff', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, color: '#fff', fontSize: 14 }}>M</div>
        <span style={{ fontWeight: 600, fontSize: 15, color: '#f0f0f0' }}>ML Platform</span>
      </header>

      <main style={{ maxWidth: 820, margin: '0 auto', padding: '64px 24px' }}>
        <div className="fade-up" style={{ textAlign: 'center', marginBottom: 56 }}>
          <div style={{ display: 'inline-block', background: 'rgba(108,99,255,0.12)', border: '1px solid rgba(108,99,255,0.3)', borderRadius: 99, padding: '4px 14px', fontSize: 12, color: '#a09af0', marginBottom: 20, letterSpacing: '0.05em' }}>
            NO-CODE ML BUILDER
          </div>
          <h1 style={{ fontSize: 48, fontWeight: 600, lineHeight: 1.15, letterSpacing: '-0.03em', margin: '0 0 16px', color: '#f0f0f0' }}>
            Build ML models.<br /><span style={{ color: '#6c63ff' }}>Get the code.</span>
          </h1>
          <p style={{ fontSize: 16, color: '#555', maxWidth: 440, margin: '0 auto', lineHeight: 1.6 }}>
            Upload a CSV, pick a model, train it, and get working Python code — no setup needed.
          </p>
        </div>

        <div className="fade-up delay-1" style={{ marginBottom: 36 }}>
          <p style={{ fontSize: 12, color: '#555', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 500, marginBottom: 14 }}>Step 1 — Choose model type</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 10 }}>
            {MODELS.map(m => (
              <button key={m.type} onClick={() => { setSelected(m.type); setError('') }}
                style={{
                  background: selected === m.type ? 'rgba(108,99,255,0.1)' : '#13131a',
                  border: `1px solid ${selected === m.type ? m.accent : 'rgba(255,255,255,0.07)'}`,
                  borderRadius: 12, padding: '18px 16px', cursor: 'pointer', textAlign: 'left', transition: 'all 0.2s',
                }}>
                <div style={{ fontWeight: 600, fontSize: 14, color: selected === m.type ? m.accent : '#e0e0e0', marginBottom: 6 }}>{m.label}</div>
                <div style={{ fontSize: 12, color: '#555', lineHeight: 1.5 }}>{m.desc}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="fade-up delay-2">
          <p style={{ fontSize: 12, color: selected ? '#555' : '#333', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 500, marginBottom: 14, transition: 'color 0.3s' }}>Step 2 — Upload your CSV</p>
          <div
            onClick={() => selected && fileRef.current.click()}
            onDragOver={e => { e.preventDefault(); if (selected) setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }}
            style={{
              border: `1.5px dashed ${dragging ? '#6c63ff' : selected ? 'rgba(108,99,255,0.35)' : 'rgba(255,255,255,0.06)'}`,
              borderRadius: 16, padding: '48px 32px', textAlign: 'center',
              cursor: selected ? 'pointer' : 'default',
              background: dragging ? 'rgba(108,99,255,0.05)' : 'transparent',
              transition: 'all 0.2s', opacity: selected ? 1 : 0.35,
            }}>
            <input ref={fileRef} type="file" accept=".csv" style={{ display: 'none' }} onChange={e => handleFile(e.target.files[0])} />
            {uploading
              ? <div><div style={{ width: 32, height: 32, border: '2px solid rgba(108,99,255,0.3)', borderTop: '2px solid #6c63ff', borderRadius: '50%', margin: '0 auto 12px', animation: 'spin 0.8s linear infinite' }} /><p style={{ color: '#555', fontSize: 14 }}>Uploading...</p></div>
              : <div>
                  <div style={{ fontSize: 32, marginBottom: 12 }}>📂</div>
                  <p style={{ color: '#e0e0e0', fontSize: 15, fontWeight: 500, marginBottom: 6 }}>{dragging ? 'Drop it!' : 'Drag & drop your CSV'}</p>
                  <p style={{ color: '#444', fontSize: 13 }}>or click to browse</p>
                  {!selected && <p style={{ color: '#6c63ff', fontSize: 12, marginTop: 10 }}>← Pick a model type first</p>}
                </div>
            }
          </div>
          {error && <div style={{ marginTop: 10, padding: '10px 14px', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 8, fontSize: 13, color: '#f87171' }}>{error}</div>}
        </div>
      </main>
    </div>
  )
}
