import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadCSV, analyseData } from '../api/client'

const MODELS = [
  { type: 'classification',  label: 'Classification',  desc: 'Predict categories',     accent: '#6c63ff', icon: '◈', sub: 'spam · churn · disease' },
  { type: 'regression',      label: 'Regression',      desc: 'Predict numbers',         accent: '#0ea5e9', icon: '◉', sub: 'price · sales · score' },
  { type: 'clustering',      label: 'Clustering',      desc: 'Group similar data',      accent: '#10b981', icon: '◎', sub: 'segments · anomalies' },
  { type: 'neural-network',  label: 'Neural Network',  desc: 'Deep learning MLP',       accent: '#f59e0b', icon: '◌', sub: 'flexible · powerful' },
]

export default function Home() {
  const [selected, setSelected]   = useState(null)
  const [dragging, setDragging]   = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress]   = useState('')
  const [error, setError]         = useState('')
  const fileRef = useRef()
  const navigate = useNavigate()

  async function handleFile(file) {
    if (!selected) { setError('Select a model type first.'); return }
    if (!file)     return
    if (!file.name.endsWith('.csv')) { setError('Only .csv files are supported.'); return }
    if (file.size > 50 * 1024 * 1024) { setError('File too large. Max 50MB.'); return }

    setError('')
    setUploading(true)

    try {
      setProgress('Reading file...')
      const base64 = await fileToBase64(file)
      if (!base64) throw new Error('File could not be read.')
      sessionStorage.setItem('csvRaw', base64)

      setProgress('Uploading dataset...')
      const data = await uploadCSV(file)
      if (!data.columns || data.columns.length === 0) throw new Error('CSV appears to be empty.')
      sessionStorage.setItem('dataset', JSON.stringify(data))
      sessionStorage.setItem('modelType', selected)

      setProgress('Analysing columns...')
      const viz = await analyseData(file)
      sessionStorage.setItem('vizData', JSON.stringify(viz))
      sessionStorage.setItem('csvFile', file.name)

      setProgress('Done!')
      setTimeout(() => navigate('/visualise'), 300)

    } catch(e) {
      console.error(e)
      setError(e.message || 'Something went wrong. Try again.')
    } finally {
      setUploading(false)
      setProgress('')
    }
  }

  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload  = () => resolve(reader.result?.split(',')[1] || null)
      reader.onerror = () => reject(new Error('File read error'))
      reader.readAsDataURL(file)
    })
  }

  const selectedModel = MODELS.find(m => m.type === selected)

  return (
    <div style={{ minHeight: '100vh' }}>
      {/* Header */}
      <header style={{
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        padding: '18px 40px',
        display: 'flex', alignItems: 'center', gap: 12,
        background: 'rgba(2,2,8,0.8)',
        backdropFilter: 'blur(20px)',
        position: 'sticky', top: 0, zIndex: 50,
      }}>
        <div style={{
          width: 32, height: 32, borderRadius: 8,
          background: 'linear-gradient(135deg, #6c63ff, #4f46e5)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 16, fontWeight: 700, color: '#fff',
          boxShadow: '0 0 16px rgba(108,99,255,0.4)',
        }}>M</div>
        <span style={{ fontWeight: 600, fontSize: 15, letterSpacing: '-0.02em' }}>ML Platform</span>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#10b981', boxShadow: '0 0 8px #10b981' }} />
          <span style={{ fontSize: 11, color: '#555', fontFamily: 'monospace' }}>v1.0 · local</span>
        </div>
      </header>

      <main style={{ maxWidth: 860, margin: '0 auto', padding: '72px 24px 48px' }}>

        {/* Hero */}
        <div className="fade-up" style={{ textAlign: 'center', marginBottom: 64 }}>
          <div style={{
            display: 'inline-block',
            background: 'rgba(108,99,255,0.1)',
            border: '1px solid rgba(108,99,255,0.25)',
            borderRadius: 99, padding: '5px 16px',
            fontSize: 11, color: '#a09af0',
            marginBottom: 24, letterSpacing: '0.1em',
            fontFamily: 'monospace',
          }}>
            NO-CODE · ML · BUILDER
          </div>

          <h1 style={{
            fontSize: 'clamp(36px, 6vw, 60px)',
            fontWeight: 700,
            lineHeight: 1.1,
            letterSpacing: '-0.04em',
            margin: '0 0 20px',
          }}>
            Build ML models.<br />
            <span className="neon">Get the code.</span>
          </h1>

          <p style={{ fontSize: 16, color: '#666', maxWidth: 420, margin: '0 auto', lineHeight: 1.7 }}>
            Upload a CSV, pick a model, configure it, train it — walk away with working Python code.
          </p>
        </div>

        {/* Step 1 — Model selection */}
        <div className="fade-up delay-1" style={{ marginBottom: 40 }}>
          <StepLabel n="1" label="Choose what to build" />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 12 }}>
            {MODELS.map(m => (
              <button
                key={m.type}
                onClick={() => { setSelected(m.type); setError('') }}
                style={{
                  background: selected === m.type ? `rgba(${hexToRgb(m.accent)},0.08)` : 'rgba(255,255,255,0.01)',
                  border: `1px solid ${selected === m.type ? m.accent : 'rgba(255,255,255,0.06)'}`,
                  borderRadius: 14,
                  padding: '20px',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.25s',
                  position: 'relative',
                  overflow: 'hidden',
                }}
                onMouseEnter={e => { if (selected !== m.type) e.currentTarget.style.borderColor = 'rgba(255,255,255,0.15)' }}
                onMouseLeave={e => { if (selected !== m.type) e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)' }}
              >
                {selected === m.type && (
                  <div style={{
                    position: 'absolute', top: 0, left: 0, right: 0, height: 2,
                    background: `linear-gradient(90deg, transparent, ${m.accent}, transparent)`,
                  }} />
                )}
                <div style={{ fontSize: 24, marginBottom: 10, color: m.accent }}>{m.icon}</div>
                <div style={{ fontWeight: 600, fontSize: 15, color: selected === m.type ? m.accent : '#e0e0e0', marginBottom: 4, letterSpacing: '-0.01em' }}>{m.label}</div>
                <div style={{ fontSize: 12, color: '#555', marginBottom: 6 }}>{m.desc}</div>
                <div style={{ fontSize: 11, color: '#3a3a4a', fontFamily: 'monospace' }}>{m.sub}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Step 2 — Upload */}
        <div className="fade-up delay-2">
          <StepLabel n="2" label="Upload your dataset" active={!!selected} />

          <div
            onClick={() => { if (selected && !uploading) fileRef.current.click() }}
            onDragOver={e => { e.preventDefault(); if (selected) setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => { e.preventDefault(); setDragging(false); if (!uploading) handleFile(e.dataTransfer.files[0]) }}
            style={{
              border: `1.5px dashed ${dragging ? (selectedModel?.accent || '#6c63ff') : selected ? 'rgba(108,99,255,0.4)' : 'rgba(255,255,255,0.06)'}`,
              borderRadius: 18,
              padding: '52px 32px',
              textAlign: 'center',
              cursor: selected && !uploading ? 'pointer' : 'default',
              background: dragging ? 'rgba(108,99,255,0.04)' : 'rgba(255,255,255,0.01)',
              transition: 'all 0.25s',
              opacity: selected ? 1 : 0.35,
              animation: selected && !uploading ? 'pulse-border 3s infinite' : 'none',
            }}
          >
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={e => { if (e.target.files[0]) handleFile(e.target.files[0]); e.target.value = '' }}
            />

            {uploading ? (
              <div>
                <div style={{
                  width: 44, height: 44,
                  border: '2px solid rgba(108,99,255,0.2)',
                  borderTop: `2px solid ${selectedModel?.accent || '#6c63ff'}`,
                  borderRadius: '50%',
                  margin: '0 auto 16px',
                  animation: 'spin 0.8s linear infinite',
                }} />
                <div style={{ fontSize: 14, color: '#a09af0', fontWeight: 500 }}>{progress}</div>
                <div style={{ fontSize: 12, color: '#444', marginTop: 6 }}>Please wait...</div>
              </div>
            ) : (
              <div>
                <div style={{
                  width: 56, height: 56,
                  background: selected ? `rgba(${hexToRgb(selectedModel?.accent || '#6c63ff')},0.1)` : 'rgba(255,255,255,0.03)',
                  border: `1px solid ${selected ? (selectedModel?.accent || '#6c63ff') + '40' : 'rgba(255,255,255,0.06)'}`,
                  borderRadius: 14,
                  margin: '0 auto 20px',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 24, transition: 'all 0.3s',
                }}>
                  {dragging ? '⬇' : '📂'}
                </div>
                <div style={{ fontSize: 16, fontWeight: 600, color: '#e0e0e0', marginBottom: 8 }}>
                  {dragging ? 'Drop your file here' : 'Drag & drop your CSV file'}
                </div>
                <div style={{ fontSize: 13, color: '#555', marginBottom: 12 }}>or click to browse your files</div>
                <div style={{ fontSize: 11, color: '#3a3a3a', fontFamily: 'monospace' }}>
                  .csv only · max 50MB · UTF-8 encoding
                </div>
                {!selected && (
                  <div style={{ marginTop: 12, fontSize: 12, color: '#6c63ff' }}>← Select a model type first</div>
                )}
              </div>
            )}
          </div>

          {error && (
            <div style={{
              marginTop: 14,
              padding: '12px 16px',
              background: 'rgba(239,68,68,0.06)',
              border: '1px solid rgba(239,68,68,0.2)',
              borderRadius: 10,
              fontSize: 13,
              color: '#f87171',
              display: 'flex', alignItems: 'center', gap: 10,
            }}>
              <span style={{ fontSize: 16 }}>⚠</span>
              {error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="fade-up delay-4" style={{ marginTop: 52, textAlign: 'center', fontSize: 11, color: '#2a2a3a', fontFamily: 'monospace' }}>
          No data is stored · files processed in memory · discarded after session
        </div>
      </main>
    </div>
  )
}

function StepLabel({ n, label, active = true }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
      <div style={{
        width: 24, height: 24, borderRadius: '50%',
        background: active ? '#6c63ff' : '#1a1a2e',
        border: `1px solid ${active ? '#6c63ff' : 'rgba(255,255,255,0.1)'}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 11, fontWeight: 700,
        color: active ? '#fff' : '#555',
        flexShrink: 0, transition: 'all 0.3s',
        boxShadow: active ? '0 0 12px rgba(108,99,255,0.4)' : 'none',
      }}>{n}</div>
      <span style={{ fontSize: 13, fontWeight: 500, color: active ? '#aaa' : '#444', letterSpacing: '0.05em', textTransform: 'uppercase' }}>{label}</span>
    </div>
  )
}

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1,3), 16)
  const g = parseInt(hex.slice(3,5), 16)
  const b = parseInt(hex.slice(5,7), 16)
  return `${r},${g},${b}`
}