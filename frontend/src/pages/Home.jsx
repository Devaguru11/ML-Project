import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadCSV, analyseData } from '../api/client'

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
  const [error, setError] = useState('')
  const fileRef = useRef()
  const navigate = useNavigate()

  async function handleFile(file) {
    if (!selected) {
      setError('Pick a model type first.')
      return
    }

    if (!file) return

    if (!file.name.endsWith('.csv')) {
      setError('Only CSV files supported.')
      return
    }

    setError('')
    setUploading(true)

    try {
      // ✅ Convert file to base64 (SAFE)
      const base64 = await fileToBase64(file)
      if (!base64) throw new Error('Failed to convert file')

      sessionStorage.setItem('csvRaw', base64)

      // ✅ Upload CSV
      const data = await uploadCSV(file)
      sessionStorage.setItem('dataset', JSON.stringify(data))
      sessionStorage.setItem('modelType', selected)

      // ✅ Analyse
      const viz = await analyseData(file)
      sessionStorage.setItem('vizData', JSON.stringify(viz))
      sessionStorage.setItem('csvFile', file.name)

      // ✅ Navigate
      navigate('/visualise')

    } catch (e) {
      console.error(e) // 🔥 IMPORTANT for debugging
      setError(e.message || 'Something went wrong')
    } finally {
      setUploading(false)
    }
  }

  // ✅ FIXED BASE64 FUNCTION
  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()

      reader.onload = () => {
        if (!reader.result) {
          reject(new Error('File read failed'))
          return
        }

        const base64 = reader.result.split(',')[1]
        resolve(base64)
      }

      reader.onerror = () => reject(new Error('File read error'))

      reader.readAsDataURL(file)
    })
  }

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0f' }}>
      <header style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', padding: '20px 40px', display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ width: 28, height: 28, background: '#6c63ff', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, color: '#fff', fontSize: 14 }}>M</div>
        <span style={{ fontWeight: 600, fontSize: 15, color: '#f0f0f0' }}>ML Platform</span>
      </header>

      <main style={{ maxWidth: 820, margin: '0 auto', padding: '64px 24px' }}>
        <div style={{ textAlign: 'center', marginBottom: 56 }}>
          <h1 style={{ fontSize: 48, fontWeight: 600, color: '#f0f0f0' }}>
            Build ML models.<br /><span style={{ color: '#6c63ff' }}>Get the code.</span>
          </h1>
        </div>

        {/* MODEL SELECT */}
        <div style={{ marginBottom: 36 }}>
          <p style={{ fontSize: 12, color: '#555', marginBottom: 14 }}>Step 1 — Choose model type</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 10 }}>
            {MODELS.map(m => (
              <button key={m.type} onClick={() => { setSelected(m.type); setError('') }}
                style={{
                  background: selected === m.type ? 'rgba(108,99,255,0.1)' : '#13131a',
                  border: `1px solid ${selected === m.type ? m.accent : 'rgba(255,255,255,0.07)'}`,
                  borderRadius: 12, padding: '18px',
                  cursor: 'pointer',
                }}>
                <div style={{ color: selected === m.type ? m.accent : '#e0e0e0' }}>{m.label}</div>
              </button>
            ))}
          </div>
        </div>

        {/* FILE UPLOAD */}
        <div>
          <p style={{ fontSize: 12, marginBottom: 14 }}>Step 2 — Upload CSV</p>

          <div
            onClick={() => selected && fileRef.current.click()}
            onDragOver={e => { e.preventDefault(); if (selected) setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => {
              e.preventDefault()
              setDragging(false)
              handleFile(e.dataTransfer.files[0])
            }}
            style={{
              border: '1.5px dashed #6c63ff',
              borderRadius: 16,
              padding: 40,
              textAlign: 'center',
              cursor: selected ? 'pointer' : 'not-allowed'
            }}
          >
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={e => handleFile(e.target.files[0])}
            />

            {uploading ? 'Uploading...' : 'Click or drag CSV here'}
          </div>

          {error && <p style={{ color: 'red' }}>{error}</p>}
        </div>
      </main>
    </div>
  )
}