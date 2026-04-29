import { useState, useRef } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { uploadCSV, analyseData, analyseDataset } from '../api/client'

const ACCEPTED = {
  '.csv':  { label: 'CSV',    mime: 'text/csv' },
  '.xlsx': { label: 'Excel',  mime: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' },
  '.xls':  { label: 'Excel',  mime: 'application/vnd.ms-excel' },
  '.tsv':  { label: 'TSV',    mime: 'text/tab-separated-values' },
  '.json': { label: 'JSON',   mime: 'application/json' },
}

const STAGES = [
  { id: 'read',     label: 'Reading file',        icon: '📄' },
  { id: 'parse',    label: 'Parsing data',         icon: '⚙' },
  { id: 'upload',   label: 'Uploading',            icon: '⬆' },
  { id: 'analyse',  label: 'Analysing dataset',    icon: '🔍' },
  { id: 'suggest',  label: 'Detecting model type', icon: '🤖' },
]

export default function UploadPage() {
  const [dragging, setDragging]   = useState(false)
  const [file, setFile]           = useState(null)
  const [stage, setStage]         = useState('')     // current stage id
  const [stagePct, setStagePct]   = useState(0)
  const [stageMsg, setStageMsg]   = useState('')
  const [error, setError]         = useState('')
  const [preview, setPreview]     = useState(null)   // {headers, rows, format, rowCount}
  const fileRef = useRef()
  const navigate = useNavigate()

  function getExt(name) {
    return ('.' + name.split('.').pop()).toLowerCase()
  }

  function isAccepted(name) {
    return getExt(name) in ACCEPTED
  }

  // ── Convert any supported format to CSV string ──────────────
  async function toCSVString(file) {
    const ext = getExt(file.name)

    if (ext === '.csv' || ext === '.tsv') {
      return new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result)
        reader.onerror = reject
        reader.readAsText(file)
      })
    }

    if (ext === '.xlsx' || ext === '.xls') {
      // Dynamically import SheetJS
      const XLSX = await import('https://cdn.jsdelivr.net/npm/xlsx@0.18.5/xlsx.mjs')
      const buf  = await file.arrayBuffer()
      const wb   = XLSX.read(buf, { type: 'array' })
      const ws   = wb.Sheets[wb.SheetNames[0]]
      return XLSX.utils.sheet_to_csv(ws)
    }

    if (ext === '.json') {
      const text = await file.text()
      const data = JSON.parse(text)
      // Handle array of objects
      const arr = Array.isArray(data) ? data : (data.data || Object.values(data)[0] || [])
      if (!arr.length) throw new Error('JSON file must contain an array of objects.')
      const keys = Object.keys(arr[0])
      const csv  = [keys.join(','), ...arr.map(row => keys.map(k => {
        const v = row[k] ?? ''
        return String(v).includes(',') ? `"${v}"` : v
      }).join(','))].join('\n')
      return csv
    }

    throw new Error(`Unsupported format: ${ext}`)
  }

  // ── Parse CSV string to preview ──────────────────────────────
  function parsePreview(csvStr) {
    const lines = csvStr.trim().split('\n').filter(Boolean)
    if (lines.length < 2) throw new Error('File appears to be empty.')
    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''))
    const rows    = lines.slice(1, 6).map(line => {
      const cells = line.split(',').map(c => c.trim().replace(/^"|"$/g, ''))
      return headers.reduce((obj, h, i) => { obj[h] = cells[i] ?? ''; return obj }, {})
    })
    return { headers, rows, rowCount: lines.length - 1 }
  }

  // ── fileToBase64 ─────────────────────────────────────────────
  function csvToBase64(csvStr) {
    return btoa(unescape(encodeURIComponent(csvStr)))
  }

  // ── Main handler ─────────────────────────────────────────────
  async function handleFile(rawFile) {
    if (!rawFile) return
    if (!isAccepted(rawFile.name)) {
      setError(`Unsupported format. Accepted: ${Object.keys(ACCEPTED).join(', ')}`)
      return
    }
    if (rawFile.size > 100 * 1024 * 1024) {
      setError('File too large. Maximum 100MB.')
      return
    }

    setFile(rawFile)
    setError('')
    setPreview(null)

    try {
      // Stage 1 — read
      setStage('read'); setStagePct(10); setStageMsg('Reading file from disk...')
      await tick()

      const csvStr = await toCSVString(rawFile)

      // Stage 2 — parse
      setStage('parse'); setStagePct(25); setStageMsg('Parsing rows and columns...')
      await tick()

      const previewData = parsePreview(csvStr)
      setPreview({ ...previewData, format: getExt(rawFile.name).replace('.','').toUpperCase() })

      const base64 = csvToBase64(csvStr)

      // Stage 3 — upload (create a CSV blob so the multipart endpoint works)
      setStage('upload'); setStagePct(45); setStageMsg('Sending to server...')
      await tick()

      const csvBlob = new Blob([csvStr], { type: 'text/csv' })
      const csvFile = new File([csvBlob], rawFile.name.replace(/\.[^.]+$/, '.csv'), { type: 'text/csv' })

      const [uploadResult, vizResult] = await Promise.all([
        uploadCSV(csvFile),
        analyseData(csvFile),
      ])

      // Stage 4 — analyse
      setStage('analyse'); setStagePct(70); setStageMsg('Analysing dataset health and structure...')
      await tick()

      const analysisResult = await analyseDataset(base64)

      // Stage 5 — detect
      setStage('suggest'); setStagePct(92); setStageMsg('Detecting best ML approach...')
      await tick()

      // Persist everything
      sessionStorage.setItem('csvRaw',      base64)
      sessionStorage.setItem('csvFile',     rawFile.name)
      sessionStorage.setItem('dataset',     JSON.stringify(uploadResult))
      sessionStorage.setItem('vizData',     JSON.stringify(vizResult))
      sessionStorage.setItem('dsAnalysis',  JSON.stringify(analysisResult))

      setStagePct(100); setStageMsg('Done! Redirecting...')
      await new Promise(r => setTimeout(r, 400))

      navigate('/dataset-report')

    } catch (e) {
      console.error(e)
      setError(e.message || 'Something went wrong.')
      setStage(''); setStagePct(0)
    }
  }

  function tick() { return new Promise(r => setTimeout(r, 180)) }

  const isProcessing = !!stage && stagePct < 100

  return (
    <div style={{ minHeight: '100vh' }}>
      {/* Header */}
      <header style={{
        borderBottom: '1px solid rgba(255,255,255,0.05)', padding: '16px 40px',
        display: 'flex', alignItems: 'center', gap: 14,
        background: 'rgba(2,2,8,0.85)', backdropFilter: 'blur(20px)',
        position: 'sticky', top: 0, zIndex: 50,
      }}>
        <Link to='/' style={{ color: '#555', fontSize: 13, textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 6, transition: 'color 0.2s' }}
          onMouseEnter={e => e.currentTarget.style.color = '#aaa'}
          onMouseLeave={e => e.currentTarget.style.color = '#555'}
        >
          ← Back
        </Link>
        <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)' }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 24, height: 24, borderRadius: 6, background: 'linear-gradient(135deg,#6c63ff,#4f46e5)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, fontWeight: 800, color: '#fff' }}>M</div>
          <span style={{ fontWeight: 600, fontSize: 14 }}>ML Platform</span>
        </div>
        {/* Step indicator */}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
          {['Upload', 'Report', 'Configure', 'Train', 'Results'].map((s, i) => (
            <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{
                width: 22, height: 22, borderRadius: '50%',
                background: i === 0 ? '#6c63ff' : 'rgba(255,255,255,0.05)',
                border: `1px solid ${i === 0 ? '#6c63ff' : 'rgba(255,255,255,0.1)'}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 10, fontWeight: 700, color: i === 0 ? '#fff' : '#555',
                boxShadow: i === 0 ? '0 0 8px rgba(108,99,255,0.5)' : 'none',
              }}>{i + 1}</div>
              <span style={{ fontSize: 11, color: i === 0 ? '#a09af0' : '#444', whiteSpace: 'nowrap' }}>{s}</span>
              {i < 4 && <div style={{ width: 16, height: 1, background: 'rgba(255,255,255,0.06)' }} />}
            </div>
          ))}
        </div>
      </header>

      <main style={{ maxWidth: 760, margin: '0 auto', padding: '56px 24px' }}>
        <div style={{ textAlign: 'center', marginBottom: 48 }}>
          <div style={{ fontSize: 11, color: '#6c63ff', textTransform: 'uppercase', letterSpacing: '0.12em', fontFamily: 'monospace', marginBottom: 14 }}>STEP 1 OF 5</div>
          <h1 style={{ fontSize: 32, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 10 }}>Upload your dataset</h1>
          <p style={{ fontSize: 15, color: '#555', lineHeight: 1.6 }}>
            We'll analyse it, detect the right ML approach, and guide you from there.
          </p>
        </div>

        {/* ── Processing overlay ── */}
        {isProcessing && (
          <div style={{
            background: 'rgba(2,2,8,0.95)', border: '1px solid rgba(108,99,255,0.25)',
            borderRadius: 20, padding: '40px', marginBottom: 28, textAlign: 'center',
          }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 8, marginBottom: 32 }}>
              {STAGES.map((s, i) => {
                const currentIdx = STAGES.findIndex(x => x.id === stage)
                const done = i < currentIdx
                const active = s.id === stage
                return (
                  <div key={s.id} style={{ textAlign: 'center' }}>
                    <div style={{
                      width: 44, height: 44, borderRadius: '50%', margin: '0 auto 8px',
                      background: done ? 'rgba(16,185,129,0.15)' : active ? 'rgba(108,99,255,0.15)' : 'rgba(255,255,255,0.03)',
                      border: `2px solid ${done ? '#10b981' : active ? '#6c63ff' : 'rgba(255,255,255,0.08)'}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 18, transition: 'all 0.3s',
                      boxShadow: active ? '0 0 16px rgba(108,99,255,0.4)' : 'none',
                    }}>
                      {done ? '✓' : s.icon}
                    </div>
                    <div style={{ fontSize: 10, color: done ? '#10b981' : active ? '#a09af0' : '#444', lineHeight: 1.3 }}>{s.label}</div>
                  </div>
                )
              })}
            </div>

            {/* Progress bar */}
            <div style={{ marginBottom: 14 }}>
              <div style={{ height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 99, overflow: 'hidden', marginBottom: 10 }}>
                <div style={{
                  height: '100%', width: `${stagePct}%`,
                  background: 'linear-gradient(90deg,#6c63ff,#0ea5e9)',
                  borderRadius: 99, transition: 'width 0.4s ease',
                  boxShadow: '0 0 12px rgba(108,99,255,0.5)',
                }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                <span style={{ color: '#888' }}>{stageMsg}</span>
                <span style={{ color: '#6c63ff', fontFamily: 'monospace', fontWeight: 700 }}>{stagePct}%</span>
              </div>
            </div>

            {/* File name */}
            {file && (
              <div style={{ fontSize: 12, color: '#444', fontFamily: 'monospace' }}>
                {file.name} · {(file.size / 1024).toFixed(1)} KB
              </div>
            )}
          </div>
        )}

        {/* ── Drop zone ── */}
        {!isProcessing && (
          <>
            <div
              onClick={() => fileRef.current.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }}
              style={{
                border: `2px dashed ${dragging ? '#6c63ff' : 'rgba(108,99,255,0.3)'}`,
                borderRadius: 20, padding: '64px 40px', textAlign: 'center',
                cursor: 'pointer', background: dragging ? 'rgba(108,99,255,0.04)' : 'rgba(255,255,255,0.01)',
                transition: 'all 0.25s',
                animation: 'pulse-border 3s infinite',
              }}
            >
              <input
                ref={fileRef} type='file'
                accept='.csv,.xlsx,.xls,.tsv,.json'
                style={{ display: 'none' }}
                onChange={e => { if (e.target.files[0]) handleFile(e.target.files[0]); e.target.value = '' }}
              />

              <div style={{
                width: 72, height: 72, borderRadius: 18, margin: '0 auto 24px',
                background: dragging ? 'rgba(108,99,255,0.2)' : 'rgba(108,99,255,0.08)',
                border: `1px solid ${dragging ? '#6c63ff' : 'rgba(108,99,255,0.25)'}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 30, transition: 'all 0.3s',
              }}>
                {dragging ? '⬇' : '📂'}
              </div>

              <div style={{ fontSize: 20, fontWeight: 700, color: '#e8e8f0', marginBottom: 10, letterSpacing: '-0.01em' }}>
                {dragging ? 'Drop it here!' : 'Drag & drop your dataset'}
              </div>
              <div style={{ fontSize: 14, color: '#555', marginBottom: 20 }}>or click to browse your files</div>

              {/* Format badges */}
              <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
                {Object.entries(ACCEPTED).map(([ext, info]) => (
                  <span key={ext} style={{
                    fontSize: 11, fontFamily: 'monospace',
                    background: 'rgba(108,99,255,0.08)', border: '1px solid rgba(108,99,255,0.2)',
                    borderRadius: 6, padding: '3px 10px', color: '#7c75cc',
                  }}>{info.label} {ext}</span>
                ))}
              </div>
              <div style={{ fontSize: 11, color: '#3a3a3a', marginTop: 16, fontFamily: 'monospace' }}>Max 100MB · UTF-8 encoding recommended</div>
            </div>

            {error && (
              <div style={{
                marginTop: 16, padding: '12px 16px',
                background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.2)',
                borderRadius: 10, fontSize: 13, color: '#f87171',
                display: 'flex', gap: 10, alignItems: 'center',
              }}>
                <span>⚠</span> {error}
              </div>
            )}

            {/* ── Preview table ── */}
            {preview && (
              <div style={{ marginTop: 28, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 16, overflow: 'hidden' }}>
                <div style={{ padding: '14px 20px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ fontSize: 12, color: '#10b981', fontFamily: 'monospace', background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.2)', borderRadius: 6, padding: '2px 10px' }}>
                    {preview.format}
                  </span>
                  <span style={{ fontSize: 13, color: '#888' }}>
                    {preview.rowCount.toLocaleString()} rows · {preview.headers.length} columns
                  </span>
                  <span style={{ marginLeft: 'auto', fontSize: 11, color: '#555' }}>Preview (first 5 rows)</span>
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ background: 'rgba(255,255,255,0.02)' }}>
                        {preview.headers.map(h => (
                          <th key={h} style={{ padding: '10px 14px', textAlign: 'left', color: '#6c63ff', fontWeight: 600, fontFamily: 'monospace', fontSize: 11, letterSpacing: '0.04em', borderBottom: '1px solid rgba(255,255,255,0.05)', whiteSpace: 'nowrap' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {preview.rows.map((row, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                          {preview.headers.map(h => (
                            <td key={h} style={{ padding: '9px 14px', color: '#aaa', fontFamily: 'monospace', fontSize: 11, whiteSpace: 'nowrap', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis' }}>{row[h]}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Tips */}
            <div style={{ marginTop: 32, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {[
                { icon: '✓', text: 'First row should be column headers' },
                { icon: '✓', text: 'Numeric columns used as features' },
                { icon: '✓', text: 'Categorical columns detected as targets' },
                { icon: '✓', text: 'Missing values handled automatically' },
              ].map((t, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#555' }}>
                  <span style={{ color: '#10b981', fontWeight: 700 }}>{t.icon}</span>
                  {t.text}
                </div>
              ))}
            </div>
          </>
        )}
      </main>
    </div>
  )
}