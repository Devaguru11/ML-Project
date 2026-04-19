import { useState } from 'react'
import { generateCode } from '../api/client'

// ─────────────────────────────────────────────────────────
// Usage — drop this inside any results page:
//
// Classification:
//   <CodeExport payload={{ model_type: 'classification', model_name: model, target, features, test_size: testSize }} />
//
// Regression:
//   <CodeExport payload={{ model_type: 'regression', model_name: model, target, features, test_size: testSize }} />
//
// Clustering:
//   <CodeExport payload={{ model_type: 'clustering', model_name: model, features, n_clusters: nClusters, eps, min_samples: minSamples }} />
//
// Neural Network:
//   <CodeExport payload={{ model_type: 'neural', model_name: 'mlp', problem_type: problemType, target, features, test_size: testSize, hidden_layers: layers, activation, max_iter: maxIter }} />
// ─────────────────────────────────────────────────────────

export default function CodeExport({ payload }) {
  const [code, setCode]     = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState('')
  const [copied, setCopied] = useState(false)

  async function handleGenerate() {
    if (!payload || !payload.model_type) {
      setError('No model config found. Train a model first.')
      return
    }
    setLoading(true)
    setError('')
    setCode('')
    try {
      const res = await generateCode(payload)
      setCode(res.code)
    } catch (e) {
      setError(e.message || 'Code generation failed.')
    } finally {
      setLoading(false)
    }
  }

  function handleCopy() {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  function handleDownload() {
    const modelLabel = payload?.model_name || payload?.model_type || 'model'
    const blob = new Blob([code], { type: 'text/plain' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = `ml_${payload.model_type}_${modelLabel}.py`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div style={{
      background: '#13131a',
      border: '1px solid rgba(108,99,255,0.2)',
      borderRadius: 12,
      padding: '20px',
      marginTop: 20,
    }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <div>
          <p style={{ fontSize: 13, fontWeight: 600, color: '#f0f0f0', marginBottom: 3 }}>
            Export Python code
          </p>
          <p style={{ fontSize: 11, color: '#555' }}>
            Ready-to-run script with your exact config
          </p>
        </div>

        {/* Show Generate button only when no code yet */}
        {!code && !loading && (
          <button
            onClick={handleGenerate}
            style={{
              padding: '9px 18px',
              background: '#6c63ff',
              border: 'none',
              borderRadius: 8,
              color: '#fff',
              fontSize: 13,
              fontWeight: 600,
              cursor: 'pointer',
              whiteSpace: 'nowrap',
            }}
          >
            Generate code
          </button>
        )}
      </div>

      {/* Loading spinner */}
      {loading && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 0', color: '#666', fontSize: 13 }}>
          <div style={{
            width: 16, height: 16,
            border: '2px solid rgba(108,99,255,0.3)',
            borderTop: '2px solid #6c63ff',
            borderRadius: '50%',
            animation: 'spin 0.8s linear infinite',
          }} />
          Generating script...
        </div>
      )}

      {/* Error message */}
      {error && (
        <div style={{
          padding: '8px 12px',
          background: 'rgba(239,68,68,0.08)',
          border: '1px solid rgba(239,68,68,0.2)',
          borderRadius: 8,
          fontSize: 12,
          color: '#f87171',
          marginBottom: 12,
        }}>
          {error}
        </div>
      )}

      {/* Code output */}
      {code && (
        <div>
          {/* Action buttons */}
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <button onClick={handleCopy} style={{
              padding: '7px 14px',
              background: copied ? 'rgba(16,185,129,0.15)' : 'rgba(255,255,255,0.05)',
              border: `1px solid ${copied ? '#10b981' : 'rgba(255,255,255,0.1)'}`,
              borderRadius: 7, color: copied ? '#10b981' : '#aaa',
              fontSize: 12, cursor: 'pointer', transition: 'all 0.2s',
            }}>
              {copied ? '✓ Copied!' : 'Copy code'}
            </button>

            <button onClick={handleDownload} style={{
              padding: '7px 14px',
              background: 'rgba(108,99,255,0.12)',
              border: '1px solid rgba(108,99,255,0.3)',
              borderRadius: 7, color: '#a09af0',
              fontSize: 12, cursor: 'pointer',
            }}>
              ↓ Download .py
            </button>

            <button onClick={() => { setCode(''); setError('') }} style={{
              padding: '7px 14px',
              background: 'transparent',
              border: '1px solid rgba(255,255,255,0.07)',
              borderRadius: 7, color: '#555',
              fontSize: 12, cursor: 'pointer',
            }}>
              Regenerate
            </button>
          </div>

          {/* Code block */}
          <pre style={{
            background: '#0d0d16',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 8,
            padding: '16px',
            fontSize: 12,
            fontFamily: "'Fira Code', 'Cascadia Code', 'Consolas', monospace",
            color: '#a9dc76',
            overflowX: 'auto',
            overflowY: 'auto',
            whiteSpace: 'pre',
            maxHeight: 480,
            lineHeight: 1.7,
            margin: 0,
          }}>
            {code}
          </pre>

          <p style={{ fontSize: 11, color: '#444', marginTop: 10 }}>
            Replace <code style={{ color: '#666', background: 'rgba(255,255,255,0.05)', padding: '1px 5px', borderRadius: 3 }}>your_dataset.csv</code> with your actual file path before running.
          </p>
        </div>
      )}

      {/* Spin keyframe — injected once */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}