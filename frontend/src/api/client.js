const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ── Upload (multipart) ──────────────────────────────────────
export async function uploadCSV(file) {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: fd })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Upload failed') }
  return res.json()
}

// ── Visualisation analysis (multipart) ─────────────────────
export async function analyseData(file) {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(`${BASE}/analyse`, { method: 'POST', body: fd })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Analysis failed') }
  return res.json()
}

// ── Deep dataset analysis + model suggestion ───────────────
export async function analyseDataset(csvData) {
  const res = await fetch(`${BASE}/analyse-dataset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ csv_data: csvData }),
  })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Analysis failed') }
  return res.json()
}

// ── Preprocess dataset ─────────────────────────────────────
export async function preprocessDataset(csvData, options = {}) {
  const res = await fetch(`${BASE}/preprocess`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ csv_data: csvData, ...options }),
  })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Preprocessing failed') }
  return res.json()
}

// ── Code export ────────────────────────────────────────────
export async function generateCode(payload) {
  const res = await fetch(`${BASE}/generate-code`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Code generation failed') }
  return res.json()
}

// ── Streaming train helpers ────────────────────────────────
export function trainClassifierStream(payload, onEvent) {
  return streamTrain(`${BASE}/classify`, payload, onEvent)
}
export function trainRegressorStream(payload, onEvent) {
  return streamTrain(`${BASE}/regress`, payload, onEvent)
}
export function trainClusterStream(payload, onEvent) {
  return streamTrain(`${BASE}/cluster`, payload, onEvent)
}
export function trainNeuralStream(payload, onEvent) {
  return streamTrain(`${BASE}/neural`, payload, onEvent)
}

async function streamTrain(url, payload, onEvent) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const e = await res.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(e.detail || 'Training failed')
  }
  const reader  = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop()
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try { onEvent(JSON.parse(line.slice(6))) } catch {} // eslint-disable-line no-empty
      }
    }
  }
}