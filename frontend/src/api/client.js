const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function uploadCSV(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form })
  if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Upload failed') }
  return res.json()
}

export async function analyseData(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/analyse`, { method: 'POST', body: form })
  if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Analysis failed') }
  return res.json()
}

export async function trainClassifier(payload) {
  const res = await fetch(`${BASE}/classify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Training failed') }
  return res.json()
}

export async function trainRegressor(payload) {
  const res = await fetch(`${BASE}/regress`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Training failed') }
  return res.json()
}

export async function trainCluster(payload) {
  const res = await fetch(`${BASE}/cluster`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Clustering failed') }
  return res.json()
}

export async function trainNeural(payload) {
  const res = await fetch(`${BASE}/neural`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Training failed') }
  return res.json()
}

export async function generateCode(payload) {
  const res = await fetch(`${BASE}/generate-code`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Code generation failed') }
  return res.json()
}