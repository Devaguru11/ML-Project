import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, CartesianGrid, Cell, ReferenceLine,
} from 'recharts'

const TABS = [
  { id: 0, label: 'Distribution',  icon: '▪', desc: 'How values spread across each column' },
  { id: 1, label: 'Correlation',   icon: '▫', desc: 'Which features move together' },
  { id: 2, label: 'Scatter',       icon: '·', desc: 'Relationship between two columns' },
  { id: 3, label: 'Missing Data',  icon: '○', desc: 'Gaps in your dataset' },
  { id: 4, label: 'Spread',        icon: '—', desc: 'Outliers and value range' },
]

const ACCENT = '#6c63ff'

const cardStyle = {
  background: 'rgba(255,255,255,0.02)',
  border: '1px solid rgba(255,255,255,0.06)',
  borderRadius: 14,
  padding: '20px 20px',
  marginBottom: 16,
}

export default function Visualise() {
  const [viz, setViz]           = useState(null)
  const [tab, setTab]           = useState(0)
  const [col, setCol]           = useState('')
  const [scatterX, setScatterX] = useState('')
  const [scatterY, setScatterY] = useState('')
  const [insights, setInsights] = useState([])
  const navigate                = useNavigate()

  useEffect(() => {
    const v = sessionStorage.getItem('vizData')
    const d = sessionStorage.getItem('dataset')
    if (!v || !d) { navigate('/'); return }
    const vizData = JSON.parse(v)
    setViz(vizData)
    if (vizData.numeric_columns.length > 0) {
      setCol(vizData.numeric_columns[0])
      setScatterX(vizData.numeric_columns[0])
      setScatterY(vizData.numeric_columns[Math.min(1, vizData.numeric_columns.length - 1)])
    }
    setInsights(generateInsights(vizData))
  }, [navigate])

  if (!viz) return null
  const modelType = sessionStorage.getItem('modelType') || 'classification'

  function generateInsights(v) {
    const tips = []
    const highMissing = v.missing.filter(m => m.pct > 20)
    if (highMissing.length > 0) tips.push({ type: 'warn', text: `${highMissing.map(m => m.column).join(', ')} ${highMissing.length > 1 ? 'have' : 'has'} >20% missing values — consider dropping or imputing.` })

    if (v.correlation?.matrix) {
      const cols = v.correlation.columns
      const mat  = v.correlation.matrix
      const highCorr = []
      for (let i = 0; i < cols.length; i++) {
        for (let j = i + 1; j < cols.length; j++) {
          if (Math.abs(mat[i][j]) > 0.85) highCorr.push(`${cols[i]} & ${cols[j]}`)
        }
      }
      if (highCorr.length > 0) tips.push({ type: 'info', text: `${highCorr[0]} are highly correlated (>0.85). Consider removing one to avoid redundancy.` })
    }

    const hasOutliers = Object.entries(v.boxplots || {}).filter(([, b]) => b.outliers?.length > 5)
    if (hasOutliers.length > 0) tips.push({ type: 'warn', text: `${hasOutliers[0][0]} has ${hasOutliers[0][1].outliers.length} outliers which may affect model accuracy.` })

    if (tips.length === 0) tips.push({ type: 'ok', text: 'Your dataset looks clean! No major issues detected. Ready to train.' })
    return tips
  }

  // ── HISTOGRAM ──
  function HistogramChart() {
    if (!col || !viz.histograms[col]) return null
    const h = viz.histograms[col]
    const data = h.bins.map((b, i) => ({ bin: parseFloat(b.toFixed(2)), count: h.counts[i] }))
    const maxCount = Math.max(...data.map(d => d.count))
    const mostCommon = data.find(d => d.count === maxCount)

    return (
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
          <ColSelect value={col} onChange={setCol} cols={viz.numeric_columns} label='Column' />
          <div style={{ display: 'flex', gap: 10 }}>
            <Stat label="Min" value={Math.min(...data.map(d => d.bin))} color="#6c63ff" />
            <Stat label="Max" value={Math.max(...data.map(d => d.bin))} color="#0ea5e9" />
            <Stat label="Most common" value={`~${mostCommon?.bin}`} color="#10b981" />
          </div>
        </div>

        <Explainer text={`Each bar shows how many rows have values in that range for "${col}". Tall bars = that value is common in your data.`} />

        <ResponsiveContainer width='100%' height={280}>
          <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 30 }}>
            <XAxis dataKey='bin' tick={{ fill: '#555', fontSize: 10 }} angle={-35} textAnchor='end' interval='preserveStartEnd' />
            <YAxis tick={{ fill: '#555', fontSize: 11 }} label={{ value: 'Rows', angle: -90, position: 'insideLeft', fill: '#444', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#0d0d18', border: '1px solid rgba(108,99,255,0.3)', borderRadius: 8, color: '#f0f0f0', fontSize: 12 }}
              formatter={(v) => [`${v} rows`, 'Count']}
              labelFormatter={(l) => `Value ~${l}`}
            />
            <Bar dataKey='count' radius={[4, 4, 0, 0]}>
              {data.map((d, i) => (
                <Cell key={i} fill={d.count === maxCount ? '#6c63ff' : 'rgba(108,99,255,0.3)'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <InsightBox text={`The most frequent value range for "${col}" is around ${mostCommon?.bin}. ${data[0].count > data[data.length-1].count ? 'Values skew lower — there are more small values.' : 'Values skew higher — there are more large values.'}`} />
      </div>
    )
  }

  // ── CORRELATION ──
  function CorrelationChart() {
    const { columns, matrix } = viz.correlation
    if (!columns || columns.length === 0) return <NoData msg='No numeric columns for correlation.' />

    function getColor(v) {
      if (v >= 1)    return '#4338ca'
      if (v > 0.7)   return '#6c63ff'
      if (v > 0.4)   return '#a09af0'
      if (v > 0.1)   return '#d4d2f8'
      if (v > -0.1)  return '#1e1e2e'
      if (v > -0.4)  return '#f8d2d2'
      if (v > -0.7)  return '#f09595'
      return '#e24b4a'
    }

    function getTextColor(v) { return Math.abs(v) > 0.4 ? '#fff' : '#666' }

    const size = Math.max(36, Math.min(72, Math.floor(520 / columns.length)))
    const strongPairs = []
    for (let i = 0; i < columns.length; i++) {
      for (let j = i + 1; j < columns.length; j++) {
        if (Math.abs(matrix[i][j]) > 0.6) strongPairs.push({ a: columns[i], b: columns[j], v: matrix[i][j] })
      }
    }

    return (
      <div>
        <Explainer text="Each cell shows how strongly two columns are related. Purple = they go up together. Red = one goes up while the other goes down. White = no relationship." />

        <div style={{ overflowX: 'auto', marginBottom: 16 }}>
          <div style={{ display: 'inline-block' }}>
            <div style={{ display: 'flex', marginLeft: size + 8 }}>
              {columns.map(c => (
                <div key={c} style={{ width: size, fontSize: 9, color: '#777', textAlign: 'center', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', padding: '0 2px', marginBottom: 4 }} title={c}>{c}</div>
              ))}
            </div>
            {matrix.map((row, ri) => (
              <div key={ri} style={{ display: 'flex', alignItems: 'center', marginBottom: 2 }}>
                <div style={{ width: size, fontSize: 9, color: '#777', textAlign: 'right', paddingRight: 8, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flexShrink: 0 }} title={columns[ri]}>{columns[ri]}</div>
                {row.map((val, ci) => (
                  <div key={ci}
                    title={`${columns[ri]} vs ${columns[ci]}: ${val} — ${val > 0.7 ? 'Strong positive' : val > 0.3 ? 'Moderate positive' : val < -0.7 ? 'Strong negative' : val < -0.3 ? 'Moderate negative' : 'Weak/no relationship'}`}
                    style={{
                      width: size, height: size, background: getColor(val),
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: Math.max(8, size / 5),
                      color: getTextColor(val),
                      margin: 1, borderRadius: 4, cursor: 'default',
                      border: ri === ci ? '1.5px solid rgba(255,255,255,0.2)' : 'none',
                      fontWeight: ri === ci ? 700 : 400,
                      transition: 'opacity 0.15s',
                    }}>
                    {val.toFixed(1)}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
          {[['#6c63ff','Strong positive'],['#a09af0','Moderate positive'],['#1e1e2e','No relation'],['#f09595','Moderate negative'],['#e24b4a','Strong negative']].map(([c, l]) => (
            <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#666' }}>
              <div style={{ width: 14, height: 14, borderRadius: 3, background: c, border: '1px solid rgba(255,255,255,0.1)' }} />
              {l}
            </div>
          ))}
        </div>

        {strongPairs.length > 0 && (
          <InsightBox text={`Strong relationship found: "${strongPairs[0].a}" and "${strongPairs[0].b}" are ${strongPairs[0].v > 0 ? 'positively' : 'negatively'} correlated (${strongPairs[0].v.toFixed(2)}). ${strongPairs[0].v > 0.85 ? 'Consider removing one to avoid redundancy.' : 'This may help the model make better predictions.'}`} />
        )}
      </div>
    )
  }

  // ── SCATTER ──
  function ScatterPlot() {
    if (viz.numeric_columns.length < 2) return <NoData msg='Need at least 2 numeric columns.' />
    const data = viz.scatter_data.map(row => ({ x: row[scatterX], y: row[scatterY] })).filter(d => d.x != null && d.y != null)
    const xVals = data.map(d => d.x)
    const yVals = data.map(d => d.y)
    const xMean = xVals.reduce((a, b) => a + b, 0) / xVals.length
    const yMean = yVals.reduce((a, b) => a + b, 0) / yVals.length

    return (
      <div>
        <div style={{ display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
          <ColSelect value={scatterX} onChange={setScatterX} cols={viz.numeric_columns} label='X axis' />
          <ColSelect value={scatterY} onChange={setScatterY} cols={viz.numeric_columns} label='Y axis' />
          <div style={{ display: 'flex', gap: 10 }}>
            <Stat label="Avg X" value={xMean.toFixed(1)} color="#6c63ff" />
            <Stat label="Avg Y" value={yMean.toFixed(1)} color="#0ea5e9" />
          </div>
        </div>

        <Explainer text={`Each dot is one row in your dataset, plotted by "${scatterX}" (left-right) and "${scatterY}" (up-down). A diagonal pattern means the two columns are related.`} />

        <ResponsiveContainer width='100%' height={300}>
          <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray='3 3' stroke='rgba(255,255,255,0.04)' />
            <XAxis dataKey='x' name={scatterX} tick={{ fill: '#555', fontSize: 11 }} label={{ value: scatterX, position: 'insideBottom', offset: -10, fill: '#555', fontSize: 11 }} />
            <YAxis dataKey='y' name={scatterY} tick={{ fill: '#555', fontSize: 11 }} label={{ value: scatterY, angle: -90, position: 'insideLeft', fill: '#555', fontSize: 11 }} />
            <ReferenceLine x={xMean} stroke="rgba(108,99,255,0.3)" strokeDasharray="4 4" />
            <ReferenceLine y={yMean} stroke="rgba(14,165,233,0.3)" strokeDasharray="4 4" />
            <Tooltip
              contentStyle={{ background: '#0d0d18', border: '1px solid rgba(108,99,255,0.3)', borderRadius: 8, color: '#f0f0f0', fontSize: 12 }}
              formatter={(v, name) => [v?.toFixed(3), name]}
            />
            <Scatter data={data} fill={ACCENT} opacity={0.6} />
          </ScatterChart>
        </ResponsiveContainer>

        <InsightBox text={`Dashed lines show the averages. ${data.length} data points plotted (first 200 rows). Look for a diagonal pattern — that means "${scatterX}" and "${scatterY}" are related and both useful as features.`} />
      </div>
    )
  }

  // ── MISSING ──
  function MissingChart() {
    const withMissing  = viz.missing.filter(d => d.missing > 0)
    const cleanCols    = viz.missing.filter(d => d.missing === 0).length
    const totalCols    = viz.missing.length

    if (withMissing.length === 0) return (
      <div>
        <Explainer text="Missing values are empty cells in your CSV — rows where a column has no data. These can cause models to fail or perform poorly." />
        <div style={{ padding: '48px 0', textAlign: 'center' }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>✓</div>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#10b981', marginBottom: 8 }} className="neon-green">No missing values!</div>
          <div style={{ fontSize: 13, color: '#555' }}>All {totalCols} columns are complete. Your dataset is clean and ready to train.</div>
        </div>
      </div>
    )

    return (
      <div>
        <Explainer text="Missing values are empty cells — rows where a column has no data. Red = serious problem. Yellow = moderate. Green = minor. Columns with lots of missing data should be dropped or filled before training." />

        <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
          <Stat label="Clean columns" value={`${cleanCols}/${totalCols}`} color="#10b981" />
          <Stat label="Affected columns" value={withMissing.length} color="#f59e0b" />
          <Stat label="Worst column" value={`${withMissing[0]?.pct}%`} color="#e24b4a" />
        </div>

        <ResponsiveContainer width='100%' height={Math.max(180, withMissing.length * 50)}>
          <BarChart data={withMissing} layout='vertical' margin={{ top: 5, right: 80, left: 10, bottom: 5 }}>
            <XAxis type='number' domain={[0, 100]} tick={{ fill: '#555', fontSize: 11 }} tickFormatter={v => v + '%'} />
            <YAxis type='category' dataKey='column' tick={{ fill: '#aaa', fontSize: 11 }} width={130} />
            <Tooltip
              contentStyle={{ background: '#0d0d18', border: '1px solid rgba(108,99,255,0.3)', borderRadius: 8, color: '#f0f0f0', fontSize: 12 }}
              formatter={(v, n, p) => [`${p.payload.missing} rows (${v}%)`, 'Missing']}
            />
            <Bar dataKey='pct' radius={[0, 6, 6, 0]} label={{ position: 'right', formatter: v => v + '%', fill: '#666', fontSize: 11 }}>
              {withMissing.map((d, i) => (
                <Cell key={i} fill={d.pct > 30 ? '#e24b4a' : d.pct > 10 ? '#f59e0b' : '#10b981'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <div style={{ display: 'flex', gap: 12, marginTop: 12, flexWrap: 'wrap' }}>
          {[['#e24b4a', '>30% — Drop this column'], ['#f59e0b', '10–30% — Fill with median/mean'], ['#10b981', '<10% — Usually safe to keep']].map(([c, l]) => (
            <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#666' }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: c }} />{l}
            </div>
          ))}
        </div>

        <InsightBox text={`"${withMissing[0]?.column}" is missing ${withMissing[0]?.pct}% of its values. ${withMissing[0]?.pct > 40 ? 'This is too many to reliably fill — consider removing this column.' : 'You can fill missing values with the column median before training.'}`} />
      </div>
    )
  }

  // ── BOX PLOT / SPREAD ──
  function SpreadChart() {
    if (!col || !viz.boxplots[col]) return null
    const b = viz.boxplots[col]
    const range = b.max - b.min || 1
    function pct(v) { return Math.max(0, Math.min(100, (v - b.min) / range * 100)) }
    const iqr = b.q3 - b.q1
    const isSkewed = Math.abs((b.median - b.min) - (b.max - b.median)) > iqr * 0.5

    return (
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
          <ColSelect value={col} onChange={setCol} cols={viz.numeric_columns} label='Column' />
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <Stat label="Min"    value={b.min}    color="#555" />
            <Stat label="Median" value={b.median} color="#a09af0" />
            <Stat label="Max"    value={b.max}    color="#555" />
            {b.outliers.length > 0 && <Stat label="Outliers" value={b.outliers.length} color="#e24b4a" />}
          </div>
        </div>

        <Explainer text={`The spread chart shows how values are distributed for "${col}". The purple box = middle 50% of rows. The line = median (middle value). Dots outside = outliers (unusual values).`} />

        <div style={{ padding: '20px 0 40px' }}>
          <div style={{ position: 'relative', height: 90, margin: '0 20px' }}>
            {/* Range line */}
            <div style={{ position: 'absolute', top: '50%', left: `${pct(b.min)}%`, right: `${100 - pct(b.max)}%`, height: 2, background: 'rgba(255,255,255,0.1)', transform: 'translateY(-50%)' }} />
            {/* Min whisker */}
            <div style={{ position: 'absolute', top: '30%', left: `${pct(b.min)}%`, width: 2, height: '40%', background: 'rgba(255,255,255,0.3)' }} />
            {/* Max whisker */}
            <div style={{ position: 'absolute', top: '30%', left: `${pct(b.max)}%`, width: 2, height: '40%', background: 'rgba(255,255,255,0.3)' }} />
            {/* IQR box */}
            <div style={{
              position: 'absolute',
              top: '20%', bottom: '20%',
              left: `${pct(b.q1)}%`,
              width: `${pct(b.q3) - pct(b.q1)}%`,
              background: 'rgba(108,99,255,0.2)',
              border: '1.5px solid #6c63ff',
              borderRadius: 6,
            }} />
            {/* Median line */}
            <div style={{
              position: 'absolute',
              top: '15%', bottom: '15%',
              left: `${pct(b.median)}%`,
              width: 3,
              background: '#a09af0',
              borderRadius: 2,
              boxShadow: '0 0 8px rgba(160,154,240,0.5)',
            }} />
            {/* Outliers */}
            {b.outliers.slice(0, 30).map((v, i) => (
              <div key={i} style={{
                position: 'absolute',
                top: '50%',
                left: `${pct(v)}%`,
                width: 8, height: 8,
                background: '#e24b4a',
                borderRadius: '50%',
                transform: 'translate(-50%, -50%)',
                boxShadow: '0 0 6px rgba(226,75,74,0.5)',
              }} />
            ))}
          </div>

          {/* Labels */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr', textAlign: 'center', marginTop: 8, fontSize: 11, color: '#555' }}>
            <div><div style={{ color: '#888', fontWeight: 600 }}>{b.min}</div><div>Min</div></div>
            <div><div style={{ color: '#6c63ff', fontWeight: 600 }}>{b.q1}</div><div>Q1 (25%)</div></div>
            <div><div style={{ color: '#a09af0', fontWeight: 600 }}>{b.median}</div><div>Median</div></div>
            <div><div style={{ color: '#6c63ff', fontWeight: 600 }}>{b.q3}</div><div>Q3 (75%)</div></div>
            <div><div style={{ color: '#888', fontWeight: 600 }}>{b.max}</div><div>Max</div></div>
          </div>
        </div>

        <InsightBox text={`Middle 50% of "${col}" falls between ${b.q1} and ${b.q3} (a range of ${(b.q3 - b.q1).toFixed(2)}).${b.outliers.length > 0 ? ` ${b.outliers.length} outlier${b.outliers.length > 1 ? 's' : ''} detected — values far outside the normal range that may skew your model.` : ' No outliers — clean distribution.'}${isSkewed ? ' Data appears skewed (not symmetric).' : ''}`} />
      </div>
    )
  }

  const charts = [<HistogramChart />, <CorrelationChart />, <ScatterPlot />, <MissingChart />, <SpreadChart />]

  return (
    <div style={{ minHeight: '100vh' }}>
      <header style={{
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        padding: '16px 40px',
        display: 'flex', alignItems: 'center', gap: 12,
        background: 'rgba(2,2,8,0.8)',
        backdropFilter: 'blur(20px)',
        position: 'sticky', top: 0, zIndex: 50,
      }}>
        <Link to='/' style={{ color: '#555', fontSize: 13, textDecoration: 'none', transition: 'color 0.2s' }}
          onMouseEnter={e => e.currentTarget.style.color = '#aaa'}
          onMouseLeave={e => e.currentTarget.style.color = '#555'}>
          ← Back
        </Link>
        <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)' }} />
        <span style={{ fontSize: 14, fontWeight: 500 }}>Explore data</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: '#444', fontFamily: 'monospace' }}>
          {sessionStorage.getItem('csvFile') || 'dataset.csv'}
        </span>
      </header>

      <main style={{ maxWidth: 900, margin: '0 auto', padding: '36px 24px' }}>

        <div className="fade-up" style={{ marginBottom: 28 }}>
          <h2 style={{ fontSize: 22, fontWeight: 700, letterSpacing: '-0.02em', marginBottom: 6 }}>Explore your data</h2>
          <p style={{ color: '#555', fontSize: 14 }}>Understand your dataset before building a model. Each chart reveals something different.</p>
        </div>

        {/* Insights panel */}
        {insights.length > 0 && (
          <div className="fade-up delay-1" style={{ marginBottom: 24 }}>
            {insights.map((ins, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'flex-start', gap: 10,
                padding: '10px 14px',
                background: ins.type === 'ok' ? 'rgba(16,185,129,0.06)' : ins.type === 'warn' ? 'rgba(245,158,11,0.06)' : 'rgba(14,165,233,0.06)',
                border: `1px solid ${ins.type === 'ok' ? 'rgba(16,185,129,0.2)' : ins.type === 'warn' ? 'rgba(245,158,11,0.2)' : 'rgba(14,165,233,0.2)'}`,
                borderRadius: 10, marginBottom: 8, fontSize: 13,
                color: ins.type === 'ok' ? '#34d399' : ins.type === 'warn' ? '#fbbf24' : '#38bdf8',
              }}>
                <span>{ins.type === 'ok' ? '✓' : ins.type === 'warn' ? '⚠' : 'ℹ'}</span>
                <span style={{ color: '#aaa', lineHeight: 1.5 }}>{ins.text}</span>
              </div>
            ))}
          </div>
        )}

        {/* Tab buttons */}
        <div className="fade-up delay-2" style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              padding: '8px 16px', borderRadius: 10, fontSize: 13, cursor: 'pointer',
              background: tab === t.id ? ACCENT : 'rgba(255,255,255,0.02)',
              border: `1px solid ${tab === t.id ? ACCENT : 'rgba(255,255,255,0.07)'}`,
              color: tab === t.id ? '#fff' : '#777', transition: 'all 0.2s',
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
              boxShadow: tab === t.id ? '0 0 16px rgba(108,99,255,0.3)' : 'none',
            }}>
              <span style={{ fontWeight: 600 }}>{t.label}</span>
              <span style={{ fontSize: 10, opacity: 0.7 }}>{t.desc}</span>
            </button>
          ))}
        </div>

        {/* Chart area */}
        <div className="fade-up delay-2" style={{
          background: 'rgba(255,255,255,0.015)',
          border: '1px solid rgba(255,255,255,0.06)',
          borderRadius: 16,
          padding: '28px 24px',
          marginBottom: 24,
          minHeight: 360,
        }}>
          {charts[tab]}
        </div>

        {/* Continue */}
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button
            onClick={() => navigate(`/${modelType}`)}
            className="btn-glow"
            style={{
              padding: '13px 32px',
              background: 'linear-gradient(135deg, #6c63ff, #4f46e5)',
              border: 'none', borderRadius: 12,
              color: '#fff', fontSize: 14, fontWeight: 600, cursor: 'pointer',
              boxShadow: '0 0 24px rgba(108,99,255,0.3)',
            }}
          >
            Continue to model config →
          </button>
        </div>
      </main>
    </div>
  )
}

function ColSelect({ value, onChange, cols, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span style={{ fontSize: 12, color: '#666', minWidth: 50, whiteSpace: 'nowrap' }}>{label}:</span>
      <select value={value} onChange={e => onChange(e.target.value)} style={{
        background: '#0d0d18', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8,
        color: '#e0e0e0', padding: '6px 28px 6px 10px', fontSize: 13, cursor: 'pointer',
        outline: 'none',
      }}>
        {cols.map(c => <option key={c} value={c}>{c}</option>)}
      </select>
    </div>
  )
}

function Stat({ label, value, color }) {
  return (
    <div style={{ textAlign: 'center', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 8, padding: '6px 12px', minWidth: 70 }}>
      <div style={{ fontSize: 14, fontWeight: 700, color, fontFamily: 'monospace' }}>{value}</div>
      <div style={{ fontSize: 10, color: '#555', marginTop: 1 }}>{label}</div>
    </div>
  )
}

function Explainer({ text }) {
  return (
    <div style={{ padding: '10px 14px', background: 'rgba(108,99,255,0.04)', border: '1px solid rgba(108,99,255,0.12)', borderRadius: 8, fontSize: 12, color: '#888', lineHeight: 1.6, marginBottom: 16 }}>
      {text}
    </div>
  )
}

function InsightBox({ text }) {
  return (
    <div style={{ marginTop: 16, padding: '10px 14px', background: 'rgba(108,99,255,0.04)', borderLeft: '3px solid rgba(108,99,255,0.5)', borderRadius: '0 8px 8px 0', fontSize: 12, color: '#999', lineHeight: 1.6 }}>
      <strong style={{ color: '#7c75cc' }}>Insight: </strong>{text}
    </div>
  )
}

function NoData({ msg }) {
  return <div style={{ padding: '60px 0', textAlign: 'center', color: '#444', fontSize: 14 }}>{msg}</div>
}