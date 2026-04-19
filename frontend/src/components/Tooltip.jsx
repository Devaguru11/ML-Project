import { useState } from 'react'

export default function Tooltip({ text }) {
  const [show, setShow] = useState(false)
  return (
    <span style={{ position: 'relative', display: 'inline-block', marginLeft: 6 }}>
      <span
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        style={{
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          width: 16, height: 16, borderRadius: '50%',
          background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.15)',
          fontSize: 10, color: '#666', cursor: 'help', flexShrink: 0,
        }}
      >?</span>
      {show && (
        <div style={{
          position: 'absolute', bottom: '120%', left: '50%',
          transform: 'translateX(-50%)',
          background: '#1c1c26', border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: 8, padding: '8px 12px', fontSize: 12, color: '#ccc',
          whiteSpace: 'nowrap', zIndex: 100, lineHeight: 1.5,
          boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
          maxWidth: 240, whiteSpace: 'normal',
        }}>
          {text}
          <div style={{
            position: 'absolute', top: '100%', left: '50%',
            transform: 'translateX(-50%)',
            border: '5px solid transparent',
            borderTopColor: '#1c1c26',
          }} />
        </div>
      )}
    </span>
  )
}