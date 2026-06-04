// components/layout/TopBar.tsx — barra superior con reloj en tiempo real
// Maneja su propio estado de reloj; no recibe datos de Bitcoin.

import { useState, useEffect } from 'react'

export const TopBar = () => {
  const [now, setNow] = useState(new Date())

  useEffect(() => {
    const tick = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(tick)
  }, [])

  return (
    <div
      style={{
        borderBottom: '1px solid #1a1a1a',
        padding: '14px 32px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        background: '#080808',
        zIndex: 10
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <span style={{ color: '#f7931a', fontSize: 20, fontWeight: 700 }}>
          ₿
        </span>
        <span style={{ fontSize: 13, letterSpacing: '0.16em', color: '#888' }}>
          BTC-USD
        </span>
        <span style={{ fontSize: 11, color: '#333', letterSpacing: '0.1em' }}>
          LSTM FORECAST DASHBOARD
        </span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: '#22c55e',
              animation: 'pulse 2s infinite'
            }}
          />
          <span style={{ fontSize: 11, color: '#444' }}>LIVE</span>
        </div>
        <span style={{ fontSize: 11, color: '#333' }}>
          {now.toUTCString().replace(' GMT', ' UTC')}
        </span>
      </div>
    </div>
  )
}
