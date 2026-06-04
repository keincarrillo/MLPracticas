// components/layout/MetricsPanel.tsx — panel detallado de métricas del modelo
// Muestra MAE, error relativo y arquitectura de la red.

import { SectionHeader } from '@/components/ui'
import { fmt } from '@/components/ui'
import type { Metrics, HistoryPoint } from '@/types'

interface Props {
  metrics: Metrics | null
  history: HistoryPoint[] | null
}

export const MetricsPanel = ({ metrics, history }: Props) => {
  const lastPrice = history?.[history.length - 1]?.price

  return (
    <div style={{ marginBottom: 48 }}>
      <SectionHeader
        title="Métricas del modelo"
        sub="Evaluación sobre el conjunto de prueba"
      />
      <div
        style={{
          background: '#111',
          border: '1px solid #1a1a1a',
          padding: '24px 28px',
          display: 'flex',
          gap: 40,
          flexWrap: 'wrap'
        }}
      >
        {/* MAE absoluto */}
        <div>
          <p
            style={{
              color: '#444',
              fontSize: 11,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              marginBottom: 6
            }}
          >
            MAE
          </p>
          <p style={{ color: '#818cf8', fontSize: 36, fontWeight: 700 }}>
            {metrics ? fmt(metrics.mae) : '—'}
          </p>
          <p style={{ color: '#444', fontSize: 11, marginTop: 6 }}>
            Mean Absolute Error en USD
          </p>
        </div>

        {/* Error relativo */}
        <div style={{ borderLeft: '1px solid #1a1a1a', paddingLeft: 40 }}>
          <p
            style={{
              color: '#444',
              fontSize: 11,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              marginBottom: 6
            }}
          >
            Error relativo
          </p>
          <p style={{ color: '#22c55e', fontSize: 36, fontWeight: 700 }}>
            {metrics && lastPrice
              ? `${((metrics.mae / lastPrice) * 100).toFixed(1)}%`
              : '—'}
          </p>
          <p style={{ color: '#444', fontSize: 11, marginTop: 6 }}>
            MAE / precio actual
          </p>
        </div>

        {/* Arquitectura */}
        <div style={{ borderLeft: '1px solid #1a1a1a', paddingLeft: 40 }}>
          <p
            style={{
              color: '#444',
              fontSize: 11,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              marginBottom: 6
            }}
          >
            Arquitectura
          </p>
          <p
            style={{
              color: '#888',
              fontSize: 14,
              fontWeight: 500,
              marginTop: 8,
              lineHeight: 1.8
            }}
          >
            LSTM(50) → Dropout(0.2) → Dense(1)
            <br />
            <span style={{ color: '#444' }}>Optimizer: Adam · Loss: MSE</span>
            <br />
            <span style={{ color: '#444' }}>
              Window: 60d · Train/Test: 80/20
            </span>
          </p>
        </div>
      </div>
    </div>
  )
}
