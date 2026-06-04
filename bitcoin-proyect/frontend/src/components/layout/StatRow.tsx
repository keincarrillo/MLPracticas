// components/layout/StatRow.tsx — fila de tarjetas de métricas resumidas
// Recibe los datos ya procesados y renderiza las 4 StatCards.

import { StatCard } from '@/components/ui'
import { fmt } from '@/components/ui'
import type { Metrics, HistoryPoint } from '@/types'

interface Props {
  history: HistoryPoint[] | null
  metrics: Metrics | null
  metricsLoading: boolean
}

export const StatRow = ({ history, metrics, metricsLoading }: Props) => {
  const lastPrice = history?.[history.length - 1]?.price
  const firstPrice = history?.[0]?.price

  const pct =
    lastPrice && firstPrice
      ? (((lastPrice - firstPrice) / firstPrice) * 100).toFixed(1)
      : null
  const pctPositive = pct !== null && Number(pct) > 0

  return (
    <div
      style={{
        display: 'flex',
        gap: 16,
        marginBottom: 40,
        flexWrap: 'wrap'
      }}
    >
      <StatCard
        label="Último precio"
        value={lastPrice ? fmt(lastPrice) : '—'}
        sub="Precio de cierre más reciente"
        accent="#f7931a"
      />
      <StatCard
        label="Variación 3 años"
        value={pct ? `${pctPositive ? '+' : ''}${pct}%` : '—'}
        sub={`${firstPrice ? fmt(firstPrice) : '—'} → ${lastPrice ? fmt(lastPrice) : '—'}`}
        accent={pctPositive ? '#22c55e' : '#ef4444'}
      />
      <StatCard
        label="MAE del modelo"
        value={metrics ? fmt(metrics.mae) : metricsLoading ? '…' : '—'}
        sub="Error absoluto medio en test set"
        accent="#818cf8"
      />
      <StatCard
        label="Horizonte"
        value="30 días"
        sub="Ventana LSTM: 60 días"
        accent="#38bdf8"
      />
    </div>
  )
}
