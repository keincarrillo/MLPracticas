import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'
import type { LossPoint, ChartTooltipProps } from '@/types'
import { Skeleton, ErrorBox, ChartLegend } from '@/components/ui'
import { SectionHeader } from '@/components/ui'

// Tooltip especializado para mostrar valores con 6 decimales (MSE loss)
const LossTooltip = ({ active, payload, label }: ChartTooltipProps) => {
  if (!active || !payload?.length) return null
  return (
    <div
      style={{
        background: '#0d0d0d',
        border: '1px solid #2a2a2a',
        padding: '10px 14px',
        fontFamily: "'IBM Plex Mono'",
        fontSize: 12
      }}
    >
      <p style={{ color: '#666', marginBottom: 4 }}>Época {label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color, margin: '2px 0' }}>
          {p.name}: {Number(p.value).toFixed(6)}
        </p>
      ))}
    </div>
  )
}

interface Props {
  data: LossPoint[] | null
  loading: boolean
  error?: boolean
}

export const LossChart = ({ data, loading, error }: Props) => (
  <div style={{ marginBottom: 48 }}>
    <SectionHeader
      title="Curva de aprendizaje"
      sub="Pérdida MSE por época — entrenamiento vs validación"
    />
    {loading ? (
      <Skeleton h={280} />
    ) : error ? (
      <ErrorBox message="Error al cargar curva de aprendizaje" />
    ) : (
      <>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart
            data={data ?? []}
            margin={{ top: 10, right: 0, left: 0, bottom: 0 }}
          >
            <CartesianGrid
              stroke="#1a1a1a"
              strokeDasharray="3 3"
              vertical={false}
            />
            <XAxis
              dataKey="epoch"
              tick={{
                fill: '#444',
                fontSize: 10,
                fontFamily: "'IBM Plex Mono'"
              }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => `E${v}`}
            />
            <YAxis
              tick={{
                fill: '#444',
                fontSize: 10,
                fontFamily: "'IBM Plex Mono'"
              }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => v.toFixed(4)}
              width={60}
            />
            <Tooltip content={<LossTooltip />} />
            <Line
              type="monotone"
              dataKey="train"
              name="Entrenamiento"
              stroke="#f7931a"
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey="val"
              name="Validación"
              stroke="#818cf8"
              strokeWidth={1.5}
              dot={false}
              strokeDasharray="4 2"
              activeDot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
        <ChartLegend
          items={[
            ['#f7931a', 'Entrenamiento'],
            ['#818cf8', 'Validación']
          ]}
        />
      </>
    )}
  </div>
)
