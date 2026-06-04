// components/charts/ForecastChart.tsx — predicción recursiva 30 días
// Recibe datos por props, no sabe nada del hook ni de la API.

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'
import type { ForecastPoint } from '@/types'
import { ChartTooltip, Skeleton, ErrorBox, fmtShort } from '@/components/ui'
import { SectionHeader } from '@/components/ui'

interface Props {
  data: ForecastPoint[] | null
  loading: boolean
  error?: boolean
}

export const ForecastChart = ({ data, loading, error }: Props) => (
  <div style={{ marginBottom: 48 }}>
    <SectionHeader
      title="Forecast — próximos 30 días"
      sub="Predicción recursiva LSTM a partir del último precio conocido"
    />
    {loading ? (
      <Skeleton h={280} />
    ) : error ? (
      <ErrorBox message="Error al cargar forecast" />
    ) : (
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart
          data={data ?? []}
          margin={{ top: 10, right: 0, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="gradForecast" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            stroke="#1a1a1a"
            strokeDasharray="3 3"
            vertical={false}
          />
          <XAxis
            dataKey="date"
            tick={{ fill: '#444', fontSize: 10, fontFamily: "'IBM Plex Mono'" }}
            tickLine={false}
            axisLine={false}
            interval={4}
          />
          <YAxis
            tick={{ fill: '#444', fontSize: 10, fontFamily: "'IBM Plex Mono'" }}
            tickLine={false}
            axisLine={false}
            tickFormatter={fmtShort}
            width={52}
          />
          <Tooltip content={<ChartTooltip />} />
          <Area
            type="monotone"
            dataKey="price"
            name="Forecast"
            stroke="#818cf8"
            strokeWidth={1.5}
            fill="url(#gradForecast)"
            dot={false}
            strokeDasharray="5 3"
            activeDot={{
              r: 4,
              fill: '#818cf8',
              stroke: '#080808',
              strokeWidth: 2
            }}
          />
        </AreaChart>
      </ResponsiveContainer>
    )}
  </div>
)
