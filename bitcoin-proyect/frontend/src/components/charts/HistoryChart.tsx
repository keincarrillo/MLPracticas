import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'
import type { HistoryPoint } from '@/types'
import { ChartTooltip, Skeleton, ErrorBox, fmtShort } from '@/components/ui'
import { SectionHeader } from '@/components/ui'

interface Props {
  data: HistoryPoint[] | null
  loading: boolean
  error?: boolean
}

export const HistoryChart = ({ data, loading, error }: Props) => (
  <div style={{ marginBottom: 48 }}>
    <SectionHeader
      title="Historial de precios"
      sub="Precio de cierre diario BTC-USD — últimos 5 años"
    />
    {loading ? (
      <Skeleton h={280} />
    ) : error ? (
      <ErrorBox message="Error al cargar datos de historial" />
    ) : (
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart
          data={data ?? []}
          margin={{ top: 10, right: 0, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="gradHistory" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f7931a" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#f7931a" stopOpacity={0} />
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
            interval={Math.floor((data?.length ?? 1) / 6)}
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
            name="Precio"
            stroke="#f7931a"
            strokeWidth={1.5}
            fill="url(#gradHistory)"
            dot={false}
            activeDot={{
              r: 4,
              fill: '#f7931a',
              stroke: '#080808',
              strokeWidth: 2
            }}
          />
        </AreaChart>
      </ResponsiveContainer>
    )}
  </div>
)
