import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'
import type { PredictPoint } from '@/types'
import {
  ChartTooltip,
  Skeleton,
  ErrorBox,
  ChartLegend,
  fmtShort
} from '@/components/ui'
import { SectionHeader } from '@/components/ui'

interface Props {
  data: PredictPoint[] | null
  loading: boolean
  error?: boolean
}

export const PredictChart = ({ data, loading, error }: Props) => (
  <div style={{ marginBottom: 48 }}>
    <SectionHeader
      title="Real vs Predicho"
      sub="Conjunto de prueba (20% de los datos) — modelo LSTM"
    />
    {loading ? (
      <Skeleton h={280} />
    ) : error ? (
      <ErrorBox message="Error al cargar predicciones" />
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
              dataKey="index"
              tick={{
                fill: '#444',
                fontSize: 10,
                fontFamily: "'IBM Plex Mono'"
              }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => `D${v}`}
              interval={Math.floor((data?.length ?? 1) / 6)}
            />
            <YAxis
              tick={{
                fill: '#444',
                fontSize: 10,
                fontFamily: "'IBM Plex Mono'"
              }}
              tickLine={false}
              axisLine={false}
              tickFormatter={fmtShort}
              width={52}
            />
            <Tooltip content={<ChartTooltip />} />
            <Line
              type="monotone"
              dataKey="real"
              name="Real"
              stroke="#38bdf8"
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              name="Predicho"
              stroke="#f7931a"
              strokeWidth={1.5}
              dot={false}
              strokeDasharray="4 2"
              activeDot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
        <ChartLegend
          items={[
            ['#38bdf8', 'Real'],
            ['#f7931a', 'Predicho (LSTM)']
          ]}
        />
      </>
    )}
  </div>
)
