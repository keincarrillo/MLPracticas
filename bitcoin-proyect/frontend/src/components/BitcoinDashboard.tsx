import { useState, useEffect } from 'react'
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'

const API = '/api'

// ── Types ────────────────────────────────────────────────────────────────────
interface HistoryPoint {
  date: string
  price: number
}
interface PredictPoint {
  index: number
  real: number
  predicted: number
}
interface Metrics {
  mae: number
}
interface LoadingState {
  history: boolean
  predict: boolean
  metrics: boolean
}
interface ErrorState {
  history?: boolean
  predict?: boolean
  metrics?: boolean
}

interface TooltipPayloadItem {
  name?: string
  value?: number
  color?: string
}

interface ChartTooltipProps {
  active?: boolean
  payload?: TooltipPayloadItem[]
  label?: string
}

// ── Formatters ───────────────────────────────────────────────────────────────
const fmt = (n: number) =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0
  }).format(n)

const fmtShort = (n: number) => {
  if (n >= 1000) return `$${(n / 1000).toFixed(0)}k`
  return `$${n}`
}

// ── Custom tooltip ────────────────────────────────────────────────────────────
const ChartTooltip = ({ active, payload, label }: ChartTooltipProps) => {
  if (!active || !payload?.length) return null
  return (
    <div
      style={{
        background: '#0d0d0d',
        border: '1px solid #2a2a2a',
        padding: '10px 14px',
        fontFamily: "'IBM Plex Mono', monospace",
        fontSize: 12
      }}
    >
      <p style={{ color: '#666', marginBottom: 4 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color, margin: '2px 0' }}>
          {p.name}: {fmt(p.value ?? 0)}
        </p>
      ))}
    </div>
  )
}

// ── Stat card ─────────────────────────────────────────────────────────────────
interface StatCardProps {
  label: string
  value: string
  sub?: string
  accent: string
}
const StatCard = ({ label, value, sub, accent }: StatCardProps) => (
  <div
    style={{
      background: '#111',
      border: `1px solid ${accent}22`,
      borderLeft: `3px solid ${accent}`,
      padding: '20px 24px',
      flex: 1,
      minWidth: 180
    }}
  >
    <p
      style={{
        color: '#555',
        fontSize: 11,
        letterSpacing: '0.12em',
        textTransform: 'uppercase',
        marginBottom: 8,
        fontFamily: "'IBM Plex Mono', monospace"
      }}
    >
      {label}
    </p>
    <p
      style={{
        color: accent,
        fontSize: 28,
        fontWeight: 700,
        fontFamily: "'IBM Plex Mono', monospace",
        margin: 0
      }}
    >
      {value}
    </p>
    {sub && (
      <p
        style={{
          color: '#444',
          fontSize: 11,
          marginTop: 6,
          fontFamily: "'IBM Plex Mono', monospace"
        }}
      >
        {sub}
      </p>
    )}
  </div>
)

// ── Section header ────────────────────────────────────────────────────────────
interface SectionHeaderProps {
  title: string
  sub?: string
}
const SectionHeader = ({ title, sub }: SectionHeaderProps) => (
  <div style={{ marginBottom: 20 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <div style={{ width: 3, height: 18, background: '#f7931a' }} />
      <h2
        style={{
          margin: 0,
          fontSize: 13,
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          color: '#ccc',
          fontFamily: "'IBM Plex Mono', monospace"
        }}
      >
        {title}
      </h2>
    </div>
    {sub && (
      <p
        style={{
          margin: '6px 0 0 15px',
          color: '#444',
          fontSize: 11,
          fontFamily: "'IBM Plex Mono', monospace"
        }}
      >
        {sub}
      </p>
    )}
  </div>
)

// ── Loading skeleton ──────────────────────────────────────────────────────────
const Skeleton = ({ h = 300 }: { h?: number }) => (
  <div
    style={{
      height: h,
      background: 'linear-gradient(90deg, #111 25%, #1a1a1a 50%, #111 75%)',
      backgroundSize: '200% 100%',
      animation: 'shimmer 1.5s infinite',
      borderRadius: 2
    }}
  />
)

// ── Main dashboard ────────────────────────────────────────────────────────────
export default function BitcoinDashboard() {
  const [history, setHistory] = useState<HistoryPoint[] | null>(null)
  const [predict, setPredict] = useState<PredictPoint[] | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [loading, setLoading] = useState<LoadingState>({
    history: true,
    predict: true,
    metrics: true
  })
  const [errors, setErrors] = useState<ErrorState>({})
  const [now, setNow] = useState(new Date())

  useEffect(() => {
    const tick = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(tick)
  }, [])

  useEffect(() => {
    fetch(`${API}/history`)
      .then(r => r.json())
      .then((d: { dates: string[]; prices: number[] }) => {
        const sampled = d.dates
          .map((date, i) => ({ date, price: d.prices[i] }))
          .filter((_, i) => i % 5 === 0)
        setHistory(sampled)
        setLoading(l => ({ ...l, history: false }))
      })
      .catch(() => {
        setErrors(e => ({ ...e, history: true }))
        setLoading(l => ({ ...l, history: false }))
      })

    fetch(`${API}/predict`)
      .then(r => r.json())
      .then((d: { real: number[]; predicted: number[] }) => {
        const data = d.real.map((r, i) => ({
          index: i,
          real: r,
          predicted: d.predicted[i]
        }))
        setPredict(data)
        setLoading(l => ({ ...l, predict: false }))
      })
      .catch(() => {
        setErrors(e => ({ ...e, predict: true }))
        setLoading(l => ({ ...l, predict: false }))
      })

    fetch(`${API}/metrics`)
      .then(r => r.json())
      .then((d: Metrics) => {
        setMetrics(d)
        setLoading(l => ({ ...l, metrics: false }))
      })
      .catch(() => {
        setErrors(e => ({ ...e, metrics: true }))
        setLoading(l => ({ ...l, metrics: false }))
      })
  }, [])

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
        minHeight: '100vh',
        background: '#080808',
        color: '#e0e0e0',
        fontFamily: "'IBM Plex Mono', monospace",
        padding: '0'
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #111; }
        ::-webkit-scrollbar-thumb { background: #333; }
      `}</style>

      {/* ── Top bar ── */}
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
          <span
            style={{ fontSize: 13, letterSpacing: '0.16em', color: '#888' }}
          >
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

      <div style={{ padding: '32px 32px 64px' }}>
        {/* ── Stat row ── */}
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
            value={metrics ? fmt(metrics.mae) : loading.metrics ? '…' : '—'}
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

        {/* ── History chart ── */}
        <div style={{ marginBottom: 48 }}>
          <SectionHeader
            title="Historial de precios"
            sub="Precio de cierre diario BTC-USD — últimos 3 años"
          />
          {loading.history ? (
            <Skeleton h={280} />
          ) : errors.history ? (
            <div
              style={{
                height: 280,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#ef4444',
                fontSize: 12
              }}
            >
              Error al cargar datos de historial
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart
                data={history ?? []}
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
                  tick={{
                    fill: '#444',
                    fontSize: 10,
                    fontFamily: "'IBM Plex Mono'"
                  }}
                  tickLine={false}
                  axisLine={false}
                  interval={Math.floor((history?.length ?? 1) / 6)}
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

        {/* ── Predict chart ── */}
        <div style={{ marginBottom: 48 }}>
          <SectionHeader
            title="Real vs Predicho"
            sub="Conjunto de prueba (20% de los datos) — modelo LSTM"
          />
          {loading.predict ? (
            <Skeleton h={280} />
          ) : errors.predict ? (
            <div
              style={{
                height: 280,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#ef4444',
                fontSize: 12
              }}
            >
              Error al cargar predicciones
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart
                data={predict ?? []}
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
                  interval={Math.floor((predict?.length ?? 1) / 6)}
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
          )}

          {/* legend */}
          <div
            style={{ display: 'flex', gap: 24, marginTop: 14, paddingLeft: 4 }}
          >
            {(
              [
                ['#38bdf8', 'Real'],
                ['#f7931a', 'Predicho (LSTM)']
              ] as [string, string][]
            ).map(([color, label]) => (
              <div
                key={label}
                style={{ display: 'flex', alignItems: 'center', gap: 8 }}
              >
                <div style={{ width: 20, height: 2, background: color }} />
                <span style={{ fontSize: 11, color: '#555' }}>{label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── MAE detail ── */}
        <div>
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
                <span style={{ color: '#444' }}>
                  Optimizer: Adam · Loss: MSE
                </span>
                <br />
                <span style={{ color: '#444' }}>
                  Window: 60d · Train/Test: 80/20
                </span>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
