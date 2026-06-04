import type { ChartTooltipProps } from '@/types'

export const fmt = (n: number) =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0
  }).format(n)

export const fmtShort = (n: number) => {
  if (n >= 1000) return `$${(n / 1000).toFixed(0)}k`
  return `$${n}`
}

export const ChartTooltip = ({ active, payload, label }: ChartTooltipProps) => {
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

interface StatCardProps {
  label: string
  value: string
  sub?: string
  accent: string
}

export const StatCard = ({ label, value, sub, accent }: StatCardProps) => (
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

interface SectionHeaderProps {
  title: string
  sub?: string
}

export const SectionHeader = ({ title, sub }: SectionHeaderProps) => (
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

export const Skeleton = ({ h = 300 }: { h?: number }) => (
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

// ── ErrorBox ──────────────────────────────────────────────────────────────────

export const ErrorBox = ({ message }: { message: string }) => (
  <div
    style={{
      height: 280,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#ef4444',
      fontSize: 12,
      fontFamily: "'IBM Plex Mono', monospace"
    }}
  >
    {message}
  </div>
)

export const ChartLegend = ({ items }: { items: [string, string][] }) => (
  <div style={{ display: 'flex', gap: 24, marginTop: 14, paddingLeft: 4 }}>
    {items.map(([color, label]) => (
      <div
        key={label}
        style={{ display: 'flex', alignItems: 'center', gap: 8 }}
      >
        <div style={{ width: 20, height: 2, background: color }} />
        <span style={{ fontSize: 11, color: '#555' }}>{label}</span>
      </div>
    ))}
  </div>
)
