// components/BitcoinDashboard.tsx — punto de ensamblaje (~35 líneas)
// Solo llama el hook y pasa props a cada sección. Sin lógica de negocio.

import { useBitcoin } from '@/hooks/useBitcoin'
import { TopBar } from '@/components/layout/TopBar'
import { StatRow } from '@/components/layout/StatRow'
import { MetricsPanel } from '@/components/layout/MetricsPanel'
import { HistoryChart } from '@/components/charts/HistoryChart'
import { PredictChart } from '@/components/charts/PredictChart'
import { LossChart } from '@/components/charts/LossChart'
import { ForecastChart } from '@/components/charts/ForecastChart'

export default function BitcoinDashboard() {
  const { history, predict, metrics, forecast, loss, loading, errors } =
    useBitcoin()

  return (
    <div
      style={{
        minHeight: '100vh',
        background: '#080808',
        color: '#e0e0e0',
        fontFamily: "'IBM Plex Mono', monospace"
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }
        @keyframes pulse   { 0%,100%{opacity:1} 50%{opacity:0.4} }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #111; }
        ::-webkit-scrollbar-thumb { background: #333; }
      `}</style>

      <TopBar />

      <div style={{ padding: '32px 32px 64px' }}>
        <StatRow
          history={history}
          metrics={metrics}
          metricsLoading={loading.metrics}
        />
        <HistoryChart
          data={history}
          loading={loading.history}
          error={errors.history}
        />
        <PredictChart
          data={predict}
          loading={loading.predict}
          error={errors.predict}
        />
        <LossChart data={loss} loading={loading.loss} error={errors.loss} />
        <MetricsPanel metrics={metrics} history={history} />
        <ForecastChart
          data={forecast}
          loading={loading.forecast}
          error={errors.forecast}
        />
      </div>
    </div>
  )
}
