import { useState, useEffect } from 'react'
import { bitcoinApi } from '@/api/bitcoin'
import type {
  HistoryPoint,
  PredictPoint,
  Metrics,
  ForecastPoint,
  LossPoint,
  LoadingState,
  ErrorState
} from '@/types'

export interface BitcoinData {
  history: HistoryPoint[] | null
  predict: PredictPoint[] | null
  metrics: Metrics | null
  forecast: ForecastPoint[] | null
  loss: LossPoint[] | null
  loading: LoadingState
  errors: ErrorState
}

export function useBitcoin(): BitcoinData {
  const [history, setHistory] = useState<HistoryPoint[] | null>(null)
  const [predict, setPredict] = useState<PredictPoint[] | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [forecast, setForecast] = useState<ForecastPoint[] | null>(null)
  const [loss, setLoss] = useState<LossPoint[] | null>(null)

  const [loading, setLoading] = useState<LoadingState>({
    history: true,
    predict: true,
    metrics: true,
    forecast: true,
    loss: true
  })
  const [errors, setErrors] = useState<ErrorState>({})

  useEffect(() => {
    bitcoinApi
      .fetchHistory()
      .then(d => {
        // Muestrear cada 5 registros para no saturar la gráfica
        const sampled = d.dates
          .map((date, i) => ({ date, price: d.prices[i]! }))
          .filter((_, i) => i % 5 === 0)
        setHistory(sampled)
      })
      .catch(() => setErrors(e => ({ ...e, history: true })))
      .finally(() => setLoading(l => ({ ...l, history: false })))

    bitcoinApi
      .fetchPredict()
      .then(d => {
        const data = d.real.map((r, i) => ({
          index: i,
          real: r,
          predicted: d.predicted[i]!
        }))
        setPredict(data)
      })
      .catch(() => setErrors(e => ({ ...e, predict: true })))
      .finally(() => setLoading(l => ({ ...l, predict: false })))

    bitcoinApi
      .fetchMetrics()
      .then(setMetrics)
      .catch(() => setErrors(e => ({ ...e, metrics: true })))
      .finally(() => setLoading(l => ({ ...l, metrics: false })))

    bitcoinApi
      .fetchForecast()
      .then(d => {
        const data = d.dates.map((date, i) => ({ date, price: d.prices[i]! }))
        setForecast(data)
      })
      .catch(() => setErrors(e => ({ ...e, forecast: true })))
      .finally(() => setLoading(l => ({ ...l, forecast: false })))

    bitcoinApi
      .fetchLoss()
      .then(d => {
        const data = d.epochs.map((epoch, i) => ({
          epoch,
          train: d.train[i]!,
          val: d.val[i]!
        }))
        setLoss(data)
      })
      .catch(() => setErrors(e => ({ ...e, loss: true })))
      .finally(() => setLoading(l => ({ ...l, loss: false })))
  }, [])

  return { history, predict, metrics, forecast, loss, loading, errors }
}
