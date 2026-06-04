import axios from 'axios'
import type {
  HistoryApiResponse,
  PredictApiResponse,
  ForecastApiResponse,
  LossApiResponse,
  Metrics
} from '@/types'

const BASE = `${import.meta.env.VITE_API_URL ?? ''}/api` || '/api'

const client = axios.create({ baseURL: BASE })

export const bitcoinApi = {
  fetchHistory: () =>
    client.get<HistoryApiResponse>('/history').then(r => r.data),

  fetchPredict: () =>
    client.get<PredictApiResponse>('/predict').then(r => r.data),

  fetchMetrics: () => client.get<Metrics>('/metrics').then(r => r.data),

  fetchForecast: () =>
    client.get<ForecastApiResponse>('/forecast').then(r => r.data),

  fetchLoss: () => client.get<LossApiResponse>('/loss').then(r => r.data)
}
