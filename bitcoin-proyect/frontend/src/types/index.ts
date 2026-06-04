export interface ForecastPoint {
  date: string
  price: number
}

export interface HistoryPoint {
  date: string
  price: number
}

export interface PredictPoint {
  index: number
  real: number
  predicted: number
}

export interface Metrics {
  mae: number
}

export interface LossPoint {
  epoch: number
  train: number
  val: number
}

export interface LoadingState {
  history: boolean
  predict: boolean
  metrics: boolean
  forecast: boolean
  loss: boolean
}

export interface ErrorState {
  history?: boolean
  predict?: boolean
  metrics?: boolean
  forecast?: boolean
  loss?: boolean
}

export interface TooltipPayloadItem {
  name?: string
  value?: number
  color?: string
}

export interface ChartTooltipProps {
  active?: boolean
  payload?: TooltipPayloadItem[]
  label?: string
}

// Respuestas crudas de la API (antes de mapear a los tipos de dominio)
export interface HistoryApiResponse {
  dates: string[]
  prices: number[]
}

export interface PredictApiResponse {
  real: number[]
  predicted: number[]
}

export interface ForecastApiResponse {
  dates: string[]
  prices: number[]
}

export interface LossApiResponse {
  epochs: number[]
  train: number[]
  val: number[]
}
