// routes/api.routes.ts
// define los endpoints REST del dashboard de Bitcoin
// cada handler es responsable solo de llamar el script correcto y devolver JSON
// la lógica de "asegurar que el modelo existe" vive en los middlewares

import Router from 'express'
import { spawnPython, parsePythonJson } from '@/utils/python'
import { ensureModel, ensureScaler } from '@/middleware/ensureModel'

const router = Router()

// GET /history
// devuelve el historial completo de precios de cierre con fechas
// solo requiere el scaler, no el modelo entrenado
router.get('/history', ensureScaler, async (_req, res) => {
  try {
    const output = await spawnPython('predict.py', ['--history'])
    res.json(parsePythonJson(output))
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
})

// GET /predict
// devuelve precios reales y predichos del conjunto de prueba
router.get('/predict', ensureModel, async (_req, res) => {
  try {
    const output = await spawnPython('predict.py', ['--predict'])
    res.json(parsePythonJson(output))
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
})

// GET /metrics
// devuelve el MAE calculado sobre el conjunto de prueba
router.get('/metrics', ensureModel, async (_req, res) => {
  try {
    const output = await spawnPython('predict.py', ['--metrics'])
    res.json(parsePythonJson(output))
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
})

// GET /forecast
// devuelve la predicción recursiva de los próximos 30 días
router.get('/forecast', ensureModel, async (_req, res) => {
  try {
    const output = await spawnPython('predict.py', ['--forecast'])
    res.json(parsePythonJson(output))
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
})

// GET /loss
// devuelve los datos de pérdida por época del último entrenamiento
router.get('/loss', ensureModel, async (_req, res) => {
  try {
    const output = await spawnPython('predict.py', ['--loss'])
    res.json(parsePythonJson(output))
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
})

export default router
