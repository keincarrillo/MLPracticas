import Router from 'express'
import path from 'path'
import { existsSync } from 'fs'
import { spawnPython, MODEL_DIR } from '../utils/python'

const router = Router()

router.get('/history', async (_req, res) => {
  try {
    const scalerPath = path.join(MODEL_DIR, 'scaler.pkl')

    // si no existe el scaler, correr preprocesamiento completo primero
    if (!existsSync(scalerPath)) {
      await spawnPython('preprocess.py')
    }

    const output = await spawnPython('predict.py', ['--history'])
    res.json(JSON.parse(output))
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
})

// -----------------------------------------------------------------------
// GET /predict
// devuelve los precios reales y predichos del conjunto de prueba
// ejecuta train.py si modelo.h5 no existe
// -----------------------------------------------------------------------
router.get('/predict', async (_req, res) => {
  try {
    const modelPath = path.join(MODEL_DIR, 'modelo.h5')

    if (!existsSync(modelPath)) {
      await spawnPython('preprocess.py')
      await spawnPython('train.py')
    }

    const output = await spawnPython('predict.py', ['--predict'])
    res.json(JSON.parse(output))
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
})

// -----------------------------------------------------------------------
// GET /metrics
// devuelve el MAE calculado sobre el conjunto de prueba
// ejecuta train.py si modelo.h5 no existe
// -----------------------------------------------------------------------
router.get('/metrics', async (_req, res) => {
  try {
    const modelPath = path.join(MODEL_DIR, 'modelo.h5')

    if (!existsSync(modelPath)) {
      await spawnPython('preprocess.py')
      await spawnPython('train.py')
    }

    const output = await spawnPython('predict.py', ['--metrics'])
    res.json(JSON.parse(output))
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
})

export default router
