// middleware/ensureModel.ts
// middleware de Express que garantiza que el modelo entrenado existe en disco
// antes de dejar pasar la request al handler de la ruta
//
// si modelo.h5 no existe ejecuta el pipeline completo de entrenamiento:
//   preprocess.py → train.py
// si scaler.pkl no existe (pero no el modelo) solo ejecuta preprocess.py
//
// uso: router.get('/ruta', ensureModel, handler)

import type { Request, Response, NextFunction } from 'express'
import path from 'path'
import { existsSync } from 'fs'
import { spawnPython, MODEL_DIR } from '@/utils/python'

const MODEL_PATH = path.join(MODEL_DIR, 'modelo.h5')
const SCALER_PATH = path.join(MODEL_DIR, 'scaler.pkl')

// ensureModel — garantiza modelo entrenado antes de pasar al handler
export async function ensureModel(
  _req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    if (!existsSync(MODEL_PATH)) {
      await spawnPython('preprocess.py')
      await spawnPython('train.py')
    }
    next()
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
}

// ensureScaler — garantiza solo el scaler (para /history que no necesita modelo)
export async function ensureScaler(
  _req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    if (!existsSync(SCALER_PATH)) {
      await spawnPython('preprocess.py')
    }
    next()
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
}
