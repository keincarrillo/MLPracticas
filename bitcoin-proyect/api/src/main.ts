import express from 'express'
import cors from 'cors'
import apiRoutes from '@/routes/api.routes.ts'

const app = express()

// Midlewares
app.use(cors())
app.use(express.json())

// Routes
app.use('/api', apiRoutes)

export default app
