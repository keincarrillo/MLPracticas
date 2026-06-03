// utils/python.ts
// helper para ejecutar scripts Python del modulo model como subprocesos
// usa Bun.spawn para lanzar el interprete del entorno virtual
// retorna stdout como string o lanza error si el proceso falla

import path from 'path'

// ruta absoluta al directorio model/ relativa a este archivo
export const MODEL_DIR = path.resolve(process.cwd(), '../model')

// ruta al interprete Python dentro del entorno virtual del modulo model
// se puede sobreescribir con la variable de entorno PYTHON_BIN
const PYTHON_BIN =
  process.env.PYTHON_BIN ?? path.join(MODEL_DIR, 'venv/bin/python3')

// spawnPython(script, args?) -> Promise<string>
// ejecuta MODEL_DIR/<script> con el interprete del venv
// captura stdout y lo retorna; lanza error con stderr si el proceso falla
export async function spawnPython(
  script: string,
  args: string[] = []
): Promise<string> {
  const scriptPath = path.join(MODEL_DIR, script)

  const proc = Bun.spawn([PYTHON_BIN, scriptPath, ...args], {
    cwd: MODEL_DIR,
    stdout: 'pipe',
    stderr: 'pipe'
  })

  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited
  ])

  if (exitCode !== 0) {
    throw new Error(`${script} fallo (exit ${exitCode}):\n${stderr}`)
  }

  return stdout.trim()
}
