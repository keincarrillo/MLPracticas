// utils/python.ts
// helper para ejecutar scripts Python del módulo model como subprocesos
// usa Bun.spawn para lanzar el intérprete del entorno virtual
// retorna stdout como string o lanza error si el proceso falla

import path from 'path'

// ruta absoluta al directorio model/ resuelta desde la ubicación de este
// archivo — estable sin importar desde dónde se arranque el proceso
export const MODEL_DIR = path.resolve(import.meta.dirname, '../../../model')

// ruta al intérprete Python dentro del entorno virtual del módulo model
// se puede sobreescribir con la variable de entorno PYTHON_BIN
const PYTHON_BIN =
  process.env.PYTHON_BIN ?? path.join(MODEL_DIR, 'venv/bin/python3')

// tiempo máximo en ms que se espera a que un script Python termine
// pasado este tiempo el proceso se mata y se lanza TimeoutError
const TIMEOUT_MS = Number(process.env.PYTHON_TIMEOUT_MS ?? 120_000)

// spawnPython(script, args?) -> Promise<string>
// ejecuta MODEL_DIR/<script> con el intérprete del venv
// captura stdout y lo retorna; lanza error con stderr si el proceso falla
// aplica un timeout para evitar que un script colgado bloquee el servidor
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

  // carrera entre la ejecución normal y el timeout
  const timeout = new Promise<never>((_, reject) =>
    setTimeout(() => {
      proc.kill()
      reject(new Error(`${script} superó el timeout de ${TIMEOUT_MS}ms`))
    }, TIMEOUT_MS)
  )

  const execution = Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited
  ])

  const [stdout, stderr, exitCode] = await Promise.race([execution, timeout])

  if (exitCode !== 0) {
    throw new Error(`${script} falló (exit ${exitCode}):\n${stderr}`)
  }

  return stdout.trim()
}

// parsePythonJson<T>(output) -> T
// parsea la salida de un script Python como JSON tipado
// lanza un error descriptivo si la salida no es JSON válido
export function parsePythonJson<T>(output: string): T {
  try {
    return JSON.parse(output) as T
  } catch {
    throw new Error(
      `El script devolvió output no-JSON:\n${output.slice(0, 200)}`
    )
  }
}
