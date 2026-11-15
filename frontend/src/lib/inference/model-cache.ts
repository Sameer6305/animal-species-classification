/**
 * Download a model file once, with progress, and keep it in Cache Storage so later
 * visits (and reloads) start instantly. Falls back to a plain fetch where Cache Storage
 * is unavailable (non-secure origins, some private windows).
 */
import type { LoadProgress } from '@/lib/inference/types'

export const MODEL_CACHE_NAME = 'wildlens-models-v1'

async function openCache(): Promise<Cache | null> {
  try {
    return 'caches' in globalThis ? await caches.open(MODEL_CACHE_NAME) : null
  } catch {
    return null
  }
}

export async function fetchModel(
  url: string,
  onProgress?: (p: LoadProgress) => void,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const cache = await openCache()
  const absolute = new URL(url, globalThis.location?.href).toString()
  const hit = await cache?.match(absolute).catch(() => undefined)
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer())
    onProgress?.({ phase: 'model', loaded: buf.byteLength, total: buf.byteLength, fromCache: true })
    return buf
  }

  const res = await fetch(absolute, { signal })
  if (!res.ok) throw new Error(`Could not download the model (${res.status} ${res.statusText}) from ${url}.`)
  const total = Number(res.headers.get('content-length')) || undefined
  let buf: Uint8Array
  if (res.body) {
    const reader = res.body.getReader()
    const chunks: Uint8Array[] = []
    let loaded = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      loaded += value.byteLength
      onProgress?.({ phase: 'model', loaded, total })
    }
    buf = new Uint8Array(loaded)
    let offset = 0
    for (const c of chunks) {
      buf.set(c, offset)
      offset += c.byteLength
    }
  } else {
    buf = new Uint8Array(await res.arrayBuffer())
  }
  if (total && buf.byteLength !== total) throw new Error('The model download was incomplete. Please retry.')
  try {
    await cache?.put(
      absolute,
      new Response(new Blob([buf as BlobPart]), {
        headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(buf.byteLength) },
      }),
    )
  } catch {
    /* quota exceeded or storage disabled: still usable for this session */
  }
  return buf
}

/** True if the model at `url` is already in Cache Storage. */
export async function isModelCached(url: string): Promise<boolean> {
  const cache = await openCache()
  if (!cache) return false
  const absolute = new URL(url, globalThis.location?.href).toString()
  return Boolean(await cache.match(absolute).catch(() => undefined))
}
