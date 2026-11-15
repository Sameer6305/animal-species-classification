import { api } from '@/lib/api'
import type { IdentifyOptions, IdentifyResult, InferenceEngine } from '@/lib/inference/types'

/**
 * Server mode: the FastAPI backend runs the ONNX model. /api/predict (threshold, image
 * metadata) and /api/explain (top-5 + amber CAM, one forward pass) are requested in parallel.
 */
export class ServerInference implements InferenceEngine {
  readonly mode = 'server' as const

  async identify(file: Blob, opts: IdentifyOptions = {}): Promise<IdentifyResult> {
    const [pred, exp] = await Promise.all([
      api.predict(file, { topK: 5, signal: opts.signal }),
      api.explain(file, { colormap: 'amber', signal: opts.signal }),
    ])
    return {
      mode: 'server',
      backend: 'server',
      model: pred.model,
      variant: null,
      predictions: pred.predictions,
      lowConfidence: pred.low_confidence,
      threshold: pred.low_confidence_threshold,
      inferenceMs: pred.inference_ms,
      explainMs: exp.explain_ms,
      image: { width: pred.image.width, height: pred.image.height },
      heatmapPng: exp.heatmap_png,
      heatmapWidth: exp.width,
      heatmapHeight: exp.height,
      method: exp.method,
      slugs: { ...exp.slugs, ...pred.slugs },
    }
  }
}
