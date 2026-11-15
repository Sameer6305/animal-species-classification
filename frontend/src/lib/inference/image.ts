/** Decode an uploaded image to RGBA pixels at its natural size, EXIF orientation applied. */
export interface DecodedImage {
  width: number
  height: number
  /** RGBA, row-major. */
  data: Uint8ClampedArray
}

export async function decodeImage(file: Blob): Promise<DecodedImage> {
  let bitmap: ImageBitmap
  try {
    // No colour management or premultiplication: keep the file's sRGB values as Pillow does.
    bitmap = await createImageBitmap(file, {
      imageOrientation: 'from-image',
      colorSpaceConversion: 'none',
      premultiplyAlpha: 'none',
    })
  } catch {
    throw new Error('This file could not be decoded as an image. Use a JPEG, PNG or WebP photograph.')
  }
  const { width, height } = bitmap
  const canvas =
    typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(width, height)
      : Object.assign(document.createElement('canvas'), { width, height })
  const ctx = canvas.getContext('2d', { willReadFrequently: true }) as
    CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null
  if (!ctx) throw new Error('Canvas 2D is not available in this browser.')
  ctx.drawImage(bitmap, 0, 0)
  bitmap.close()
  const data = ctx.getImageData(0, 0, width, height).data
  return { width, height, data }
}

/** Grayscale uint8 map -> PNG data URL (R = G = B = value, opaque), like heatmap_png. */
export async function grayscalePng(
  values: Uint8ClampedArray,
  width: number,
  height: number,
): Promise<string> {
  const canvas = Object.assign(document.createElement('canvas'), { width, height })
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas 2D is not available in this browser.')
  const img = ctx.createImageData(width, height)
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number
    img.data[i * 4] = v
    img.data[i * 4 + 1] = v
    img.data[i * 4 + 2] = v
    img.data[i * 4 + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
  return canvas.toDataURL('image/png')
}
