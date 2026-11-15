import type { SpeciesImage } from '@/lib/api'
import { cn } from '@/lib/utils'

export type AttributionImage = Pick<
  SpeciesImage,
  'artist' | 'license' | 'license_url' | 'source_url' | 'provider'
>

interface AttributionProps {
  image: AttributionImage
  /** `caption` sits under a photo; `overlay` is light text for use on a dark scrim. */
  variant?: 'caption' | 'overlay'
  className?: string
}

/** Image credit: author, licence (linked) and source (linked). Required for Commons imagery. */
export function Attribution({ image, variant = 'caption', className }: AttributionProps) {
  const linkCls =
    variant === 'overlay'
      ? 'underline decoration-white/40 underline-offset-2 hover:decoration-white'
      : 'underline decoration-border-strong underline-offset-2 hover:text-fg hover:decoration-accent'
  const author = image.artist?.trim() || 'Unknown author'
  return (
    <p
      className={cn(
        'text-caption leading-snug',
        variant === 'overlay' ? 'text-white/90' : 'text-fg-muted',
        className,
      )}
    >
      <span className="sr-only">Image credit: </span>
      {image.source_url ? (
        <a href={image.source_url} target="_blank" rel="noreferrer" className={linkCls}>
          {author}
        </a>
      ) : (
        author
      )}
      {image.license && (
        <>
          <span aria-hidden="true"> · </span>
          {image.license_url ? (
            <a href={image.license_url} target="_blank" rel="noreferrer license" className={linkCls}>
              {image.license}
            </a>
          ) : (
            image.license
          )}
        </>
      )}
      {image.provider && (
        <>
          <span aria-hidden="true"> · </span>
          {image.provider}
        </>
      )}
    </p>
  )
}
