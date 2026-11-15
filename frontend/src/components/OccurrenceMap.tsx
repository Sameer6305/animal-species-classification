/**
 * World occurrence map: GBIF basemap tiles + GBIF Maps API v2 occurrence density for one
 * taxon. Leaflet (and its CSS) live in this module, which pages load with React.lazy so the
 * map library never lands in the initial bundle. No API key needed for either tile source.
 */
import 'leaflet/dist/leaflet.css'

import L from 'leaflet'
import { useEffect, useRef, useState } from 'react'

import { useTheme } from '@/lib/theme'
import { cn } from '@/lib/utils'

interface OccurrenceMapProps {
  taxonKey: number
  /** For the accessible description, e.g. "Red panda (Ailurus fulgens)". */
  label: string
  className?: string
}

const BASEMAP = (style: string) => `https://tile.gbif.org/3857/omt/{z}/{x}/{y}@1x.png?style=${style}`
const DENSITY = (taxonKey: number) =>
  `https://api.gbif.org/v2/map/occurrence/density/{z}/{x}/{y}@1x.png?taxonKey=${taxonKey}&bin=hex&hexPerTile=64&style=classic.poly`

const ATTRIBUTION =
  '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors, © <a href="https://openmaptiles.org/" target="_blank" rel="noreferrer">OpenMapTiles</a>, <a href="https://www.gbif.org/citation-guidelines" target="_blank" rel="noreferrer">GBIF</a>'

export default function OccurrenceMap({ taxonKey, label, className }: OccurrenceMapProps) {
  const el = useRef<HTMLDivElement>(null)
  const map = useRef<L.Map | null>(null)
  const base = useRef<L.TileLayer | null>(null)
  const { resolved } = useTheme()
  const [tileError, setTileError] = useState(false)

  useEffect(() => {
    if (!el.current) return
    // Show the whole world once, at the largest whole zoom that fits the width (whole zooms
    // avoid hairline seams between tiles that fractional zooms produce).
    const fit = Math.max(1, Math.floor(Math.log2(el.current.clientWidth / 256)))
    const world = L.latLngBounds([-85, -180], [85, 180])
    const m = L.map(el.current, {
      center: [28, 10],
      zoom: fit,
      minZoom: fit,
      maxZoom: 9,
      zoomSnap: 1,
      zoomDelta: 1,
      maxBounds: world,
      maxBoundsViscosity: 1,
      scrollWheelZoom: false, // page scroll stays page scroll; use the +/- controls or drag
      attributionControl: true,
    })
    m.attributionControl.setPrefix(false)
    const density = L.tileLayer(DENSITY(taxonKey), {
      opacity: 0.75,
      zIndex: 2,
      crossOrigin: true,
      noWrap: true,
      bounds: world,
    })
    // (GBIF answers empty density tiles with 204, which Leaflet reports as tile errors: only
    // basemap errors are treated as a connectivity problem.)
    density.addTo(m)
    map.current = m
    return () => {
      m.remove()
      map.current = null
      base.current = null
    }
  }, [taxonKey])

  useEffect(() => {
    const m = map.current
    if (!m) return
    base.current?.remove()
    base.current = L.tileLayer(BASEMAP(resolved === 'dark' ? 'gbif-dark' : 'gbif-light'), {
      attribution: ATTRIBUTION,
      zIndex: 1,
      crossOrigin: true,
      noWrap: true,
      bounds: L.latLngBounds([-85, -180], [85, 180]),
    })
      .on('tileerror', () => setTileError(true))
      .addTo(m)
  }, [resolved, taxonKey])

  return (
    <figure className={cn('relative', className)}>
      <div
        ref={el}
        role="region"
        aria-label={`Map of GBIF occurrence records for ${label}`}
        data-testid="occurrence-map"
        className="wildlens-map h-[28rem] w-full overflow-hidden rounded border border-border bg-surface-sunken"
      />
      {tileError && (
        <p className="absolute right-3 top-3 z-[500] rounded-sm bg-bg/90 px-3 py-1.5 text-caption text-fg-secondary">
          Some map tiles could not be loaded (offline?).
        </p>
      )}
    </figure>
  )
}
