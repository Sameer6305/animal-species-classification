/**
 * Context handoff into the Field Guide (/ask) from Identify and species profiles.
 * Encoded in the URL (`/ask?species=<slug>&name=<common name>`) so it survives reloads
 * and can be linked; the Field Guide pre-fills a first question and sends the slug as
 * chat `context` with every request.
 */

export interface AskHandoff {
  slug: string
  name: string
  /** `identify` adds "the photo I identified" framing to the suggested question. */
  from?: 'identify' | 'profile'
}

export function askLink(h: AskHandoff): string {
  const p = new URLSearchParams({ species: h.slug, name: h.name })
  if (h.from) p.set('from', h.from)
  return `/ask?${p.toString()}`
}

const SLUG = /^[a-z0-9]+(?:-[a-z0-9]+)*$/

export function parseHandoff(search: string): AskHandoff | null {
  const p = new URLSearchParams(search)
  const slug = p.get('species')?.trim().toLowerCase() ?? ''
  if (!SLUG.test(slug) || slug.length > 64) return null
  const name = (p.get('name') ?? '').trim().slice(0, 80) || slug.replace(/-/g, ' ')
  const fromRaw = p.get('from')
  const from = fromRaw === 'identify' || fromRaw === 'profile' ? fromRaw : undefined
  return { slug, name, from }
}

/** Lower-case a common name for mid-sentence use, keeping proper nouns like "Harbor seal" sensible. */
function midSentence(name: string): string {
  return name.length > 1 && name[1] === name[1]?.toLowerCase()
    ? name.charAt(0).toLowerCase() + name.slice(1)
    : name
}

/** The pre-filled first question for a handoff. */
export function suggestedQuestion(h: AskHandoff): string {
  const n = midSentence(h.name)
  return h.from === 'identify'
    ? `The model identified a ${n} in my photo. Where do they live, and how are they doing in the wild?`
    : `Tell me about the ${n}: where does it live, what does it eat, and what is its conservation status?`
}
