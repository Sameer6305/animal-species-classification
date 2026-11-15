import { Section, SectionHeader } from '@/components/Section'
import type { TeamMember } from '@/content/team'
import { safeHttpUrl } from '@/lib/url'

/** About-page team list; renders nothing while `src/content/team.ts` is empty. */
export function TeamSection({ members }: { members: TeamMember[] }) {
  if (members.length === 0) return null
  return (
    <Section spacing="compact" aria-labelledby="team-title" data-testid="team-section">
      <SectionHeader id="team-title" size="md" title="Team" />
      <ul className="grid grid-cols-3 gap-x-12 gap-y-6">
        {members.map((m) => {
          const href = safeHttpUrl(m.link)
          return (
            <li key={m.name} className="border-t border-border pt-4">
              <p className="font-serif text-h4 text-fg">
                {href ? (
                  <a href={href} target="_blank" rel="noreferrer" className="link">
                    {m.name}
                  </a>
                ) : (
                  m.name
                )}
              </p>
              {m.role && <p className="mt-1 text-small text-fg-secondary">{m.role}</p>}
            </li>
          )
        })}
      </ul>
    </Section>
  )
}
