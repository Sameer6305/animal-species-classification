import { ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'

import { Section, SectionHeader } from '@/components/Section'
import { Button } from '@/components/ui/button'

export interface PlaceholderPageProps {
  eyebrow: string
  title: string
  lead: string
  phase: number
  /** What the finished page will contain. */
  planned: string[]
}

/** Stub for routes built in later phases. */
export function PlaceholderPage({ eyebrow, title, lead, phase, planned }: PlaceholderPageProps) {
  return (
    <Section>
      <SectionHeader as="h1" eyebrow={eyebrow} title={title} lead={lead} />
      <div className="grid grid-cols-12 gap-10 border-t border-border pt-10">
        <div className="col-span-4">
          <p className="type-overline !text-fg-muted">In preparation</p>
          <p className="mt-3 font-serif text-h3 text-fg">Arrives in Phase {phase}</p>
          <p className="mt-3 text-small text-fg-muted">
            This page is part of the WildLens build plan and will be completed in a later phase.
          </p>
          <Button asChild variant="secondary" className="mt-6">
            <Link to="/">
              Back to home <ArrowRight />
            </Link>
          </Button>
        </div>
        <div className="col-span-7 col-start-6">
          <p className="type-overline mb-4 !text-fg-muted">What it will include</p>
          <ol className="divide-y divide-border border-y border-border">
            {planned.map((item, i) => (
              <li key={item} className="flex gap-6 py-4 text-body text-fg-secondary">
                <span className="tabular w-6 shrink-0 pt-0.5 font-mono text-caption text-fg-muted">
                  {String(i + 1).padStart(2, '0')}
                </span>
                {item}
              </li>
            ))}
          </ol>
        </div>
      </div>
    </Section>
  )
}
