import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { TeamSection } from '@/components/about/TeamSection'
import { TEAM } from '@/content/team'

describe('TeamSection', () => {
  it('ships empty, so the About page shows no team section by default', () => {
    expect(TEAM).toEqual([])
    const { container } = render(<TeamSection members={TEAM} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders members with a role and only http(s) links', () => {
    render(
      <TeamSection
        members={[
          { name: 'A. Person', role: 'Model training', link: 'https://example.org/a' },
          { name: 'B. Person', link: 'javascript:alert(1)' },
        ]}
      />,
    )
    expect(screen.getByRole('heading', { name: 'Team' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'A. Person' })).toHaveAttribute('href', 'https://example.org/a')
    expect(screen.getByText('Model training')).toBeInTheDocument()
    expect(screen.getByText('B. Person').closest('a')).toBeNull()
  })
})
