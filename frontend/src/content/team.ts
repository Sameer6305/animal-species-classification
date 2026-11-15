/**
 * Team section on the About page. Empty by default: the section is hidden until entries are
 * added here (see frontend/README.md, "Content"). Only add details people agreed to publish.
 *
 * Example:
 *   { name: 'A. Person', role: 'Model training and evaluation', link: 'https://example.org' }
 */
export interface TeamMember {
  name: string
  /** What they worked on, e.g. "Frontend and design". */
  role?: string
  /** Optional profile or homepage URL (http/https). */
  link?: string
}

export const TEAM: TeamMember[] = []
