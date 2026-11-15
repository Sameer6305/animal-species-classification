import { ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'

import { Attribution } from '@/components/Attribution'
import { Container } from '@/components/layout/Container'
import { ScientificName } from '@/components/ScientificName'
import { Section } from '@/components/Section'
import { StatRow, StatTile } from '@/components/StatTile'
import { Button } from '@/components/ui/button'

/** CC0 photograph from the knowledge base (Wikimedia Commons). */
const HERO = {
  src: 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Red_Panda_%2824986761703%29.jpg/1280px-Red_Panda_%2824986761703%29.jpg',
  alt: 'A red panda resting on a branch',
  image: {
    artist: 'Mathias Appel',
    license: 'CC0',
    license_url: 'https://creativecommons.org/publicdomain/zero/1.0/',
    source_url: 'https://commons.wikimedia.org/wiki/File:Red_Panda_(24986761703).jpg',
    provider: 'Wikimedia Commons',
  },
}

/** Home placeholder — the scroll-driven narrative arrives in Phase 5. */
export function HomePage() {
  return (
    <>
      <Container width="content" className="grid grid-cols-12 items-center gap-12 pb-20 pt-20">
        <div className="col-span-5">
          <p className="type-overline mb-6 flex items-center gap-3">
            <span aria-hidden="true" className="h-px w-8 bg-accent" />
            Species identification · 80 classes
          </p>
          <h1 className="type-display">See the animal. Understand the model.</h1>
          <p className="type-lead mt-6 max-w-md">
            WildLens identifies animals in photographs, shows which parts of the image the model relied on,
            and connects each result to a field-guide profile of the species.
          </p>
          <div className="mt-10 flex items-center gap-4">
            <Button asChild size="lg">
              <Link to="/identify">
                Identify a photo <ArrowRight />
              </Link>
            </Button>
            <Button asChild size="lg" variant="secondary">
              <Link to="/species">Browse the atlas</Link>
            </Button>
          </div>
        </div>
        <figure className="col-span-7">
          <div className="grain aspect-[3/2] overflow-hidden rounded bg-surface-sunken">
            <img src={HERO.src} alt={HERO.alt} className="size-full object-cover" />
          </div>
          <figcaption className="mt-3 flex items-baseline justify-between gap-6">
            <span className="text-small text-fg-secondary">
              Red panda, <ScientificName name="Ailurus fulgens" /> — Endangered
            </span>
            <Attribution image={HERO.image} />
          </figcaption>
        </figure>
      </Container>

      <Section spacing="compact" width="content" className="pt-0">
        <StatRow>
          <StatTile value="80" label="Animal classes" footnote="Mammals, birds, reptiles, insects and more" />
          <StatTile
            value="79.9"
            unit="%"
            label="Top-1 test accuracy"
            footnote="ResNet50 on 7,576 held-out images"
          />
          <StatTile
            value="97.3"
            unit="%"
            label="Top-5 test accuracy"
            footnote="Correct species among the top five"
          />
          <StatTile
            value="27"
            unit="ms"
            label="Inference on a laptop CPU"
            footnote="Median server round-trip"
          />
        </StatRow>
        <p className="type-caption mt-6">
          The full story — the problem, the dataset, the pipeline and the results — arrives with Phase 5.
        </p>
      </Section>
    </>
  )
}
