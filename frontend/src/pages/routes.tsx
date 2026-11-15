import type { PlaceholderPageProps } from '@/pages/PlaceholderPage'

/** Placeholder content for routes from section 3 of docs/platform/PLAN.md. */
export const PLACEHOLDERS: Record<string, PlaceholderPageProps> = {
  research: {
    eyebrow: 'Research',
    title: 'How the models compare',
    lead: 'Four architectures trained on the same data, evaluated on the same test split — and an honest account of where they fall short.',
    phase: 5,
    planned: [
      'Comparison of ResNet50, EfficientNetV2-S, ConvNeXt-Tiny and Swin-Tiny',
      'Confusion matrices, ROC curves and training histories',
      'Dataset composition and preprocessing',
      'Limitations: class imbalance, tightly cropped training images, hard classes',
    ],
  },
  about: {
    eyebrow: 'About',
    title: 'About WildLens',
    lead: 'The team, the technology and the architecture behind the project.',
    phase: 5,
    planned: [
      'Team and acknowledgements',
      'Technology stack',
      'Architecture diagram',
      'Links to the source code',
    ],
  },
}
