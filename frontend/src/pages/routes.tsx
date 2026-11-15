import type { PlaceholderPageProps } from '@/pages/PlaceholderPage'

/** Placeholder content for routes from section 3 of docs/platform/PLAN.md. */
export const PLACEHOLDERS: Record<string, PlaceholderPageProps> = {
  identify: {
    eyebrow: 'Identify',
    title: 'Identify an animal from a photograph',
    lead: 'Upload, drop or paste a photo and the model returns its five most likely species, with a heatmap showing which parts of the image informed the decision.',
    phase: 4,
    planned: [
      'Upload, drag-and-drop, paste, camera and curated sample photographs',
      'Top-5 predictions with confidence and inference time',
      'Class-activation heatmap overlay with an opacity control',
      'A clear warning when the photo may not show one of the 80 species',
      'Links to the species profile and to the Field Guide',
    ],
  },
  species: {
    eyebrow: 'Species Atlas',
    title: 'Eighty animals, one atlas',
    lead: 'Every class the model recognises, with photographs, scientific names and conservation status.',
    phase: 4,
    planned: [
      'A grid of all 80 classes with licensed photographs',
      'Search by common or scientific name',
      'Filters for animal group and IUCN Red List status',
      'The model’s test accuracy for each class',
    ],
  },
  speciesProfile: {
    eyebrow: 'Species profile',
    title: 'Species profile',
    lead: 'A field-guide page for a single animal: summary, taxonomy, status and where it has been observed.',
    phase: 4,
    planned: [
      'Hero photograph with full attribution',
      'Summary, taxonomy and key facts',
      'IUCN Red List status on the LC → EX scale',
      'World occurrence map from GBIF',
      'How well the model recognises this class',
    ],
  },
  ask: {
    eyebrow: 'Field Guide',
    title: 'Ask about any animal',
    lead: 'A conversational guide grounded in Wikipedia, Wikidata and iNaturalist, with every claim linked to its source.',
    phase: 4,
    planned: [
      'Streaming answers with numbered citations',
      'Questions about any animal, not only the 80 classes',
      'Context handed over from an identification',
      'A retrieval-only answer when no language model is available',
    ],
  },
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
