# WildLens frontend

React 18 + Vite + TypeScript (strict) + Tailwind CSS v3 + shadcn/ui (Radix), with Framer Motion,
React Router, TanStack Query, lucide-react and Recharts. Phase 3 of `docs/platform/PLAN.md`:
design tokens, typography, the layout shell, reusable components and a component preview page.
Real pages arrive in Phases 4–5; their routes currently show placeholders.

Laptop and desktop only (designed for 1280 px and wider). Below 1024 px a notice replaces the app.

## Run

Requires Node 22+ and npm (`nvm use` picks it up from `.nvmrc`). Node 20 fails to start the Vitest workers.

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173  (/__ui = component preview)
```

The dev and preview servers proxy `/api` to `http://localhost:8000`, so start the backend in a
second terminal (see `backend/README.md`):

```bash
cd backend && .venv/bin/uvicorn app.main:app --reload
```

Use `WILDLENS_API_URL=http://host:port npm run dev` to proxy elsewhere, or set `VITE_API_BASE`
at build time to call an API on another origin (that origin must be in `WILDLENS_CORS_ORIGINS`).
Phase 3 pages do not call the API yet; the typed client is in `src/lib/api.ts`.

| Script                               | What it does                                                                                                                                                                               |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `npm run dev`                        | Vite dev server with HMR                                                                                                                                                                   |
| `npm run build`                      | Type-check, then production build to `dist/`                                                                                                                                               |
| `npm run preview`                    | Serve `dist/` (also proxies `/api`)                                                                                                                                                        |
| `npm run lint` / `npm run typecheck` | ESLint (flat config) / `tsc -b`                                                                                                                                                            |
| `npm test`                           | Vitest + Testing Library (jsdom)                                                                                                                                                           |
| `npm run format`                     | Prettier                                                                                                                                                                                   |
| `npm run screenshots -- <outDir>`    | Full-page Playwright screenshots of `/`, `/__ui`, `/identify` and a 404 at 1280×800 and 1920×1080, light and dark (needs `npm run dev` running and `npx playwright install chromium` once) |

The `/__ui` preview is included only in development (`import.meta.env.DEV`); set
`VITE_UI_PREVIEW=true` to include it in a production build for review.

Fonts are self-hosted from `@fontsource-variable/*` packages and bundled by Vite, so local mode
works offline. Species photographs on the preview page are hotlinked Wikimedia Commons thumbnails
(they need a network connection; cards fall back to a placeholder); the heatmap demo photo
(`src/fixtures/red-panda.jpg`, CC0) is bundled.

## Layout

```
src/
  main.tsx, App.tsx            providers (theme, query, motion, tooltip, toaster) and routes
  styles/globals.css           design tokens (CSS variables), fonts, type scale, utilities
  lib/                         api.ts (typed client + SSE parser), theme.tsx, iucn.ts, format.ts,
                               images.ts (Commons thumbnails), heatmap.ts (CAM colouriser), upload.ts
  components/ui/               shadcn/ui primitives themed to the tokens
  components/layout/           AppShell, SiteHeader, SiteFooter, Container, Wordmark, ThemeToggle,
                               ScreenSizeNotice
  components/                  Section, StatTile, SpeciesCard, IucnBadge, IucnScale, ConfidenceBar,
                               PredictionList, HeatmapViewer, UploadDropzone, Attribution,
                               ScientificName, States (EmptyState/ErrorState), charts/, chat/
  pages/                       HomePage, PlaceholderPage (+ routes.tsx content), NotFoundPage,
                               dev/UiPreview.tsx
  fixtures/                    small copies of data/species_kb.json, one /api/explain response
                               (amber colormap, overlay omitted), a metrics summary
```

## Design tokens

Tokens are RGB channels on `:root` (light) and `:root[data-theme='dark']`; with no stored choice,
`prefers-color-scheme: dark` applies the dark set. The theme toggle stores `wildlens-theme` in
`localStorage` and an inline script in `index.html` applies it before first paint. Tailwind maps the
tokens to classes (`bg-surface`, `text-fg-muted`, `bg-iucn-en`, …), so components never use raw hex.

Base palette: forest `#0F1F17`, moss `#2F4A3A`, paper `#F4F1EA`, stone `#8A8F87`, ochre `#C8913A`.

### Contrast (WCAG 2.1)

Text on backgrounds (AA: 4.5 normal text, 3.0 large text and UI boundaries):

| Pair                                                | Light              | Dark               |
| --------------------------------------------------- | ------------------ | ------------------ |
| text primary on background / surface / raised       | 15.2 / 16.1 / 17.1 | 15.3 / 14.2 / 12.6 |
| text secondary on background / raised               | 8.3 / 9.4          | 10.4 / 8.5         |
| text muted on background / sunken / raised          | 5.5 / 5.0 / 6.2    | 6.8 / 7.1 / 5.6    |
| accent text (ochre for text) on background / sunken | 5.2 / 4.8          | 9.3 / 9.8          |
| success / warning / danger on background            | 5.5 / 5.0 / 5.6    | 7.2 / 9.0 / 6.9    |
| primary-button label on primary                     | 8.6                | 13.9               |
| accent-button label on ochre                        | 6.2                | 7.3                |
| focus ring vs background / surface (non-text)       | 4.1 / 4.3          | 7.7 / 7.1          |
| input border vs background / raised (non-text)      | 3.2 / 3.6          | 3.8 / 3.1          |

Stone `#8A8F87` is 2.9:1 on paper, so it is used only for decorative rules and icons, never text.
Ochre `#C8913A` itself is used for fills, rules and highlights; ochre text uses `--accent-text`.

IUCN code chips (code text on fill; the category name always appears next to the chip in body ink,
so status never relies on colour):

| Code  | LC  | NT  | VU  | EN  | CR  | EW  | EX   | DD  | NE  |
| ----- | --- | --- | --- | --- | --- | --- | ---- | --- | --- |
| Light | 5.0 | 5.1 | 7.0 | 5.2 | 7.1 | 9.8 | 15.1 | 7.1 | 7.2 |
| Dark  | 5.9 | 7.8 | 9.0 | 6.3 | 4.7 | 5.2 | 14.2 | 5.6 | 7.4 |

In dark mode EX is inverted (bone chip) so it stays visible on the forest background.

## Typography

- **Fraunces** (variable, with an optical-size axis; OFL) for display and headings. Chosen over
  Playfair Display because the `opsz` axis keeps it sturdy at h3/h4 sizes and refined at display
  sizes, and its softer, old-style construction reads as natural-history editorial rather than
  fashion-magazine high contrast.
- **Inter** (variable; OFL) for interface and body text; **JetBrains Mono** (OFL) for technical
  labels (model method, catalogue numbers, API status codes).
- Scale: display 72, h1 52, h2 38, h3 28, h4 21 (Inter semibold), lead 20, body 16/1.65, small 14,
  caption 13, overline 12 caps, mono 13. Classes `type-display` … `type-mono`; `tabular` for
  metric numerals.
- `ScientificName` renders `lang="la"` italics; ranks above genus (family, order, …) render upright,
  following zoological convention.

## Notes for later phases

- `api.explain()` defaults to the `amber` colormap; `HeatmapViewer` colourises `heatmap_png`
  client-side, so `overlay_png` is optional.
- Predict/explain responses carry `slugs` (`{label: slug}`); `PredictionList` takes them with a
  `renderLink` prop to link to `/species/:slug`.
- `ChatMessage` has a `searching` status for live lookups (~4.5 s for non-KB animals) and a
  provider badge (`fallback` = "Retrieval-only answer").
- KB image URLs are full-resolution originals; `commonsThumb()` requests a standard Commons
  thumbnail width instead.
