/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** API origin; empty = same origin (dev/preview proxy). */
  readonly VITE_API_BASE?: string
  /** Include the /__ui component preview in a production build ("true"). */
  readonly VITE_UI_PREVIEW?: string
  readonly VITE_REPO_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
