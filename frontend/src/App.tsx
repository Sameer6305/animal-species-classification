import { lazy, Suspense } from 'react'
import { createBrowserRouter, RouterProvider, type RouteObject } from 'react-router-dom'

import { AppShell } from '@/components/layout/AppShell'
import { Spinner } from '@/components/ui/spinner'
import { HomePage } from '@/pages/HomePage'
import { NotFoundPage } from '@/pages/NotFoundPage'
import { PlaceholderPage } from '@/pages/PlaceholderPage'
import { PLACEHOLDERS } from '@/pages/routes'

/** Dev-only component preview; the import is dropped from production bundles. */
const SHOW_UI_PREVIEW = import.meta.env.DEV || import.meta.env.VITE_UI_PREVIEW === 'true'
const UiPreview = SHOW_UI_PREVIEW ? lazy(() => import('@/pages/dev/UiPreview')) : null

const p = PLACEHOLDERS

const children: RouteObject[] = [
  { index: true, element: <HomePage /> },
  { path: 'identify', element: <PlaceholderPage {...p.identify!} /> },
  { path: 'species', element: <PlaceholderPage {...p.species!} /> },
  { path: 'species/:slug', element: <PlaceholderPage {...p.speciesProfile!} /> },
  { path: 'ask', element: <PlaceholderPage {...p.ask!} /> },
  { path: 'research', element: <PlaceholderPage {...p.research!} /> },
  { path: 'about', element: <PlaceholderPage {...p.about!} /> },
  ...(UiPreview
    ? [
        {
          path: '__ui',
          element: (
            <Suspense
              fallback={
                <div className="flex justify-center py-32">
                  <Spinner label="Loading preview" />
                </div>
              }
            >
              <UiPreview />
            </Suspense>
          ),
        },
      ]
    : []),
  { path: '*', element: <NotFoundPage /> },
]

const router = createBrowserRouter([{ path: '/', element: <AppShell />, children }])

export function App() {
  return <RouterProvider router={router} />
}
