/**
 * Full-page review screenshots of the running dev server.
 *   npm run dev            # in another terminal
 *   npm run screenshots -- [outDir] [baseUrl]
 * Captures /, /__ui and /identify at 1280×800 and 1920×1080 in light and dark.
 */
import { mkdir } from 'node:fs/promises'
import path from 'node:path'
import { chromium } from '@playwright/test'

const outDir = path.resolve(process.argv[2] ?? 'screenshots')
const base = process.argv[3] ?? 'http://localhost:5173'
const pages = [
  ['home', '/'],
  ['ui', '/__ui'],
  ['identify-placeholder', '/identify'],
  ['not-found', '/does-not-exist'],
]
const viewports = [
  [1280, 800],
  [1920, 1080],
]
const themes = ['light', 'dark']

await mkdir(outDir, { recursive: true })
const browser = await chromium.launch()
try {
  for (const [w, h] of viewports) {
    for (const theme of themes) {
      const context = await browser.newContext({
        viewport: { width: w, height: h },
        colorScheme: theme,
        reducedMotion: 'reduce',
      })
      await context.addInitScript((t) => localStorage.setItem('wildlens-theme', t), theme)
      const page = await context.newPage()
      for (const [name, route] of pages) {
        if (name === 'not-found' && w !== 1280) continue
        await page.goto(base + route, { waitUntil: 'networkidle' })
        await page.evaluate(() => document.fonts.ready)
        // Let lazy images load: scroll through the page once.
        await page.evaluate(async () => {
          for (let y = 0; y < document.body.scrollHeight; y += 600) {
            window.scrollTo(0, y)
            await new Promise((r) => setTimeout(r, 60))
          }
          window.scrollTo(0, 0)
        })
        await page.waitForLoadState('networkidle')
        await page.waitForTimeout(600)
        const file = path.join(outDir, `${name}-${w}-${theme}.png`)
        await page.screenshot({ path: file, fullPage: true })
        console.log(file)
      }
      await context.close()
    }
  }
  // Narrow viewport: the "designed for laptop and desktop" notice.
  const context = await browser.newContext({ viewport: { width: 900, height: 700 } })
  const page = await context.newPage()
  await page.goto(base + '/', { waitUntil: 'networkidle' })
  const file = path.join(outDir, 'narrow-notice-900.png')
  await page.screenshot({ path: file })
  console.log(file)
  await context.close()
} finally {
  await browser.close()
}
