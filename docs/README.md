# Tissology public image workspace

A static adaptation of the original Tissology frontend, hosted by GitHub Pages from `main /docs`. The original application's base, presentation, and collection styles are retained in `original/`. Backend application files remain unchanged.

## Working features

- Browse 36 actual public tissue tiles; filter by class, filename, and review status.
- Inspect and zoom images; navigate with buttons or arrow keys.
- Confirm, change, or mark reference labels unsure; record notes.
- Select image evidence and save findings with review snapshots.
- Export reviews/findings as JSON and standalone HTML reports with embedded images.
- Rank images by browser-computed color and texture distance.
- Import PNG, JPEG, and WebP files locally (100 images / 50 MB per selection); no uploads or inference.
- Persist sample reviews/findings in localStorage, with a tab-only fallback if storage fails. Local imports and findings containing them last only for the current tab.

There are no model predictions for individual sample tiles, live inference, arbitrary text semantic search, retraining, accounts, or shared storage. Reference labels are explicitly distinguished from predictions. Aggregate benchmark exploration remains at `evaluation.html`.

## Resources and data

Static files and 36 PNG images (~3.5 MB combined) are served by GitHub Pages. Images load lazily. All computation and storage happen in the visitor's browser. `connect-src 'none'` blocks backend requests. There are no external image/font assets, telemetry, paid APIs, or connections to the research server. Source attribution and modifications: [ATTRIBUTION.md](ATTRIBUTION.md). Machine-readable manifest: `samples.json`.

## Preview

Serve this directory with a static HTTP server, e.g. `python3 -m http.server 8765 --directory docs`, then open localhost:8765. Serving over HTTP avoids browser file-origin restrictions on canvas-based similarity and report export. No package installation or build step is needed.
