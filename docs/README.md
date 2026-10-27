# Tissology public image workspace

A static adaptation of the original Tissology frontend, hosted by GitHub Pages from `main /docs`. The original application's base, presentation, and collection styles are retained in `original/`. Backend application files remain unchanged.

## Working features

- Browse 36 actual public tissue tiles; filter by class, filename, and review status.
- Inspect and zoom images; navigate with buttons or arrow keys.
- Confirm, change, or mark reference labels unsure; record notes.
- Select image evidence and save findings with review snapshots.
- Export reviews/findings as JSON and standalone HTML reports with embedded images.
- Rank images by browser-computed color and texture distance.
- Import PNG, JPEG, and WebP files locally (100 images / 50 MB per selection); no uploads; optional local CPU inference.
- Persist sample reviews/findings in localStorage, with a tab-only fallback if storage fails. Local imports and findings containing them last only for the current tab.

Optional TIA MobileNetV3-Small inference generates actual predictions locally after explicit opt-in. It uses one CPU worker, batches up to 20 tiles, and supports cancellation, cached model downloads, unloading, and JSON export. Predictions and review labels remain separate. There is no arbitrary text semantic search, retraining, account system, or shared storage. Reference labels are explicitly distinguished from predictions. Aggregate benchmark exploration remains at `evaluation.html`.

## Resources and data

Static files and 36 PNG images (~3.5 MB combined) are served by GitHub Pages. Images load lazily. All computation and storage happen in the visitor's browser. `connect-src 'self'` permits downloading local static model/runtime assets; no remote inference endpoints are called. There are no external image/font assets, telemetry, paid APIs, or connections to the research server. Source attribution and modifications: [ATTRIBUTION.md](ATTRIBUTION.md). Machine-readable manifest: `samples.json`.

## Local CPU model

First-use download: about 15.1 MB (3.7 MB model plus 11.3 MB runtime). Input: square colorectal H&E tiles, at least 224 px per side; resized to 224 × 224, RGB / 255. Class scores are uncalibrated. On the 36 bundled examples, 23 predictions agree with their reference labels; this is a demonstration sample, not an independent accuracy estimate. This model is distinct from CONCH. See [model provenance and validation](models/README.md).

Model snapshots accompany newly saved findings and JSON review exports. Standalone predictions are held in the current tab and can be exported separately.

## Preview

Serve this directory with a static HTTP server, e.g. `python3 -m http.server 8765 --directory docs`, then open localhost:8765. Serving over HTTP avoids browser file-origin restrictions on canvas-based similarity and report export. No package installation or build step is needed.
