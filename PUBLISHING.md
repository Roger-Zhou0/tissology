# Tissology public image workspace

Hosted at https://roger-zhou0.github.io/tissology/ through GitHub Pages, from `main /docs`. Merged changes publish through the Pages workflow.

The frontend adapts the original application styles. It includes 36 licensed public tissue tiles, browser-only image import, reference-label review, appearance similarity, findings, and standalone report export. See `docs/ATTRIBUTION.md` for image provenance and `docs/README.md` for supported functionality.

GitHub Pages serves static files only. No calls go to the research server, private collections, analytics, or paid APIs. Optional MobileNet inference runs on the visitor’s CPU after an explicit enable/download step. The visitor's browser performs lightweight image comparisons and keeps sample reviews/findings in localStorage. Imported images and their findings stay in memory for the current tab.

Validate browsing, zoom, filtering, label correction, browser persistence, similarity, import, export, opt-in model loading, cancellation, and CPU prediction parity after changes. The archived model benchmark is separate at `evaluation.html`; sample reference labels must never be presented as model predictions.
