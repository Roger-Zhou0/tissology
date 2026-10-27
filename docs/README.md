# Tissology public research workspace

A static workspace for the archived `conch-retention-001` evaluation. Hosted with GitHub Pages from `main /docs`. No backend, inference, account, uploads, analytics, or calls to the private research server. Browser-side interaction only.

## Explore

- Filter and sort the nine tissue classes by recall, precision, or support.
- Inspect class prediction distributions and individual confusion-matrix cells.
- Switch between tile counts and row percentages.
- Review all nonzero error pairs, ranked by frequency.
- Write findings linked to a class or confusion pair and export them as JSON.

Findings live in memory in the current tab. Closing or reloading clears them; export first. The app warns before leaving when there is work in memory. No original tissue images or per-image predictions are included, and there are no simulated images or invented predictions.

## Provenance

`benchmark.json` is copied from `studio/static/collection-validation-conch-retention-001.json`. The UI's embedded counts reproduce that record. Matrix total: 7,180; correct: 6,905; label agreement: 96.17%; macro F1: 94.62%. Precision, recall, and F1 are computed from the embedded matrix. This is reused benchmark evidence, not fresh patient-level validation or a clinical tool.

## Run

Open `index.html` directly, or serve this directory with any static server. There is no build or dependency installation. Content Security Policy blocks connections via `connect-src 'none'`. All assets are local.
