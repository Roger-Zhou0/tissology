# Tissology public research preview

A **static, login-free, resource-light** counterpart to the authenticated Tissology research workspace. It is designed for GitHub Pages, with no backend, no paid inference, no login, and no collection data.

## What this demo includes

- A summary of the already-published `conch-retention-001` classifier evaluation.
- Interactive per-class recall and confusion-matrix exploration.
- An illustrative review workflow using **synthetic abstract patterns** (not real tissue images), with session-only annotations and local JSON export.

## What it deliberately excludes

No model weights, live predictions, image uploads, real tissue tiles, remote APIs, private lab data, credentials, analytics, or persistent storage. The full private workspace remains unchanged and should keep its authentication. Notes are kept in JavaScript memory within the open tab only.

## Deploy without your compute resources

Copy the contents of this `docs/` directory into the `docs/` directory of your public [tissology](https://github.com/Roger-Zhou0/tissology) repository. On GitHub, open **Settings → Pages → Build and deployment → Deploy from a branch**, and choose **main /docs**. The empty `.nojekyll` file disables Jekyll preprocessing. The expected project address after deployment is `https://roger-zhou0.github.io/tissology/`, but **do not treat it as live until Pages reports success**.

GitHub serves the static files. The demo sends **no requests to your private server and runs zero inference**; ordinary browser-side rendering is used for charts and synthetic illustrations. Pages bandwidth and anti-abuse limits still apply. Since this is a research portfolio, not a paid SaaS, it fits the intended static-site use case.

## Data integrity

The class supports, correct counts, and full confusion matrix in `app.js` were transcribed from `studio/static/collection-validation-conch-retention-001.json` in the original repo. They represent reused benchmark evaluation, not fresh independent validation. The demo also labels limitations clearly.

To check: the diagonal sums to **6,905**, the matrix totals **7,180**, the reported label-agreement rate is **96.17%**, and macro-F1 is **94.62%**. No per-image records or patient information are included.

## Quick local preview

Open `docs/index.html` directly in a browser; no development server is necessary. Select a class, inspect confusions, annotate a synthetic illustration, and export session notes.
