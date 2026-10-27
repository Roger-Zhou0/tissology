# Publishing Tissology's public preview

This package is ready for an isolated static deployment. It has **not** been pushed or activated in GitHub Pages.

1. In your **Roger-Zhou0/tissology** repository, commit the `docs/` folder from this package to the default branch. Keep the original application, data, access controls, and deployment configuration unchanged.
2. In repository **Settings → Pages**, choose **Deploy from a branch → main → /docs → Save**.
3. After GitHub confirms successful publication, visit `https://roger-zhou0.github.io/tissology/` in an incognito browser and verify that no sign-in is requested.
4. Check the Network tab: no calls should go to the original Tissology server or to external services. The static files originate from GitHub Pages only.
5. Link **that URL** from the Tissology project entry on your resume—not the existing private workspace.

If you prefer not to use Pages, any static-file host works. Avoid enabling `TISSUE_PUBLIC_READONLY=1` on the existing live server solely for this demo: the underlying GET endpoints are not guaranteed to be computationally free or safe to expose to anonymous users.

Important: The preview uses no real tissue images. It is an interactive **research showcase**, not a fully functional public deployment of the model. You would need a public, appropriately licensed, precomputed image sample to demonstrate actual image inspection without live inference.
