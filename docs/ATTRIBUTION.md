# Public sample image attribution

Kather, Jakob Nikolas; Halama, Niels; Marx, Alexander (2018). *100,000 histological images of human colorectal cancer and healthy tissue*, version v0.1. Zenodo. https://doi.org/10.5281/zenodo.1214456

The 36 images in `samples/` are from `CRC-VAL-HE-7K.zip`, distributed under Creative Commons Attribution 4.0 International: https://creativecommons.org/licenses/by/4.0/ . The license was verified in the original Zenodo record's API metadata.

Selection: the first four TIFF filenames alphabetically within each of the nine reference-class folders. This small, balanced selection is for workflow demonstration, not a representative evaluation sample. All tiles retain their original 224 × 224 pixel resolution. TIFF files were converted losslessly to PNG for browser compatibility. Original archive paths and TIFF SHA-256 checksums are in `samples.json`.

Reference labels come from dataset folders. No per-image CONCH predictions or confidence scores are provided or inferred. The separate benchmark page reproduces the repository's archived aggregate evaluation; it does not evaluate this selected sample.

Similarity ranking uses RGB histograms and horizontal intensity differences calculated in the visitor's browser. It is a lightweight appearance comparison, not a neural embedding, semantic search, or tissue classification.

## Optional browser model

TIA Centre MobileNetV3-Small / Kather100K weights are adapted to ONNX under CC BY 4.0. ONNX Runtime Web 1.20.1 is included under MIT with third-party notices. Model source, modifications, checksums, citations, and numerical checks are in [models/README.md](models/README.md). Predictions from this optional model are separate from the dataset labels and CONCH evaluation.
