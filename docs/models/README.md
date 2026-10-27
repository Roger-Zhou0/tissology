# Optional local CPU model

**Model:** TIA Toolbox MobileNetV3-Small, trained on Kather100K. This is not CONCH and does not inherit the CONCH benchmark results.

**Source:** [TIACentre/TIAToolbox_pretrained_weights](https://huggingface.co/TIACentre/TIAToolbox_pretrained_weights/tree/ea909bdcce3aa409bf436c09b4495cca90a8e99b), `mobilenet_v3_small-kather100k.pth`, pinned revision `ea909bdcce3aa409bf436c09b4495cca90a8e99b`. Source and exported checksums: [manifest.json](manifest.json).

**License:** CC BY 4.0. The official weight repository states that its pretrained weights use the respective training dataset's license; Kather100K is CC BY 4.0. Credit the TIA Centre authors and Kather, Halama & Marx. [License](https://creativecommons.org/licenses/by/4.0/) · [Official weight terms](https://huggingface.co/TIACentre/TIAToolbox_pretrained_weights/blob/ea909bdcce3aa409bf436c09b4495cca90a8e99b/README.md).

Pocock, J., Graham, S., Vu, Q. D., et al. (2022). *TIAToolbox as an end-to-end library for advanced tissue image analytics*. Communications Medicine 2, 120. https://doi.org/10.1038/s43856-022-00186-5

Kather, J. N., Halama, N., Marx, A. (2018). *100,000 histological images of human colorectal cancer and healthy tissue*. https://doi.org/10.5281/zenodo.1214456

## Adaptation

The original floating-point model is exported to ONNX opset 17. No retraining, quantization, or calibration was performed. Architecture: torchvision MobileNetV3-Small feature extractor, adaptive average pooling, and a 576-to-9 linear classifier with softmax, matching TIA Toolbox's CNNModel. Original source code: [vanilla.py](https://github.com/TissueImageAnalytics/tiatoolbox/blob/c0bbb1472b494f8093820dc39d6080abac4e9dc9/tiatoolbox/models/architecture/vanilla.py).

Inputs are RGB float32 tensors of shape `[1, 3, 224, 224]`, scaled by 1/255 without mean/std normalization. The public app accepts square tiles at least 224 × 224 pixels; larger tiles are resized using the browser canvas. The original training resolution is 0.5 microns/pixel; the app does not verify physical magnification or perform stain normalization. Non-colorectal images, other stains, rectangular photographs, and whole slides are outside the intended scope.

Class order: BACK, NORM, DEB, TUM, ADI, MUC, MUS, STR, LYM. Verified against the [official patch-prediction notebook](https://github.com/TissueImageAnalytics/tiatoolbox/blob/c0bbb1472b494f8093820dc39d6080abac4e9dc9/examples/05-patch-prediction.ipynb). Preprocessing: [classification.py](https://github.com/TissueImageAnalytics/tiatoolbox/blob/c0bbb1472b494f8093820dc39d6080abac4e9dc9/tiatoolbox/models/dataset/classification.py).

Scores are uncalibrated softmax outputs, not probabilities of diagnostic correctness. No clinical use.

## Validation and limitations

Original PyTorch versus ONNX Runtime CPU: all 36 bundled tiles have matching top predictions; maximum absolute score difference is 0.00000346. Reference vectors are in `reference-outputs.json` for testing only; the app does not load or use them for inference.

On the 36 bundled examples, the model agrees with **23 reference labels**. These examples were selected by filename, and this result is not an independent or representative accuracy estimate. A larger MobileNetV2 candidate also agreed on 23/36, so the smaller model was retained. This comparison must not be presented as a clinical evaluation or the original 96.17% CONCH result.

## Runtime and privacy

ONNX Runtime Web **1.20.1**, self-hosted in `../vendor/onnxruntime/`, runs with the WASM CPU execution provider and one thread inside a dedicated worker. The runtime is MIT-licensed; its license and third-party notices are included beside the runtime files. No cross-origin isolation headers or GPU are required.

Nothing downloads until the visitor presses Enable. First use transfers about 15.1 MB: 3,741,332 model bytes plus about 11.3 MB runtime code/WASM. The model is SHA-256 verified on download and cache reads. Cache Storage failure falls back to an in-memory model. Stop terminates the worker immediately, cancelling model initialization or current inference; completed results remain available. Unload releases the worker/model memory. Remove saved model deletes the explicit model cache, but ordinary browser HTTP caching may retain runtime files.

Images and tensors are never uploaded. CSP allows same-origin static asset downloads and WebAssembly execution; there are no external inference services. Main-thread image preprocessing uses canvas; neural computation uses the worker. Predictions stay in tab memory, with explicit export and optional snapshots in saved findings.

## Reproduce conversion

In an isolated Python 3.12 environment install `torch==2.6.0 torchvision==0.21.0 onnx==1.17.0 onnxruntime==1.20.1 pillow requests numpy`, then run `python scripts/export_browser_model.py` from the repository root. It verifies the pinned source checksum, loads weights with `weights_only=True`, exports ONNX, and verifies all bundled reference outputs. No GPU or training is used.

### Browser verification (2026-10-27)

Headless Chrome on the development machine produced the same top predictions as Python ONNX Runtime for all 36 tiles, with maximum absolute score difference 0.00000215. Neural execution took 5.1–12.7 ms/tile after initialization in that run (mean 5.67 ms); these timings exclude download and preprocessing and do not predict other devices' performance.

Integration checks covered opt-in loading, 20-image batch limits, resuming without duplicates, cancelling initialization/inference, imported-image classification, domain acknowledgement, unsupported shape rejection, cache-corruption recovery, explicit cache deletion, exported model provenance, finding snapshots, and mobile page overflow. Re-run `scripts/test_browser_cpu.py` using the isolated-browser setup described in its module docstring. This is numerical and functional verification, not independent clinical validation.
