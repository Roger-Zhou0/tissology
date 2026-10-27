# ONNX Runtime Web 1.20.1

Files copied unchanged from the official npm release:
https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.20.1.tgz

Included: `dist/ort.wasm.min.js`, `dist/ort-wasm-simd-threaded.mjs`, and `dist/ort-wasm-simd-threaded.wasm`.

MIT license and third-party notices are copied from the corresponding official `v1.20.1` Git tag:
https://github.com/microsoft/onnxruntime/tree/v1.20.1

The application requests these files only after the user enables local CPU analysis. Execution is in a dedicated worker with `numThreads = 1`, `proxy = false`, and the `wasm` execution provider. No WebGPU execution provider is enabled.
