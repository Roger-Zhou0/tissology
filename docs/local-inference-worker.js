/* Inference is isolated in this worker so cancellation can stop CPU work immediately. */
'use strict';
let session;
let manifest;
const cacheName = 'tissology-local-model-v1';
const modelURL = new URL('models/mobilenet-v3-small-kather.onnx', self.location.href).href;
const send = (type, detail = {}) => self.postMessage({type, ...detail});
async function digest(buffer) {
  const hash = await crypto.subtle.digest('SHA-256', buffer);
  return [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2, '0')).join('');
}
async function modelBytes() {
  let cache;
  try {
    cache = await caches.open(cacheName);
    const hit = await cache.match(modelURL);
    if (hit) {
      const bytes = await hit.arrayBuffer();
      if (bytes.byteLength === manifest.bytes && await digest(bytes) === manifest.sha256) {
        send('progress', {message: 'Verified cached model. Loading the CPU runtime…', fraction: 1});
        return bytes;
      }
      await cache.delete(modelURL);
    }
  } catch { /* Private browsing may disable Cache Storage. In-memory loading still works. */ }
  const response = await fetch(modelURL, {cache: 'no-store'});
  if (!response.ok) throw Error('The model download failed (' + response.status + ').');
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const {value, done} = await reader.read();
    if (done) break;
    received += value.length;
    if (received > manifest.bytes) throw Error('The model file has an unexpected size.');
    chunks.push(value);
    send('progress', {message: `Downloading model: ${(received / 1e6).toFixed(1)} / ${(manifest.bytes / 1e6).toFixed(1)} MB`, fraction: received / manifest.bytes});
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  if (received !== manifest.bytes || await digest(bytes) !== manifest.sha256) {
    throw Error('Model verification failed. Please retry the download.');
  }
  if (cache) {
    try { await cache.put(modelURL, new Response(bytes, {headers: {'Content-Type': 'application/octet-stream'}})); }
    catch { send('cache-warning', {message: 'Model cache is unavailable; it may download again next time.'}); }
  }
  return bytes.buffer;
}
self.onmessage = async ({data}) => {
  try {
    if (data.type === 'load') {
      const response = await fetch(new URL('models/manifest.json', self.location.href));
      if (!response.ok) throw Error('Model information could not be loaded.');
      manifest = await response.json();
      const bytes = await modelBytes();
      send('progress', {message: 'Loading the CPU runtime (about 11.3 MB on first use)…', fraction: 1});
      importScripts('vendor/onnxruntime/ort.wasm.min.js');
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.wasm.wasmPaths = new URL('vendor/onnxruntime/', self.location.href).href;
      session = await ort.InferenceSession.create(bytes, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
      send('ready', {manifest});
    } else if (data.type === 'predict') {
      if (!session) throw Error('Enable local analysis before running predictions.');
      if (!(data.pixels instanceof ArrayBuffer) || data.pixels.byteLength !== 3 * 224 * 224 * 4) throw Error('Invalid image tensor.');
      const started = performance.now();
      const tensor = new ort.Tensor('float32', new Float32Array(data.pixels), [1, 3, 224, 224]);
      const result = await session.run({pixels: tensor});
      const scores = Array.from(result.scores.data);
      result.scores.dispose();
      tensor.dispose();
      if (scores.length !== 9 || !scores.every(x => Number.isFinite(x) && x >= 0 && x <= 1) || Math.abs(scores.reduce((a,b) => a+b,0)-1) > .001) throw Error('The model returned invalid scores.');
      send('prediction', {requestId: data.requestId, scores, elapsedMs: performance.now() - started});
    }
  } catch (error) {
    send('error', {requestId: data.requestId, message: error.message || String(error)});
  }
};
