/* Optional CPU inference. No model/runtime is requested until the Enable button is pressed. */
'use strict';
(() => {
  const predictions = new Map();
  let worker = null, state = 'idle', manifest = null, pending = null, generation = 0;
  let processed = 0, total = 0, sequence = 0;
  const get = id => document.getElementById(id);
  const isBusy = () => state === 'loading' || state === 'running';
  function status(message) { get('analysis-status').textContent = message; }
  function updateControls() {
    get('enable-analysis').hidden = state === 'ready' || state === 'running';
    get('enable-analysis').disabled = state === 'loading';
    get('run-analysis').disabled = state !== 'ready';
    get('stop-analysis').hidden = !isBusy();
    get('release-analysis').hidden = !worker || isBusy();
    get('forget-model').disabled = isBusy();
    get('export-predictions').disabled = !predictions.size;
    get('analysis-scope').disabled = isBusy();
    if (get('analyze-image')) get('analyze-image').disabled = state !== 'ready';
  }
  function repaint() {
    renderGallery();
    if (current) renderPrediction(current);
    updateControls();
  }
  function stop(message = 'Stopped. Completed predictions are kept. Enable local analysis to resume.') {
    generation++;
    if (worker) worker.terminate();
    worker = null;
    state = 'idle';
    if (pending) { pending.reject(new Error('Cancelled')); pending = null; }
    get('analysis-progress').hidden = true;
    status(message);
    updateControls();
  }
  function fail(message) {
    stop('Local analysis could not finish: ' + message + ' You can retry.');
  }
  function enable() {
    if (isBusy() || state === 'ready') return;
    if (!window.Worker || !window.WebAssembly || !window.crypto?.subtle) {
      status('This browser cannot run local analysis. Open the HTTPS site in a recent browser.'); return;
    }
    const version = ++generation;
    state = 'loading';
    get('analysis-progress').hidden = false;
    get('analysis-progress').value = 0;
    status('Preparing the model download. Images stay on your device.');
    updateControls();
    try {
      worker = new Worker('local-inference-worker.js');
      worker.onerror = event => { event.preventDefault(); if (version === generation) fail('The CPU worker could not start or ran out of memory.'); };
      worker.onmessage = ({data}) => {
        if (version !== generation) return;
        if (data.type === 'progress') { status(data.message); get('analysis-progress').value = data.fraction; }
        else if (data.type === 'ready') {
          manifest = data.manifest; state = 'ready';
          status('Ready. Inference uses one CPU worker on this device. Choose images, then run analysis.');
          get('analysis-progress').hidden = true; updateControls();
        } else if (data.type === 'prediction' && pending?.requestId === data.requestId) {
          const resolve = pending.resolve; pending = null; resolve(data);
        } else if (data.type === 'error') fail(data.message);
        else if (data.type === 'cache-warning') get('analysis-cache-note').textContent = data.message;
      };
      worker.postMessage({type: 'load'});
    } catch (error) { fail(error.message); }
  }
  async function pixelsFor(image) {
    const decoded = await loadImage(image);
    if (decoded.naturalWidth !== decoded.naturalHeight || decoded.naturalWidth < 224) {
      throw Error('Use square tissue tiles at least 224 × 224 pixels. Whole slides and rectangular photos are not supported.');
    }
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 224;
    const context = canvas.getContext('2d', {willReadFrequently: true});
    context.fillStyle = '#fff'; context.fillRect(0, 0, 224, 224);
    context.imageSmoothingEnabled = true; context.imageSmoothingQuality = 'high';
    context.drawImage(decoded, 0, 0, 224, 224);
    const rgba = context.getImageData(0, 0, 224, 224).data;
    const pixels = new Float32Array(3 * 224 * 224);
    for (let i = 0; i < 224 * 224; i++) for (let channel = 0; channel < 3; channel++) pixels[channel * 224 * 224 + i] = rgba[i * 4 + channel] / 255;
    return pixels;
  }
  async function run(explicitImages) {
    if (state !== 'ready') return;
    const list = explicitImages || (get('analysis-scope').value === 'selected' ? [...selected].map(item).filter(Boolean) : visible);
    const queue = list.filter(image => !predictions.has(image.id)).slice(0, 20);
    if (!queue.length) { status('No new images to analyze. Change the filters or selection; completed predictions are kept.'); return; }
    if (queue.some(image => image.collection === 'local') && !get('confirm-domain').checked) {
      status('For local images, confirm they are colorectal H&E tissue tiles before running this model.'); return;
    }
    const version = generation;
    state = 'running'; processed = 0; total = queue.length;
    get('analysis-progress').hidden = false; get('analysis-progress').value = 0;
    updateControls();
    try {
      for (const image of queue) {
        status(`Analyzing ${processed + 1} of ${total}: ${image.filename}`);
        const pixels = await pixelsFor(image);
        if (version !== generation) return;
        const requestId = ++sequence;
        const result = await new Promise((resolve, reject) => {
          pending = {requestId, resolve, reject};
          worker.postMessage({type: 'predict', requestId, pixels: pixels.buffer}, [pixels.buffer]);
        });
        if (version !== generation) return;
        const top = result.scores.indexOf(Math.max(...result.scores));
        predictions.set(image.id, {
          image_id: image.id, filename: image.filename,
          model_id: manifest.id, model_sha256: manifest.sha256,
          predicted_label: manifest.labels[top],
          scores: Object.fromEntries(manifest.labels.map((code,i) => [code,result.scores[i]])),
          score_interpretation: 'Uncalibrated softmax scores; not clinical confidence',
          input_size: [224,224], execution: 'Browser CPU / WASM, single worker',
          elapsed_ms: result.elapsedMs, created_at: new Date().toISOString()
        });
        processed++; get('analysis-progress').value = processed / total;
        repaint();
      }
      state = 'ready'; get('analysis-progress').hidden = true;
      status(`Finished ${processed} image${processed === 1 ? '' : 's'} on this device. Run again for the next batch (up to 20).`);
      updateControls();
    } catch (error) { if (version === generation) fail(error.message); }
  }
  function renderPrediction(id) {
    const root = get('local-prediction'); if (!root) return;
    root.replaceChildren();
    root.append(el('h3', 'Local model prediction'));
    const prediction = predictions.get(id);
    if (prediction) {
      const sorted = Object.entries(prediction.scores).sort((a,b) => b[1]-a[1]);
      for (const [code,score] of sorted.slice(0,3)) {
        const row = el('p', undefined, 'prediction-score');
        row.append(el('span',names[code]),el('strong',(score*100).toFixed(1)+'%')); root.append(row);
      }
      if (item(id)?.label && item(id).label !== prediction.predicted_label) root.append(el('p', 'Prediction differs from the published reference label.', 'scope-note'));
      root.append(el('p', 'TIA MobileNetV3-Small · uncalibrated model scores. Separate from the reference label and your review.', 'scope-note'));
    } else {
      root.append(el('p', 'No prediction yet. Enable local CPU analysis in the collection to run this model.', 'scope-note'));
      const b = button('Analyze this image', () => run([item(id)]), 'secondary');
      b.id = 'analyze-image'; b.disabled = state !== 'ready'; root.append(b);
    }
  }
  function decorateGallery() {
    document.querySelectorAll('#gallery .tile').forEach((tile,index) => {
      tile.querySelectorAll('.model-label').forEach(node => node.remove());
      const prediction = predictions.get(visible[index]?.id);
      if (prediction) tile.querySelector('.tile-copy').append(el('small', 'Model: '+names[prediction.predicted_label], 'model-label'));
    });
  }
  function snapshot(ids) { return Object.fromEntries(ids.filter(id => predictions.has(id)).map(id => [id, predictions.get(id)])); }
  window.LocalAnalysis = {enable, run, stop, renderPrediction, decorateGallery, snapshot, pixelsFor,
    get state() {return state;}, get predictions() {return predictions;}};
  get('enable-analysis').addEventListener('click',enable);
  get('run-analysis').addEventListener('click',()=>run());
  get('stop-analysis').addEventListener('click',()=>stop());
  get('release-analysis').addEventListener('click',()=>stop('CPU model unloaded from memory. Cached files and completed predictions are kept.'));
  get('forget-model').addEventListener('click',async()=>{
    stop('Clearing the saved model…');
    try { await caches.delete('tissology-local-model-v1'); status('Saved model removed. A new model download is required next time. The browser may retain runtime files in its normal HTTP cache.'); }
    catch { status('Could not access model storage. You can clear this site’s storage in your browser settings.'); }
  });
  get('export-predictions').addEventListener('click',()=>exportJSON({model:manifest,predictions:[...predictions.values()]},'tissology-local-predictions.json'));
  window.addEventListener('beforeunload',()=>worker?.terminate());
  updateControls();
})();
