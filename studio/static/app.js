'use strict';
const $ = (selector) => document.querySelector(selector);
// Adapted studies keep their own audits; teaching workspaces live at the parent app.
const studyMount = location.pathname.indexOf('/feedback-studies/');
if (studyMount >= 0) {
  const parentPath = location.pathname.slice(0, studyMount + 1);
  for (const route of ['feedback', 'semantic']) {
    for (const link of document.querySelectorAll(`a[href="${route}"]`)) link.href = parentPath + route;
  }
}

const state = {summary: null, quality: null, testEvaluations: null, tiles: [], audit: null, selected: null, shown: 48, neighbors: null, example: null};
const endpoints = ['TUM', 'STR', 'LYM'];
const classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'];
const names = {
  ADI: 'Adipose', BACK: 'Background', DEB: 'Debris / necrosis',
  LYM: 'Lymphocyte tissue', MUC: 'Mucus', MUS: 'Smooth muscle',
  NORM: 'Normal mucosa', STR: 'Stroma', TUM: 'Tumor epithelium'
};
const colors = {ADI: '#b4a17c', BACK: '#9ba3ae', DEB: '#ac7e51', LYM: '#7374b3', MUC: '#bc8ea6', MUS: '#978460', NORM: '#55849c', STR: '#568c80', TUM: '#b35f87'};
const methods = {manual: 'Manual random sampling', uniform: 'Uniform model-assisted sampling', stratified: 'Random sampling within model classes', shared: 'Shared pilot allocation (experimental)'};
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const pct = (value, digits = 1) => `${(100 * value).toFixed(digits)}%`;
const imageUrl = id => `api/images/${encodeURIComponent(id)}`;
const sourceName = source => source === 'human_entered' ? 'Human-entered label' : 'Dataset label (simulated review)';
// Older benchmark records retain the encoder name from their first evaluation.
// Keep those frozen records intact while using model-neutral scope language here.
const displayScope = value => String(value || '').replace(/\bPLIP pretraining overlap\b/g, 'Encoder pretraining overlap');
function encoderName() {
  const provenance = state.summary?.provenance || {};
  const name = provenance.encoder_name || provenance.model?.split('/').pop() || 'Image encoder';
  return /^(plip|conch)$/i.test(name) ? name.toUpperCase() : name;
}
function predictorName() {
  return state.summary?.provenance.classifier?.name || `${encoderName()} with text prompts`;
}
function weakestClassText(metrics) {
  const rows = [...(metrics.per_class || [])].filter(row => row.support > 0 && row.recall != null)
    .sort((a, b) => a.recall - b.recall || b.support - a.support);
  if (!rows.length) return '';
  const row = rows[0];
  if (row.recall === 1) return 'Every reference class was classified correctly in this collection.';
  return `Lowest class recall: ${names[row.label] || row.label} — ${pct(row.recall)} (${row.correct} of ${row.support} reference images correct).`;
}
function predictionScores(row) {
  if (row.score_kind === 'rbf_svm_votes') {
    return {values: row.class_vote_counts, kind: 'votes', title: 'Pairwise class votes',
      explanation: 'Pairwise vote counts, not probabilities. Each vote is a win against another tissue class.'};
  }
  if (row.score_kind === 'linear_svm_decision') {
    return {values: row.decision_scores, kind: 'decision', title: 'Model decision scores',
      explanation: 'Raw decision scores, not probabilities. Higher scores rank the model’s preferred tissue classes.'};
  }
  if (row.score_kind === 'linear_classifier_probability' || (!row.score_kind && row.classification_scores)) {
    return {values: row.classification_scores, kind: 'classifier', title: 'Model scores',
      explanation: 'Classifier scores; not validated as calibrated probabilities.'};
  }
  // Text-image similarities cannot stand in for missing classifier scores.
  if (!state.summary?.provenance.classifier && row.cosine_scores) {
    return {values: row.cosine_scores, kind: 'cosine', title: 'Model scores',
      explanation: 'Cosine similarities, not probabilities.'};
  }
  return {values: null, kind: 'unavailable', title: 'Model scores',
    explanation: 'Scores for this classifier are unavailable.'};
}
function rankedScores(row) {
  return Object.entries(predictionScores(row).values || {}).sort((a, b) => b[1] - a[1]
    || (a[0] === row.predicted_label ? -1 : b[0] === row.predicted_label ? 1 : a[0].localeCompare(b[0])));
}
function contrastPanel(row, panelId) {
  const labels = rankedScores(row).slice(0, 2).map(([label]) => label);
  return `<details class="settings contrast-examples" id="${panelId}" data-example-labels="${labels.join(',')}">
    <summary>Compare labeled examples</summary>
    <p class="small muted">Dataset-labeled development examples from the model’s two highest-scoring tissue classes. These are separate from study and pilot images; viewing them does not change your review.</p>
    <div class="contrast-content"><p class="small muted">Open to load examples.</p></div>
  </details>`;
}
function bindContrastPanel(panelId) {
  const panel = $(`#${panelId}`);
  panel.ontoggle = async () => {
    if (!panel.open || panel.dataset.loaded) return;
    panel.dataset.loaded = 'loading';
    const labels = panel.dataset.exampleLabels.split(',').filter(Boolean);
    const target = panel.querySelector('.contrast-content');
    if (labels.length < 2) {
      target.innerHTML = '<p class="small muted">Competing-class examples are unavailable for this prediction.</p>';
      return;
    }
    try {
      const groups = await Promise.all(labels.map(label => api(`/api/reference-examples/${label}`)));
      if (groups.some(group => !group.available)) {
        target.innerHTML = '<p class="small muted">Development examples are not available for this study yet.</p>';
        return;
      }
      target.innerHTML = `<div class="contrast-grid">${groups.map((group, i) => `<section><h3>${esc(names[labels[i]] || labels[i])}</h3><div class="contrast-images">${group.examples.map(example => `<figure><img loading="lazy" src="api/reference-example-images/${encodeURIComponent(example.id)}" alt="Dataset-labeled development example: ${esc(names[example.reference_label])}"><figcaption>Development example ${example.rank}</figcaption></figure>`).join('')}</div></section>`).join('')}</div>`;
      panel.dataset.loaded = 'true';
    } catch (_) {
      delete panel.dataset.loaded;
      target.innerHTML = '<p class="small muted">Development examples could not be loaded. Close and reopen to try again.</p>';
    }
  };
}

function notice(message, error = false) {
  const target = $('#notice');
  target.textContent = message;
  target.setAttribute('role', error ? 'alert' : 'status');
  clearTimeout(notice.timer);
  if (error) target.scrollIntoView({block: 'nearest'});
  else notice.timer = setTimeout(() => target.textContent = '', 8000);
}
async function api(path, body) {
  // Resolve under the page URL so Delta Open OnDemand's /rnode/ proxy works.
  const response = await fetch(path.replace(/^\//, ''), body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}
function rememberAudit(id) {
  try {
    if (id) sessionStorage.setItem((window.TissueLab?.storageKey('tissue-study-review') || 'tissue-study-review'), id);
    else sessionStorage.removeItem((window.TissueLab?.storageKey('tissue-study-review') || 'tissue-study-review'));
  } catch (_) { /* The workflow also works with browser storage disabled. */ }
}
function screen(name, focus = true) {
  if (!['study', 'atlas', 'audit', 'results'].includes(name)) name = 'study';
  document.querySelectorAll('.screen').forEach(el => el.hidden = el.id !== `screen-${name}`);
  document.querySelectorAll('[data-screen]').forEach(el => {
    const active = el.dataset.screen === name;
    el.classList.toggle('active', active);
    if (active) el.setAttribute('aria-current', el.closest('.steps') ? 'step' : 'page');
    else el.removeAttribute('aria-current');
  });
  if (location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);
  if (focus) $(`#screen-${name} h1`).focus({preventScroll: true});
  window.scrollTo({top: 0, behavior: 'instant'});
}
function renderStudy() {
  const {study, population, pilot, provenance} = state.summary;
  $('#spatial-link').hidden = !state.summary.spatial_available;
  const modelName = predictorName();
  $('#collection-label').textContent = `CRC study · ${population} images`;
  $('#study-intro').textContent = `${modelName} has labeled ${population} colorectal tissue images. Check a sample of its labels to correct the estimates for the whole collection and see how much uncertainty remains.`;
  $('#initial-estimates tbody').innerHTML = endpoints.map(k => `
    <tr><td>${names[k]}${k === 'STR' ? '<span class="muted small"> (supporting tissue)</span>' : ''}</td>
      <td class="numeric">${pct(state.summary.initial[k])}</td>
      <td class="count">${Math.round(population * state.summary.initial[k])} of ${population}</td></tr>`).join('');
  $('#definitions').innerHTML = endpoints.map(k => `
    <div class="definition"><strong>${names[k]} image percentage</strong>
      Number of images with reference class ${esc(k)}, divided by all ${population} study images, multiplied by 100.</div>`).join('');
  $('#population-description').textContent = `${study.target_population}. Every retained image has equal weight; background and debris remain in the denominator. The results describe this image collection. Patient IDs, physical adjacency, and clinical groups are unavailable.`;
  $('#study-details').innerHTML = `
    <dt>Dataset</dt><dd>NCT-CRC-HE-100K · shared Delta data</dd>
    <dt>Image size</dt><dd>224 × 224 pixels, RGB</dd>
    <dt>Physical scale</dt><dd>0.5 µm / pixel per dataset documentation; no physical measurements are computed here</dd>
    <dt>Exclusions</dt><dd>Unreadable or duplicate images before the collection was frozen</dd>
    <dt>Predictor</dt><dd>${esc(modelName)} · fixed predictions</dd>
    <dt>Image encoder</dt><dd>${esc(provenance.model)}</dd>
    <dt>Classifier features</dt><dd>${esc(provenance.classifier?.representation || provenance.representation)}</dd>
    <dt>Image search features</dt><dd>${esc(provenance.representation)}</dd>
    <dt>Model revision</dt><dd class="code-wrap">${esc(provenance.model_revision)}</dd>
    <dt>Collection hash</dt><dd class="code-wrap">${esc(state.summary.manifest_sha256)}</dd>`;
  $('#budget').max = population;
  $('#budget-context').textContent = `of ${population} study images`;
  $('#budget-note').textContent = `${pilot} separate pilot labels are available to every method and charged equally in method comparisons. This review budget is additional. Model predictions, sample selection, and inclusion probabilities are fixed before review.`;
  if (study.development_labels) $('#budget-note').textContent += ` The predictor also used ${study.development_labels} separate development labels; these are outside the review budget.`;
  // An explicit example action replaces silently loading a completed saved run.
  state.example = [...state.summary.audits].reverse().find(a => a.reviewed === a.budget) || null;
  $('#open-example').hidden = !state.example;
  $('#example-note').hidden = !state.example;
  if (state.example) $('#example-note').textContent = `Or explore the completed example with ${state.example.budget} checked images.`;
  renderAuditList();
}
function renderQuality() {
  const quality = state.quality;
  $('#model-quality').hidden = !quality?.available;
  if (!quality?.available) return;
  const active = quality.active_model;
  const developmentLabels = state.summary.study.development_labels || state.summary.provenance.classifier?.refit_labels;
  $('#model-explanation').textContent = state.summary.provenance.classifier
    ? `The pretrained ${encoderName()} encoder describes image appearance. A tissue classifier trained on ${developmentLabels ? `${developmentLabels.toLocaleString()} separate labeled images` : 'separate development images'} uses those features to assign one of nine tissue classes. The pretrained encoder stays fixed.`
    : `The pretrained ${encoderName()} model compares each image with text descriptions of the nine tissue classes.`;
  $('#accuracy-summary').innerHTML = `<div class="accuracy-metric current">
    <div class="label-source">Current model · existing ${active.total}-image study</div>
    <strong class="accuracy-value">${pct(active.accuracy)}</strong>
    <div>${active.correct} of ${active.total} images correct</div>
    <div class="small muted">${esc(active.name)}</div>
  </div><p class="small" id="study-class-weakness">${esc(weakestClassText(active))}</p>`;
  $('#accuracy-scope').textContent = `This is the study collection already evaluated${state.summary.provenance.classifier ? ', held out from classifier development' : ''}. Accuracy is measured against dataset labels. Reviewing a sample separately corrects the tissue percentages; it does not retrain the model.`;
  const classRows = active.per_class || [];
  const perClassValue = row => row?.support ? pct(row.recall) : 'No reference images';
  const labels = active.labels || classes;
  const matrix = active.confusion_matrix;
  const mistakes = matrix ? labels.flatMap((reference, i) => labels.map((prediction, j) => ({
    reference, prediction, count: i === j ? 0 : matrix[i][j]
  }))).filter(row => row.count > 0).sort((a, b) => b.count - a.count) : [];
  $('#accuracy-breakdown').innerHTML = `
    ${developmentLabels ? `<p class="small">The classifier used ${developmentLabels} separate development images. The ${esc(encoderName())} image encoder stays fixed. Test images and final review labels were not used to fit or select the classifier.</p>` : ''}
    <p class="scope-note">${esc(quality.scope)}</p>
    <h3>Accuracy within each reference class</h3>
    <p class="small muted">Of images with each dataset label, how many did the model classify correctly? This is recall for that class. Classes with few images have less evidence.</p>
    <div class="table-scroll"><table class="class-accuracy-table"><thead><tr><th scope="col">Reference tissue class</th><th scope="col">Class recall</th><th scope="col">Correct / reference images</th></tr></thead><tbody>
      ${classRows.map(row => `<tr><th scope="row">${esc(names[row.label] || row.label)}</th><td>${perClassValue(row)}</td><td>${row.correct} / ${row.support}</td></tr>`).join('')}
    </tbody></table></div>
    ${mistakes.length ? `<p class="small"><strong>Most common current mistake:</strong> ${mistakes[0].count} images with reference label “${esc(names[mistakes[0].reference] || mistakes[0].reference)}” were predicted as “${esc(names[mistakes[0].prediction] || mistakes[0].prediction)}”.</p>` : matrix ? '<p class="small">Every prediction matched its reference label in this finite test collection.</p>' : ''}
    ${matrix ? `<details class="settings" id="confusion-details"><summary>Show all prediction mistakes (confusion matrix)</summary>
      <p class="small muted">Each row is a dataset reference class; each column is the current model’s prediction. Counts on the diagonal are correct. Other cells show which classes are confused.</p>
      <div class="table-scroll"><table class="confusion-table"><caption>Current model · reference classes in rows, predicted classes in columns</caption><thead><tr><th scope="col">Reference ↓ / predicted →</th>${labels.map(label => `<th scope="col" title="${esc(names[label] || label)}">${esc(label)}</th>`).join('')}</tr></thead><tbody>
      ${labels.map((label, i) => `<tr><th scope="row">${esc(label)} · ${esc(names[label] || label)}</th>${labels.map((_, j) => `<td${i === j ? ' class="correct-cell"' : ''}>${matrix[i][j]}</td>`).join('')}</tr>`).join('')}
      </tbody></table></div></details>` : ''}`;
}
function renderTestEvaluations() {
  const data = state.testEvaluations;
  const evaluations = data?.evaluations || [];
  const available = data?.available && evaluations.length > 0;
  const hasInspected = evaluations.some(row => row.already_inspected === true);
  const allInspected = evaluations.length > 0 && evaluations.every(row => row.already_inspected === true);
  const allFresh = evaluations.length > 0 && evaluations.every(row => row.already_inspected === false);
  const freshness = row => row.already_inspected === true ? 'Previously inspected benchmark'
    : row.already_inspected === false ? 'Fresh test collection' : 'Test collection';
  const testName = row => {
    const round = String(row.id).match(/(?:^|[-_])round[-_]?(\d+)(?:$|[-_])/i);
    return row.name + (round ? ` · Round ${round[1]}` : '');
  };
  const taxonomy = row => {
    const supported = row.supported_reference_classes;
    const outputs = row.prediction_classes;
    return Array.isArray(supported) && Array.isArray(outputs) && supported.length < outputs.length
      ? {supported, outputs, unsupported: outputs.filter(label => !supported.includes(label))} : null;
  };
  const taxonomyNote = row => {
    const subset = taxonomy(row);
    return subset ? `<span class="test-set-scope test-taxonomy">${subset.supported.length} reference classes · all ${subset.outputs.length} model outputs retained</span>` : '';
  };
  $('#test-evaluation-title').textContent = allInspected ? 'Performance on known benchmarks'
    : hasInspected ? 'New test sets and known benchmarks'
    : allFresh ? 'Performance on new test sets' : 'Performance on test collections';
  $('#test-evaluation-status').textContent = available
    ? (hasInspected
      ? 'All rows use the same frozen classifier. Previously inspected benchmarks are performance checks on known images. Only rows marked “Fresh test collection” were newly selected for this evaluation.'
      : (displayScope(data.scope) || 'These collections use the same frozen classifier. Their labels were used only for evaluation.'))
    : (data?.error ? 'Test results could not be loaded. Reload the page to try again.' : 'No test results are available yet.');
  $('#test-evaluation-results').hidden = !available;
  if (!available) return;
  const weakestClassNote = row => {
    const metric = [...(row.per_class || [])].filter(value => value.support > 0)
      .sort((a, b) => a.recall - b.recall)[0];
    return metric ? `<span class="test-set-scope"><strong>Lowest class recall:</strong> ${esc(names[metric.label] || metric.label)} · ${pct(metric.recall)} (${metric.correct} / ${metric.support} correct).</span>` : '';
  };
  $('#test-evaluation-table tbody').innerHTML = evaluations.map(row => `<tr data-test-set="${esc(row.id)}">
    <th scope="row">${esc(testName(row))}<span class="test-set-scope test-set-freshness"><strong>${freshness(row)}</strong></span>
      ${taxonomyNote(row)}${weakestClassNote(row)}</th>
    <td data-label="Images">${row.total}</td>
    <td data-label="Accuracy"><strong>${pct(row.accuracy)}</strong><span class="test-set-count">${row.correct} correct</span></td>
    <td data-label="Balanced accuracy">${row.balanced_accuracy == null ? 'Unavailable' : pct(row.balanced_accuracy)}</td>
  </tr>`).join('');
  $('#test-evaluation-details').innerHTML = evaluations.map(row => {
    const subset = taxonomy(row);
    return `<details class="settings test-class-details" data-test-details="${esc(row.id)}">
    <summary>${esc(testName(row))}: dataset scope and class accuracy</summary>
    <p class="small test-set-freshness">${freshness(row)}${row.already_inspected === true ? ': these images were inspected before this model evaluation, so they are not an untouched test set.' : ''}</p>
    <p class="scope-note">${esc(displayScope(row.scope))}</p>
    ${subset ? `<p class="small taxonomy-note">This collection evaluates ${subset.supported.length} reference classes with the unchanged ${subset.outputs.length}-class predictor. Predictions of ${esc(subset.unsupported.map(label => names[label] || label).join(' or '))} count as errors. Balanced accuracy averages the supported reference classes.</p>
      ${['taxonomy_scope', 'scale_scope', 'cohort_scope'].filter(key => row[key]).map(key => `<p class="small" data-taxonomy-scope="${key}">${esc(row[key])}</p>`).join('')}
      ${row.supported_reference_macro_f1 != null ? `<p class="small supported-macro-f1">Macro F1 (${subset.supported.length} supported reference classes): ${pct(row.supported_reference_macro_f1)}. ${esc(row.supported_reference_macro_f1_definition || '')}</p>` : ''}` : ''}
    <div class="table-scroll"><table class="test-class-table"><thead><tr><th scope="col">Reference tissue class</th><th scope="col">Class recall</th><th scope="col">Correct / reference images</th></tr></thead><tbody>
      ${(row.per_class || []).map(metric => `<tr><th scope="row">${esc(names[metric.label] || metric.label)}</th><td>${subset && !subset.supported.includes(metric.label) ? 'Not evaluated in this reference taxonomy' : metric.support ? pct(metric.recall) : 'No reference images'}</td><td>${metric.correct} / ${metric.support}</td></tr>`).join('')}
    </tbody></table></div>
  </details>`;
  }).join('');
}
function renderAuditList() {
  $('#audit-select').innerHTML = '<option value="">Choose a review</option>' + state.summary.audits.map(a => `
    <option value="${a.id}">${a.id === state.example?.id ? 'Completed example · ' : ''}${a.reviewed} of ${a.budget} checked · ${esc(methods[a.method])} · ${a.id.slice(0,6)}</option>`).join('');
  if (state.audit) $('#audit-select').value = state.audit.plan.id;
  $('#start-review').textContent = state.audit && !state.audit.result ? 'Continue review' : 'Start a review';
}
function openNewReview() {
  state.audit = null;
  state.selected = null;
  rememberAudit(null);
  $('#plan-panel').hidden = false;
  $('#new-review').hidden = true;
  $('#audit-workspace').innerHTML = '';
  $('#saved-reviews').open = false;
  renderAuditList();
  renderResults();
}
async function openExample() {
  if (!state.example) return;
  await loadAudit(state.example.id);
  screen('results');
}
function tileCard(row, reference = null) {
  const label = reference || row.predicted_label;
  return `<button class="tile" data-image="${row.id}" aria-label="Inspect image ${row.id.slice(0,8)}, ${reference ? 'reviewed' : 'model label'} ${esc(names[label])}">
    <img src="${imageUrl(row.id)}" loading="lazy" alt="H&E tissue image ${row.id.slice(0,8)}">
    <span class="tile-meta">${esc(names[label])}<small>${reference ? `Model: ${esc(names[row.predicted_label])}` : 'Model label'}</small></span></button>`;
}
function renderAtlas() {
  $('#atlas-description').textContent = `Each point is an image. Nearby points have similar model features; they are not physically adjacent tissue. This is a two-dimensional PCA projection of ${encoderName()} image features.`;
  $('#tile-order option[value="margin"]').textContent = state.tiles.some(row => row.score_kind === 'rbf_svm_votes')
    ? 'Smallest vote-count gap' : state.tiles.some(row => row.score_kind === 'linear_svm_decision')
      ? 'Smallest raw decision-score gap' : 'Smallest model score gap';
  const plot = $('#atlas-plot');
  const rows = state.tiles.filter(r => r.coordinates);
  if (!rows.length) plot.innerHTML = '<text x="30" y="160">Image features are not yet available.</text>';
  else {
    const xs = rows.map(r => r.coordinates[0]), ys = rows.map(r => r.coordinates[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const x = v => 25 + 530 * (v - minX) / (maxX - minX || 1);
    const y = v => 295 - 270 * (v - minY) / (maxY - minY || 1);
    plot.innerHTML = rows.map(r => `<circle data-image="${r.id}" cx="${x(r.coordinates[0])}" cy="${y(r.coordinates[1])}" r="3.6" fill="${colors[r.predicted_label]}" opacity="0.7"><title>${esc(names[r.predicted_label])} · ${r.id.slice(0,8)}</title></circle>`).join('');
  }
  renderGallery();
}
function renderGallery() {
  let rows = state.neighbors || state.tiles;
  const filter = $('#class-filter').value;
  if (filter !== 'ALL') rows = rows.filter(r => r.predicted_label === filter);
  const byScoreGap = $('#tile-order').value === 'margin';
  if (byScoreGap) rows = [...rows].sort((a,b) => a.margin - b.margin);
  $('#tile-order option[value="id"]').textContent = state.neighbors ? 'Most similar first' : 'Collection order';
  $('#gallery').innerHTML = rows.slice(0,state.shown).map(r => tileCard(r)).join('');
  $('#tile-count').textContent = `${rows.length} images`;
  $('#load-more').hidden = state.shown >= rows.length;
  $('#reset-neighbors').hidden = state.neighbors === null;
  $('#gallery-caption').textContent = state.neighbors
    ? (byScoreGap
      ? `Similar-image shortlist from ${encoderName()} features, now ordered by smallest model score gap. Captions show model labels.`
      : `Study images ranked by cosine similarity of ${encoderName()} image features. Similarity is not a tissue-class probability. Captions show model labels.`)
    : 'Captions show model labels, which may be wrong. Select an image to inspect it.';
}
async function inspectTile(id) {
  const row = state.tiles.find(r => r.id === id);
  if (!row) return;
  const annotation = state.audit?.annotations.find(a => a.id === id);
  const scoreInfo = predictionScores(row);
  const scores = rankedScores(row).slice(0,4);
  const rawGap = ['decision', 'votes'].includes(scoreInfo.kind) && scores.length >= 2 ? scores[0][1] - scores[1][1] : null;
  const gapExplanation = scoreInfo.kind === 'votes'
    ? `Vote-count gap: ${rawGap}. ${row.vote_tie ? 'The highest vote count is tied. The displayed prediction uses the frozen model’s tie-breaking rule.' : 'This is the difference between the two highest vote counts.'}`
    : `Raw decision-score gap: ${rawGap?.toFixed(3)}. This is the difference between the two highest decision scores.`;
  $('#tile-inspector').innerHTML = `
    <h2 id="inspector-title">Image ${id.slice(0,8)}</h2>
    <div class="inspector-top"><img src="${imageUrl(id)}" alt="Selected H&E tissue image">
      <div><div class="label-source">Model label</div><div class="label-value">${names[row.predicted_label]}</div>
      ${annotation ? `<p class="label-source" style="margin-top:20px">${sourceName(annotation.source)}</p><div class="label-value">${names[annotation.label]}</div>` : '<p class="muted small" style="margin-top:20px">The reference label has not been revealed in this review.</p>'}
      <details class="settings" id="model-score-details"><summary>${scoreInfo.title}</summary><div class="score-list">${scores.map(([k,v]) => `<div><span>${names[k]}</span><span>${scoreInfo.kind === 'votes' ? v : v.toFixed(3)}</span></div>`).join('')}</div><p class="small muted">${scoreInfo.explanation}</p>${rawGap === null ? '' : `<p class="small" id="model-score-gap">${gapExplanation}</p>`}</details></div></div>
    <p class="small muted">Entire image · 224 × 224 pixels</p>
    ${contrastPanel(row, 'inspector-contrast')}
    <button class="secondary" id="find-neighbors">Find similar images</button>`;
  $('#tile-dialog').showModal();
  bindContrastPanel('inspector-contrast');
  $('#find-neighbors').onclick = async () => {
    try {
      state.neighbors = await api(`/api/neighbors/${id}`);
      state.shown = 48;
      $('#class-filter').value = 'ALL';
      $('#tile-order').value = 'id';
      $('#tile-dialog').close();
      screen('atlas');
      renderGallery();
    } catch (error) { notice(error.message, true); }
  };
}
async function loadAudit(id) {
  state.audit = await api(`/api/audits/${id}`);
  state.selected = null;
  rememberAudit(id);
  renderAuditList();
  renderAudit();
  renderResults();
}
function renderAudit() {
  if (!state.audit) return;
  $('#plan-panel').hidden = true;
  $('#new-review').hidden = false;
  const {plan, annotations, predictions, result} = state.audit;
  const annotated = new Map(annotations.map(a => [a.id,a]));
  if (!state.selected) state.selected = (plan.units.find(u => !annotated.has(u.id)) || plan.units[0]).id;
  const current = plan.units.find(u => u.id === state.selected);
  const pred = predictions[current.id], reviewed = annotated.get(current.id);
  const next = plan.units.find(u => !annotated.has(u.id));
  const remaining = plan.budget - annotations.length;
  const currentIndex = plan.units.findIndex(u => u.id === current.id) + 1;
  const labelMatched = reviewed && reviewed.label === pred.predicted_label;
  $('#audit-workspace').innerHTML = `
    <div class="audit-summary">
      <strong>${annotations.length} of ${plan.budget} images checked</strong>
      <div class="progress-track" role="progressbar" aria-label="Images checked" aria-valuemin="0" aria-valuemax="${plan.budget}" aria-valuenow="${annotations.length}"><div class="progress-fill" style="width:${100 * annotations.length / plan.budget}%"></div></div>
      ${result ? '<button class="primary" data-go="results">See corrected results</button>' : ''}
    </div>
    ${result ? '<p class="muted">Review complete. The corrected estimates are ready; you can also inspect the saved labels below.</p>' : '<p class="simulation-note"><strong>Simulated review:</strong> this dataset already has reference labels. Reveal them to check the model; no pathologist-time savings are being measured.</p>'}
    <section class="panel" aria-label="Current image review">
      <div class="review-position">Selected image ${currentIndex} of ${plan.budget} · ${current.id.slice(0,8)}</div>
      <div class="review-top">
        <img class="review-image" src="${imageUrl(current.id)}" alt="Tissue image selected for review">
        <div>
          <div class="label-pair">
            <div><div class="label-source">Model label</div><div class="label-value">${names[pred.predicted_label]}</div></div>
            <div><div class="label-source">${reviewed ? sourceName(reviewed.source) : 'Dataset reference label'}</div><div class="label-value" id="reference-label">${reviewed ? names[reviewed.label] : 'Not revealed yet'}</div></div>
          </div>
          ${reviewed ? `<div class="review-feedback"><strong>${labelMatched ? 'Labels match.' : 'Labels differ.'}</strong> ${plan.method === 'manual' ? 'This reviewed label contributes to the estimate for the collection.' : labelMatched ? 'This agreement contributes to the estimate along with the other checked images.' : 'This difference contributes to the correction for the collection.'}</div>` : '<p class="small muted" style="margin-top:16px">Use the stored label to simulate checking this image.</p>'}
          <div class="review-buttons">
            ${!reviewed ? '<button class="primary" id="reveal-one">Use dataset label for this image</button>' : ''}
            ${reviewed && next ? `<button class="primary" data-review="${next.id}" data-next-review>Next image</button>` : ''}
          </div>
          ${!reviewed ? `<details class="settings"><summary>Enter your own label instead</summary><p class="small muted">A manual label is recorded separately from simulated review. Saved labels cannot be changed.</p><div class="manual-row"><label for="manual-label">Tissue class<select id="manual-label"><option value="">Choose a class</option>${classes.map(k => `<option value="${k}">${names[k]}</option>`).join('')}</select></label><button class="secondary" id="save-manual">Save label</button></div></details>` : ''}
        </div>
      </div>
      <div class="review-support">
        ${contrastPanel(pred, 'review-contrast')}
        <details><summary>Image and selection details</summary><dl><dt>Image extent</dt><dd>Entire 224 × 224 pixel image; one reference class</dd><dt>Selection probability</dt><dd>${current.inclusion_probability.toFixed(4)}</dd><dt>Sampling stratum</dt><dd>${esc(current.stratum)}</dd><dt>Image ID</dt><dd class="code-wrap">${current.id}</dd></dl></details>
      </div>
    </section>
    ${!result ? `<div class="simulate-rest"><div><strong>Finish the demo with existing labels</strong><p>Use the dataset labels for all ${remaining} remaining images in this fixed sample.</p></div><button class="secondary" id="simulate-all">Use all remaining labels</button></div>` : ''}
    <details class="details-block"><summary>See all ${plan.budget} selected images</summary><div class="queue">${plan.units.map(u => `<button data-review="${u.id}" class="${u.id === current.id ? 'selected' : ''}" aria-label="Open selected image ${u.id.slice(0,8)}${annotated.has(u.id) ? ', checked' : ', not checked'}"><img loading="lazy" src="${imageUrl(u.id)}" alt="Selected tissue image">${annotated.has(u.id) ? '<span class="tick">✓</span>' : ''}</button>`).join('')}</div></details>
    <details class="details-block"><summary>Sampling plan and label budget</summary>
      <p>${methods[plan.method]}. The selection was saved before review. All ${plan.budget} selected images must be checked before uncertainty ranges are reported.</p>
      <table><thead><tr><th>Model class</th><th>Study images</th><th>Selected images</th></tr></thead><tbody>${plan.strata.map(s => `<tr><td>${esc(names[s.name] || s.name)}</td><td>${s.N}</td><td>${s.n}</td></tr>`).join('')}</tbody></table>
      <p class="small muted">Seed ${plan.seed}. Sampling without replacement within each stratum. ${plan.total_labels_including_common_pilot} labels including ${plan.pilot_labels_available} common pilot labels. Each label informs every measurement. Review ID: ${plan.id}.</p>
    </details>`;
  if ($('#reveal-one')) $('#reveal-one').onclick = () => submitReview({image_id: current.id, simulate: true});
  if ($('#simulate-all')) $('#simulate-all').onclick = () => submitReview({simulate_all: true});
  if ($('#save-manual')) $('#save-manual').onclick = () => {
    if (!$('#manual-label').value) return notice('Choose a tissue class before saving.', true);
    submitReview({image_id: current.id, label: $('#manual-label').value});
  };
  bindContrastPanel('review-contrast');
}
async function submitReview(body) {
  const buttons = $('#audit-workspace').querySelectorAll('button');
  buttons.forEach(b => b.disabled = true);
  try {
    state.audit = await api(`/api/audits/${state.audit.plan.id}/review`, body);
    state.summary = await api('/api/study');
    renderAuditList();
    renderAudit();
    renderResults();
    if (state.audit.result) {
      notice('All selected images have been checked. Corrected results are ready.');
      $('#audit-workspace [data-go="results"]').focus({preventScroll: true});
      $('#audit-workspace').scrollIntoView({block: 'start'});
    } else if ($('[data-next-review]')) $('[data-next-review]').focus({preventScroll: true});
  } catch (error) {
    notice(error.message, true);
    buttons.forEach(b => b.disabled = false);
  }
}
function renderResults() {
  const result = state.audit?.result;
  $('#export-link').hidden = !result;
  if (!result) {
    const remaining = state.audit ? state.audit.plan.budget - state.audit.annotations.length : null;
    $('#results-content').innerHTML = `<div class="empty-state"><h2>${remaining === null ? 'Start a review to get corrected estimates' : `${remaining} selected images still need a label`}</h2><p>${remaining === null ? 'Checking a sample gives us an estimate of the model’s errors across the collection.' : 'The uncertainty calculation requires the complete sample. Your checked images are saved.'}</p><div class="actions"><button class="primary" data-go="audit">${remaining === null ? 'Start a review' : 'Continue review'}</button>${state.example ? '<button class="secondary" data-example>See completed example</button>' : ''}</div></div>`;
    return;
  }
  const {plan, annotations, predictions} = state.audit;
  $('#export-link').href = `api/audits/${plan.id}/export`;
  const annotationMap = new Map(annotations.map(a => [a.id, a]));
  const evidence = plan.units.map(u => ({row: predictions[u.id], ref: annotationMap.get(u.id).label}));
  const errors = evidence.filter(e => e.row.predicted_label !== e.ref);
  const largest = [...result.endpoints].sort((a,b) => Math.abs(b.estimate - b.initial) - Math.abs(a.estimate - a.initial))[0];
  const noAdjustment = Math.abs(largest.estimate - largest.initial) < 1e-12;
  const sources = result.annotation_sources.map(sourceName).join(' / ');
  $('#results-content').innerHTML = `
    <p class="result-context" id="result-context">${plan.id === state.example?.id ? 'Completed example · ' : ''}${result.reviewed} images checked · ${esc(sources)}</p>
    <section class="panel"><h2>Percentage of all ${result.population} study images</h2>
      <div class="table-scroll"><table class="measurement-table results-table"><thead><tr><th scope="col">Tissue class</th><th scope="col">Model estimate</th><th scope="col">After review</th><th scope="col">95% uncertainty range</th></tr></thead><tbody>
        ${result.endpoints.map(e => `<tr data-endpoint="${e.endpoint}"><td>${names[e.endpoint]}</td><td class="numeric" data-label="Model estimate">${pct(e.initial)}</td><td class="numeric after" data-label="After review">${pct(e.estimate)}</td><td class="numeric" data-label="95% range">${pct(e.lower)}–${pct(e.upper)}</td></tr>`).join('')}
      </tbody></table></div>
      <p class="scope-note">The ranges reflect uncertainty from reviewing a sample. In repeated random reviews, this method covers all three true percentages together at least 95% of the time, treating reference labels as correct. These are image percentages, not tissue areas or cell counts.</p>
    </section>
    <div class="result-summary">${noAdjustment
      ? '<strong>No adjustment from this review</strong><p>The checked labels produce no net change in these three estimates. Uncertainty remains because only a sample was reviewed; the ranges above still apply.</p>'
      : `<strong>Largest adjustment: ${names[largest.endpoint].toLowerCase()}</strong><p>The estimate changes from ${pct(largest.initial)} to ${pct(largest.estimate)}. The remaining uncertainty range is ${pct(largest.lower)}–${pct(largest.upper)}.</p>`}</div>
    <h2>What was checked?</h2>
    <p>${errors.length} of ${result.reviewed} reviewed labels differ from the model. Both differences and agreements contribute to the estimate for the whole collection.</p>
    <details class="details-block" id="reviewed-evidence"><summary>Inspect the ${result.reviewed} reviewed images</summary><p class="small muted">Each image shows the reviewed label above the model label. Select an image to inspect the evidence.</p><div class="evidence-grid">${evidence.map(e => tileCard(e.row, e.ref)).join('')}</div></details>
    <details class="details-block"><summary>How the correction and uncertainty are calculated</summary>
      <p>${result.method === 'manual' ? 'This method estimates class percentages directly from the random review sample.' : 'We start with the model’s totals, then add the sampled differences between reference labels and predictions, weighted by each image’s selection probability.'}</p>
      <p>The conservative 95% intervals use exact hypergeometric inversion and a multiplicity adjustment for the three predefined measurements. They concern this fixed image collection, not variation between patients. Zero observed errors do not imply certainty. Selecting a favorable result from repeated reviews is outside this fixed-plan guarantee.</p>
      <p class="small muted">${esc(result.model_assistance_note)}.</p>
    </details>
    <details class="details-block"><summary>Study record and reproducibility</summary>
      <p>The download includes all predictions, the sampling plan, selected reference labels, image checksums, and covariance. Unreviewed study labels are excluded.</p>
      <dl><dt>Sampling method</dt><dd>${methods[result.method]}</dd><dt>Predictor</dt><dd>${esc(predictorName())}</dd><dt>Image encoder</dt><dd>${esc(state.summary.provenance.model)}</dd><dt>Checkpoint</dt><dd class="code-wrap">${esc(state.summary.provenance.model_revision)}</dd><dt>Prediction hash</dt><dd class="code-wrap">${esc(result.prediction_sha256)}</dd><dt>Review plan hash</dt><dd class="code-wrap">${esc(result.plan_sha256)}</dd></dl>
    </details>`;
}
async function renderEvaluation() {
  $('#load-evaluation').disabled=true;
  try {
    const evaluation=await api('/api/evaluation');
    if (evaluation.available === false) {
      $('#evaluation-content').textContent = evaluation.reason;
      $('#load-evaluation').hidden = true;
      return;
    }
    const rows=Object.entries(evaluation.methods);
    $('#evaluation-content').innerHTML=`<p>${evaluation.repetitions} simulated audits per method, each reviewing ${rows[0][1].budget} of ${evaluation.population} real tiles. ${rows[0][1].pilot_labels} common pilot labels are charged to all methods. No pathologist-time savings have been measured.</p><div class="table-scroll"><table><thead><tr><th>Method</th><th>Joint coverage</th><th>Coverage MC 95% interval</th><th>Mean widest interval</th><th>Labels incl. pilot</th></tr></thead><tbody>${rows.map(([key,m])=>`<tr><td>${methods[key]}</td><td>${pct(m.simultaneous_coverage)}</td><td>${pct(m.simultaneous_coverage_mc95[0])}–${pct(m.simultaneous_coverage_mc95[1])}</td><td>${pct(m.mean_maximum_width)}</td><td>${m.total_labels_including_common_pilot}</td></tr>`).join('')}</tbody></table></div><div class="table-scroll"><table><thead><tr><th>Method / endpoint</th><th>Reference truth</th><th>Bias (pp)</th><th>RMSE (pp)</th><th>Mean width (pp)</th><th>Coverage</th></tr></thead><tbody>${rows.flatMap(([key,m])=>m.endpoints.map(e=>`<tr><td>${methods[key]} · ${e.endpoint}</td><td>${pct(e.truth)}</td><td>${(e.bias*100).toFixed(2)}</td><td>${(e.rmse*100).toFixed(2)}</td><td>${(e.mean_width*100).toFixed(2)}</td><td>${pct(e.coverage)}</td></tr>`)).join('')}</tbody></table></div><div class="callout">${esc(evaluation.interval_comparison)}. ${esc(evaluation.shared_objective_limit)}.</div><p class="caption">${esc(evaluation.interpretation)}. Model agreement with all study reference labels: ${pct(evaluation.descriptive_accuracy)}.</p>`;
    if (evaluation.predictor_development_labels) $('#evaluation-content').insertAdjacentHTML('afterbegin', `<p class="small muted">This comparison holds the trained predictor fixed. Its ${evaluation.predictor_development_labels} separate development labels are additional to the audit and pilot counts below.</p>`);
    $('#load-evaluation').hidden=true;
  } catch(e) { notice(e.message); $('#load-evaluation').disabled=false; }
}
document.addEventListener('click', event => {
  const nav = event.target.closest('[data-screen],[data-go]');
  if (nav) {
    if (nav.id === 'start-review' && state.audit?.result) openNewReview();
    screen(nav.dataset.screen || nav.dataset.go);
  }
  const tile = event.target.closest('[data-image]');
  if (tile) inspectTile(tile.dataset.image).catch(error => notice(error.message, true));
  const review = event.target.closest('[data-review]');
  if (review) {
    state.selected = review.dataset.review;
    renderAudit();
    if ($('#reveal-one')) $('#reveal-one').focus({preventScroll: true});
    $('#audit-workspace').scrollIntoView({block: 'start'});
  }
  if (event.target.closest('[data-example]')) openExample().catch(error => notice(error.message, true));
});
$('.dialog-close').onclick = () => $('#tile-dialog').close();
$('#tile-dialog').addEventListener('click', event => { if (event.target === $('#tile-dialog')) $('#tile-dialog').close(); });
$('#class-filter').innerHTML += classes.map(k => `<option value="${k}">${names[k]}</option>`).join('');
$('#class-filter').onchange = () => { state.shown = 48; renderGallery(); };
$('#tile-order').onchange = () => { state.shown = 48; renderGallery(); };
$('#load-more').onclick = () => { state.shown += 48; renderGallery(); };
$('#reset-neighbors').onclick = () => { state.neighbors = null; state.shown = 48; renderGallery(); };
$('#audit-select').onchange = event => {
  if (event.target.value) loadAudit(event.target.value).catch(error => notice(error.message, true));
};
$('#open-example').onclick = () => openExample().catch(error => notice(error.message, true));
$('#new-review').onclick = () => { openNewReview(); $('#budget').focus(); };
$('#load-evaluation').onclick = renderEvaluation;
$('#method').onchange = () => {
  const descriptions = {
    stratified: 'We’ll select a random sample within the model’s predicted tissue classes. Complete the selected sample to get corrected estimates.',
    uniform: 'We’ll select a random sample across all study images. The checked labels will correct the model’s estimates for the collection.',
    manual: 'We’ll select a random sample across all study images. Estimates will use the checked labels directly, without a model-based correction.',
    shared: 'We’ll use the separate pilot labels to allocate the review budget across model classes, then randomly select images within each class. This allocation is experimental.'
  };
  $('#sampling-description').textContent = descriptions[$('#method').value];
};
$('#budget').oninput = () => $('#select-images').textContent = $('#budget').value ? `Select ${$('#budget').value} images` : 'Select images';
$('#plan-form').onsubmit = async event => {
  event.preventDefault();
  const button = $('#select-images');
  button.disabled = true;
  try {
    const plan = await api('/api/audits', {budget: Number($('#budget').value), method: $('#method').value, seed: Number($('#seed').value)});
    state.summary = await api('/api/study');
    await loadAudit(plan.id);
    $('#reveal-one').focus({preventScroll: true});
    notice(`${plan.budget} images selected. Use the dataset labels to check the model.`);
  } catch (error) { notice(error.message, true); }
  finally { button.disabled = false; }
};
window.addEventListener('hashchange', () => screen(location.hash.slice(1)));
(async () => {
  try {
    [state.summary, state.tiles, state.quality, state.testEvaluations] = await Promise.all([
      api('/api/study'), api('/api/tiles'), api('/api/model-quality').catch(() => null),
      api('/api/test-evaluations').catch(() => ({available: false, error: true}))
    ]);
    renderStudy();
    renderQuality();
    renderTestEvaluations();
    renderAtlas();
    renderResults();
    let saved;
    try { saved = sessionStorage.getItem((window.TissueLab?.storageKey('tissue-study-review') || 'tissue-study-review')); } catch (_) { /* Optional. */ }
    if (saved && state.summary.audits.some(a => a.id === saved)) await loadAudit(saved);
    screen(location.hash.slice(1) || 'study', false);
  } catch (error) { notice(`Could not open study: ${error.message}`, true); }
})();
