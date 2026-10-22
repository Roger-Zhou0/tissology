'use strict';
const $ = selector => document.querySelector(selector);
const ENDPOINTS = ['epithelial_fraction', 'dense_lymphocytic_fraction'];
const STEPS = ['home', 'setup', 'overview', 'initial', 'budget', 'review', 'results'];
const STORAGE = window.TissueLab?.storageKey('tissue-study-research-v1') || 'tissue-study-research-v1';
const draftBaseKey = 'tissue-study-research-draft-v1:' + encodeURIComponent(new URL('api/research', document.baseURI).href);
const DRAFT_STORAGE = window.TissueLab?.storageKey(draftBaseKey) || draftBaseKey;
const state = {collections: [], savedStudies: [], collectionId: null, endpointIds: [...ENDPOINTS], study: null, audit: null,
  roiId: null, unitId: null, step: 'home', comparison: null, comparisonBudget: null, example: null,
  busy: false, budget: null, method: 'uniform', seed: 1729, preview: null, viewMode: 'compare', draft: null};
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const number = value => typeof value === 'number' && Number.isFinite(value);
const pct = value => number(value) ? `${(100 * value).toFixed(1)}%` : 'Undefined';
const pp = value => number(value) ? `${(100 * value).toFixed(1)} pp` : 'Not available';
const count = value => number(value) ? value.toLocaleString() : 'Not yet available';
const text = value => typeof value === 'string' || typeof value === 'number' ? String(value)
  : Array.isArray(value) ? value.map(text).filter(Boolean).join('; ')
    : value && typeof value === 'object' ? text(value.description || value.name || value.label || value.definition) : '';
const list = value => Array.isArray(value) ? value : value && typeof value === 'object'
  ? Object.entries(value).map(([id, row]) => typeof row === 'object' ? {id, ...row} : {id, description: row}) : [];
const definitions = value => list(value?.endpoint_definitions).filter(row => ENDPOINTS.includes(row.id));
const selectedCollection = () => state.collections.find(row => row.id === state.draft?.collectionId);
const studyPath = () => `studies/${encodeURIComponent(state.study.id)}`;
const auditPath = () => `${studyPath()}/audits/${encodeURIComponent(state.audit.plan.id)}`;
function resourceUrl(value) {
  const path = String(value || '').replace(/^\/+/, '');
  return path.startsWith('api/research/') && !path.includes('..') && !path.includes('\\') ? path : '';
}
function annotations() {
  const source = state.audit?.annotations || {};
  return new Map(Array.isArray(source) ? source.map(row => [row.id || row.unit_id, row])
    : Object.entries(source).filter(([, row]) => row != null));
}
function readSaved() {
  try { const value = JSON.parse(localStorage.getItem(STORAGE) || '{}'); return value && typeof value === 'object' ? value : {}; }
  catch (_) { return {}; }
}
function setupDraft() {
  if (state.draft) return state.draft;
  let saved = {};
  try { saved = JSON.parse(sessionStorage.getItem(DRAFT_STORAGE) || '{}') || {}; } catch (_) { /* Keep the draft in memory when browser storage is unavailable. */ }
  const collection = state.collections.find(row => row.id === saved.collectionId)
    || state.collections.find(row => row.status === 'ready') || state.collections[0];
  const available = definitions(collection).map(row => row.id);
  state.draft = {name: typeof saved.name === 'string' ? saved.name.slice(0, 120) : '',
    collectionId: collection?.id || null,
    endpointIds: Array.isArray(saved.endpointIds) ? saved.endpointIds.filter(id => available.includes(id)) : available};
  return state.draft;
}
function saveSetupDraft() {
  try { sessionStorage.setItem(DRAFT_STORAGE, JSON.stringify(state.draft)); } catch (_) { /* Navigation within this page still retains the draft. */ }
}
function clearSetupDraft() {
  state.draft = null;
  try { sessionStorage.removeItem(DRAFT_STORAGE); } catch (_) { /* A disabled browser store does not prevent creation. */ }
}
function persist() {
  const query = new URLSearchParams(location.search);
  for (const [key, value] of Object.entries({study: state.study?.id, audit: state.audit?.plan.id, step: state.step})) {
    if (value) query.set(key, value); else query.delete(key);
  }
  history.replaceState(null, '', `${location.pathname}?${query.toString()}${location.hash}`);
  if (!state.study) return;
  try {
    const previous = readSaved();
    const savedStudies = Array.isArray(previous.studies) ? previous.studies.filter(row => row.id !== state.study.id) : [];
    savedStudies.unshift({id: state.study.id, name: state.study.name || state.study.collection?.name || 'Tissue area study',
      audit_id: state.audit?.plan.id || null});
    localStorage.setItem(STORAGE, JSON.stringify({study: state.study.id, audit: state.audit?.plan.id || null,
      step: state.step, studies: savedStudies.slice(0, 12)}));
  } catch (_) { /* A disabled browser store does not prevent server-side progress. */ }
}
async function api(path, body) {
  const response = await fetch(`api/research/${path}`, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  let result;
  try { result = await response.json(); }
  catch (_) { throw new Error('The server returned an unreadable response. Reload this page when the server is available.'); }
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'The request could not be completed. Your saved review has not been reset.');
  return result;
}
function error(message = '') {
  $('#research-error').hidden = !message;
  $('#research-error').textContent = message;
}
function savedNotice(message = '') {
  $('#research-saved-notice').hidden=!message;
  $('#research-saved-notice').textContent=message;
}
function rememberAuditResponse() {
  const plan=state.audit.plan;
  const summary={id:plan.id,method:plan.method,budget:plan.budget,created_at:plan.created_at,reviewed:annotations().size};
  state.study={...state.study,audits:[summary,...(state.study.audits||[]).filter(row=>row.id!==plan.id)].sort((a,b)=>String(b.created_at||'').localeCompare(String(a.created_at||'')))};
  rememberStudy(state.study);
  persist();
}
async function refreshAfterSave(message) {
  try {await refreshStudy();}
  catch (_) {savedNotice(message+' The study summary could not refresh. Your saved review is available and you can continue.');}
}
async function action(message, work) {
  if (state.busy) return;
  const previousStep = state.step;
  state.busy = true; error(); savedNotice(); $('#research-status').textContent = message;
  $('#research-screen').setAttribute('aria-busy','true');
  document.querySelectorAll('#research-screen button, #research-screen input, #research-screen select, #research-screen textarea, .research-steps button, #research-new-study').forEach(button => button.disabled = true);
  try { await work(); }
  catch (failure) { error(failure.message || 'The request failed.'); }
  finally {
    state.busy = false; $('#research-status').textContent = ''; $('#research-screen').setAttribute('aria-busy','false'); render();
    if (state.step !== previousStep) {
      window.scrollTo({top: 0, behavior: 'instant'});
      $('#research-screen h2')?.focus({preventScroll: true});
    }
  }
}
function stepAllowed(step) {
  return ['home', 'setup'].includes(step) || (['overview', 'initial', 'budget', 'review'].includes(step) && state.study) || (step === 'results' && state.audit?.result);
}
function move(step, focus = true) {
  if (state.busy || !stepAllowed(step)) return;
  if (step === 'review' && !state.audit) step = 'budget';
  state.step = step; error(); savedNotice(); persist(); render();
  if (focus) { $('#research-screen h2')?.focus({preventScroll: true}); window.scrollTo({top: 0, behavior: 'instant'}); }
}
function legend() {
  return '<div class="research-legend"><span><i style="background:#be5b86"></i>Epithelium</span><span><i style="background:#478b7c"></i>Stroma</span><span><i style="background:#725dac"></i>Dense lymphocytic tissue, within stroma</span><span><i style="background:#e1e5ea"></i>Other / excluded</span></div>';
}
function figure(url, alt, caption, mask = false) {
  const valid = resourceUrl(url);
  return `<figure>${valid ? `<img ${mask ? 'class="mask"' : ''} src="${esc(valid)}" alt="${esc(alt)}">`
    : '<div class="research-placeholder">Image not available</div>'}<figcaption>${esc(caption)}</figcaption></figure>`;
}
function formula(row) {
  if (row.formula) return text(row.formula);
  const numerator = text(row.numerator_definition || row.numerator);
  const denominator = text(row.denominator_definition || row.denominator);
  return numerator && denominator ? `${numerator} ÷ ${denominator}` : text(row.definition || row.description);
}
function definitionDetails(row) {
  const values = [['Numerator', row.numerator_definition || row.numerator], ['Denominator', row.denominator_definition || row.denominator],
    ['Eligible tissue', row.eligible_tissue || row.eligible], ['Excluded regions', row.excluded_regions || row.excluded],
    ['Scale', row.physical_scale || row.scale], ['Aggregation', row.aggregation || row.aggregation_rule],
    ['Boundaries', row.boundary_rule], ['Zero denominator', row.zero_denominator]];
  const visible = values.filter(([, value]) => text(value));
  return visible.length ? `<details><summary>Exact measurement definition</summary><dl>${visible.map(([label, value]) =>
    `<dt>${esc(label)}</dt><dd>${esc(text(value))}</dd>`).join('')}</dl></details>` : '';
}
function measurementLabel(row) {
  return ({epithelial_fraction: 'Epithelial tissue area', dense_lymphocytic_fraction: 'Dense lymphocytic tissue area'})[row.id] || row.name;
}
function measurementDescription(row) {
  return ({epithelial_fraction: 'Share of eligible tissue covered by epithelium.',
    dense_lymphocytic_fraction: 'Share of eligible tissue covered by densely packed lymphocytic tissue.'})[row.id] || formula(row);
}
function studyTitle(value) {
  if (!value) return 'Tissue composition study';
  if (value.name && value.name !== value.collection?.name) return value.name;
  return (value.name || value.collection?.name || 'Tissue composition study') + (value.id ? ' · ' + value.id.slice(0,6) : '');
}

function savedStudyRows() {
  const browserSaved = readSaved().studies;
  // Server records carry the latest review, including work saved in another browser.
  const records = new Map(state.savedStudies.map(row => [row.id, row]));
  for (const row of Array.isArray(browserSaved) ? browserSaved : []) if (row?.id && !records.has(row.id)) records.set(row.id, row);
  return [...records.values()];
}
function studyLink(row) {
  const audit = row.latest_audit, auditId = audit?.id || row.audit_id;
  const complete = number(audit?.budget) && audit.budget > 0 && audit.reviewed >= audit.budget;
  const query = new URLSearchParams({study: row.id, step: auditId ? complete ? 'results' : 'review' : 'overview'});
  if (auditId) query.set('audit', auditId);
  return `?${query.toString()}`;
}
function studyProgress(row) {
  const audit = row.latest_audit;
  if (number(audit?.budget) && number(audit.reviewed)) return audit.reviewed >= audit.budget
    ? 'Review complete · open results' : `${count(audit.reviewed)} of ${count(audit.budget)} patches reviewed`;
  return row.audit_id ? 'Resume saved review' : 'Review not started';
}
function renderHome() {
  const preview = state.preview, roi = preview?.rois?.[0];
  const saved = savedStudyRows().filter(row => row.id !== state.example?.study_id);
  $('#research-screen').innerHTML = `${state.example ? `<article class="study-feature">
    <div class="study-feature-image">${roi ? figure(roi.image_url, 'Breast tissue from the example study', 'Actual study image · measurement area outlined') : '<div class="research-placeholder">Image preview unavailable</div>'}</div>
    <div class="study-feature-copy"><span class="status-pill">Example study · review complete</span><h2 tabindex="-1">${esc(studyTitle(preview))}</h2>
    <p>Inspect the tissue, follow a simulated review, and see the measurements with their image evidence.</p>
    <div class="collection-facts"><span><strong>${count(preview?.collection?.roi_count)}</strong> regions</span><span><strong>${count(preview?.collection?.slide_count)}</strong> slides</span><span><strong>${preview?.endpoint_ids?.length || 2}</strong> measurements</span></div>
    <a class="primary" id="research-open-example" href="?study=${encodeURIComponent(state.example.study_id)}&audit=${encodeURIComponent(state.example.audit_id)}&step=initial">Explore example <span aria-hidden="true">→</span></a>
    <p class="small muted">Existing dataset annotations simulate expert review. Uploaded collections remain separate.</p></div></article>` : '<section class="panel"><h2 tabindex="-1">Explore the prepared collection</h2><p>Choose New example run to define measurements on the prepared breast regions. Uploaded image collections are separate.</p></section>'}
    <section class="collection-scan-entry" data-feature-link="collections" hidden aria-labelledby="home-scan-heading"><div><h2 id="home-scan-heading">Scan your image collection</h2><p>Import images, see predicted tissue-class counts, and inspect each image.</p><span>Preliminary colorectal tile inspection · separate from an audited study</span></div><a id="research-open-scanner" class="secondary" href="collections?import=1">Scan collection <span aria-hidden="true">→</span></a></section>
    <div class="study-home-bottom"><section><h2 class="section-label">Saved example runs</h2>${saved.length ? `<div class="study-list">${saved.slice(0, 8).map(row => `<a href="${esc(studyLink(row))}"><span>${esc(row.name || 'Tissue study')}<small>${esc(studyProgress(row))}</small></span><span aria-hidden="true">↗</span></a>`).join('')}</div>` : '<p class="muted small">Runs you create on the prepared collection appear here.</p>'}</section>
    <aside class="library-entry"><span class="eyebrow">Explore another collection</span><h2>Find tissue by appearance</h2><p>Describe a pattern, review matches, and guide the search with examples.</p><a href="semantic">Open image library →</a><p class="small muted">Separate colorectal tile library · 4,500 development images</p></aside></div>`;
  window.TissueNavigation?.refresh();
}
function renderSetup() {
  const draft = setupDraft();
  const collection = selectedCollection(), ready = collection?.status === 'ready';
  const options = state.collections.map(row => `<option value="${esc(row.id)}" ${row.id === draft.collectionId ? 'selected' : ''}>${esc(row.name)}${row.status === 'ready' ? '' : ' · preparing'}</option>`).join('');
  const preview = state.preview?.collection_id === draft.collectionId ? state.preview?.rois?.[0] : null;
  $('#research-screen').innerHTML = `<div class="research-setup-layout"><section><h2 class="research-stage-title" tabindex="-1">Define the example measurements</h2><p class="stage-intro">Choose tissue areas to measure on the prepared breast collection. This example uses existing annotations to simulate review.</p>
    ${collection ? `<form id="research-create-form" class="panel"><label for="research-name">Study name</label><input id="research-name" required maxlength="120" value="${esc(draft.name)}" placeholder="For example: Breast composition pilot"><label for="research-collection">Prepared image collection</label><select id="research-collection">${options}</select>
      <div class="research-collection-summary"><div class="collection-facts"><span><strong>${count(collection.roi_count)}</strong> regions</span><span><strong>${count(collection.slide_count)}</strong> slides</span><span><strong>${count(collection.population_size)}</strong> review patches</span></div>
      <details class="collection-details"><summary>Collection scope and predictor</summary><p>${esc(text(collection.scope))}</p><p class="scope-note">${esc(text(collection.heldout_status))}</p>
      ${number(collection.pilot_roi_count) ? `<p class="scope-note">${count(collection.pilot_roi_count)} separate pilot fields inform the review design and are excluded from this study population.</p>` : ''}
      <p class="scope-note">Results describe these image regions; no patient-level generalization is claimed.</p><p class="scope-note">Frozen predictor: ${esc(text(collection.model) || 'Available when ready')}. Review adjusts measurements, not this predictor.</p></details></div>
      <h3>Measure tissue area</h3><div class="research-definition-list">${definitions(collection).map(row => `<div class="research-definition"><label><input type="checkbox" name="endpoint" value="${esc(row.id)}" ${draft.endpointIds.includes(row.id) ? 'checked' : ''}><span>${esc(measurementLabel(row))}<span class="formula">${esc(measurementDescription(row))}</span></span></label>${definitionDetails(row)}</div>`).join('')}</div>
      <p class="scope-note">Percentages use eligible tissue area. These are not individual cell counts.</p>
      ${!ready ? `<p class="research-pending">This collection is ${esc(collection.status || 'preparing')}. Creation becomes available after its assets are verified.</p>` : ''}
      <div class="research-start-actions"><button class="primary" type="submit" ${!ready || !draft.endpointIds.length ? 'disabled' : ''}>Create example run →</button><button type="button" class="text-button" id="research-cancel-create">Back to example</button></div></form>` : '<p class="research-pending">No prepared study collection is configured yet.</p>'}</section>
      <aside class="setup-preview">${preview ? figure(preview.image_url, 'Preview of the selected study collection', 'Breast tissue · one region from the selected collection') : ''}<h3>What happens next</h3><ol class="setup-path"><li>Inspect the tissue and predicted maps.</li><li>Choose a fixed sample to review.</li><li>See corrected measurements and evidence.</li></ol><p class="small muted">The demo reveals existing annotations for the selected sample.</p></aside></div>`;
  if ($('#research-name')) $('#research-name').oninput = event => { draft.name = event.target.value; saveSetupDraft(); };
  if ($('#research-collection')) $('#research-collection').onchange = event => {
    draft.collectionId = event.target.value;
    const available = definitions(selectedCollection()).map(row => row.id);
    draft.endpointIds = draft.endpointIds.filter(id => available.includes(id));
    saveSetupDraft(); renderSetup();
  };
  document.querySelectorAll('input[name=endpoint]').forEach(input => input.onchange = () => { draft.endpointIds = [...document.querySelectorAll('input[name=endpoint]:checked')].map(item => item.value); saveSetupDraft(); $('#research-create-form button[type=submit]').disabled = !ready || !draft.endpointIds.length; });
  if ($('#research-cancel-create')) $('#research-cancel-create').onclick = () => move('home');
  if ($('#research-create-form')) $('#research-create-form').onsubmit = event => {
    event.preventDefault(); if (!draft.endpointIds.length) return error('Choose at least one measurement.');
    draft.name = $('#research-name').value; saveSetupDraft();
    action('Creating the study record…', async () => {
      const created = await api('studies', {collection_id: draft.collectionId, endpoint_ids: [...draft.endpointIds], name: draft.name.trim()});
      // Creation already returns the complete study; another request could fail after a successful save.
      adoptStudy(created); clearSetupDraft(); state.step = 'initial'; persist();
    });
  };
}
function renderOverview() {
  const collection = state.study.collection || {}, roi = state.study.rois?.[0];
  const checked = annotations().size, budget = state.audit?.plan.budget;
  $('#research-screen').innerHTML = `<div class="study-workbench overview-workbench"><section class="tissue-canvas"><div class="canvas-toolbar"><h2 tabindex="-1">Study overview</h2><span class="status-pill">${state.audit?.result ? 'Review complete' : state.audit ? 'Review in progress' : 'Ready to review'}</span></div>
    ${roi ? `<div class="overview-image">${figure(roi.image_url, 'Actual tissue from this study', 'Selected study region · measurement area outlined')}</div>` : ''}
    <div class="collection-facts"><span><strong>${count(collection.roi_count)}</strong> regions</span><span><strong>${count(collection.slide_count)}</strong> slides</span><span><strong>${count(state.study.population_size)}</strong> sampling windows</span></div></section>
    <aside class="workspace-inspector"><p class="eyebrow">Your measurements</p>${definitions(state.study).filter(row => state.endpointIds.includes(row.id)).map(row => `<section class="measurement-definition"><h3>${esc(measurementLabel(row))}</h3><p>${esc(measurementDescription(row))}</p>${definitionDetails(row)}</section>`).join('')}
    <div class="inspector-next"><p class="small">${state.audit ? `${checked} of ${budget} selected patches reviewed.` : 'Inspect the AI maps, then choose your review budget.'}</p><button class="primary" id="research-overview-next">${state.audit?.result ? 'Open results →' : state.audit ? 'Continue review →' : 'Explore tissue →'}</button></div>
    <details class="settings"><summary>Study scope and source</summary><p>${esc(text(collection.scope))}</p><p>${esc(text(collection.model))}</p><p>Selected regions only. Tissue area fractions; no cell counts or patient-level conclusions.</p></details>${savedAudits()}</aside></div>`;
  $('#research-overview-next').onclick = () => move(state.audit?.result ? 'results' : state.audit ? 'review' : 'initial'); bindSavedAudits();
}
function renderInitial() {
  const rois = state.study.rois || [], roi = rois.find(row => row.id === state.roiId) || rois[0];
  if (roi) state.roiId = roi.id;
  $('#research-screen').innerHTML = `<div class="study-workbench"><section class="tissue-canvas"><div class="canvas-toolbar"><h2 class="research-stage-title" tabindex="-1">Explore tissue</h2><div class="view-switch" aria-label="Tissue view">${[['compare','Compare'],['original','Image'],['prediction','AI map']].map(([id,label]) => `<button data-view-mode="${id}" aria-pressed="${state.viewMode === id}">${label}</button>`).join('')}</div></div>
    ${roi ? `<label class="research-viewer-controls">Region<select id="research-roi">${rois.map((row, index) => `<option value="${esc(row.id)}" ${row.id === roi.id ? 'selected' : ''}>${index + 1} / ${rois.length} · ${esc(row.slide_id || row.id)}</option>`).join('')}</select></label>
    <div class="research-images" data-view="${state.viewMode}">${figure(roi.image_url, 'Original tissue field with the measurement area identified', 'Original tissue · measurement area outlined')}${figure(roi.prediction_url, 'Frozen predicted tissue-region map', 'AI tissue map · frozen prediction', true)}</div>${legend()}` : '<p>Image previews have not been supplied.</p>'}
    <p class="canvas-scope">Only the outlined measurement area belongs to the study.</p></section>
    <aside class="workspace-inspector"><p class="eyebrow">Across the collection</p><h3>Initial measurements</h3><table class="research-endpoint-table"><tbody>${(state.study.initial || []).map(row => `<tr><td>${esc(measurementLabel(row))}</td><td class="numeric">${pct(row.estimate)}</td></tr>`).join('')}</tbody></table>
    <p class="scope-note">AI estimates of eligible tissue area across the selected image regions.</p><div class="inspector-next"><button class="primary" id="research-choose-budget">${state.audit ? 'Open review →' : 'Plan the review →'}</button><p class="small muted">${state.audit ? 'Inspect the saved sample and revealed annotations.' : 'Check a fixed sample to estimate a correction.'}</p>${state.audit ? '<button class="text-button" id="research-new-review">Plan a separate review</button>' : ''}</div>
    <details class="settings"><summary>Measurement definitions</summary>${definitions(state.study).filter(row => state.endpointIds.includes(row.id)).map(row => `<h3>${esc(measurementLabel(row))}</h3><p class="small">${esc(measurementDescription(row))}</p>${definitionDetails(row)}`).join('')}</details>
    <details class="settings"><summary>Exact pixel totals</summary><div class="table-scroll"><table><thead><tr><th>Measurement</th><th>Numerator / eligible pixels</th></tr></thead><tbody>${(state.study.initial || []).map(row => `<tr><td>${esc(row.name)}</td><td>${count(row.numerator)} / ${count(row.denominator)}</td></tr>`).join('')}</tbody></table></div></details>
    <details class="settings"><summary>Image source and boundaries</summary>${roi ? `<p class="scope-note research-roi-provenance">${roi.patient_id ? `Recorded patient: ${esc(roi.patient_id)}. ` : ''}${roi.hospital_code ? `TCGA hospital: ${esc(roi.hospital_code)}. ` : ''}Identifiers document source; results apply to selected regions.</p><p class="scope-note">${esc(text(roi.scope) || 'Positions represent adjacency within this field. Reference annotations are revealed only for sampled review windows.')}</p>` : ''}</details>${savedAudits()}</aside></div>`;
  if ($('#research-roi')) $('#research-roi').onchange = event => { state.roiId = event.target.value; renderInitial(); };
  document.querySelectorAll('[data-view-mode]').forEach(button => button.onclick = () => { state.viewMode = button.dataset.viewMode; renderInitial(); $(`[data-view-mode="${state.viewMode}"]`).focus({preventScroll:true}); });
  $('#research-choose-budget').onclick = () => move(state.audit ? 'review' : 'budget');
  if ($('#research-new-review')) $('#research-new-review').onclick = () => move('budget'); bindSavedAudits();
}
function methods() {
  const supplied = state.study.audit_methods || state.study.collection?.audit_methods;
  return Array.isArray(supplied) && supplied.length ? supplied : [
    {id: 'uniform', name: 'Uniform model-assisted', description: 'Every fixed window has the same chance of selection. Reviewed differences correct the frozen model totals.', min_budget: 2},
    {id: 'manual', name: 'Manual-only random sample', description: 'Use the same random sampling design, estimating totals from the reviewed annotations alone.', min_budget: 2}];
}
function savedAudits() {
  const audits = state.study.audits || [];
  return audits.length ? `<details class="details-block research-saved"><summary>Resume a saved review</summary><label for="research-saved-audit">Saved review</label>
    <select id="research-saved-audit"><option value="">Choose a review</option>${audits.map(row => `<option value="${esc(row.id)}">${count(row.reviewed ?? row.reviewed_count)} of ${count(row.budget)} windows · ${esc(row.method)} · ${esc(row.id.slice(0, 8))}</option>`).join('')}</select></details>` : '';
}
function bindSavedAudits() {
  if ($('#research-saved-audit')) $('#research-saved-audit').onchange = event => {
    if (!event.target.value) return;
    action('Loading the saved review…', async () => { await loadAudit(event.target.value); state.step = state.audit.result ? 'results' : 'review'; persist(); });
  };
}
function renderBudget() {
  const options = methods();
  const selected = options.find(row => row.id === state.method) || options.find(row => row.default) || options[0];
  state.method = selected.id;
  const minimum = selected.min_budget ?? 2, maximum = state.study.population_size;
  const budgets = (state.study.collection?.evaluation_budgets || [48, 96, 160]).filter(value => number(value) && value >= minimum && value <= maximum);
  if (!budgets.includes(state.budget)) state.budget = budgets.includes(96) ? 96 : budgets[0];
  const pilotCost = state.study.collection?.pilot_window_count ?? state.study.allocation_pilot_cost;
  const accounting = () => number(pilotCost) && number(state.budget)
    ? `${count(pilotCost)} separate pilot windows + ${count(state.budget)} final review windows = ${count(pilotCost + state.budget)} total annotation units. The same pilot cost is charged to every method.`
    : 'The same recorded pilot cost is charged to every comparison method.';
  $('#research-screen').innerHTML = `<h2 class="research-stage-title" tabindex="-1">How many patches will you check?</h2>
    <p class="stage-intro">Each patch informs every selected measurement. Fix the sample size before revealing annotations, then complete the selected sample.</p>
    <div class="budget-layout"><div><form id="research-budget-form" class="panel"><div class="research-fields">
      <label>New patches to review<select id="research-budget" required>${budgets.map(value => `<option value="${value}" ${value === state.budget ? 'selected' : ''}>${value} patches</option>`).join('')}</select></label></div>
      <p id="research-method-explanation">${state.method === 'uniform' ? 'Check a random sample to correct the AI estimates. Every patch has the same chance of selection.' : esc(text(selected.description))}</p><p class="scope-note">${count(maximum)} patches in this study. Budgets count image patches, not minutes of expert time.</p>
      <p id="research-budget-accounting" class="research-selection-note">${esc(accounting())}</p>
      <details class="settings"><summary>Sampling method &amp; reproducibility</summary><label>Review method<select id="research-method">${options.map(row => `<option value="${esc(row.id)}" ${row.id === state.method ? 'selected' : ''}>${esc(row.name || row.id)}</option>`).join('')}</select></label><label>Random seed<input id="research-seed" type="number" min="0" step="1" required value="${state.seed}"></label><p class="scope-note">The sample, inclusion probabilities, model version and seed are recorded. Every method can reuse annotations across measurements. Complete the whole sample; do not stop early because a result looks favorable.</p></details>
      <div class="actions"><button class="primary" type="submit" ${!budgets.length ? 'disabled' : ''}>Select ${count(state.budget)} review patches</button></div>
      ${state.audit ? '<p class="scope-note">This creates a separate saved review. The existing review remains available.</p>' : ''}</form>${savedAudits()}</div><aside class="budget-preview">${state.study.rois?.[0] ? figure(state.study.rois[0].image_url, 'Tissue from this study before review patches are selected', 'Study region · the review sample is selected after you confirm') : ''}<h3>One patch, multiple measurements</h3><p>The same reviewed annotation informs each selected tissue-area measurement.</p><p>The complete sample stays fixed. Existing annotations simulate the review; the budget counts patches, not minutes.</p></aside></div>`;
  $('#research-budget').onchange = event => {
    state.budget = Number(event.target.value);
    $('#research-budget-form button[type=submit]').textContent = `Select ${count(state.budget)} review patches`;
    $('#research-budget-accounting').textContent = accounting();
  };
  $('#research-method').onchange = event => {
    state.seed = Number($('#research-seed').value); state.method = event.target.value;
    const next = options.find(row => row.id === state.method);
    state.budget = Math.max(next.min_budget ?? 2, state.budget); renderBudget();
  };
  $('#research-budget-form').onsubmit = event => {
    event.preventDefault(); state.seed = Number($('#research-seed').value);
    action('Freezing the probability sample…', async () => {
      const created = await api(`${studyPath()}/audits`, {budget: state.budget, method: state.method, seed: state.seed, alpha: .05});
      if (created.plan) { state.audit = created; state.unitId = null; state.comparison = null; }
      else await loadAudit(created.id);
      state.step = 'review'; rememberAuditResponse();
      await refreshAfterSave('The fixed review sample was saved.');
    });
  };
  bindSavedAudits();
}
function allocationDetails() {
  const allocation = state.audit.plan.allocation;
  const rows = list(allocation);
  return `<details class="settings"><summary>Sampling record</summary><p>${esc(text(state.audit.plan.design_description))}</p>
    <p class="small">Frozen final budget: ${count(state.audit.plan.budget)} windows. ${number(state.audit.plan.pilot_cost) ? `Pilot cost: ${count(state.audit.plan.pilot_cost)} windows. Total annotation units: ${count(state.audit.plan.total_review_units)}. ` : ''}Seed: ${esc(state.audit.plan.seed)}.</p>
    ${rows.length ? `<div class="table-scroll"><table><thead><tr><th>Stratum</th><th>Population windows</th><th>Selected windows</th></tr></thead><tbody>${rows.map(row => `<tr><td>${esc(row.name || row.stratum || row.id)}</td><td>${count(row.population ?? row.population_size ?? row.N)}</td><td>${count(row.allocated ?? row.sample_size ?? row.n ?? row.budget)}</td></tr>`).join('')}</tbody></table></div>` : ''}</details>`;
}
function componentTable(unit, reference) {
  const predicted = unit.predicted_components || unit.prediction_components || unit.prediction || state.audit.predictions?.[unit.id];
  const checked = reference?.components || reference?.reference_components || reference?.reference;
  const componentNames = Array.isArray(predicted) && predicted.length === 3
    ? ['Epithelial pixels', 'Dense lymphocytic tissue pixels', 'Eligible pixels']
    : ['Epithelial pixels', 'Stromal pixels', 'Dense lymphocytic tissue pixels', 'Eligible pixels'];
  if (!Array.isArray(predicted)) return '';
  return `<div class="table-scroll"><table class="research-component-table"><thead><tr><th>Component</th><th>Predicted pixels</th><th>Revealed reference pixels</th></tr></thead><tbody>${componentNames.map((label, i) => `<tr><td>${label}</td><td>${count(predicted[i])}</td><td>${reference ? count(checked?.[i]) : 'Not revealed'}</td></tr>`).join('')}</tbody></table></div>`;
}
function renderReview() {
  const plan = state.audit.plan, units = plan.units || [], checked = annotations();
  const next = units.find(row => !checked.has(row.id)), unit = units.find(row => row.id === state.unitId) || next || units[0];
  if (!unit) { $('#research-screen').innerHTML = '<p class="research-pending">The audit contains no review windows.</p>'; return; }
  state.unitId = unit.id;
  const reference = checked.get(unit.id), remaining = plan.budget - checked.size;
  const referenceUrl = unit.reference_url || `api/research/${auditPath()}/reference/${encodeURIComponent(unit.id)}`;
  $('#research-screen').innerHTML = `<div class="study-workbench"><section class="tissue-canvas"><div class="canvas-toolbar"><h2 class="research-stage-title" tabindex="-1">Review tissue</h2><span class="status-pill">${reference ? 'Annotation revealed' : 'Awaiting review'}</span></div>
    <label class="research-viewer-controls">Selected patch<select id="research-window">${units.map((row,i) => `<option value="${esc(row.id)}" ${row.id === unit.id ? 'selected' : ''}>Patch ${i+1} / ${plan.budget}${checked.has(row.id) ? ' · reviewed' : ''}</option>`).join('')}</select></label>
    <div class="research-images research-window-images">${figure(unit.context_url,'Tissue context with the central measurement window outlined','Original context · central patch outlined')}${figure(unit.prediction_url,'Frozen predicted tissue labels for the central window','AI map · central patch',true)}${reference ? figure(referenceUrl,'Revealed manual tissue-region annotation for the central window','Dataset annotation · central patch',true) : '<figure><div class="research-placeholder"><span>Annotation hidden</span><small>Reveal the existing annotation to compare this patch.</small></div><figcaption>Dataset annotation · simulated review</figcaption></figure>'}</div>${legend()}<p class="canvas-scope">Only the outlined central patch counts. Surrounding context is excluded.</p></section>
    <aside class="workspace-inspector"><p class="eyebrow">Fixed review sample</p><div class="research-progress"><strong>${checked.size} of ${plan.budget} patches reviewed</strong><progress value="${checked.size}" max="${plan.budget}" aria-label="Review progress"></progress></div><h3>${reference ? 'Compare the tissue maps' : 'Check this patch'}</h3><p class="scope-note">Compare the frozen AI map with the existing dataset annotation.</p>
    <div class="research-review-actions">${!reference ? '<button class="primary" id="research-reveal">Reveal annotation</button>' : next ? '<button class="primary" id="research-next">Next unreviewed patch →</button>' : '<button class="primary" id="research-see-results">See corrected results →</button>'}${remaining > 0 ? `<button class="secondary" id="research-reveal-all">Reveal remaining ${remaining} (demo)</button>` : ''}</div>
    <p class="small muted">Simulated review. The predictor and selected sample stay fixed.</p>${remaining > 0 ? '<p class="small muted">Results appear after the whole selected sample is reviewed.</p>' : ''}
    <details class="settings"><summary>Selection and pixel counts</summary><p class="small">${esc(text(unit.reason) || text(plan.design_description) || 'Selected by the recorded probability design before annotations were revealed.')}</p><p class="scope-note">${unit.bounds ? `Central native-pixel bounds: [${unit.bounds.left}, ${unit.bounds.right}) × [${unit.bounds.top}, ${unit.bounds.bottom}). ` : ''}${number(unit.inclusion_probability) ? `Inclusion probability: ${unit.inclusion_probability.toFixed(4)}.` : ''}</p>${componentTable(unit,reference)}</details>${allocationDetails()}</aside></div>`;
  $('#research-window').onchange = event => {state.unitId = event.target.value; renderReview();};
  if ($('#research-reveal')) $('#research-reveal').onclick = () => review({unit_id:unit.id,simulate:true});
  if ($('#research-reveal-all')) $('#research-reveal-all').onclick = () => review({simulate_all:true});
  if ($('#research-next')) $('#research-next').onclick = () => {state.unitId=next.id;renderReview();};
  if ($('#research-see-results')) $('#research-see-results').onclick = () => move('results');
}
async function review(body) {
  await action('Revealing only the selected reference annotations…', async () => {
    state.audit = await api(`${auditPath()}/review`, body);
    rememberAuditResponse();
    await refreshAfterSave('Your annotation review was saved.');
    if (state.audit.result) await loadComparison();
  });
}
function renderEvidence(result) {
  const evidence = Array.isArray(result.evidence) ? result.evidence : [];
  if (!evidence.length) return '<p>No ranked correction evidence has been supplied.</p>';
  const explanation = state.audit.plan.method === 'manual'
    ? 'Showing up to three reviewed windows with the largest inverse-probability-weighted reference contributions to the manual estimate.'
    : 'Showing up to three windows with the largest recorded component corrections among those reviewed. Weighted pixel-total corrections are not isolated changes in an area fraction.';
  return `<p class="scope-note">Open a reviewed patch to inspect its image and annotation.</p><details class="evidence-selection"><summary>How these examples were selected</summary><p>${esc(text(result.evidence_description) || explanation)}</p></details><ol class="research-evidence">${evidence.slice(0, 3).map(row => {
    const unit = state.audit.plan.units.find(item => item.id === (row.unit_id || row.id));
    const label = row.endpoint_name || state.study.endpoint_definitions?.find?.(item => item.id === row.endpoint_id)?.name || row.name;
    return `<li>${unit ? figure(unit.context_url, 'Tissue surrounding a reviewed patch linked to the correction', 'Reviewed tissue · central patch outlined') : ''}
      ${label ? `<strong>${esc(label)}</strong>` : ''}
      ${unit && annotations().has(unit.id) ? `<button class="text-button research-inspect-evidence" data-unit="${esc(unit.id)}">Inspect this reviewed patch <span aria-hidden="true">→</span></button>` : '<p class="small muted">No accessible reviewed patch is linked.</p>'}
      <details><summary>Correction details</summary><p>${esc(text(row.description || row.reason))}</p>
      ${number(row.correction) && row.correction_units ? `<p>${esc(row.correction_label || 'Recorded correction contribution')}: ${esc(row.correction)} ${esc(row.correction_units)}</p>` : ''}
      ${Array.isArray(row.correction) && row.correction.length === 3 ? `<p>Weighted correction to totals: ${['epithelial', 'dense lymphocytic tissue', 'eligible'].map((name, i) => `${name} ${number(row.correction[i]) ? `${row.correction[i] >= 0 ? '+' : ''}${row.correction[i].toLocaleString(undefined, {maximumFractionDigits: 1})}` : 'undefined'}`).join('; ')} ${esc(row.units || 'pixels')}.</p>` : ''}
      </details></li>`;
  }).join('')}</ol>`;
}
function renderComparison() {
  const value = state.comparison;
  if (!value || !value.available) return `<p class="research-pending">${esc(text(value?.status || value?.message) || 'Fixed-budget comparison results are not available yet. No method advantage is claimed.')}</p>`;
  const allRows = value.rows || value.methods || [];
  const budgets = [...new Set(allRows.map(row => row.budget).filter(number))].sort((a, b) => a - b);
  if (!budgets.includes(state.comparisonBudget)) state.comparisonBudget = budgets.includes(state.audit.plan.budget) ? state.audit.plan.budget : budgets[0] ?? value.budget;
  const rows = allRows.filter(row => !number(row.budget) || row.budget === state.comparisonBudget);
  const assessment = code => ({evidence_of_undercoverage: 'Monte Carlo evidence of coverage below 95%',
    estimate_below_nominal: 'Coverage estimate below 95%'}[code] || text(code));
  return `<p>${esc(text(value.scope))}</p>
    ${budgets.length ? `<label class="research-comparison-controls">Comparison final budget<select id="research-comparison-budget">${budgets.map(budget => `<option value="${budget}" ${budget === state.comparisonBudget ? 'selected' : ''}>${budget} windows</option>`).join('')}</select></label>` : ''}
    <p class="scope-note">${count(state.comparisonBudget)} final reviewed windows per simulated audit. ${number(value.repetitions) ? `${count(value.repetitions)} repeated simulated audits. ` : ''}Every method uses the same population, measurements and final review budget, is charged the same pilot cost, and may reuse annotations across endpoints. Changing this comparison does not alter your saved audit.</p>
    ${rows.length ? `<div class="table-scroll"><table id="research-comparison-table"><thead><tr><th>Audit method</th><th>Mean endpoint RMSE</th><th>Mean widest confidence-set hull</th><th>Simultaneous coverage</th></tr></thead><tbody>${rows.map(row => `<tr><td>${esc(row.name || row.method)}</td><td>${pp(row.mean_root_mean_squared_error)}</td><td>${pp(row.mean_widest_interval)}</td><td>${pct(row.simultaneous_coverage)}${Array.isArray(row.coverage_mc95) && row.coverage_mc95.length === 2 && row.coverage_mc95.every(number) ? `<br><span class="small muted">MC interval: ${pct(row.coverage_mc95[0])}–${pct(row.coverage_mc95[1])}</span>` : ''}${row.coverage_assessment ? `<br><span class="small muted">${esc(assessment(row.coverage_assessment))}</span>` : ''}</td></tr>`).join('')}</tbody></table></div>` : '<p>The report contains no supported comparison rows.</p>'}
    <p class="scope-note">RMSE means root mean squared estimation error, conditional on a defined estimate. Percentage points (pp) describe errors and range widths. A confidence-set hull spans separated segments; it does not fill the gap between them. Coverage checks the complete simultaneous family. These simulations do not establish pathologist time savings.</p>
    <details class="settings"><summary>Error and failures for each measurement</summary><div class="table-scroll"><table><thead><tr><th>Method / measurement</th><th>Conditional RMSE</th><th>Conditional bias</th><th>Coverage</th><th>Undefined estimates</th></tr></thead><tbody>${rows.flatMap(row => (row.endpoints || []).map(endpoint => `<tr><td>${esc(row.name || row.method)} / ${esc(endpoint.name || definitions(state.study).find(item => item.id === endpoint.id)?.name || endpoint.id)}</td><td>${pp(endpoint.conditional_rmse)}</td><td>${pp(endpoint.conditional_bias)}</td><td>${pct(number(endpoint.coverage) ? endpoint.coverage : endpoint.coverage?.estimate)}</td><td>${count(endpoint.undefined_estimates)}</td></tr>`)).join('')}</tbody></table></div></details>
    ${value.warnings?.length ? listWarnings(value.warnings) : ''}`;
}
function listWarnings(warnings) {
  const labels = {
    nonpositive_estimated_denominator: 'The corrected eligible-area total is nonpositive, so its area fractions are undefined.',
    conservative_component_intersection_fallback: 'A conservative component bound conflicted with already reviewed area. The result uses the recorded deterministic fallback.',
    approximate_fieller_unbounded: 'The approximate confidence set is unbounded before restricting it to feasible area fractions. A precise range is not supported.',
    approximate_fieller_disjoint: 'The approximate confidence set has separated segments. Do not replace them with a precise single range.',
    approximate_fieller_empty: 'The approximate confidence set is empty. Do not interpret this as a precise result.',
    point_estimate_outside_physical_range: 'The raw corrected estimate falls outside 0–100%. It is retained without clipping and requires caution.'
  };
  return (Array.isArray(warnings) ? warnings : [warnings]).filter(Boolean).map(value => `<p class="research-warning">${esc(labels[text(value)] || text(value))}</p>`).join('');
}
function rangeFigure(estimate, interval) {
  if (!interval.every(value => number(value) && value >= 0 && value <= 1)) return '';
  const x = value => 10 + value * 280;
  return `<svg class="range-plot" viewBox="0 0 300 36" role="img" aria-label="Approximate interval ${pct(interval[0])} to ${pct(interval[1])}; corrected estimate ${pct(estimate)}. Axis zero to one hundred percent."><line class="range-track" x1="10" x2="290" y1="12" y2="12"/><line class="range-interval" x1="${x(interval[0])}" x2="${x(interval[1])}" y1="12" y2="12"/>${number(estimate) && estimate >= 0 && estimate <= 1 ? `<circle class="range-estimate" cx="${x(estimate)}" cy="12" r="4"/>` : ''}<text x="10" y="32">0%</text><text x="290" y="32" text-anchor="end">100%</text></svg>`;
}
function renderResults() {
  const result = state.audit.result;
  const endpoints = list(result.endpoints);
  const manual = state.audit.plan.method === 'manual';
  const estimateLabel = manual ? 'Manual sample estimate' : 'Corrected estimate';
  const confidence = number(result.simultaneous_confidence) ? `${Math.round(100 * result.simultaneous_confidence)}% ` : '';
  $('#research-screen').innerHTML = `<div class="heading-row"><h2 class="research-stage-title" tabindex="-1">Study results</h2><a id="research-open-report" class="primary" href="report?kind=study&id=${encodeURIComponent(state.study.id)}&audit=${encodeURIComponent(state.audit.plan.id)}">Open report</a></div>
    <p class="stage-intro">${count(state.audit.plan.budget)} selected patches reviewed. Estimated tissue composition in these image regions.</p>
    <p class="research-uncertainty-note"><strong>Approximate ${confidence}uncertainty ranges.</strong> Nominal coverage is not guaranteed. Reliability checks and conservative bounds are available below.</p>
    ${listWarnings(result.warnings || [])}<div class="results-layout"><div class="results-measurements"><section class="panel"><div class="table-scroll"><table class="research-result-table"><thead><tr><th>Measurement</th><th>${estimateLabel}</th><th>Approximate uncertainty range</th></tr></thead><tbody>
    ${endpoints.map(row => {
      const segments = Array.isArray(row.confidence_set) ? row.confidence_set.filter(segment => Array.isArray(segment) && segment.length === 2 && segment.every(number)) : [];
      const complicated = segments.length > 1 || String(row.status || row.interval_status || '').includes('unbounded');
      const interval = !complicated && Array.isArray(row.interval) && row.interval.length === 2 && row.interval.every(number) ? row.interval : null;
      const conflict = row.estimate_outside_interval || (interval && number(row.estimate) && (row.estimate < interval[0] || row.estimate > interval[1]));
      const noVariation = row.standard_error === 0 && row.status !== 'exact_census'
        && state.audit.plan.budget < state.study.population_size && result.interval_method?.approximate !== false
        && !(row.warnings || []).some(value => text(value).toLowerCase().includes('no observed variation'));
      const label = measurementLabel(row) || definitions(state.study).find(item => item.id === row.id)?.name || row.id;
      return `<tr><td>${esc(label)}<span class="result-definition">${esc(measurementDescription(row))}</span></td><td class="result-estimate" data-label="${estimateLabel}"><strong>${pct(row.estimate)}</strong>
      ${row.estimate_status && row.estimate_status !== 'defined' ? `<p class="small muted">${esc(text(row.estimate_note) || row.estimate_status.replaceAll('_', ' '))}</p>` : ''}
      ${conflict ? '<p class="research-warning">The raw estimate conflicts with the reported bounds. Keep it as a diagnostic; use the uncertainty range. More review may be needed.</p>' : ''}${row.warning || row.warnings ? listWarnings(row.warning || row.warnings) : ''}</td>
      <td class="result-range" data-label="Approximate uncertainty range"><strong>${interval ? `${pct(interval[0])}–${pct(interval[1])}` : 'Range unresolved'}</strong>${interval ? rangeFigure(row.estimate, interval) : ''}
      ${!interval ? '<p class="research-warning">No single finite range is supported. Inspect the reported confidence set and warnings; more review may be needed.</p>' : ''}
      ${noVariation ? '<p class="research-warning">No observed variation; this approximate range may be unreliable.</p>' : ''}
      ${segments.length && (!interval || complicated) ? `<p class="small">Reported confidence set: ${segments.map(segment => `${pct(segment[0])}–${pct(segment[1])}`).join(' or ')}</p>` : ''}
      ${row.interval_status && !['defined', 'valid'].includes(row.interval_status) ? `<p class="small muted">${esc(text(row.interval_note) || row.interval_status.replaceAll('_', ' '))}</p>` : ''}</td></tr>`;
    }).join('')}</tbody></table></div>
    <p class="scope-note">${manual ? 'Estimated from the complete reference sample.' : 'Adjusted using the complete selected sample; the predictor stays fixed.'} These are tissue-area measurements, not model accuracy scores.</p>
    </section></div>
    <aside class="results-evidence"><h3>Image evidence</h3>${renderEvidence(result)}</aside></div>
    <div class="results-secondary"><section class="panel research-export-panel"><div><h3>Keep the result and its evidence</h3><p class="scope-note">Download the measurements, review records and image references for reproducibility.</p>
      <details class="settings research-provenance"><summary>Recorded study and audit identifiers</summary><dl><dt>Study</dt><dd><code>${esc(state.study.id)}</code></dd><dt>Audit</dt><dd><code>${esc(state.audit.plan.id)}</code></dd>
      ${result.plan_sha256 ? `<dt>Plan SHA256</dt><dd><code>${esc(result.plan_sha256)}</code></dd>` : ''}${result.prediction_sha256 ? `<dt>Prediction SHA256</dt><dd><code>${esc(result.prediction_sha256)}</code></dd>` : ''}</dl></details></div><a class="secondary" id="research-export" href="${esc(resourceUrl(result.export_url) || `api/research/${auditPath()}/export`)}">Download study record</a></section>
    <h3 class="results-detail-heading">Methods and reliability</h3>
    <details class="panel research-calculation"><summary>Initial estimates and calculation details</summary>
    <div class="table-scroll"><table class="research-initial-comparison"><thead><tr><th>Measurement</th><th>AI estimate before review</th><th>${estimateLabel}</th></tr></thead><tbody>${endpoints.map(row => `<tr><td>${esc(measurementLabel(row) || row.name || row.id)}</td><td>${pct(row.predicted_fraction ?? state.study.initial?.find(item => item.id === row.id)?.estimate)}</td><td>${pct(row.estimate)}</td></tr>`).join('')}</tbody></table></div>
    <p class="scope-note">Primary simultaneous ranges use the approximate Fieller method. ${manual ? 'Numerator and denominator totals are estimated jointly from the sampled references.' : 'Numerator and denominator totals are corrected jointly.'} Ratio estimates are not exactly unbiased. Raw estimates are shown without clipping.</p>
    <p class="scope-note">${esc(text(result.interval_description) || 'Weakly estimated denominators may leave uncertainty unresolved. Conservative ranges may remain wide at a small review budget.')}</p>
    ${state.study.interval_family ? `<p class="scope-note">${esc(state.study.interval_family)}</p>` : ''}
    <p class="scope-note">${esc(text(result.scope || result.design_scope) || 'Inference concerns these selected image fields, not new patients, whole slides, biological cell counts or treatment outcomes.')}</p></details>
    <details class="panel research-conservative"><summary>Conservative finite-population bounds</summary><p class="scope-note">These auxiliary bounds are separate from the approximate Fieller ranges. They may cover most or all of 0–100% at this budget.</p>
      <div class="table-scroll"><table><thead><tr><th>Measurement</th><th>Conservative bound</th></tr></thead><tbody>${endpoints.map(row => `<tr><td>${esc(row.name || definitions(state.study).find(item => item.id === row.id)?.name || row.id)}</td><td>${Array.isArray(row.conservative_interval) && row.conservative_interval.length === 2 && row.conservative_interval.every(number) ? `${pct(row.conservative_interval[0])}–${pct(row.conservative_interval[1])}` : 'Not supplied'}</td></tr>`).join('')}</tbody></table></div></details>
    <details class="panel research-comparison"><summary>Audit-method benchmarks and coverage checks</summary>${renderComparison()}</details>
    <div class="actions"><button class="secondary" id="research-return-review">Inspect reviewed windows</button><button class="text-button" id="research-another-audit">Start a separate fixed-budget review</button></div></div>`;
  document.querySelectorAll('.research-inspect-evidence').forEach(button => button.onclick = () => { state.unitId = button.dataset.unit; move('review'); });
  $('#research-return-review').onclick = () => move('review');
  $('#research-another-audit').onclick = () => move('budget');
  if ($('#research-comparison-budget')) $('#research-comparison-budget').onchange = event => {
    state.comparisonBudget = Number(event.target.value); renderResults(); $('#research-comparison-budget').closest('.research-comparison').open = true; $('#research-comparison-budget').focus();
  };
}
function render() {
  if (!stepAllowed(state.step)) state.step = state.study ? 'initial' : 'home';
  if (state.step === 'review' && !state.audit) state.step = 'budget';
  const inStudy = !!state.study && !['home','setup'].includes(state.step);
  document.body.classList.toggle('research-in-study',inStudy);
  document.body.dataset.researchStep=state.step;
  $('.research-steps').hidden=!inStudy;
  document.querySelectorAll('.research-steps button').forEach(button => {
    const current=button.dataset.step===state.step || (button.dataset.step==='review' && state.step==='budget');
    button.disabled=state.busy || !stepAllowed(button.dataset.step);
    if(current)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');
  });
  $('#research-context').hidden=!inStudy;
  $('#research-title').textContent=inStudy ? studyTitle(state.study) : state.step==='setup' ? 'New example run' : 'Example tissue study';
  $('#research-breadcrumb').innerHTML=inStudy ? `<a href="?step=home">Example study</a><span aria-hidden="true"> / </span>${esc(state.study.collection?.name || 'Tissue study')}` : 'Tissology';
  $('#research-subtitle').hidden=inStudy || state.step==='setup';
  $('#research-new-study').hidden=state.step==='setup';
  $('#research-new-study').disabled=state.busy;
  $('.research-simulation').hidden=state.step==='home';
  if(inStudy)$('#research-context').innerHTML=`<span>${count(state.study.collection?.roi_count)} regions</span><span>${count(state.study.collection?.slide_count)} slides</span><span>${state.endpointIds.length} measurements</span>${state.audit ? `<span>${annotations().size} / ${count(state.audit.plan.budget)} reviewed</span>` : '<span>Review not started</span>'}`;
  document.title=`${$('#research-title').textContent} · Tissology`;
  ({home:renderHome,setup:renderSetup,overview:renderOverview,initial:renderInitial,budget:renderBudget,review:renderReview,results:renderResults})[state.step]();
  if(state.busy)document.querySelectorAll('#research-screen button, #research-screen input, #research-screen select, #research-screen textarea').forEach(control=>control.disabled=true);
}
function rememberStudy(study) {
  const summary = {...study, latest_audit: study.audits?.[0] || null};
  state.savedStudies = [summary, ...state.savedStudies.filter(row => row.id !== study.id)];
}
async function refreshStudy() { state.study = await api(studyPath()); rememberStudy(state.study); }
function adoptStudy(study) {
  state.study = study; state.audit = null; state.comparison = null; state.comparisonBudget = null;
  state.collectionId = state.study.collection_id || state.study.collection?.id || state.collectionId;
  state.endpointIds = state.study.endpoint_ids || definitions(state.study).map(row => row.id);
  state.roiId = null; state.unitId = null; state.budget = null;
  rememberStudy(study);
}
async function loadStudy(id) { adoptStudy(await api(`studies/${encodeURIComponent(id)}`)); }
async function loadComparison() {
  try { state.comparison = await api(`${studyPath()}/comparison?budget=${encodeURIComponent(state.audit.plan.budget)}`); }
  catch (_) { state.comparison = {available: false, status: 'Comparison results could not be loaded. The saved audit results remain available.'}; }
}
async function loadAudit(id) {
  state.audit = await api(`${studyPath()}/audits/${encodeURIComponent(id)}`); state.unitId = null; state.comparison = null; state.comparisonBudget = null;
  if (state.audit.result) await loadComparison();
}
document.querySelectorAll('.research-steps button').forEach(button => button.onclick = () => move(button.dataset.step));
$('#research-new-study').onclick = () => move('setup');
(async () => {
  try {
    const [response, overview] = await Promise.all([api('collections'), api('comparison').catch(() => null)]);
    state.collections = response.collections || []; state.savedStudies = response.studies || []; state.example = overview?.demo || null;
    state.preview = state.example ? await api(`studies/${encodeURIComponent(state.example.study_id)}`).catch(() => null) : null;
    state.collectionId = (state.collections.find(row => row.status === 'ready') || state.collections[0])?.id || null;
    const query = new URLSearchParams(location.search), saved = readSaved();
    const entryStep = query.get('step');
    const id = ['home', 'setup'].includes(entryStep) ? null : query.get('study') || saved.study;
    if (entryStep === 'setup') state.step = 'setup';
    if (id) {
      await loadStudy(id);
      const auditId = query.has('study') ? query.get('audit') : saved.audit;
      if (auditId) await loadAudit(auditId);
      const wanted = query.get('step') || saved.step;
      state.step = STEPS.includes(wanted) && stepAllowed(wanted) ? wanted : state.audit?.result ? 'results' : state.audit ? 'review' : 'initial';
      persist();
    }
    render();
  } catch (failure) { error(failure.message); render(); }
})();
