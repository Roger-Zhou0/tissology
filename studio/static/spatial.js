'use strict';
const $ = selector => document.querySelector(selector);
const state = {summary: null, audit: null, selected: null};
const names = ['Epithelial area', 'Stromal area', 'Dense lymphocytic tissue area', 'Eligible area'];
const pct = value => value == null ? 'Undefined' : `${(100 * value).toFixed(1)}%`;
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const imageUrl = (id, kind) => `api/spatial/windows/${encodeURIComponent(id)}/${kind}`;
async function api(path, body) {
  const response = await fetch(`api/spatial/${path}`, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail));
  return result;
}
function notice(message) { $('#spatial-notice').textContent = message; }
function renderSummary() {
  const data = state.summary;
  $('#spatial-intro').textContent = `${data.regions} selected regions contain ${data.population} fixed image windows. Compare predicted tissue areas with a limited sample of manual annotations, then inspect corrected area fractions and uncertainty.`;
  $('#spatial-budget').max = data.population;
  $('#spatial-initial tbody').innerHTML = data.initial.map((value, i) => `<tr><td>${names[i]} / eligible area</td><td>${pct(value)}</td></tr>`).join('');
  $('#spatial-saved').innerHTML = '<option value="">Choose a review</option>' + data.audits.map(row => `<option value="${esc(row.id)}">${row.reviewed} of ${row.budget} windows checked · ${esc(row.method)} · ${row.id.slice(0, 6)}</option>`).join('');
  if (state.audit) $('#spatial-saved').value = state.audit.plan.id;
}
function renderReview() {
  const audit = state.audit;
  $('#spatial-plan-panel').hidden = !!audit;
  if (!audit) { $('#spatial-review').innerHTML = ''; return; }
  const reviewed = new Map(audit.annotations.map(row => [row.id, row]));
  const next = audit.plan.units.find(row => !reviewed.has(row.id));
  if (!state.selected) state.selected = (next || audit.plan.units[0]).id;
  const unit = audit.windows.find(row => row.id === state.selected);
  const reference = reviewed.get(unit.id);
  const prediction = audit.predictions[unit.id];
  const remaining = audit.plan.budget - reviewed.size;
  const referenceUrl = `api/spatial/audits/${audit.plan.id}/reference/${encodeURIComponent(unit.id)}`;
  $('#spatial-review').innerHTML = `<section class="panel">
    <div class="heading-row"><h2>2. Review selected windows</h2><button class="secondary" id="spatial-new">Start another review</button></div>
    <p><strong>${reviewed.size} of ${audit.plan.budget} windows checked</strong> · ${audit.plan.method === 'uniform' ? 'Uniform model-assisted' : 'Manual-only random sample'}</p>
    <label>Selected window<select id="spatial-window">${audit.plan.units.map((row, index) => `<option value="${esc(row.id)}" ${row.id === unit.id ? 'selected' : ''}>${index + 1}. ${esc(row.id)}${reviewed.has(row.id) ? ' · checked' : ''}</option>`).join('')}</select></label>
    <div class="spatial-images">
      <figure><img src="${imageUrl(unit.id, 'context')}" alt="Tissue context with central measurement window outlined in red"><figcaption>Image context. Only the red central window contributes pixels.</figcaption></figure>
      <figure><img class="region-mask" src="${imageUrl(unit.id, 'prediction')}" alt="Frozen predicted region labels for the central window"><figcaption>Frozen MuTILs prediction · central window</figcaption></figure>
      <figure>${reference ? `<img class="region-mask" src="${referenceUrl}" alt="Revealed manual reference region labels for the central window">` : '<div class="spatial-placeholder">Reference not revealed</div>'}<figcaption>Manual dataset annotation · central window</figcaption></figure>
    </div>
    <div class="spatial-legend"><span><i style="background:#be5b86"></i>Epithelium</span><span><i style="background:#478b7c"></i>Stroma</span><span><i style="background:#725dac"></i>Dense lymphocytic tissue (within stroma)</span><span><i style="background:#e1e5ea"></i>Excluded</span></div>
    <p class="scope-note">Central bounds: [${unit.bounds.left}, ${unit.bounds.right}) × [${unit.bounds.top}, ${unit.bounds.bottom}) in native image pixels; ${unit.area_pixels} pixels. Surrounding context is for interpretation and contributes no pixels. Windows do not overlap or double count.</p>
    <div class="table-scroll"><table class="spatial-components"><thead><tr><th>Pixel component</th><th>Prediction</th><th>Reviewed reference</th></tr></thead><tbody>${names.map((name, i) => `<tr><td>${name}</td><td>${prediction[i]}</td><td>${reference ? reference.components[i] : 'Not revealed'}</td></tr>`).join('')}</tbody></table></div>
    <div class="spatial-review-actions">${!reference ? '<button class="primary" id="spatial-reveal">Reveal this window’s reference</button>' : next ? '<button class="primary" id="spatial-next">Next unchecked window</button>' : ''}${remaining ? `<button class="secondary" id="spatial-reveal-all">Use annotations for all ${remaining} remaining windows</button>` : '<a class="primary" href="#spatial-results">See corrected results</a>'}</div>
    <p class="small muted">Simulated review uses existing manual annotations. Each selected window has inclusion probability ${audit.plan.inclusion_probability.toFixed(4)}. Frozen predictions and sample selection remain unchanged.</p>
  </section>`;
  $('#spatial-window').onchange = event => { state.selected = event.target.value; renderReview(); };
  $('#spatial-new').onclick = () => { state.audit = null; state.selected = null; renderReview(); renderResults(); };
  if ($('#spatial-reveal')) $('#spatial-reveal').onclick = () => review({unit_id: unit.id, simulate: true});
  if ($('#spatial-reveal-all')) $('#spatial-reveal-all').onclick = () => review({simulate_all: true});
  if ($('#spatial-next')) $('#spatial-next').onclick = () => { state.selected = next.id; renderReview(); };
}
function renderResults() {
  const result = state.audit?.result;
  $('#spatial-results').hidden = !result;
  if (!result) return;
  $('#spatial-result-content').innerHTML = `<p>${result.reviewed} of ${result.population_size} fixed windows reviewed. The ranges below are conservative ${(1 - result.alpha) * 100}% simultaneous uncertainty bounds for this finite image collection.</p>
    <div class="table-scroll"><table class="spatial-result-table"><thead><tr><th>Area fraction</th><th>Simultaneous range</th><th>Model estimate</th><th>Raw corrected estimate</th></tr></thead><tbody>${result.endpoints.map((row, i) => {
      const conflicts = row.estimate_outside_interval || (row.estimate != null && row.interval && (row.estimate < row.interval[0] || row.estimate > row.interval[1]));
      return `<tr><td>${names[i]} / eligible area</td><td data-label="Simultaneous range"><strong>${row.interval ? `${pct(row.interval[0])}–${pct(row.interval[1])}` : 'No eligible area'}</strong>${row.interval_status === 'conditional_on_positive_eligible_area' ? '<br><span class="small">Conditional on positive eligible area</span>' : ''}</td><td data-label="Model estimate">${pct(row.predicted_fraction)}</td><td data-label="Raw corrected estimate">${pct(row.estimate)}${row.estimate_status !== 'defined' ? `<br><span class="small">${esc(row.estimate_status.replaceAll('_', ' '))}</span>` : ''}${conflicts ? '<p class="spatial-estimate-warning">Raw estimate conflicts with already reviewed area; more review is needed. Use the simultaneous range; retain this raw estimate as a diagnostic.</p>' : ''}</td></tr>`;
    }).join('')}</tbody></table></div>
    <p class="scope-note">Bounds may be wide, including 0–100%, at a small budget. Numerator and denominator totals are estimated jointly; the resulting ratios are not exactly unbiased. Raw model-assisted point estimates can leave 0–100% and are shown without clipping. Epithelial and stromal fractions are complementary. These bounds concern the selected image collection, not new patients or cell counts.</p>
    <p><a class="secondary" id="spatial-export" href="api/spatial/audits/${state.audit.plan.id}/export">Download spatial study record</a></p>
    <details class="settings"><summary>Sampling and uncertainty method</summary><p class="small">Fixed-size simple random sampling without replacement. All four component totals share the same annotations. Finite-population concentration bounds and simultaneous component bounds produce conservative ratio ranges. Optional stopping is not supported.</p><p class="small"><a href="${esc(result.interval_source.url)}" target="_blank" rel="noreferrer">${esc(result.interval_source.authors)} (${result.interval_source.year}), ${esc(result.interval_source.theorem)}</a></p><p class="small code-wrap">Prediction hash: ${esc(state.summary.study.prediction_sha256)}<br>Plan hash: ${esc(result.plan_sha256)}</p></details>`;
}
async function loadAudit(id) {
  state.audit = await api(`audits/${id}`);
  state.selected = null;
  renderReview(); renderResults();
}
async function review(body) {
  $('#spatial-review').querySelectorAll('button').forEach(button => button.disabled = true);
  try {
    state.audit = await api(`audits/${state.audit.plan.id}/review`, body);
    state.summary = await api('study');
    renderSummary(); renderReview(); renderResults();
  } catch (error) { notice(error.message); renderReview(); }
}
$('#spatial-plan-form').onsubmit = async event => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true;
  try {
    notice('');
    const plan = await api('audits', {budget: Number($('#spatial-budget').value), method: $('#spatial-method').value, seed: Number($('#spatial-seed').value)});
    await loadAudit(plan.id);
    state.summary = await api('study'); renderSummary();
  } catch (error) { notice(error.message); }
  finally { button.disabled = false; }
};
$('#spatial-saved').onchange = event => { if (event.target.value) loadAudit(event.target.value).catch(error => notice(error.message)); };
(async () => {
  try { state.summary = await api('study'); if (!state.summary.available) throw new Error('Spatial pilot is not configured.'); renderSummary(); }
  catch (error) { notice(error.message); }
})();
