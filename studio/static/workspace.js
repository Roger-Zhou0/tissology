'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const guide = $('workflow-guide'), guideToggle = $('workspace-guide-toggle');
  const wideGuide = window.matchMedia('(min-width: 1180px)');
  const guidePreferenceKey = 'tissue-studio:workflow-guide:auto-hidden:v1';
  const guideSessionKey = 'tissue-studio:workflow-guide:dismissed:session';
  let guideDismissedThisSession = false;
  try { guideDismissedThisSession = sessionStorage.getItem(guideSessionKey) === '1'; } catch (_) {}
  const guideDismiss = $('workflow-guide-dismiss');
  function guidePreferenceNote(saved=true) {
    $('workflow-guide-preference-note').textContent = saved
      ? 'Saved in this browser. Reopen the guide from How it works.'
      : 'This browser could not save your choice. You can close the guide now, but it may open again on your next visit.';
  }
  try { guideDismiss.checked = localStorage.getItem(guidePreferenceKey) === '1'; }
  catch (_) { guidePreferenceNote(false); }
  guideDismiss.addEventListener('change', () => {
    try {
      if (guideDismiss.checked) localStorage.setItem(guidePreferenceKey, '1');
      else localStorage.removeItem(guidePreferenceKey);
      guidePreferenceNote();
    } catch (_) { guidePreferenceNote(false); }
  });
  function openGuide() {
    if (guide.open) return;
    const modal = !wideGuide.matches;
    document.body.classList.add('workflow-guide-open');
    document.body.classList.toggle('workflow-guide-modal', modal);
    guideToggle.setAttribute('aria-expanded', 'true');
    guide.setAttribute('aria-modal', String(modal));
    if (modal) guide.showModal(); else guide.show();
  }
  function finishGuide() {
    const returnFocus = guideToggle.getAttribute('aria-expanded') === 'true';
    document.body.classList.remove('workflow-guide-open', 'workflow-guide-modal');
    guideToggle.setAttribute('aria-expanded', 'false');
    if (returnFocus) guideToggle.focus({preventScroll:true});
  }
  function closeGuide() {
    guideDismissedThisSession = true;
    try { sessionStorage.setItem(guideSessionKey, '1'); } catch (_) {}
    guide.close(); finishGuide();
  }
  guideToggle.addEventListener('click', () => guide.open ? closeGuide() : openGuide());
  $('workspace-guide-close').addEventListener('click', closeGuide);
  $('workflow-guide-done').addEventListener('click', closeGuide);
  guide.addEventListener('cancel', event => { event.preventDefault(); closeGuide(); });
  guide.addEventListener('close', () => { if (!guide.open) finishGuide(); });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && wideGuide.matches && guide.open && !event.defaultPrevented) { event.preventDefault(); closeGuide(); }
  });
  wideGuide.addEventListener('change', () => {
    if (!guide.open) return;
    const focused = document.activeElement;
    // Reopen in the appropriate native mode; the small-screen dialog traps focus.
    guide.close(); openGuide();
    if (guide.contains(focused)) focused.focus({preventScroll:true});
  });
  function updateGuideAvailability(data, pending=false) {
    const availability=$('workspace-scan-availability');
    availability.textContent=pending ? 'Checking scan availability…' : data?.scan_available===false
      ? 'Browse images · new scans paused' : data?.scan_available===true ? 'Scan colorectal H&E tiles' : 'Scan availability unconfirmed';
    const importCopy=document.querySelector('#workspace-add-collection .action-copy p');
    if(importCopy) importCopy.textContent=data?.scan_available===false ? 'Browse scanned collections or import images for viewing.' : 'Upload images and review tissue predictions.';
    const sampleId = '307b6b38b73f455390059a997ec2ceb2';
    $('workflow-guide-sample').hidden = !(data?.collections || []).some(row => row.id === sampleId);
    $('workflow-guide-status').textContent = pending ? 'Checking scan availability…' : data?.scan_available === false
      ? 'New image scans are currently paused. You can still explore collections that have already been scanned.'
      : data?.scan_available === true
        ? 'New scans are available for supported colorectal H&E tiles.'
        : 'Scan availability could not be confirmed. Use Refresh in Saved work to reconnect.';
  }
  const node = (tag, text, className) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (className) el.className = className; return el; };
  let items = [], loading = false, exampleStudyId = null, page = null, requestNumber = 0, searchTimer = null;
  const kinds = {collection:'Image collection',study:'Tissue study',concept:'Saved search',description:'Finding',draft:'Browser draft'};
  const number = value => Number(value || 0).toLocaleString();
  async function get(path) { const response = await fetch(path); if (!response.ok) throw new Error('Could not load '+path); return response.json(); }
  async function loadBenchmark() {
    try {
      const record = await get('api/collections/validation');
      const measured = record.known_benchmark_reproduction;
      if (!record.available || !measured || !Number.isInteger(measured.total) || measured.total < 1 || !Number.isInteger(measured.correct) || measured.correct < 0 || measured.correct > measured.total || !Array.isArray(measured.labels)) throw new Error('Benchmark unavailable');
      $('workspace-accuracy').textContent = (100 * measured.correct / measured.total).toFixed(2) + '%';
      $('workspace-accuracy-scope').textContent = `${number(measured.correct)} / ${number(measured.total)} supplied labels matched · ${record.dataset}`;
      document.querySelector('.benchmark-limit').textContent = `${measured.labels.length} tissue classes · previously inspected benchmark`;
      const facts = $('workspace-benchmark-record');
      facts.replaceChildren(node('p', record.scope_note), node('p', 'This benchmark does not estimate accuracy on your collection. Use Check this scan for a separate random image check.'), node('p', 'This classification result does not measure image-search quality or breast tissue-area accuracy.'));
      if (Number.isFinite(record.balanced_accuracy)) facts.append(node('p', `Balanced accuracy: ${(100 * record.balanced_accuracy).toFixed(1)}%.`));
      if (record.weakest_class && Number.isFinite(record.weakest_class.recall)) facts.append(node('p', `Lowest class recall: ${record.weakest_class.name}, ${(100 * record.weakest_class.recall).toFixed(1)}%.`));
      if (record.fresh_image_check?.passed) facts.append(node('p', record.fresh_image_check.summary));
      if (record.saved_examples_note) facts.append(node('p', record.saved_examples_note));
      const download = $('workspace-benchmark-download');
      if (!/^static\/collection-validation(?:-[a-z0-9-]+)?\.json$/.test(record.report_url || '')) throw new Error('Benchmark record link unavailable');
      download.href = record.report_url;
      $('workspace-benchmark-details').hidden = false;
    } catch (_) {
      $('workspace-accuracy').textContent = '—';
      $('workspace-accuracy-scope').textContent = 'The benchmark could not be loaded. Your tools and saved work are still available.';
      $('workspace-benchmark-details').hidden = true;
    }
  }
  function render() {
    $('workspace-items').replaceChildren();
    for (const item of items) {
      const row = node('article', undefined, 'workspace-row'); row.dataset.kind = item.kind;
      const main = node('div', undefined, 'work-identity');
      const title = node(item.href ? 'a' : 'span', item.name || 'Untitled record', 'work-title'); if(item.href)title.href = item.href;
      const savedDate = item.created_at ? new Date(item.created_at).toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric'}) : '';
      main.append(node('span', kinds[item.kind]+(savedDate?' · '+savedDate:''), 'work-kind'), title, node('p', item.source, 'work-source'));
      if(item.prediction_version?.name)main.append(node('p',item.prediction_version.name,'work-version'));
      const progress = node('div', undefined, 'work-progress'); progress.append(node('span', item.status), node('small', item.detail || ''));
      const actions = node('div',undefined,'work-actions');
      if(item.href){const open=node('a','Open','work-open');open.href=item.href;open.setAttribute('aria-label','Open '+item.name);actions.append(open);}
      if(item.report_href){const report=node('a','View report','work-report');report.href=item.report_href;report.setAttribute('aria-label','View report for '+item.name);actions.append(report);}
      row.append(main, progress, actions); $('workspace-items').append(row);
    }
    if (!items.length) {
      const filtered=$('workspace-filter').value.trim()||$('workspace-kind').value!=='all';
      const empty = node('div',undefined,'workspace-empty');
      empty.append(node('h3',page?.warnings?.length?'Some saved work is unavailable':filtered?'No matching work':'Your work will appear here'), node('p',page?.warnings?.length?'Use Refresh to reconnect.':filtered?'Try a different name or type of work.':'Imported collections, searches, findings and studies are saved on this server.'));
      if(!filtered&&!page?.warnings?.length){const start=node('a','Start with an example →');start.href='#examples-heading';empty.append(start);}
      $('workspace-items').append(empty);
    }
    const total=page?.total||0,offset=page?.offset||0;
    $('workspace-status').textContent=total?`${offset+1}–${Math.min(offset+items.length,total)} of ${number(total)} ${total===1?'record':'records'}${page.complete===false?' · some sources need attention':''}`:'';
    $('workspace-pagination').hidden=!total||total<=page.limit;
    $('workspace-page').textContent=total?`Page ${Math.floor(offset/page.limit)+1} of ${Math.ceil(total/page.limit)}`:'';
    $('workspace-previous').disabled=page?.previous_offset==null;
    $('workspace-next').disabled=page?.next_offset==null;
  }
  function savedPending(message) {
    $('workspace-status').textContent=message;
    $('workspace-previous').disabled=true;$('workspace-next').disabled=true;
    $('workspace-items').setAttribute('aria-busy','true');
  }
  async function loadSaved(offset=0) {
    clearTimeout(searchTimer);searchTimer=null;
    const request=++requestNumber;
    savedPending('Loading saved work…');
    const query=new URLSearchParams({q:$('workspace-filter').value.trim(),kind:$('workspace-kind').value,offset:String(offset),limit:'12',include_examples:String($('workspace-show-examples').checked)});
    try {
      const data=await get('api/workspace?'+query);
      if(request!==requestNumber)return;
      if(!Array.isArray(data.items)||!Number.isInteger(data.total))throw new Error('Saved-work response is unavailable');
      page=data;items=data.items;
      $('workspace-errors').hidden=!data.warnings?.length;
      $('workspace-errors').textContent=(data.unavailable_count?`${number(data.unavailable_count)} saved ${data.unavailable_count===1?'record needs':'records need'} attention. `:'')+(data.warnings||[]).join(' ');
      render();
    } catch (_) {
      if(request!==requestNumber)return;
      $('workspace-errors').hidden=false;
      $('workspace-errors').textContent='Saved work could not be refreshed. '+(items.length?'Showing the previous page. ':'')+'Your saved records are unchanged. Use Refresh to reconnect.';
      $('workspace-status').textContent=items.length?'Previously loaded records · refresh unavailable':'Records unavailable';
    } finally {if(request===requestNumber)$('workspace-items').setAttribute('aria-busy','false');}
  }
  async function load(retry=false) {
    if (loading) return; loading = true; $('workspace-refresh').disabled = true;
    const saved=loadSaved(0);
    const capabilities = await (retry ? window.TissueNavigation.reload() : window.TissueNavigation.ready);
    if(!capabilities){
      updateGuideAvailability(null);
      $('workspace-actions-status').hidden = false;
      $('workspace-actions-status').textContent = 'Tools could not connect. Use Refresh below to reconnect.';
      await saved;
      loading=false;$('workspace-refresh').disabled=false;return;
    }
    $('workspace-actions-status').hidden = !!(capabilities.collections_available || capabilities.semantic_available || capabilities.research_available);
    $('workspace-actions-status').textContent = 'No tools are currently available. Use Refresh below to check again.';
    $('workspace-example-study').hidden = !capabilities.research_available || !exampleStudyId;
    $('workspace-example-library').hidden = !capabilities.semantic_available;
    updateGuideAvailability(null, !!capabilities.collections_available);
    await Promise.allSettled([saved,
      capabilities.collections_available?get('api/collections').then(data=>{updateGuideAvailability(data);for(const element of document.querySelectorAll('[data-example-collection]'))element.hidden=!(data.collections||[]).some(row=>row.id===element.dataset.exampleCollection);}).catch(()=>updateGuideAvailability(null)):Promise.resolve(),
      capabilities.research_available?get('api/research/comparison').then(({demo})=>{if(demo?.study_id&&demo.audit_id){exampleStudyId=demo.study_id;$('workspace-example-study').hidden=false;$('workspace-example').href='research?study='+encodeURIComponent(demo.study_id)+'&audit='+encodeURIComponent(demo.audit_id)+'&step=results';}}).catch(()=>{}):Promise.resolve()
    ]);
    document.querySelector('.workspace-examples').hidden = !Array.from(document.querySelectorAll('.prepared-grid article')).some(element => !element.hidden);
    loading = false; $('workspace-refresh').disabled = false;
  }
  $('workspace-filter').addEventListener('input',()=>{
    clearTimeout(searchTimer);++requestNumber;
    savedPending('Searching saved work…');
    searchTimer=setTimeout(()=>loadSaved(0),250);
  });
  $('workspace-kind').addEventListener('change',()=>loadSaved(0));
  $('workspace-show-examples').addEventListener('change',()=>loadSaved(0));
  $('workspace-previous').addEventListener('click',()=>{if(!$('workspace-previous').disabled&&page?.previous_offset!=null)loadSaved(page.previous_offset);});
  $('workspace-next').addEventListener('click',()=>{if(!$('workspace-next').disabled&&page?.next_offset!=null)loadSaved(page.next_offset);});
  $('workspace-refresh').addEventListener('click',()=>{clearTimeout(searchTimer);load(true);loadBenchmark();});
  loadBenchmark();
  load().catch(() => { updateGuideAvailability(null); $('workspace-status').textContent='The workspace could not be loaded. Refresh to try again.'; loading=false; $('workspace-refresh').disabled=false; });
  if (!guideDismiss.checked && !guideDismissedThisSession) openGuide();
})();
