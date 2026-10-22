'use strict';

// The collection remains explicit in every URL and storage identity. A missing
// collection index never falls back to the prepared example library.
const semanticParams = new URLSearchParams(location.search);
const semanticCollection = semanticParams.get('collection');
const semanticBase = semanticParams.has('collection') ? 'api/collections/'+encodeURIComponent(semanticCollection)+'/semantic' : 'api/semantic';
const semanticURL = values => {
  const params = new URLSearchParams(values || {});
  if (semanticCollection) params.set('collection',semanticCollection);
  if (semanticCollection && !params.has('prediction_version') && state.predictionVersion) params.set('prediction_version',state.predictionVersion);
  return 'semantic'+(params.size ? '?'+params : '');
};
const element = id => document.getElementById(id);
const labels = {ADI:'Adipose',BACK:'Background',DEB:'Debris',LYM:'Lymphocytes',MUC:'Mucus',MUS:'Muscle',NORM:'Normal mucosa',STR:'Stroma',TUM:'Tumor'};
const state = {
  gallery:null, galleryLoading:null, galleryExpanded:false, summary:null, feedbackAvailable:false, search:null, positive:new Set(), negative:new Set(),
  evidence:new Set(), regionEvidence:new Set(), regions:new Map(), regionQuery:null, library:new Map(), claims:[], parentId:null, busy:false, savingDescription:false, savingConcept:false, simpleFinding:Boolean(semanticCollection),
  predictionVersion:semanticCollection ? semanticParams.get('prediction_version') : null, imageQuery:null
};

// Recovery copies stay in this browser. Server records are saved explicitly and
// remain immutable; restoring a draft never submits a search or an annotation.
const semanticDraft = (() => {
  const schema = 'semantic-browser-draft-v1';
  const writer = globalThis.crypto?.randomUUID?.() || Date.now()+'-'+Math.random();
  const fields = ['search-text','description-text','description-reviewer','concept-name','concept-reviewer','concept-note'];
  let key = null, identity = null, revision = null, signature = null, timer = null;
  let ready = false, restoring = false, unsaved = false, conflict = false, earlyEdits = false;
  let status, controls, options, restoreButton, keepButton, downloadButton, discardButton, otherDrafts, otherDraftLinks;
  const context = (() => {
    const params = new URLSearchParams(location.search);
    if (params.has('description')) return 'description:'+params.get('description');
    if (params.has('concept')) return 'concept:'+params.get('concept');
    const version=params.get('prediction_version');
    return semanticCollection && version && version!=='original' ? 'new:'+version : 'new';
  })();
  const scratch=context==='new' || context.startsWith('new:');
  function mount() {
    status = element('semantic-draft-status') || make('p',undefined,'scope-note');
    status.id = 'semantic-draft-status'; status.setAttribute('role','status');
    const session = make('div',undefined,'semantic-session');
    session.id='semantic-session';
    element('semantic-scope').after(session);
    session.append(status);
    controls = make('div',undefined,'actions'); controls.id = 'semantic-draft-actions';
    downloadButton = make('button','Download browser draft'); downloadButton.type = 'button';
    downloadButton.onclick = download;
    restoreButton = make('button','Use other tab’s draft'); restoreButton.type = 'button';
    restoreButton.onclick = () => act(async () => {
      const value = read();
      if (!value) throw new Error('The other browser draft is no longer available. Download this tab’s draft before leaving.');
      await restore(value); conflict = false; unsaved = false; show('Recovered the other tab’s browser draft. Server records are unchanged.');
    });
    keepButton = make('button','Keep this tab’s draft'); keepButton.type = 'button';
    keepButton.onclick = () => { conflict = false; signature = null; save(true); };
    discardButton = make('button',scratch ? 'Discard browser draft' : 'Discard local edits and open saved version');
    discardButton.type='button'; discardButton.id='semantic-discard-draft';
    discardButton.onclick=()=>{
      const prompt=scratch
        ? 'Discard this browser draft and start fresh? Shared server records will be kept. Download a copy first if you want to retain these local edits.'
        : 'Discard these local edits and reopen the originally opened saved version? All shared server records will be kept. Download a copy first if you want to retain the local edits.';
      if (!window.confirm(prompt)) return;
      try {
        localStorage.removeItem(key); clearTimeout(timer);
        ready=false; unsaved=false; conflict=false; location.reload();
      } catch (_) {
        show('The browser would not remove this draft. Your page is unchanged. Download a copy before leaving.','warning');
      }
    };
    controls.append(downloadButton,discardButton,restoreButton,keepButton);
    options=make('details',undefined,'semantic-details'); options.id='semantic-draft-options';
    options.append(make('summary','Draft options'),controls); status.after(options);
    otherDrafts=make('details',undefined,'semantic-details'); otherDrafts.id='semantic-other-drafts'; otherDrafts.hidden=true;
    otherDraftLinks=make('div',undefined,'actions'); otherDraftLinks.id='semantic-other-draft-links';
    otherDrafts.append(make('summary','Other browser drafts'),otherDraftLinks); options.after(otherDrafts);
    show('Draft recovery is loading.');
  }
  function show(text, kind='ready') {
    if (!status) return;
    if (status.textContent!==text) status.textContent = text;
    status.dataset.state = kind;
    status.setAttribute('role',kind === 'warning' ? 'alert' : 'status');
    restoreButton.hidden = !conflict; keepButton.hidden = !conflict;
    discardButton.hidden = !key;
    downloadButton.hidden = !key && kind !== 'warning';
    controls.hidden = !key && kind !== 'warning';
    options.hidden=controls.hidden;
    element('semantic-session').classList.toggle('has-warning',kind==='warning');
    if (kind==='warning') options.open=true;
  }
  function snapshot() {
    return {
      fields:Object.fromEntries(fields.map(id => [id,element(id).value])),
      positive:[...state.positive], negative:[...state.negative], evidence:[...state.evidence], regionEvidence:[...state.regionEvidence],
      claims:state.claims.map(claim => ({...claim,image_ids:[...claim.image_ids],region_ids:[...(claim.region_ids||[])]})),
      parentId:state.parentId, searchId:state.search?.id || null,
      simpleFinding:state.simpleFinding,
      predictionVersion:state.predictionVersion,
      imageQueryId:state.imageQuery?.id || null, regionQueryId:state.regionQuery?.id || null,
      panel:element('describe-panel').hidden ? 'find' : 'describe', galleryExpanded:state.galleryExpanded,
      guidance:window.TaskGuidance?.draft?.() || null
    };
  }
  function valid(value) {
    const data = value?.draft;
    const ids = (values,limit) => Array.isArray(values) && values.length<=limit && values.every(id => typeof id==='string' && /^[A-Za-z0-9_-]{1,200}$/.test(id));
    const text = (v,max) => typeof v==='string' && v.length<=max;
    const regions = values => values===undefined || (Array.isArray(values) && (!values.length || Boolean(semanticCollection)) && values.length<=8 && values.every(id=>/^[a-f0-9]{64}$/.test(id)));
    return value?.schema===schema && value.identity===identity && typeof value.revision==='string' && data &&
      fields.every(id => text(data.fields?.[id],id==='description-text' ? 6000 : 2000)) &&
      ids(data.positive,40) && ids(data.negative,40) && ids(data.evidence,8) && regions(data.regionEvidence) && data.evidence.length+(data.regionEvidence?.length||0)<=8 &&
      Array.isArray(data.claims) && data.claims.length<=20 && data.claims.every(claim =>
        text(claim.text,1000) && text(claim.note,2000) && ids(claim.image_ids,8) && regions(claim.region_ids) && claim.image_ids.length+(claim.region_ids?.length||0)<=8 &&
        ['unresolved','supported_by_reviewer','contradicted_by_reviewer'].includes(claim.assessment)) &&
      (data.parentId===null || ids([data.parentId],1)) && (data.searchId===null || ids([data.searchId],1)) &&
      (data.imageQueryId==null || Boolean(semanticCollection && /^[a-f0-9]{32}$/.test(data.imageQueryId))) &&
      (data.regionQueryId==null || Boolean(semanticCollection && /^[a-f0-9]{64}$/.test(data.regionQueryId) && !data.imageQueryId)) &&
      (data.predictionVersion==null || /^(original|[a-f0-9]{32})$/.test(data.predictionVersion));
  }
  function read() {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const value = JSON.parse(raw);
    if (!valid(value)) throw new Error('The stored browser draft could not be read safely. It has not been overwritten. Download your current work before leaving.');
    return value;
  }
  function listOtherDrafts() {
    if (!identity) return;
    const prefix='tissue-study-studio:semantic-draft:'+encodeURIComponent(identity)+':';
    const rows=[];
    try {
      for (let i=0;i<localStorage.length;i++) {
        const candidate=localStorage.key(i);
        if (!candidate?.startsWith(prefix) || candidate===key) continue;
        try {
          const value=JSON.parse(localStorage.getItem(candidate));
          if (valid(value) && typeof value.context==='string') rows.push(value);
        } catch (_) { /* Do not erase an unreadable recovery copy. */ }
      }
    } catch (_) { return; }
    rows.sort((a,b)=>String(b.updatedAt).localeCompare(String(a.updatedAt)));
    otherDraftLinks.replaceChildren();
    for (const row of rows.slice(0,8)) {
      const parts=row.context.split(':');
      const rowScratch=parts[0]==='new';
      if (!rowScratch && !['description','concept'].includes(parts[0])) continue;
      const link=make('a',rowScratch ? (semanticCollection ? 'Resume collection draft' : 'Resume image-library draft') :
        (parts[0]==='description' ? 'Resume description: ' : 'Resume named search: ')+
        (parts[0]==='description' ? row.draft.fields['description-text'] : row.draft.fields['concept-name']).slice(0,70));
      link.href=semanticURL({...(!rowScratch ? {[parts[0]]:parts.slice(1).join(':')} : {}),
        ...(semanticCollection ? {prediction_version:row.draft.predictionVersion || 'original'} : {})});
      link.onclick=()=>save();
      otherDraftLinks.append(link);
    }
    otherDrafts.hidden=!otherDraftLinks.childElementCount;
  }
  function save(force=false) {
    clearTimeout(timer);
    if (!ready || restoring || !key) return false;
    const draft = snapshot(), current = JSON.stringify(draft);
    if (!force && current===signature) { unsaved=false; return true; }
    unsaved = true;
    if (conflict && !force) {
      show('Another tab has a newer browser draft. Download this tab’s work, then choose which draft to keep. Nothing has been overwritten.','warning');
      return false;
    }
    try {
      const previous = read();
      if (!force && previous && previous.revision!==revision && previous.writer!==writer) {
        conflict = true; return save();
      }
      const next = writer+':'+Date.now()+':'+Math.random();
      localStorage.setItem(key,JSON.stringify({schema,identity,context,writer,revision:next,updatedAt:new Date().toISOString(),draft}));
      revision = next; signature = current; unsaved = false;
      listOtherDrafts();
      show(semanticCollection ? 'Draft saved on this device.' : 'Draft saved in this browser. Save a server version to share.');
      return true;
    } catch (_) {
      show('Browser recovery is unavailable. Your latest changes are only on this page. Download your browser draft or save a server record before leaving.','warning');
      return false;
    }
  }
  function schedule() {
    if (restoring) return;
    unsaved = true; clearTimeout(timer); timer = setTimeout(save,150);
  }
  async function restore(value) {
    if (!valid(value)) throw new Error('This browser draft belongs to a different image library or could not be read.');
    restoring = true;
    try {
      const data=value.draft;
      for (const id of fields) element(id).value=data.fields[id];
      state.positive=new Set(data.positive); state.negative=new Set(data.negative); state.evidence=new Set(data.evidence);
      state.regionEvidence=new Set(data.regionEvidence||[]);
      state.claims=data.claims.map(claim => ({...claim,image_ids:[...claim.image_ids],region_ids:[...(claim.region_ids||[])]}));
      state.simpleFinding=Boolean(semanticCollection && (data.simpleFinding ?? !state.claims.length));
      if(element('finding-statements'))element('finding-statements').open=!state.simpleFinding;
      state.parentId=data.parentId; state.galleryExpanded=Boolean(data.galleryExpanded);
      if(semanticCollection)state.predictionVersion=data.predictionVersion || 'original';
      state.imageQuery=null;state.regionQuery=null;
      await loadRegionEvidence([...state.regionEvidence,...state.claims.flatMap(claim=>claim.region_ids||[])]);
      if(data.regionQueryId){
        try{state.regionQuery=await resolveRegion(data.regionQueryId);}
        catch(_){state.regionQuery={id:data.regionQueryId,filename:'Selected region',unavailable:true};}
      }
      if(data.imageQueryId){
        try{state.imageQuery=await resolveImageReference(data.imageQueryId);}
        catch(_){
          // A failed read must not let the deep-link loader replace recovered
          // local words/evidence with an older saved record. Retain the source
          // identity too; explicit retry or text mode is required before search.
          state.imageQuery={id:data.imageQueryId,filename:'Image '+data.imageQueryId,image_url:semanticBase+'/images/'+encodeURIComponent(data.imageQueryId),unavailable:true};
        }
      }
      await window.TaskGuidance?.restoreDraft?.(data.guidance);
      state.search=null;
      element('search-summary').textContent='';
      element('search-query').hidden=true;
      element('search-notices').replaceChildren();
      element('search-more-notices').replaceChildren();
      element('search-details').hidden=true;
      element('search-interpretation').hidden=true;
      element('description-export').hidden=true;
      element('description-report').hidden=true;
      element('description-saved-result').hidden=true;
      element('description-version').textContent='Saving preserves your wording, image evidence and judgments as a new version.';
      if (data.searchId) {
        try { showSearch(await api('/searches/'+encodeURIComponent(data.searchId))); }
        catch (_) { message('Recovered your words, image choices and claims. The previous search result is unavailable; search explicitly when ready.'); }
      }
      if (state.parentId) {
        element('description-version').textContent='Browser draft based on saved version '+state.parentId.slice(0,8)+'. Saving creates a new server version.';
        element('description-export').href=semanticBase+'/descriptions/'+encodeURIComponent(state.parentId)+'/export';
        element('description-export').hidden=false;
        showDescriptionReport(state.parentId);
      }
      renderChoices(); renderResults(); renderEvidence(); renderClaims(); tab(data.panel==='describe' ? 'describe' : 'find');
      if(semanticCollection)renderCollectionContext();
      revision=value.revision; signature=JSON.stringify(snapshot()); unsaved=false;
    } finally { restoring=false; }
  }
  function download() {
    const record={schema,identity,context,exportedAt:new Date().toISOString(),note:'Browser recovery copy, not a shared server record or verified finding',draft:snapshot()};
    const url=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)],{type:'application/json'}));
    const link=make('a'); link.href=url; link.download='tissue-browser-draft.json'; link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  async function initialize() {
    const catalog=state.gallery?.catalog_sha256;
    if (!catalog || !state.summary?.model) {
      show('Browser recovery is unavailable until this image library is identified. Save a server record or download your draft before leaving.','warning');
      return false;
    }
    identity=JSON.stringify({...((window.TissueLab?.memberId())?{member:window.TissueLab.memberId()}:{}),endpoint:new URL(semanticBase,document.baseURI).href,catalog,model:state.summary.model,retrieval:state.gallery.retrieval_version || null});
    key='tissue-study-studio:semantic-draft:'+encodeURIComponent(identity)+':'+encodeURIComponent(context);
    ready=true;
    try {
      listOtherDrafts();
      const saved=read();
      if (saved && earlyEdits) {
        conflict=true; unsaved=true;
        show('A browser draft was found after you started editing. Your typing has been kept on this page. Download a copy, then choose which draft to keep.','warning');
        return true;
      }
      if (saved) { await restore(saved); show(semanticCollection ? 'Draft restored on this device. Save your finding to share it.' : 'Browser draft recovered. No search run or server save. Guidance is not reapplied.'); return true; }
      // Persist input entered while the gallery/library identity was loading.
      // An untouched deep link must first load its server record, so a temporary
      // GET failure cannot leave an empty draft that masks that record on reload.
      if (earlyEdits) { signature=null; unsaved=true; save(); }
      else {
        signature=JSON.stringify(snapshot()); unsaved=false;
        show(semanticCollection ? 'Drafts are kept on this device.' : 'Drafts recover in this browser. Save a server version to share.');
      }
      if (earlyEdits && !scratch) {
        message('Your typing was preserved. Discard local edits to reopen the saved version linked here.');
        return true;
      }
    } catch (error) {
      // A corrupt/blocked store is not an empty store. Preserve it and warn.
      ready=false; unsaved=true; show('Browser recovery is unavailable. Your work stays on this page; download your browser draft or save a server record before leaving. Existing recovery copies have not been overwritten.','warning');
    }
    return false;
  }
  mount();
  document.addEventListener('input',event=>{if(event.target.closest('.semantic-main')) {if(!ready) earlyEdits=true;schedule();}},true);
  document.addEventListener('change',event=>{if(event.target.closest('.semantic-main')) {if(!ready) earlyEdits=true;schedule();}},true);
  window.addEventListener('pagehide',()=>save());
  document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='hidden') save();});
  window.addEventListener('beforeunload',event=>{save();if(unsaved || conflict){event.preventDefault();event.returnValue='';}});
  window.addEventListener('storage',event=>{
    if (ready) listOtherDrafts();
    if (!ready || event.key!==key) return;
    if (event.newValue===null) {
      conflict=true; unsaved=true;
      show('Another tab discarded this browser draft. Your page is unchanged. Download your work or keep this tab’s draft to save it again.','warning');
      restoreButton.hidden=true;
      return;
    }
    try {
      const value=JSON.parse(event.newValue);
      if (valid(value) && value.writer!==writer && value.revision!==revision) {
        conflict=true; unsaved=true;
        show('Another tab has a newer browser draft. Download this tab’s work, then choose which draft to keep. Nothing has been overwritten.','warning');
      }
    } catch (_) { /* Existing state remains available for download. */ }
  });
  return {initialize,schedule,save,context};
})();

function make(tag, text, className) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
  };
  const response = await fetch(semanticBase + path, options);
  const value = await response.json();
  if (!response.ok) {
    const detail = value.detail;
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail)
      ? detail.map(item => item.msg).join('; ') : 'The request could not be completed.');
  }
  return value;
}

async function act(work) {
  element('semantic-error').hidden = true;
  try { await work(); }
  catch (error) {
    element('semantic-error').textContent = error.message;
    element('semantic-error').hidden = false;
    element('semantic-error').scrollIntoView({block:'nearest'});
  }
}

function message(text) { element('semantic-message').textContent = text; }

async function resolveImageReference(id) {
  if(!semanticCollection || !state.summary?.image_search_available || !/^[a-f0-9]{32}$/.test(id))throw new Error('Choose an image from a scanned collection to find similar images. Your draft is kept.');
  const response=await fetch('api/collections/'+encodeURIComponent(semanticCollection)+'/images/'+encodeURIComponent(id)+'?prediction_version='+encodeURIComponent(state.predictionVersion || 'original'));
  if(!response.ok)throw new Error('That source image or prediction version is unavailable in this collection. Your existing draft is kept.');
  const row=await response.json();
  if(row.id!==id || row.selected_prediction_version?.id!==(state.predictionVersion || 'original'))throw new Error('The source image does not match this collection and prediction version. Your existing draft is kept.');
  const reference={id,filename:row.filename,image_url:semanticBase+'/images/'+encodeURIComponent(id)};
  state.library.set(id,reference);return reference;
}

function evidenceCount() { return state.evidence.size+state.regionEvidence.size; }
function querySourceId() { return state.regionQuery?.image_id || state.imageQuery?.id || null; }
function regionCoordinates(region) {
  if(!region?.bounds_px || !region.source_size_px)return 'Source bounds unavailable';
  const [x0,y0,x1,y1]=region.bounds_px;
  return (x1-x0)+' × '+(y1-y0)+' px at ('+x0+', '+y0+') in '+region.source_size_px[0]+' × '+region.source_size_px[1]+' px source';
}
function checkRegion(record,identifier=record?.id) {
  const bounds=record?.bounds_px,size=record?.source_size_px;
  if(!semanticCollection || !/^[a-f0-9]{64}$/.test(identifier||'') || record?.id!==identifier || record.collection_id!==semanticCollection || !/^[a-f0-9]{32}$/.test(record.image_id||'') || record.prediction_version?.id!==(state.predictionVersion||'original') || !Array.isArray(bounds) || bounds.length!==4 || !bounds.every(Number.isInteger) || !Array.isArray(size) || size.length!==2 || !size.every(value=>Number.isInteger(value)&&value>0) || bounds[0]<0 || bounds[1]<0 || bounds[2]>size[0] || bounds[3]>size[1] || bounds[2]-bounds[0]<32 || bounds[3]-bounds[1]<32)throw new Error('This region does not match the collection, source pixels, or prediction version. Your draft is kept.');
  for(const value of [record.image_url,record.source_image_url]){if(typeof value!=='string'||!value)throw new Error('The region preview is unavailable.');const url=new URL(value,document.baseURI);if(url.origin!==location.origin || !['http:','https:'].includes(url.protocol))throw new Error('The region preview could not be loaded safely.');}
  state.regions.set(record.id,record);return record;
}
async function resolveRegion(identifier) { return checkRegion(await api('/regions/'+encodeURIComponent(identifier)),identifier); }
async function loadRegionEvidence(ids) {
  const unique=[...new Set(ids)];if(!unique.length)return;
  await Promise.all(unique.map(async id=>{try{await resolveRegion(id);}catch(_){state.regions.set(id,{id,unavailable:true,filename:'Unavailable region'});}}));
}
async function createRegion(imageId,bounds) {
  if(!semanticCollection || state.summary?.region_evidence_available!==true)throw new Error('Region selection is unavailable for this collection.');
  return checkRegion(await api('/regions',{image_id:imageId,prediction_version:state.predictionVersion||'original',bounds_px:bounds}));
}
async function startRegionSearch(identifier,run=false) {
  if(state.busy)throw new Error('Wait for the current search before changing its selected region.');
  const record=await resolveRegion(identifier);
  state.regionQuery=record;state.imageQuery=null;state.positive.clear();state.negative.clear();
  window.TaskGuidance?.clearApplied();element('search-text').value='';selectPanel('find');renderChoices();renderResults();refreshInspection();semanticDraft.schedule();
  message('Selected region ready. Search compares this crop with whole images in this collection.');
  semanticDraft.save();
  if(run)await search();
}
function addRegionEvidence(region) {
  checkRegion(region);
  if(!state.regionEvidence.has(region.id)&&evidenceCount()>=8)throw new Error('Select up to 8 images or regions for a finding. Remove one before adding another.');
  state.regionEvidence.add(region.id);renderEvidence();renderClaims();semanticDraft.schedule();
}
function regionThumbnail(id,remove,context='region-evidence') {
  const record=state.regions.get(id),item=make('div',undefined,'evidence-item region-evidence');item.dataset.regionEvidence=id;
  if(record&&!record.unavailable){
    const image=make('img');image.src=record.image_url;image.alt='Selected region from '+record.filename;
    const inspect=make('button',undefined,'semantic-inspect-image');inspect.type='button';inspect.setAttribute('aria-label','Inspect selected region from '+record.filename);inspect.append(image);inspect.onclick=()=>openInspection(record.image_id,inspect,context,record);item.append(inspect,make('span','Region · '+(record.bounds_px[2]-record.bounds_px[0])+' × '+(record.bounds_px[3]-record.bounds_px[1])+' px','region-evidence-label'));item.title=regionCoordinates(record);
  } else item.append(make('p','Region preview unavailable','scope-note'));
  if(remove){const button=make('button','Remove region');button.type='button';button.dataset.removeRegionEvidence=id;button.setAttribute('aria-label','Remove region from '+(record?.filename||'source image')+' from selected evidence');button.onclick=remove;item.append(button);}
  return item;
}
element('use-region-query').onclick=()=>act(async()=>{if(state.regionQuery&&!state.regionQuery.unavailable){addRegionEvidence(state.regionQuery);selectPanel('describe');message('Selected region added to your finding.');}});
element('region-query-source').onclick=()=>{if(state.regionQuery&&!state.regionQuery.unavailable)openInspection(state.regionQuery.image_id,element('region-query-source'),'region-query',state.regionQuery);};

async function startImageSearch(id) {
  if(state.busy)throw new Error('Wait for the current search to finish before changing its source image.');
  const reference=await resolveImageReference(id);
  state.regionQuery=null;state.imageQuery=reference;state.positive.clear();state.negative.clear();
  window.TaskGuidance?.clearApplied();element('search-text').value='';
  tab('find');renderChoices();renderResults();refreshInspection();
  message('Source image ready. Choose Search similar images. Previous relevance choices cleared; your finding and selected evidence are kept.');
  semanticDraft.schedule();
}

function renderImageQuery() {
  const reference=state.regionQuery||state.imageQuery,panel=element('image-query');panel.hidden=!reference;
  const entry=element('search-form').closest('.search-entry');
  entry.classList.toggle('has-results',Boolean(state.search));
  entry.classList.toggle('has-text-results',Boolean(state.search) && !reference);
  entry.classList.toggle('has-current-region-results',Boolean(state.regionQuery && !reference?.unavailable && state.search?.region_query?.id===state.regionQuery.id));
  // Example phrases start a separate text query. Keep them out of image mode,
  // including a source awaiting retry; absent metadata must stay absent too.
  element('search-starters').hidden=Boolean(reference) || !element('search-starter-buttons').childElementCount;
  if(reference){
    const picture=element('image-query-picture');
    picture.hidden=Boolean(reference.unavailable);
    if(!reference.unavailable && picture.getAttribute('src')!==reference.image_url)picture.src=reference.image_url;
    picture.alt=(state.regionQuery?'Selected region from ':'Search source: ')+reference.filename;element('image-query-name').textContent=reference.filename;
  }
  element('image-query-kicker').textContent=state.regionQuery?'Selected region':'Find similar';
  element('use-region-query').hidden=!state.regionQuery || Boolean(reference?.unavailable);
  element('use-region-query').disabled=state.busy;
  const context=element('region-query-context');context.hidden=!state.regionQuery || Boolean(reference?.unavailable);
  if(state.regionQuery&&!reference.unavailable){
    element('region-query-bounds').textContent=regionCoordinates(reference);
    element('region-query-source-picture').src=reference.source_image_url;
    element('region-query-source-picture').alt='Source image: '+reference.filename;
  }
  element('image-query-note').textContent=reference?.unavailable
    ? 'Your draft is restored, but its source image could not be checked. Retry the image or switch to text search. Your words and evidence are kept.'
    : state.regionQuery ? 'Search uses your selected crop. Results are whole images; matching areas inside them are not located.' : 'This image starts the search. It is not a review or selected evidence.';
  element('retry-image-query').hidden=!reference?.unavailable;
  element('retry-image-query').disabled=state.busy;
  element('search-text').required=!reference;
  element('search-text-label').textContent=reference?'Add a description (optional)':'Describe what you want to see';
  element('search-text').placeholder=reference?'Leave blank to search by this image alone':'Try a tissue name, color, shape, or texture…';
  if(!state.busy)element('search-button').textContent=state.regionQuery?'Search this region':reference?'Search similar images':'Search images';
  element('search-button').disabled=state.busy || !state.summary?.available || Boolean(reference?.unavailable) || Boolean(state.regionQuery && state.summary.region_search_available!==true);
  element('clear-image-query').disabled=state.busy;
  element('search-limit-note').textContent=state.regionQuery
    ? 'The first search may take a moment. Keep this page open; later searches reuse the saved region.'
    : reference
    ? 'Matches use the source image. Optional words add a smaller text contribution; matching and nonmatching examples together replace this ranking with your feedback.'
    : 'Use a tissue name or describe its appearance. Refine suggested matches with examples.';
}

element('clear-image-query').onclick=()=>{
  if(state.busy)return;
  state.imageQuery=null;state.regionQuery=null;state.positive.clear();state.negative.clear();renderChoices();renderResults();refreshInspection();
  message('Text search ready. Your finding and selected evidence are kept.');element('search-text').focus();
};
element('retry-image-query').onclick=()=>act(async()=>{
  if(state.busy || (!state.imageQuery&&!state.regionQuery))return;
  if(state.regionQuery){const id=state.regionQuery.id,reference=await resolveRegion(id);if(state.regionQuery?.id!==id)return;state.regionQuery=reference;renderChoices();message('Region restored. Your draft and image evidence are kept.');return;}
  const id=state.imageQuery.id,reference=await resolveImageReference(id);
  if(state.imageQuery?.id!==id)return;
  state.imageQuery=reference;renderChoices();
  message('Source image restored. Your query, relevance choices, finding and evidence are kept.');
});

function chooseSearchStarter(starter) {
  if(state.busy){message('Wait for the current search to finish before choosing another example.');return;}
  if(!state.summary?.available){message(state.summary?.reason || 'Search is unavailable for this image collection.');return;}
  const feedbackCount=state.positive.size+state.negative.size;
  const stoppedGuidance=Boolean(window.TaskGuidance?.clearApplied());
  state.imageQuery=null;state.regionQuery=null;
  state.positive.clear();state.negative.clear();
  element('search-text').value=starter.query;
  renderChoices();renderResults();refreshInspection();
  const changes=[];
  if(feedbackCount)changes.push('Previous matching and nonmatching choices cleared.');
  if(stoppedGuidance)changes.push('Previous guidance stopped; its draft is kept.');
  if(evidenceCount())changes.push('Your selected finding evidence is kept.');
  message('Example loaded. Edit the wording or choose Search images. '+changes.join(' '));
  element('search-text').focus();
}

async function loadSearchStarters() {
  const panel=element('search-starters'),buttons=element('search-starter-buttons');
  try{
    const response=await fetch('static/search-starters.json?v=1');
    if(!response.ok)throw new Error('Example searches unavailable');
    const value=await response.json();
    const bounded=(text,limit)=>typeof text==='string' && text.trim().length>0 && text.length<=limit;
    const rows=value.starters;
    if(value.schema!=='tissue-search-starters-v1' || !Array.isArray(rows) || rows.length!==3 ||
      new Set(rows.map(row=>row?.id)).size!==rows.length || !rows.every(row=>row &&
        typeof row.id==='string' && /^[a-z0-9_-]{1,60}$/.test(row.id) &&
        bounded(row.title,80) && bounded(row.term,100) && bounded(row.description,400) &&
        bounded(row.query,state.summary?.query_limit || 600) && bounded(row.target,20))){
      throw new Error('Example searches could not be read');
    }
    buttons.replaceChildren(...rows.map(starter=>{
      const button=make('button');button.type='button';button.dataset.searchStarter=starter.id;
      button.title=starter.description;
      button.setAttribute('aria-describedby','search-starters-hint');
      button.append(make('strong',starter.title),make('span',starter.term));
      button.onclick=()=>chooseSearchStarter(starter);
      return button;
    }));
    renderChoices();
  }catch(_){
    // Optional example metadata must never block manual search or restore an old
    // list of weaker raw suggestions. Loading it does not submit a query.
    panel.hidden=true;buttons.replaceChildren();
  }
}

function showDescriptionReport(identifier) {
  const link=element('description-report');
  link.href='report?kind=description&id='+encodeURIComponent(identifier)+(semanticCollection ? '&collection='+encodeURIComponent(semanticCollection) : '');
  link.hidden=false;
  element('description-saved-result').hidden=false;
  element('description-saved-label').textContent='Saved version '+identifier.slice(0,8);
  if(!state.savingDescription)element('save-description').textContent='Save new version';
}

const descriptionReport=make('a','Open report','finding-report-link');
descriptionReport.id='description-report'; descriptionReport.hidden=true;
descriptionReport.title='Shows the last saved description. Save your edits to include them in a new report.';
const descriptionSaved=make('section',undefined,'finding-saved-result');
descriptionSaved.id='description-saved-result';descriptionSaved.hidden=true;descriptionSaved.setAttribute('aria-label','Saved finding');
const descriptionSavedLabel=make('p');descriptionSavedLabel.id='description-saved-label';
const descriptionSavedActions=make('div',undefined,'actions');
element('description-version').before(descriptionSaved);
descriptionSavedActions.append(descriptionReport,element('description-export'));
descriptionSaved.append(descriptionSavedLabel,descriptionSavedActions,make('p','The report shows this saved version. Save again to include later edits.','scope-note'));

function tab(name) {
  element('find-panel').hidden = name !== 'find';
  element('describe-panel').hidden = name !== 'describe';
  element('find-tab').setAttribute('aria-pressed', String(name === 'find'));
  element('describe-tab').setAttribute('aria-pressed', String(name === 'describe'));
  if (name === 'describe') renderEvidence();
  element('semantic-title').textContent = name === 'describe' ? 'Findings' : 'Search images';
  semanticDraft.schedule();
}

// Explicit task changes belong in the URL so a reload reopens the visible panel.
// Restoration and incoming deep links use tab() directly and never rewrite context.
function selectPanel(name) {
  tab(name);
  const url = new URL(location.href);
  url.searchParams.set('panel',name);
  history.replaceState(history.state,'',url);
}

function updateSelectionToolbar() {
  element('selection-toolbar').hidden = !evidenceCount() && !state.positive.size && !state.negative.size;
}

async function loadGallery() {
  if (state.gallery || !state.summary?.gallery_available) return;
  if (state.galleryLoading) return state.galleryLoading;
  state.galleryLoading = (async () => {
    element('gallery-message').textContent = 'Loading varied image examples…';
    state.gallery = await api('/gallery');
    for (const row of state.gallery.results) state.library.set(row.id,row);
    renderResults();
    element('gallery-message').textContent = '';
  })();
  try { await state.galleryLoading; }
  finally { state.galleryLoading = null; }
}

function imageFor(id) {
  const known = state.library.get(id);
  return known?.image_url || semanticBase+'/images/' + encodeURIComponent(id);
}

const inspection = {id:null,trigger:null,context:null,region:null,selector:null,request:0,loaded:false};

// A failed preview must not silently become a visual judgment. A previous
// choice may always be removed, including when its image is unavailable.
function loadJudgmentImage(image,status,url,update) {
  const load=(retry=false)=>{
    const token=Symbol();image.visualLoadToken=token;image.visualReady=false;
    image.hidden=false;status.textContent='Loading image…';update(false);
    const failed=()=>{
      if(image.visualLoadToken!==token)return;
      image.visualReady=false;image.hidden=true;update(false);
      const button=make('button','Retry image');button.type='button';button.dataset.retryImage='';button.onclick=()=>load(true);
      status.replaceChildren(document.createTextNode('Image unavailable. New judgments are paused. '),button);
    };
    image.onload=async()=>{
      try {await image.decode();if(image.visualLoadToken!==token)return;image.visualReady=true;status.textContent='';update(true);}
      catch(_){failed();}
    };
    image.onerror=failed;
    const source=new URL(url,location.href);if(retry)source.searchParams.set('_image_retry',String(Date.now()));
    image.src=source.href;
  };
  load();
}

function judgmentImageReady(id) {
  if(inspection.id===id && element('semantic-image-dialog').open)return inspection.loaded;
  return [...document.querySelectorAll('.semantic-tile')].some(card=>card.dataset.imageId===id&&card.querySelector('img')?.visualReady===true);
}

function inspectionButton(id, context, image) {
  const button=make('button',undefined,'semantic-inspect-image');button.type='button';
  button.dataset.inspectImage=id;button.dataset.inspectContext=context;
  button.setAttribute('aria-label','Inspect '+(state.library.get(id)?.filename || 'image '+id));
  button.title='Inspect image';button.append(image);
  button.onclick=()=>openInspection(id,button,context);
  return button;
}

function inspectionMessage(text) {
  if(element('semantic-image-dialog').open)element('semantic-image-status').textContent=text;
  else message(text);
}

function refreshInspection() {
  if(!inspection.id || !element('semantic-image-dialog').open)return;
  const id=inspection.id;
  document.querySelector('.semantic-image-actions').hidden=Boolean(inspection.region);
  element('semantic-image-relevant').disabled=id===querySourceId()||(!inspection.loaded&&!state.positive.has(id));
  element('semantic-image-negative').disabled=id===querySourceId()||(!inspection.loaded&&!state.negative.has(id));
  element('semantic-image-evidence').disabled=!inspection.loaded&&!state.evidence.has(id);
  element('semantic-image-similar').hidden=!state.summary?.image_search_available || Boolean(inspection.region);
  element('semantic-image-similar').disabled=state.busy || id===querySourceId() || !inspection.loaded;
  element('semantic-image-relevant').setAttribute('aria-pressed',String(state.positive.has(id)));
  element('semantic-image-negative').setAttribute('aria-pressed',String(state.negative.has(id)));
  element('semantic-image-evidence').setAttribute('aria-pressed',String(state.evidence.has(id)));
  element('semantic-image-evidence').textContent=state.evidence.has(id)?'Evidence selected':'Select as evidence';
  element('semantic-image-status').textContent=inspection.region ? 'Selected region · '+regionCoordinates(inspection.region) : [state.positive.has(id)?'Relevant example':state.negative.has(id)?'Not a match':'No search feedback for this image',state.evidence.has(id)?'Selected for your finding':'Not selected as evidence'].join(' · ');
}

function openInspection(id,trigger,context,region=null) {
  inspection.selector?.destroy();inspection.selector=null;inspection.request++;
  inspection.id=id;inspection.trigger=trigger;inspection.context=context;inspection.region=region;inspection.loaded=false;
  element('semantic-region-tools').hidden=true;element('semantic-region-tools').replaceChildren();
  const row=state.library.get(id),picture=element('semantic-image-picture'),load=element('semantic-image-load');
  element('semantic-image-heading').textContent=region?.filename || row?.filename || 'Image '+id;
  element('semantic-image-id').textContent=id;
  picture.alt=region?.filename || row?.filename || 'Selected tissue image '+id;picture.hidden=false;
  element('semantic-image-dialog').showModal();refreshInspection();element('semantic-image-close').focus();
  const request=inspection.request;
  loadJudgmentImage(picture,load,region?.source_image_url || imageFor(id),ready=>{if(request===inspection.request){inspection.loaded=ready;refreshInspection();}});
  void mountInspectionRegion(inspection.request,id,picture,region);
}

async function mountInspectionRegion(request,id,picture,region) {
  if(!semanticCollection || !state.summary?.region_evidence_available || !window.TissueRegionSelector)return;
  const version=state.predictionVersion||'original';
  try{
    const response=await fetch('api/collections/'+encodeURIComponent(semanticCollection)+'/images/'+encodeURIComponent(id)+'?prediction_version='+encodeURIComponent(version));
    if(!response.ok)throw new Error('Source image details are unavailable for region selection.');
    const source=await response.json();
    if(request!==inspection.request || inspection.id!==id || !element('semantic-image-dialog').open)return;
    if(source.id!==id || source.selected_prediction_version?.id!==version || version!==(state.predictionVersion||'original'))throw new Error('Source image and prediction version do not match.');
    const host=element('semantic-region-tools');host.hidden=false;
    const current=()=>request===inspection.request && inspection.id===id && version===(state.predictionVersion||'original') && element('semantic-image-dialog').open;
    inspection.selector=window.TissueRegionSelector.mount({host,image:picture,width:source.width,height:source.height,minimum:state.summary.region_min_size_px||32,maximum:state.summary.region_max_size_px||4096,initialBounds:region?.bounds_px,
      onSearch:state.summary.region_search_available===true?async bounds=>{
        if(!current())throw new Error('Open the source image again before selecting its region.');
        const saved=await createRegion(id,bounds);if(!current())return;
        element('semantic-image-dialog').close();await act(()=>startRegionSearch(saved.id,true));
      }:null,
      onEvidence:async bounds=>{
        if(!current())throw new Error('Open the source image again before selecting its region.');
        const saved=await createRegion(id,bounds);if(!current())return;addRegionEvidence(saved);selectPanel('describe');element('semantic-image-dialog').close();
        message('Selected region added to your finding. Save the finding to keep its description and evidence together.');
      }});
  }catch(error){if(request===inspection.request)element('semantic-image-status').textContent=error.message;}
}

element('semantic-image-close').onclick=()=>element('semantic-image-dialog').close();
element('semantic-image-dialog').addEventListener('close',()=>{
  const trigger=inspection.trigger?.isConnected?inspection.trigger:
    [...document.querySelectorAll('[data-inspect-image]')].find(node=>node.dataset.inspectImage===inspection.id&&node.dataset.inspectContext===inspection.context);
  (trigger || element(element('describe-panel').hidden?'find-tab':'describe-tab')).focus({preventScroll:true});
  inspection.selector?.destroy();inspection.selector=null;inspection.request++;
  inspection.id=null;inspection.trigger=null;inspection.context=null;inspection.region=null;
});
element('semantic-image-relevant').onclick=()=>toggleRelevance(inspection.id,true);
element('semantic-image-negative').onclick=()=>toggleRelevance(inspection.id,false);
element('semantic-image-evidence').onclick=()=>toggleEvidence(inspection.id);
element('semantic-image-similar').onclick=()=>act(async()=>{
  const id=inspection.id;await startImageSearch(id);selectPanel('find');element('semantic-image-dialog').close();element('image-query').scrollIntoView({block:'nearest'});element('search-button').focus();
});

function thumbnail(id, remove, context='selected-evidence') {
  const item = make('div', undefined, 'evidence-item');
  const image = make('img');
  image.src = imageFor(id);
  image.alt = 'Selected tissue image ' + id;
  item.append(inspectionButton(id,context,image));
  if (remove) {
    const button = make('button', 'Remove image');
    button.type = 'button';
    button.dataset.removeEvidence=id;
    button.setAttribute('aria-label','Remove '+(state.library.get(id)?.filename || 'image '+id)+' from selected evidence');
    button.onclick = remove;
    item.append(button);
  }
  return item;
}

function searchMatchesDraft() {
  if(state.imageQuery?.unavailable || state.regionQuery?.unavailable)return false;
  if (!state.search || state.search.text !== element('search-text').value) return false;
  if((state.search.image_query?.id || null)!==(state.imageQuery?.id || null))return false;
  if((state.search.region_query?.id || null)!==(state.regionQuery?.id || null))return false;
  if(semanticCollection && state.search.prediction_version && state.search.prediction_version.id!==state.predictionVersion)return false;
  const same = (a,b) => JSON.stringify([...a].sort()) === JSON.stringify([...b].sort());
  return same(state.positive,state.search.positive_ids) && same(state.negative,state.search.negative_ids) &&
    (state.search.guidance_id || null)===(window.TaskGuidance?.current() || null);
}

function renderChoices() {
  renderImageQuery();
  for(const button of element('search-starter-buttons').querySelectorAll('button'))button.disabled=state.busy || !state.summary?.available;
  element('chosen-examples').replaceChildren();
  for (const [name, values] of [['Relevant',state.positive],['Not a match',state.negative]]) {
    for (const id of values) {
      const button = make('button', name + ' · ' + id.slice(0,7) + ' · remove');
      button.type = 'button';
      button.onclick = () => { values.delete(id); renderChoices(); renderResults(); };
      element('chosen-examples').append(button);
    }
  }
  window.TaskGuidance?.sync();
  element('save-concept').disabled = state.savingConcept || state.busy || !searchMatchesDraft();
  element('results-stale').hidden = !state.search || searchMatchesDraft();
  const count = state.positive.size + state.negative.size;
  element('chosen-examples-panel').hidden=count===0;
  element('clear-examples').hidden=count===0;
  element('refine-button').hidden=count===0;
  element('refine-button').disabled = !state.summary?.available || state.busy || count === 0;
  element('example-feedback').hidden=count===0;
  element('example-feedback').textContent = count
    ? state.positive.size+' matching · '+state.negative.size+' nonmatching · '+count+' selected. '+(state.positive.size && state.negative.size ? 'Ready to refine this search.' : 'Choose at least one of each to guide the search.')
    : 'Mark images Relevant or Not a match, then refine. This guides your search without changing tissue labels.';
  updateSelectionToolbar();
  semanticDraft.schedule();
}

function toggleRelevance(id, positive) {
  if(id===querySourceId()){inspectionMessage('This image is the search source. Choose other images as matching or nonmatching examples.');return;}
  const selected = positive ? state.positive : state.negative;
  const opposite = positive ? state.negative : state.positive;
  if (selected.has(id)) selected.delete(id);
  else {
    if(!judgmentImageReady(id)){inspectionMessage('Load this image before recording a new visual judgment.');return;}
    if (!state.summary?.guidance_available && selected.size>=20) { inspectionMessage('This running search version accepts at most 20 examples of each type.'); return; }
    if (state.positive.size+state.negative.size+(opposite.has(id)?0:1)>40) { inspectionMessage('Use at most 40 relevant and not-a-match examples combined. Remove one before adding another.'); return; }
    selected.add(id); opposite.delete(id);
  }
  renderChoices();
  renderResults();
  refreshInspection();
}

function toggleEvidence(id) {
  if(state.evidence.has(id))state.evidence.delete(id);
  else {
    if(!judgmentImageReady(id)){inspectionMessage('Load this image before selecting it as evidence.');return;}
    if(evidenceCount()>=8){inspectionMessage('Select up to 8 images or regions for a finding. Remove evidence to choose another.');return;}
    state.evidence.add(id);
  }
  renderEvidence();renderResults();refreshInspection();
}

function renderResults() {
  // Rebuilding the cards must not strand a keyboard user after a judgment.
  const active=document.activeElement, card=active?.closest('.semantic-tile');
  const focused=card ? {id:card.dataset.imageId,target:card.parentElement.id,choice:active.dataset.choice} : null;
  renderCards(state.search?.results || [], 'search-results', 'Match');
  const galleryRows=state.gallery?.results || [];
  renderCards(state.galleryExpanded ? galleryRows : galleryRows.slice(0,state.gallery?.primary_count || 20), 'gallery-results', 'Example');
  element('gallery-more').hidden=state.galleryExpanded || galleryRows.length <= (state.gallery?.primary_count || 20);
  if(focused?.choice){
    const replacement=[...element(focused.target).querySelectorAll('.semantic-tile')].find(node=>node.dataset.imageId===focused.id);
    [...(replacement?.querySelectorAll('[data-choice]')||[])].find(node=>node.dataset.choice===focused.choice)?.focus({preventScroll:true});
  }
}

function renderCards(rows, target, prefix) {
  const cards = rows.map((row, index) => {
    const card = make('article', undefined, 'semantic-tile');
    card.dataset.imageId=row.id;
    const image = make('img');
    image.alt = prefix + ' tissue image ' + (index + 1);
    image.loading = 'lazy';
    const status=make('p',undefined,'image-load-status');status.setAttribute('role','status');
    card.append(inspectionButton(row.id,target,image), status, make('p', prefix + ' ' + (index + 1)));
    if (row.development_label) card.append(make('p', 'Dataset label: ' + (labels[row.development_label] || row.development_label)));
    const choices = make('div', undefined, 'actions');
    for (const [text, positive] of [['Relevant',true],['Not a match',false]]) {
      const button = make('button', text);
      button.type = 'button';
      button.dataset.choice=positive?'relevant':'negative';
      button.disabled=row.id===querySourceId();
      button.setAttribute('aria-pressed', String((positive ? state.positive : state.negative).has(row.id)));
      button.onclick = () => toggleRelevance(row.id, positive);
      choices.append(button);
    }
    const evidence = make('button', state.evidence.has(row.id) ? 'Evidence selected' : 'Select as evidence');
    evidence.type = 'button';
    evidence.dataset.choice='evidence';
    evidence.dataset.evidence = row.id;
    evidence.setAttribute('aria-pressed', String(state.evidence.has(row.id)));
    evidence.onclick = () => toggleEvidence(row.id);
    card.append(choices, evidence);
    loadJudgmentImage(image,status,row.image_url,ready=>{
      choices.querySelector('[data-choice="relevant"]').disabled=row.id===querySourceId()||(!ready&&!state.positive.has(row.id));
      choices.querySelector('[data-choice="negative"]').disabled=row.id===querySourceId()||(!ready&&!state.negative.has(row.id));
      evidence.disabled=!ready&&!state.evidence.has(row.id);
    });
    return card;
  });
  element(target).replaceChildren(...cards);
}

function showInterpretation(search) {
  const panel=element('search-interpretation');
  const interpretation=search.ranking_provenance?.language_interpretation;
  panel.replaceChildren();panel.hidden=!interpretation;
  if (!interpretation) return;
  const describe=values=>(values || []).map(value=>labels[value] || value).join(', ');
  panel.append(make('h3','Interpreted as'),make('p',describe(interpretation.targets) || 'No supported broad tissue concept identified.'));
  if (interpretation.excluded_targets?.length) panel.append(make('p','Avoid: '+describe(interpretation.excluded_targets)));
  if (interpretation.explanation) panel.append(make('p',interpretation.explanation));
  const scope=make('details',undefined,'interpretation-scope');
  scope.append(make('summary','Interpretation scope'),make('p','This interpretation covers nine broad colorectal tissue concepts. It does not verify every visual detail or spatial relationship.'));
  panel.append(scope);
  const edit=make('button','Edit my query');edit.type='button';edit.onclick=()=>{element('search-text').focus();element('search-text').scrollIntoView({block:'center'});};panel.append(edit);
  if (interpretation.requires_examples) {
    panel.append(make('p','Choose relevant and not-a-match examples to show what you mean.'));
    const browse=make('button','Browse examples');browse.type='button';browse.onclick=()=>{element('gallery-panel').open=true;element('gallery-panel').scrollIntoView({block:'start'});};panel.append(browse);
  }
}

function showSearch(search) {
  window.TaskGuidance?.loadedSearch(search);
  state.search = search;
  element('search-starters').open=false;
  showInterpretation(search);
  for (const row of search.results) state.library.set(row.id, row);
  element('search-summary').textContent = search.results.length + ' suggested matches' + (search.guidance_id ? ' · guidance applied' : '');
  const queryLabel=search.region_query ? 'Region from '+search.region_query.filename+' · '+regionCoordinates(search.region_query)+(search.text?' · “'+search.text+'”':'') : search.image_query ? 'Similar to '+search.image_query.filename+(search.text?' · “'+search.text+'”':'') : 'For “'+search.text+'”';
  element('search-summary').title = queryLabel;
  element('search-query').textContent=queryLabel;
  element('search-query').hidden=false;
  element('gallery-panel').open = false;
  const provenance=search.ranking_provenance || {};
  const actionable=[];
  if (provenance.feedback_overrides_text_ranking) {
    actionable.push('Guided by '+search.positive_ids.length+' relevant and '+search.negative_ids.length+' not-a-match examples. Scores show relevance, not certainty.');
    if (search.exclusions) actionable.push('Your image choices control this ranking. Written exclusions are recorded but are not enforced by the example-trained scorer.');
  } else if (provenance.abstain) {
    actionable.push('These are exploratory matches. Add relevant and not-a-match examples to define the pattern you need.');
  } else if (!provenance.language_interpretation) {
    actionable.push('Suggested visual matches: inspect the images before using them as evidence.');
  }
  element('search-notices').replaceChildren(...actionable.slice(0,2).map(text=>make('p',text,'scope-note')));
  element('search-more-notices').replaceChildren(...(search.notices || []).map(text=>make('li',text)));
  element('search-details').hidden = false;
  element('score-description').textContent = search.score_interpretation;
  element('search-provenance').textContent = JSON.stringify({id:search.id,sha256:search.sha256,scope:search.scope,retrieval_version:search.retrieval_version,ranking_provenance:search.ranking_provenance,guidance_id:search.guidance_id,guidance:search.guidance,
    scores:search.results.map(row => ({id:row.id,similarity:row.similarity,text_similarity:row.text_similarity}))}, null, 2);
  element('save-concept').disabled = false;
  renderResults();
  renderChoices();
}

async function search() {
  if (state.busy) return;
  if(state.imageQuery?.unavailable || state.regionQuery?.unavailable)throw new Error('Retry the source image or region, or switch to text search. Your draft is kept.');
  if(state.regionQuery && state.summary.region_search_available!==true)throw new Error('Region search is unavailable on this server. You can still save the region as finding evidence.');
  const text = element('search-text').value;
  if (!text.trim() && !state.imageQuery && !state.regionQuery) throw new Error('Describe what a useful match should look like, or choose Find similar on an image.');
  if (text.length > (state.summary.query_limit || 600)) throw new Error('Select a shorter phrase for this search; your description stays unchanged.');
  state.busy = true;
  element('search-button').disabled = true;
  const searchLabel=element('search-button').textContent;
  element('search-button').textContent=state.regionQuery?'Preparing region…':'Searching…';
  element('search-form').setAttribute('aria-busy','true');
  renderChoices();
  message(state.regionQuery ? 'Encoding the selected region on CPU if needed, then searching this collection. This may take a little while; keep this page open.' : semanticCollection ? 'Searching this collection…' : 'Searching the development image library…');
  try {
    const result = await api('/search', {text,positive_ids:[...state.positive],negative_ids:[...state.negative],limit:24,...(semanticCollection ? {image_id:state.imageQuery?.id || null,region_id:state.regionQuery?.id || null,prediction_version:state.predictionVersion || 'original'} : {}),...(window.TaskGuidance?.current() ? {guidance_id:window.TaskGuidance.current()} : {})});
    showSearch(result);
    const resultStart=element('search-interpretation').hidden ? element('search-summary') : element('search-interpretation');
    const bounds = resultStart.getBoundingClientRect();
    if (bounds.top < 0 || bounds.top > window.innerHeight - 150) resultStart.scrollIntoView({block:'nearest'});
    message(searchMatchesDraft()
      ? ''
      : 'Search complete for the wording shown above the results. Your newer words and example choices are kept; search again to use them.');
  } finally {
    state.busy = false;
    element('search-button').disabled = !state.summary.available;
    element('search-button').textContent=searchLabel;
    element('search-form').setAttribute('aria-busy','false');
    renderChoices();
  }
}

function selectedWords() {
  const input = element('description-text');
  return input.selectionStart !== input.selectionEnd
    ? input.value.slice(input.selectionStart,input.selectionEnd) : input.value;
}

async function searchWords(text) {
  if(state.busy)throw new Error('Wait for the current search to finish, then search these words. Your description is unchanged.');
  if (!text.trim()) throw new Error('Write a description or select a claim to search.');
  if (text.length > (state.summary.query_limit || 600)) throw new Error('Select a phrase of at most ' + (state.summary.query_limit || 600) + ' characters. Your full draft is preserved.');
  window.TaskGuidance?.clearApplied();
  state.imageQuery=null;state.regionQuery=null;
  element('search-text').value = text;
  // A new description phrase begins without relevance judgments from another query.
  state.positive.clear();
  state.negative.clear();
  selectPanel('find');
  await search();
}

element('search-evidence').addEventListener('focusin', event => {
  const shelf=event.currentTarget,button=event.target.closest('button');
  if(!button || !shelf.contains(button))return;
  const viewport=shelf.getBoundingClientRect(),target=button.getBoundingClientRect();
  const style=getComputedStyle(shelf);
  const left=viewport.left+shelf.clientLeft+parseFloat(style.paddingLeft);
  const right=viewport.left+shelf.clientLeft+shelf.clientWidth-parseFloat(style.paddingRight);
  // Reveal the focused control inside this shelf only. Focus restoration after
  // removing evidence must not move the document vertically.
  if(target.left<left)shelf.scrollLeft-=left-target.left;
  else if(target.right>right)shelf.scrollLeft+=target.right-right;
});

function renderEvidence() {
  const count=evidenceCount();
  element('go-description').hidden=count===0;
  element('evidence-count').textContent = semanticCollection && state.simpleFinding
    ? (count ? count+' selected '+(count===1?'image or region will':'images or regions will')+' be attached when you save.' : 'Find images, choose Select as evidence, then return here to save your finding.')
    : count
    ? count + ' selected '+(count===1?'image or region. Attach it to the claim it helps':'images or regions. Attach them to the claims they help')+' you assess.'
    : 'Select specific tiles in the search results, then attach them to a claim below.';
  const remove=id=>{
    const active=document.activeElement,container=active?.closest('#selected-evidence,#search-evidence');
    const index=[...state.evidence].indexOf(id);
    state.evidence.delete(id);renderEvidence();renderResults();refreshInspection();
    if(container){
      const buttons=[...container.querySelectorAll('[data-remove-evidence]')];
      (buttons[Math.min(index,buttons.length-1)] || element(container.id==='search-evidence'?'find-tab':'editor-find-images')).focus({preventScroll:true});
    }
  };
  const removeRegion=id=>{state.regionEvidence.delete(id);renderEvidence();renderResults();};
  element('selected-evidence').replaceChildren(...[...state.evidence].map(id=>thumbnail(id,()=>remove(id))),...[...state.regionEvidence].map(id=>regionThumbnail(id,()=>removeRegion(id))));
  element('search-evidence-panel').hidden=count===0;
  element('search-evidence-count').textContent='Evidence for your finding · '+count+' of 8 selected';
  element('search-evidence-count').title='Selected images and regions are kept when you change the search.';
  element('search-evidence').replaceChildren(...[...state.evidence].map(id=>thumbnail(id,()=>remove(id),'search-evidence')),...[...state.regionEvidence].map(id=>regionThumbnail(id,()=>removeRegion(id),'search-region-evidence')));
  updateSelectionToolbar();
  semanticDraft.schedule();
}

function useStatementMode() {
  if(!semanticCollection)return;
  state.simpleFinding=false;
  element('description-text').maxLength=6000;
  semanticDraft.schedule();
}

function renderClaims() {
  element('claims').replaceChildren();
  element('claim-count').textContent = state.claims.length ? state.claims.length + (state.claims.length === 1 ? ' claim' : ' claims') : 'No claims yet';
  element('claims-empty').hidden = state.claims.length > 0;
  state.claims.forEach((claim, index) => {
    const card = make('section', undefined, 'claim-card');
    const heading = make('h4','Claim ' + (index + 1));
    const assessmentStatus = make('span', undefined, 'claim-assessment');
    const updateAssessmentStatus = () => { assessmentStatus.textContent = claim.assessment === 'unresolved' ? 'Unresolved' : claim.assessment === 'supported_by_reviewer' ? 'Reviewer: supported' : 'Reviewer: contradicted'; };
    updateAssessmentStatus();
    heading.append(assessmentStatus);
    card.append(heading);
    const textLabel = make('label','Words from your description');
    const input = make('textarea');
    input.value = claim.text;
    input.rows = 2; input.maxLength = 1000;
    input.oninput = () => { useStatementMode();claim.text = input.value; };
    textLabel.append(input);
    card.append(textLabel);
    const actions = make('div',undefined,'actions');
    const find = make('button','Search this claim');
    find.type = 'button';
    find.onclick = () => act(() => searchWords(claim.text));
    const attach = make('button','Attach selected evidence');
    attach.type = 'button';
    attach.onclick = () => {
      const combined = [...new Set([...claim.image_ids,...state.evidence])];
      const regions=[...new Set([...(claim.region_ids||[]),...state.regionEvidence])];
      if (combined.length+regions.length > 8) { message('A claim can link up to 8 images or regions. Remove evidence before adding more.'); return; }
      useStatementMode();
      claim.image_ids = combined;claim.region_ids=regions;
      renderClaims();
    };
    const remove = make('button','Remove claim');
    remove.type = 'button';
    remove.onclick = () => { useStatementMode();state.claims.splice(index,1); renderClaims(); };
    actions.append(find,attach,remove);
    card.append(actions);
    const images = make('div',undefined,'evidence-strip');
    images.append(...claim.image_ids.map(id => thumbnail(id, () => {
      useStatementMode();claim.image_ids = claim.image_ids.filter(value => value !== id); renderClaims();
    },'claim-evidence-'+index)));
    images.append(...(claim.region_ids||[]).map(id=>regionThumbnail(id,()=>{useStatementMode();claim.region_ids=claim.region_ids.filter(value=>value!==id);renderClaims();},'claim-region-evidence-'+index)));
    card.append(images);
    if (!claim.image_ids.length && !(claim.region_ids||[]).length) card.append(make('p','No image evidence attached to this claim.'));
    const assessmentLabel = make('label','Your assessment');
    const assessment = make('select');
    for (const [value,text] of [
      ['unresolved','Unresolved — needs more evidence'],
      ['supported_by_reviewer','Supported by the images I reviewed'],
      ['contradicted_by_reviewer','Contradicted by the images I reviewed']
    ]) {
      const option = make('option',text); option.value = value; assessment.append(option);
    }
    assessment.value = claim.assessment;
    assessment.onchange = () => { useStatementMode();claim.assessment = assessment.value; updateAssessmentStatus(); };
    assessmentLabel.append(assessment);
    const noteLabel = make('label','Reason / limits of this evidence','claim-note');
    const note = make('textarea'); note.rows = 2; note.maxLength = 2000; note.value = claim.note;
    note.oninput = () => { useStatementMode();claim.note = note.value; };
    noteLabel.append(note);
    card.append(assessmentLabel,noteLabel);
    element('claims').append(card);
  });
  semanticDraft.schedule();
}

async function openConcept(identifier) {
  const record = await api('/concepts/'+encodeURIComponent(identifier));
  const saved = await api('/searches/'+encodeURIComponent(record.search_id));
  element('search-text').value = saved.text;
  element('concept-name').value = record.name;
  element('concept-reviewer').value = record.reviewer || '';
  element('concept-note').value = record.note || '';
  state.positive = new Set(saved.positive_ids);
  state.negative = new Set(saved.negative_ids);
  if(semanticCollection && saved.prediction_version)state.predictionVersion=saved.prediction_version.id;
  state.imageQuery=saved.image_query || null;
  state.regionQuery=saved.region_query?await resolveRegion(saved.region_query.id):null;
  showSearch(saved); tab('find');
  message('Opened the saved search and its existing results. No new search was run. Guidance has not been reapplied.');
}

async function openDescription(identifier) {
  const record = await api('/descriptions/'+encodeURIComponent(identifier));
  element('description-text').value = record.text;
  element('description-reviewer').value = record.reviewer;
  state.claims = record.claims.map(claim => ({text:claim.text,assessment:claim.assessment,note:claim.note,image_ids:[...claim.image_ids],region_ids:[...(claim.region_ids||[])]}));
  state.simpleFinding=Boolean(semanticCollection && state.claims.length===1 && state.claims[0].text===record.text && state.claims[0].assessment==='unresolved' && !state.claims[0].note);
  if(element('finding-statements'))element('finding-statements').open=!state.simpleFinding;
  state.parentId = record.id;
  if(semanticCollection)state.predictionVersion=record.prediction_version?.id || 'original';
  if(semanticCollection)renderCollectionContext();
  state.evidence = new Set(state.claims.flatMap(claim => claim.image_ids).slice(0,8));
  state.regionEvidence=new Set(state.claims.flatMap(claim=>claim.region_ids||[]).slice(0,8-state.evidence.size));
  await loadRegionEvidence(state.claims.flatMap(claim=>claim.region_ids||[]));
  element('description-version').textContent = 'Editing a new version based on '+record.id.slice(0,8)+'. The saved original stays unchanged.';
  element('description-export').href = semanticBase+'/descriptions/'+record.id+'/export';
  element('description-export').hidden = false;
  showDescriptionReport(record.id);
  renderClaims(); tab('describe'); message('Saved description opened as an editable draft. Save explicitly to create the next version.');
}

function navigateSaved(kind,identifier) {
  semanticDraft.save();
  location.assign(semanticURL({[kind]:identifier}));
}

function renderCollectionContext() {
  if (!semanticCollection) return;
  window.TissueNavigation?.setCollection(semanticCollection,state.predictionVersion);
  const name = state.summary.collection?.name || 'Image collection';
  document.title = name+' · Tissology';
  element('semantic-collection-name').textContent = name;
  const back = element('semantic-collection-back');
  back.href = 'collections?collection='+encodeURIComponent(semanticCollection)+'&prediction_version='+encodeURIComponent(state.predictionVersion || 'original');
  back.hidden = false;
  element('find-tab').textContent = 'Search images';
  element('describe-tab').textContent = 'Findings';
  element('go-description').textContent = 'Write a finding';
  element('save-description').textContent = state.savingDescription ? 'Saving…' : state.parentId ? 'Save new version' : 'Save finding and evidence';
  element('describe-heading').textContent = 'Write a finding';
  element('find-panel').setAttribute('aria-label','Search images in '+name);
  element('semantic-scope').textContent = state.summary.available
    ? Number(state.summary.population || 0).toLocaleString()+' images in this collection'
    : (state.summary.reason || 'Search is not ready for this collection. Return to Images to inspect its status.');
  document.querySelector('.gallery-intro').textContent = 'A sample from your collection. Search above to see suggested matches.';
  document.querySelector('.library-scope > summary').textContent = 'About this collection';
  document.querySelector('.semantic-footer').textContent = 'Image-level exploration · Researcher assessments stay separate from model predictions';
  let versionNote=element('finding-prediction-version');
  if(!versionNote){versionNote=make('p',undefined,'scope-note');versionNote.id='finding-prediction-version';element('description-version').after(versionNote);}
  versionNote.textContent='Finding context: '+(state.predictionVersion && state.predictionVersion!=='original' ? 'taught prediction version '+state.predictionVersion.slice(0,8) : 'original scan')+'.';
  if (!element('finding-statements')) {
    const advanced=make('details',undefined,'semantic-details');advanced.id='finding-statements';
    advanced.append(make('summary','Review individual statements'));
    const start=element('search-description').parentElement;
    start.before(advanced);
    for(const node of [start,document.querySelector('.editor-hint'),document.querySelector('.claims-heading'),element('claims-empty'),element('claims')])advanced.append(node);
    // Opening a disclosure is inspection, not a decision to require statement-level review.
    advanced.ontoggle=()=>{element('description-text').maxLength=advanced.open||!state.simpleFinding?6000:1000;semanticDraft.schedule();};
    element('description-text').maxLength=1000;
    element('description-text').previousElementSibling.textContent='Your finding';
    element('description-text').placeholder='What do you notice in the selected images?';
    document.querySelector('.editor-heading p').textContent='Write an observation. Your selected images will be attached when you save.';
    element('description-version').textContent='Saved with this collection and your selected images. An assessment stays unresolved unless you review it explicitly.';
    const key=window.TissueLab?.storageKey('tissue-reviewer:'+new URL('api/collections',document.baseURI).pathname)||'tissue-reviewer:'+new URL('api/collections',document.baseURI).pathname;
    try{const reviewer=localStorage.getItem(key)||'';for(const id of ['description-reviewer','concept-reviewer'])if(!element(id).value)element(id).value=reviewer;}catch(_){}
  }
}

async function refreshSaved() {
  state.summary = await api('');
  renderCollectionContext();
  element('gallery-panel').hidden = !state.summary.gallery_available;
  element('browse-examples').disabled = !state.summary.gallery_available;
  element('saved-concepts').replaceChildren();
  for (const concept of state.summary.concepts || []) {
    const item = make('div',undefined,'saved-item');
    item.append(make('p',concept.name));
    const button = make('button','Open named search'); button.type = 'button';
    button.onclick = () => navigateSaved('concept',concept.id);
    item.append(button); element('saved-concepts').append(item);
  }
  if (!(state.summary.concepts || []).length) element('saved-concepts').append(make('p','No named searches saved yet.'));
  element('saved-descriptions').replaceChildren();
  for (const description of state.summary.descriptions || []) {
    const item = make('div',undefined,'saved-item');
    item.append(make('p',description.text),make('p',description.reviewer));
    const button = make('button','Open as next version'); button.type = 'button';
    button.onclick = () => navigateSaved('description',description.id);
    item.append(button); element('saved-descriptions').append(item);
  }
  if (!(state.summary.descriptions || []).length) element('saved-descriptions').append(make('p','No descriptions saved yet.'));
}

async function saveDescription() {
  if(state.savingDescription)return;
  const text = element('description-text').value;
  if (!text.trim()) throw new Error('Write a description before saving.');
  if (semanticCollection && state.simpleFinding) {
    if(text.length>1000)throw new Error('Keep this finding within 1,000 characters, or review individual statements in a longer description.');
    if(!evidenceCount())throw new Error('Select at least one image or region to include with this finding.');
    state.claims=[{text,image_ids:[...state.evidence],region_ids:[...state.regionEvidence],assessment:'unresolved',note:''}];
    renderClaims();
  }
  if (!state.claims.length) throw new Error('Add at least one claim to review, even if it remains unresolved.');
  for (const claim of state.claims) {
    if (!claim.text.trim() || !text.includes(claim.text)) throw new Error('Each claim must use words that occur exactly in the current description. Update claims after editing your draft.');
    for(const id of claim.region_ids||[])if(state.regions.get(id)?.unavailable || !state.regions.has(id))throw new Error('A selected region is unavailable. Reopen or remove it before saving this finding.');
    if (claim.assessment !== 'unresolved' && !claim.image_ids.length && !(claim.region_ids||[]).length) throw new Error('Attach image evidence to every supported or contradicted claim, or mark it unresolved.');
  }
  state.savingDescription=true;
  const button=element('save-description'),label=button.textContent;
  button.disabled=true;button.textContent='Saving…';element('description-form').setAttribute('aria-busy','true');
  try {
    const record = await api('/descriptions',{
      text,reviewer:element('description-reviewer').value,parent_id:state.parentId,claims:state.claims.map(claim=>{const {region_ids,...base}=claim;return semanticCollection?{...base,region_ids:region_ids||[]}:base;}),
      ...(semanticCollection ? {prediction_version:state.predictionVersion || 'original'} : {})
    });
    state.parentId = record.id;
    if(semanticCollection)try{localStorage.setItem(window.TissueLab?.storageKey('tissue-reviewer:'+new URL('api/collections',document.baseURI).pathname)||'tissue-reviewer:'+new URL('api/collections',document.baseURI).pathname,element('description-reviewer').value);}catch(_){}
    element('description-version').textContent = 'Saved version ' + record.id.slice(0,8) + '. Further edits will create another version.';
    element('description-export').href = semanticBase+'/descriptions/'+record.id+'/export';
    element('description-export').hidden = false;
    showDescriptionReport(record.id);
    semanticDraft.schedule();
    // The save is already complete; a list refresh failure must not suggest retrying the POST.
    try {await refreshSaved();message(semanticCollection?'Finding and image evidence saved. Open the report to review this version.':'Description and reviewer-linked evidence saved. Open the report to review this version.');}
    catch(_){message('Your finding was saved. The saved list could not refresh; you can still open its report.');}
  } finally {
    state.savingDescription=false;button.disabled=!state.summary?.available;button.textContent=state.parentId?'Save new version':label;
    element('description-form').setAttribute('aria-busy','false');
  }
}

element('gallery-more').onclick=()=>{state.galleryExpanded=true;renderResults();semanticDraft.schedule();};
element('gallery-panel').ontoggle = () => {
  if (!element('gallery-panel').open || state.gallery) return;
  act(loadGallery);
};
element('browse-examples').onclick = () => {
  element('gallery-panel').open = true;
  const bounds = element('gallery-panel').getBoundingClientRect();
  if (bounds.top < 0 || bounds.top > window.innerHeight - 150) element('gallery-panel').scrollIntoView({block:'nearest'});
  act(loadGallery);
};

// Search and description panels retain their DOM and draft state while switching.
element('find-tab').onclick = () => selectPanel('find');
element('describe-tab').onclick = () => selectPanel('describe');
element('go-description').onclick = () => selectPanel('describe');
element('editor-find-images').onclick = () => selectPanel('find');
element('search-text').oninput = renderChoices;
element('add-microscopy-context').onclick = () => {
  const input = element('search-text');
  const prefix = 'a microscopic image of tissue showing ';
  if (!input.value.startsWith(prefix)) {
    if ((prefix + input.value).length > (state.summary?.query_limit || 600)) { message('Shorten the query before adding microscopy context.'); return; }
    input.value = prefix + input.value;
  }
  renderChoices(); input.focus();
  message('Context added visibly. Edit the wording before searching; this does not verify a match.');
};
element('search-form').onsubmit = event => { event.preventDefault(); act(search); };
element('refine-button').onclick = () => act(search);
element('clear-examples').onclick = () => {
  state.positive.clear(); state.negative.clear(); renderChoices(); renderResults();
};
element('search-description').onclick = () => act(() => searchWords(selectedWords()));
element('add-claim').onclick = () => act(async () => {
  const text = selectedWords();
  if (!text.trim()) throw new Error('Write a description or select words to review first.');
  if (text.length > 1000) throw new Error('Select a claim of at most 1,000 characters.');
  if (state.claims.length >= 20) throw new Error('A description can contain up to 20 reviewed claims.');
  useStatementMode();
  state.claims.push({text,image_ids:[],region_ids:[],assessment:'unresolved',note:''});
  renderClaims();
});
element('concept-form').onsubmit = event => {
  event.preventDefault();
  act(async () => {
    if(state.savingConcept)return;
    if (!searchMatchesDraft()) throw new Error('Run or refine this search before saving the updated words and examples.');
    state.savingConcept=true;
    const button=element('save-concept');button.textContent='Saving…';button.disabled=true;
    element('concept-form').setAttribute('aria-busy','true');
    try{
      await api('/concepts',{name:element('concept-name').value,search_id:state.search.id,
        reviewer:element('concept-reviewer').value,note:element('concept-note').value});
      try{await refreshSaved();message('Named search saved. Reopen it from Saved work or the list below.');}
      catch(_){message('Your named search was saved. The list could not refresh; reopen it from Saved work when the connection is ready.');}
    }finally{
      state.savingConcept=false;button.textContent='Save named search';
      element('concept-form').setAttribute('aria-busy','false');renderChoices();
    }
  });
};
element('description-form').onsubmit = event => { event.preventDefault(); act(saveDescription); };

act(async () => {
  if(semanticCollection && state.predictionVersion && !/^(original|[a-f0-9]{32})$/.test(state.predictionVersion))throw new Error('Choose a saved prediction version from this collection.');
  if(semanticCollection && state.predictionVersion && state.predictionVersion!=='original' && !semanticParams.has('description')){
    // A well-formed identifier is not evidence that this saved version exists.
    // Saved descriptions verify and restore their own version when opened.
    const response=await fetch('api/collections/'+encodeURIComponent(semanticCollection)+'?prediction_version='+encodeURIComponent(state.predictionVersion));
    if(!response.ok)throw new Error('This prediction version could not be loaded. Return to the collection and choose a saved version, or reload when the connection is available.');
    const collection=await response.json();
    if(collection.selected_prediction_version?.id!==state.predictionVersion)throw new Error('This collection did not return the requested prediction version. Return to the collection and choose a saved version.');
  }
  await refreshSaved();
  if(semanticCollection && !state.predictionVersion)state.predictionVersion=state.summary.selected_prediction_version?.id || 'original';
  // Populate the canvas through a read-only gallery request; no query or review is submitted.
  await loadGallery();
  const recovered = await semanticDraft.initialize();
  const params = new URLSearchParams(location.search);
  if (!recovered) {
    if (params.has('description')) await act(()=>openDescription(params.get('description')));
    else if (params.has('concept')) await act(()=>openConcept(params.get('concept')));
  }
  // A direct workspace entry chooses the visible panel after restoring all work.
  // It shares the existing draft identity and never submits a search or server save.
  const requestedPanel = params.get('panel');
  if (requestedPanel === 'describe' || requestedPanel === 'find') tab(requestedPanel);
  if(semanticCollection)renderCollectionContext();
  if(params.has('similar')){
    await act(async()=>{
      await startImageSearch(params.get('similar'));
      const url=new URL(location.href);url.searchParams.delete('similar');history.replaceState({},'',url);
    });
  }
  if(semanticCollection && params.has('region')){
    await act(async()=>{
      const id=params.get('region'),run=params.get('run_region')==='1';
      const url=new URL(location.href);url.searchParams.delete('run_region');history.replaceState(history.state,'',url);
      if(params.has('similar') || !/^[a-f0-9]{64}$/.test(id||''))throw new Error('Choose one source region from this collection.');
      await startRegionSearch(id,false);
      url.searchParams.delete('region');url.searchParams.set('panel','find');history.replaceState(history.state,'',url);
      if(run)await search();
    });
  }
  if(semanticCollection && params.has('region_evidence')){
    await act(async()=>{
      const id=params.get('region_evidence');if(!/^[a-f0-9]{64}$/.test(id||''))throw new Error('This region link could not be read.');
      const record=await resolveRegion(id);addRegionEvidence(record);selectPanel('describe');semanticDraft.save();
      const url=new URL(location.href);url.searchParams.delete('region_evidence');history.replaceState(history.state,'',url);
      message('Selected region added. Save your finding to keep the description and image evidence together.');
    });
  }
  // An explicit image-inspection handoff adds evidence to the existing draft.
  // The collection API verifies membership before it is shown or retained.
  if (semanticCollection && state.summary.available && params.has('evidence')) {
    const id=params.get('evidence');
    if(!/^[A-Za-z0-9_-]{1,200}$/.test(id))throw new Error('This image link could not be read. Return to the collection and choose an image.');
    if(evidenceCount()>=8&&!state.evidence.has(id))throw new Error('Your draft already has eight selected images or regions. Remove one before adding this image.');
    const response=await fetch('api/collections/'+encodeURIComponent(semanticCollection)+'/images/'+encodeURIComponent(id)+'?prediction_version='+encodeURIComponent(state.predictionVersion || 'original'));
    if(!response.ok)throw new Error('That image is unavailable in this collection. Your existing draft has been kept.');
    const selected=await response.json();
    state.library.set(id,{id,image_url:selected.image_url || semanticBase+'/images/'+encodeURIComponent(id),filename:selected.filename});
    state.evidence.add(id);renderEvidence();renderResults();semanticDraft.schedule();
    const url=new URL(location.href);url.searchParams.delete('evidence');history.replaceState({},'',url);
  }
  try {
    const response = await fetch('api/health');
    if (response.ok) state.feedbackAvailable = Boolean((await response.json()).feedback_available);
  } catch (_) { /* Search still works without the optional correction workflow. */ }
  if (state.summary.scope) element('semantic-library-scope').textContent = state.summary.scope;
  element('search-text').maxLength = state.summary.query_limit || 600;
  element('search-button').disabled = !state.summary.available;
  element('save-description').disabled = !state.summary.available;
  // Prepared wording is an optional aid. Restoring a page or draft never chooses
  // an example, changes feedback, or starts a search.
  void loadSearchStarters();
  if (!state.summary.available) message(state.summary.reason || 'The image-search workspace is not configured on this server.');
  renderChoices();
});
