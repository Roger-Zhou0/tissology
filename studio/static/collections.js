'use strict';

(() => {
  const el = id => document.getElementById(id);
  const state = {version:null,teaching:null,teachingRequest:0,teachingPoll:null,teachingBusy:false,checkCandidate:null,checkRequest:0,checkDirty:false,validationPromise:null,config:null, collection:null, semantic:null, semanticRequest:0, images:[], total:0, offset:0, limit:24, label:'', review:'all', query:'', source:'upload', files:[], upload:null, resume:null, createRequest:null, busy:false, poll:null, request:0, imageRequest:0, imageAbort:null, imageSignature:null, detailRequest:0, detailIndex:-1};
  const colors = {ADI:'#d6b867',BACK:'#c8cdd1',DEB:'#a48772',LYM:'#896aa8',MUC:'#79b7bd',MUS:'#b67489',NORM:'#c18da9',STR:'#8eafa0',TUM:'#9a718d'};
  const fallbackNames = {ADI:'Adipose',BACK:'Background',DEB:'Debris',LYM:'Lymphocytes',MUC:'Mucus',MUS:'Muscle',NORM:'Normal mucosa',STR:'Stroma',TUM:'Tumor'};
  const statusNames = {importing:'Importing images',ready:'Ready to scan',queued:'Scan queued',scanning:'Scanning images',complete:'Scan complete',failed:'Scan needs attention'};
  const activeStatuses = new Set(['importing','queued','scanning']);
  const fmt = number => Number(number || 0).toLocaleString();
  const score = value => Number.isFinite(Number(value)) && value !== null && value !== undefined ? (100 * Number(value)).toFixed(1) + '%' : '—';
  const name = code => state.config?.classes?.find(item => item.code === code)?.name || fallbackNames[code] || code || 'Not scanned';
  const make = (tag, text, className) => { const node=document.createElement(tag); if(text !== undefined) node.textContent=text; if(className) node.className=className; return node; };
  const safeURL = value => { if (!value) return ''; try {const url=new URL(value, document.baseURI); return url.origin === location.origin && ['http:','https:'].includes(url.protocol) ? url.href : '';} catch {return '';} };
  const collectionURL = id => 'collections?collection=' + encodeURIComponent(id);
  const statusName = collection => collection.status === 'ready' && (collection.declared_domain !== 'colorectal_he'||state.config?.scan_available===false) ? 'Ready to browse' : statusNames[collection.status] || collection.status || 'Imported';
  const imageName = row => row.filename || row.name || row.original_filename || row.id;
  const imageNeedsReview = row => Boolean(row.uncertain || row.needs_review);
  const imageFailed = row => Boolean(row.error || row.status === 'failed');
  const humanReviewFilters = ['unreviewed','reviewed','confirmed','corrected','unsure'];
  const reviewNames = {confirmed:'Confirmed',corrected:'Corrected label',unsure:'Unsure'};
  const reviewerStorageKey=window.TissueLab?.storageKey('tissue-reviewer:'+new URL('api/collections',document.baseURI).pathname) || 'tissue-reviewer:'+new URL('api/collections',document.baseURI).pathname;
  let reviewDirty=false, reviewSaving=false, reviewChanged=false, pendingReviewRequest=null;
  let findings={collectionId:null,rows:[],total:0,nextOffset:null,busy:false,request:0,abort:null};
  const teachingCloseDrafts=new Map();
  let overview=null,workflowRequest=0,workflowLoading=false,batch=null,batchActive=false,batchBusy=false,batchRequestKey=null,pendingBatchId=null;
  const browseStorageKey=window.TissueLab?.storageKey('tissue-collection-view:'+new URL('api/collections',document.baseURI).href) || 'tissue-collection-view:'+new URL('api/collections',document.baseURI).href;
  const browseMemory=new Map();
  const validImageId=value=>typeof value==='string'&&/^[a-zA-Z0-9_-]{1,128}$/.test(value)?value:null;
  let restoreBrowseView=false,pendingImageId=null,detailImageId=null,skipDetailCloseRefresh=false,displayedViewURL=location.href;
  function closeImageForNavigation() {
    if(!el('image-dialog').open)return;
    skipDetailCloseRefresh=true;state.detailRequest++;detailImageId=null;el('image-dialog').close();
  }
  function savedBrowseView(id,version) {
    const key=id+':'+version;
    try {
      const stored=JSON.parse(sessionStorage.getItem(browseStorageKey)||'[]');
      if(Array.isArray(stored))for(const entry of stored.slice(-32))if(entry&&typeof entry.key==='string')browseMemory.set(entry.key,entry);
    } catch (_) { /* Browsing remains available without browser storage. */ }
    const entry=browseMemory.get(key);
    if(!entry||entry.collection!==id||entry.version!==version||![24,48,96].includes(entry.limit)||!Number.isSafeInteger(entry.offset)||entry.offset<0||entry.offset>96000000||entry.offset%entry.limit||typeof entry.query!=='string'||typeof entry.label!=='string'||!['all','needs_review',...humanReviewFilters].includes(entry.review))return null;
    return {...entry,query:entry.query.slice(0,250),image:validImageId(entry.image)};
  }
  function rememberBrowseView() {
    if(!state.collection)return;
    const collection=state.collection.id,version=versionId(),key=collection+':'+version;
    browseMemory.delete(key);browseMemory.set(key,{key,collection,version,limit:state.limit,offset:state.offset,query:state.query,label:state.label,review:state.review,image:detailImageId});
    while(browseMemory.size>32)browseMemory.delete(browseMemory.keys().next().value);
    try {sessionStorage.setItem(browseStorageKey,JSON.stringify([...browseMemory.values()]));} catch (_) {}
  }
  function browseNotice(text='') {el('collection-view-note').textContent=text;el('collection-view-resume').hidden=!text;}
  function reviewerName() {try{return localStorage.getItem(reviewerStorageKey)||'';}catch{return '';}}
  function freshRequestId() {const bytes=new Uint8Array(16);globalThis.crypto.getRandomValues(bytes);return [...bytes].map(value=>value.toString(16).padStart(2,'0')).join('');}
  function canLeaveReview() {
    if(reviewSaving)return false;
    if(!reviewDirty)return true;
    const status=el('image-review-status');if(status){status.focus();status.textContent='You have unsaved changes. Save this review or discard your changes before moving on.';}
    if(el('discard-review'))el('discard-review').hidden=false;
    return false;
  }

  async function api(path='', body, options={}) {
    const init = {...options};
    if(body !== undefined) { init.method = init.method || 'POST'; init.headers = {'Content-Type':'application/json', ...init.headers}; init.body=JSON.stringify(body); }
    const response=await fetch('api/collections'+path,init);
    let data; try {data=await response.json();} catch {throw new Error('The collection service did not return a readable response. Refresh and try again.');}
    if(!response.ok) { const detail=data.detail || data.error; const error=new Error(typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(row=>row.msg).join('; ') : 'The request could not be completed.'); error.status=response.status; error.retryAfter=Number(response.headers.get('Retry-After'))||null;throw error; }
    return data;
  }
  function error(message, id='collection-error') { el(id).textContent=message || ''; el(id).hidden=!message; }
  async function act(work) { error(''); try {await work();} catch(problem) {error(problem.message);} }
  function message(text) {el('collection-message').textContent=text || '';}
  function busy(value) {
    state.busy=value;
    for(const id of ['add-collection','close-import','refresh-collection','finish-import','resume-upload','resume-folder']) el(id).disabled=value;
    for(const node of el('import-form').querySelectorAll('button,input,select')) node.disabled=value;
    el('start-scan').disabled=value || state.config?.scan_available===false;
    el('pause-upload').disabled=!value || state.upload?.phase!=='uploading';
    if(!value) configureImport();
    lockImportIdentity();
  }
  function lockImportIdentity() {
    const locked=Boolean(state.resume?.id);
    for(const id of ['import-name','import-domain','upload-tab','folder-tab']) el(id).disabled=state.busy || locked || (id==='folder-tab' && !state.config?.import_roots?.length);
  }
  function stopTeachingPoll() {if(state.teachingPoll)clearTimeout(state.teachingPoll);state.teachingPoll=null;}
  const versionId = () => state.collection?.selected_prediction_version?.id || state.version || 'original';
  const withVersion = path => path+(path.includes('?')?'&':'?')+'prediction_version='+encodeURIComponent(versionId());
  function stopPoll() {if(state.poll) clearTimeout(state.poll); state.poll=null;}
  function setLocation(id) {const url=new URL(location.href); if(id) url.searchParams.set('collection',id); else url.searchParams.delete('collection'); for(const key of ['import','q','label','review','page','size','prediction_version','image','batch']) url.searchParams.delete(key); history.pushState({},'',url);}
  function progressBar(processed,total) {const node=make('progress'); node.max=Math.max(Number(total)||1,1); if(Number.isFinite(Number(processed))) node.value=Math.max(0,Number(processed)); node.setAttribute('aria-label','Images processed'); return node;}
  function date(value) {if(!value) return ''; const parsed=new Date(value); return Number.isNaN(parsed.getTime()) ? '' : parsed.toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'});}
  function bytes(value) {const amount=Number(value)||0; return amount>=1024**3 ? (amount/1024**3).toFixed(1)+' GB' : amount>=1024**2 ? (amount/1024**2).toFixed(1)+' MB' : (amount/1024).toFixed(0)+' KB';}
  function loadImage(node,url,description) {node.alt=description; node.src=safeURL(url); node.loading='lazy'; node.onerror=()=>{const fallback=make('div','Preview unavailable','image-placeholder'); node.replaceWith(fallback);};}
  const readinessWarning='Wait for the image to display, or retry it, before saving.';
  // A review is only actionable after this exact full image has decoded. Retry
  // changes the image request, not the form, so an unsaved assessment survives.
  function reviewImage(node,url,description,current,id) {
    let attempt=0,decoded=false,source='';const listeners=new Set();
    const status=make('p','Loading image. Saving is available once it is ready.','scope-note');status.id=id+'-status';status.setAttribute('role','status');status.setAttribute('aria-live','polite');
    const retry=make('button','Retry image','secondary');retry.id=id+'-retry';retry.type='button';retry.hidden=true;
    const controls=make('div',undefined,'review-image-status');controls.append(status,retry);
    node.alt=description;node.loading='eager';node.setAttribute('aria-describedby',status.id);
    const ready=()=>current()&&decoded&&node.isConnected&&node.complete&&node.naturalWidth>0&&node.naturalHeight>0&&node.src===source;
    const notify=()=>listeners.forEach(callback=>callback());
    const start=()=>{
      if(!current())return;
      const request=++attempt;decoded=false;retry.hidden=true;status.hidden=false;status.textContent='Loading image. Saving is available once it is ready.';node.setAttribute('aria-busy','true');notify();
      const failed=()=>{if(request!==attempt||!current())return;decoded=false;node.setAttribute('aria-busy','false');status.textContent='The image could not be displayed. Retry it before saving your assessment. Your entries are kept.';retry.hidden=false;notify();};
      const value=safeURL(url);if(!value){failed();return;}
      const next=new URL(value);if(request>1)next.searchParams.set('image_retry',String(request));source=next.href;
      node.onerror=failed;
      node.onload=async()=>{
        try{if(typeof node.decode==='function')await node.decode();}catch{failed();return;}
        if(request!==attempt||!current()||node.src!==source)return;
        if(!node.complete||!node.naturalWidth||!node.naturalHeight){failed();return;}
        decoded=true;node.setAttribute('aria-busy','false');status.hidden=true;retry.hidden=true;notify();
      };
      node.src=source;
    };
    retry.onclick=start;
    return {controls,ready,current,start,subscribe(callback){listeners.add(callback);callback();}};
  }

  async function loadValidation() {
    if(state.validationPromise)return state.validationPromise;
    state.validationPromise=(async()=>{
      try {
        const response=await fetch('api/collections/validation',{cache:'no-cache'});if(!response.ok)return;
        const data=await response.json();
        const validFraction=value=>typeof value==='number'&&Number.isFinite(value)&&value>=0&&value<=1;
        if(!data.available||!validFraction(data.label_agreement)||!Number.isInteger(data.image_count)||data.image_count<1||typeof data.dataset!=='string')return;
        const target=el('scanner-validation-content');target.replaceChildren();
        target.append(make('p','Current scanner · '+data.dataset+' · '+fmt(data.image_count)+' labeled images','validation-dataset'));
        const metrics=make('div',undefined,'validation-metrics');const agreement=make('p');agreement.append(make('strong',score(data.label_agreement)),make('span','benchmark accuracy'));metrics.append(agreement);
        if(data.weakest_class&&validFraction(data.weakest_class.recall)){const weak=make('p');weak.append(make('strong',score(data.weakest_class.recall)),make('span','lowest class recall · '+String(data.weakest_class.name||data.weakest_class.code||'reported class')));metrics.append(weak);}
        target.append(metrics);
        if(typeof data.model_name==='string')target.append(make('p',data.model_name,'scope-note'));
        if(typeof data.scope_note==='string')target.append(make('p',data.scope_note,'scope-note'));
        target.append(make('p','Benchmark accuracy describes the configured scanner. Saved collections keep their recorded model versions; this number does not establish accuracy on your images.','scope-note'));
        const limitations=(data.limitations||[]).filter(item=>typeof item==='string');if(limitations.length){const list=make('ul');limitations.forEach(item=>list.append(make('li',item)));target.append(list);}
        if(safeURL(data.report_url)){const link=make('a','View the validation record');link.href=safeURL(data.report_url);target.append(link);}
        el('scanner-validation').hidden=false;
      } catch(_) { /* Validation evidence is optional; imports remain available. */ }
    })();return state.validationPromise;
  }
  async function home() {
    window.TissueScanCheck?.clear();
    overview=null;workflowRequest++;workflowLoading=false;batch=null;batchActive=false;pendingBatchId=null;
    closeImageForNavigation();
    resetFindings();stopTeachingPoll();state.teaching=null;state.teachingRequest++;
    window.TissueNavigation?.setCollection(null);
    stopPoll(); ++state.request; ++state.imageRequest; state.collection=null; el('collection-workspace').hidden=true; el('collections-home').hidden=false; el('collection-title').textContent='Image collections'; el('collection-subtitle').textContent='Add images, scan for tissue types, and inspect the results.'; el('collection-breadcrumb').textContent='Your workspace';el('add-collection').className='primary';el('collections-loading').hidden=false;el('collection-list').setAttribute('aria-busy','true');
    const next=new URLSearchParams(location.search).get('next');
    const choosing=['search','describe'].includes(next);
    if(choosing){el('collection-title').textContent='Choose a collection';el('collection-subtitle').textContent=next==='search'?'Find tissue in your images, then keep the useful matches.':'Write a finding with images from the same collection.';}
    let config;try{config=await api();}finally{el('collections-loading').hidden=true;el('collection-list').setAttribute('aria-busy','false');}state.config=config;configureImport();loadValidation();
    const rows=config.collections || []; const list=el('collection-list'); list.replaceChildren();
    if(!config.available) { const empty=make('div',undefined,'collections-empty'); empty.append(make('h2','Collection import is not configured'),make('p','This server has not enabled image collection storage. Existing study and search workspaces are still available.')); list.append(empty); el('add-collection').disabled=true; return; }
    if(!rows.length) { const empty=make('div',undefined,'collections-empty'); empty.append(make('h2','Add your first collection'),make('p','Upload image tiles or choose a folder. A scan groups supported images by predicted tissue type, ready for inspection.')); const add=make('button','Add your first collection','primary'); add.type='button'; add.onclick=openImport; empty.append(add); list.append(empty); return; }
    const choices=[];
    rows.forEach(collection=>{
      const row=make('article',undefined,'collection-row'); const preview=collection.thumbnail_url || collection.preview_url || collection.cover_url;
      if(preview) {const image=make('img',undefined,'collection-cover'); loadImage(image,preview,'Preview from '+collection.name); row.append(image);} else {const placeholder=make('div',fmt(collection.image_count)+' images','collection-cover collection-cover-placeholder'); row.append(placeholder); if(collection.image_count) api('/'+encodeURIComponent(collection.id)+'/images?limit=1').then(data=>{const first=data.images?.[0]; if(first && placeholder.isConnected) {const image=make('img',undefined,'collection-cover'); loadImage(image,first.thumbnail_url,'Preview from '+collection.name); placeholder.replaceWith(image);}}).catch(()=>{});}
      const text=make('div',undefined,'collection-row-detail'); const heading=make('h2'); const link=make('a',collection.name); link.href=collectionURL(collection.id); link.onclick=event=>{event.preventDefault(); act(()=>openCollection(collection.id));}; heading.append(link); const metadata=make('div',undefined,'collection-row-meta');metadata.append(make('p',[fmt(collection.image_count)+' images',date(collection.created_at)].filter(Boolean).join(' · ')),make('span',statusName(collection),'status-pill status-'+collection.status));text.append(heading,metadata); row.append(text);
      const open=make('a',choosing?(next==='search'?'Find tissue':'Write a finding'):'Open collection','secondary');
      if(choosing){
        open.href='semantic?collection='+encodeURIComponent(collection.id)+(next==='describe'?'&panel=describe':'&panel=find');open.classList.add('collection-choice-action');open.hidden=true;
        const note=make('p','Checking search availability…','collection-choice-status');note.setAttribute('role','status');text.append(note);
        const actions=make('div',undefined,'collection-choice-actions'),browse=make('a','Browse images','secondary');browse.href=collectionURL(collection.id);browse.onclick=link.onclick;actions.append(open,browse);row.append(actions);
        choices.push({collection,open,note});
      }
      else {open.href=collectionURL(collection.id);open.onclick=link.onclick;}
      if(!choosing)row.append(open);list.append(row);
    });
    // Verify cached search support before inviting a researcher into that tool.
    // Limit concurrent checks when a lab has many imported collections.
    let nextChoice=0;
    const checkChoice=async()=>{
      while(nextChoice<choices.length){
        const {collection,open,note}=choices[nextChoice++];if(!note.isConnected)return;
        try {
          const summary=await api('/'+encodeURIComponent(collection.id)+'/semantic');if(!note.isConnected)continue;
          open.hidden=summary.available!==true||collection.declared_domain!=='colorectal_he';
          note.textContent=!open.hidden?'Search and findings available.':collection.declared_domain!=='colorectal_he'?'Browse these images. The current search pipeline supports scanned colorectal H&E collections.':state.config?.scan_available===false&&collection.status!=='complete'?'Browse these images now. New scanning is paused; tissue search needs a completed scan.':summary.reason||'Search is unavailable for this collection. You can still browse its images.';
        } catch (_) {if(note.isConnected)note.textContent='Search availability could not be checked. You can still browse these images.';}
      }
    };
    for(let worker=0;worker<Math.min(3,choices.length);worker++)checkChoice();
  }

  function configureImport() {
    const config=state.config; if(!config) return;
    const limits=config.limits || {}; const roots=config.import_roots || [];
    el('allowed-roots').replaceChildren(...roots.map(root=>make('li',root)));
    el('folder-unavailable').hidden=roots.length>0;
    if(!roots.length && state.source==='folder') setSource('upload');
    const limitsText=[]; if(limits.max_images) limitsText.push(fmt(limits.max_images)+' images per collection'); if(limits.max_file_bytes) limitsText.push(bytes(limits.max_file_bytes)+' per image'); if(limits.max_collection_bytes) limitsText.push(bytes(limits.max_collection_bytes)+' per collection');
    el('import-limits').textContent=limitsText.join(' · ')+(limits.max_side ? '. Up to '+fmt(limits.max_side)+' pixels per side'+(limits.max_pixels?' and '+(limits.max_pixels/1000000).toFixed(1)+' megapixels per image':'')+'.' : '.');
    el('choose-upload-directory').hidden=!('webkitdirectory' in el('import-directory'));
    updateDomain(); lockImportIdentity();
  }
  function updateDomain() {
    const domain=el('import-domain').value; const supported=domain==='colorectal_he';
    el('import-submit').textContent=state.resume?.id ? 'Resume upload' : !domain ? 'Add collection' : supported && state.config?.scan_available !== false ? 'Import and scan' : 'Import for browsing';
    el('domain-note').textContent=!domain ? 'Choose the organ and stain to determine which analysis is supported.' : supported ? (state.config?.scan_available === false ? 'The colorectal scanner is currently unavailable. Images can be imported now and scanned when it is available.' : 'Uses the colorectal tissue classifier. Select Other if the organ or stain differs.') : 'Import and browse these images. The current classifier is not validated for this collection type.';
  }
  function setSource(source) {
    state.source=source; el('upload-fields').hidden=source!=='upload'; el('folder-fields').hidden=source!=='folder';
    el('upload-tab').setAttribute('aria-pressed',String(source==='upload')); el('folder-tab').setAttribute('aria-pressed',String(source==='folder')); el('import-path').required=source==='folder';
  }
  function selectedFiles(files, directory=false) {
    const selected=[...files]; state.files=directory ? selected.filter(file=>/\.(jpe?g|png|tiff?|svs|ndpi)$/i.test(file.name)) : selected;
    const skipped=selected.length-state.files.length;
    el('upload-selection').textContent=state.files.length ? fmt(state.files.length)+' selected · '+bytes(state.files.reduce((sum,file)=>sum+file.size,0))+(directory?' · folder paths preserved':'')+(skipped?' · '+fmt(skipped)+' non-image files skipped':'') : 'No image files selected.';
    if(state.upload && !state.busy) state.upload.items=null;
  }
  function openImport(resume=null) {
    if(state.busy) return;
    const current=resume?.id ? resume : null;
    if(!current || state.resume?.id!==current.id) {
      state.upload=null; state.files=[]; el('import-form').reset(); selectedFiles([]);
    }
    state.resume=current;
    error('','import-error'); el('import-progress').hidden=!state.upload; el('upload-outcomes').hidden=true; el('pause-upload').hidden=true;
    el('import-heading').textContent=current ? 'Resume this collection’s upload' : 'Add an image collection';
    el('import-resume-note').hidden=!current;
    if(current) {
      el('import-name').value=current.name; el('import-domain').value=current.declared_domain; setSource('upload');
      el('import-resume-note').textContent=fmt(current.image_count)+' images already retained in “'+current.name+'”. Reselect the same files or folder. Matching completed uploads are skipped; this resumes the same collection.';
    } else setSource('upload');
    configureImport(); if(state.upload) renderUploadProgress(); el('import-dialog').showModal(); (current ? el('import-files') : el('import-name')).focus();
  }
  function closeImport() {
    if(state.busy) return;
    el('import-dialog').close();
    if(state.resume?.id && state.collection?.id!==state.resume.id) act(()=>openCollection(state.resume.id));
  }

  // The secure portal uses native SHA-256. The small fallback also permits
  // reproducible upload identities on an HTTP compute-node connection.
  async function digestBytes(input) {
    const bytes=input instanceof Uint8Array ? input : new Uint8Array(input);
    if(globalThis.crypto?.subtle) return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(value=>value.toString(16).padStart(2,'0')).join('');
    return sha256Fallback(bytes);
  }
  function sha256Fallback(bytes) {
    const k=[0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    const hash=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    const size=Math.ceil((bytes.length+9)/64)*64, padded=new Uint8Array(size); padded.set(bytes); padded[bytes.length]=0x80;
    const view=new DataView(padded.buffer); view.setUint32(size-8,Math.floor(bytes.length/0x20000000)); view.setUint32(size-4,(bytes.length*8)>>>0);
    const words=new Uint32Array(64), rotate=(value,bits)=>(value>>>bits)|(value<<(32-bits));
    for(let offset=0;offset<size;offset+=64) {
      for(let i=0;i<16;i++) words[i]=view.getUint32(offset+i*4);
      for(let i=16;i<64;i++) {const x=words[i-15],y=words[i-2]; words[i]=(words[i-16]+(rotate(x,7)^rotate(x,18)^(x>>>3))+words[i-7]+(rotate(y,17)^rotate(y,19)^(y>>>10)))>>>0;}
      let [a,b,c,d,e,f,g,h]=hash;
      for(let i=0;i<64;i++) {const first=(h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+k[i]+words[i])>>>0; const second=((rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c)))>>>0; h=g;g=f;f=e;e=(d+first)>>>0;d=c;c=b;b=a;a=(first+second)>>>0;}
      [a,b,c,d,e,f,g,h].forEach((value,index)=>hash[index]=(hash[index]+value)>>>0);
    }
    return hash.map(value=>value.toString(16).padStart(8,'0')).join('');
  }
  async function uploadReceipts(id) {
    const receipts=new Map(); if(!state.config?.capabilities?.resumable_uploads) return receipts;
    let offset=0;
    while(true) {const data=await api('/'+encodeURIComponent(id)+'/uploads?offset='+offset+'&limit=500'); for(const row of data.uploads||[]) receipts.set(row.upload_key,row); offset+=(data.uploads||[]).length; if(offset>=data.total || !data.uploads?.length) return receipts;}
  }
  let uploadProgressTimer=null, uploadProgressAt=0;
  function renderUploadProgress(force=false) {
    const elapsed=performance.now()-uploadProgressAt;
    if(!force && elapsed<200) {
      if(!uploadProgressTimer)uploadProgressTimer=setTimeout(()=>{uploadProgressTimer=null;renderUploadProgress(true);},200-elapsed);
      return;
    }
    if(uploadProgressTimer){clearTimeout(uploadProgressTimer);uploadProgressTimer=null;}
    uploadProgressAt=performance.now();
    const upload=state.upload; if(!upload) return;
    const items=upload.items||[], done=items.filter(item=>['accepted','duplicate','rejected','retained'].includes(item.status));
    const failures=items.filter(item=>item.status==='failed'); const current=items.find(item=>item.status==='sending'||item.status==='hashing');
    const node=el('import-progress'); node.hidden=false; node.replaceChildren();
    const heading=upload.phase==='finalizing' ? 'Finishing the import…' : upload.phase==='paused' ? 'Upload paused' : upload.phase==='stopped' ? 'Import needs another try' : upload.phase==='preparing' ? 'Checking previous uploads…' : 'Importing your images';
    node.append(make('strong',heading),progressBar(done.length,items.length||state.files.length));
    node.append(make('p',fmt(done.length)+' of '+fmt(items.length||state.files.length)+' selected files processed.'));
    const counts=[['accepted','accepted'],['retained','already retained'],['duplicate','duplicates skipped'],['rejected','not imported'],['failed','awaiting retry']].map(([status,label])=>{const count=items.filter(item=>item.status===status).length;return count?fmt(count)+' '+label:null;}).filter(Boolean);
    if(counts.length) node.append(make('p',counts.join(' · '),'scope-note'));
    if(current && upload.phase==='uploading') node.append(make('p',current.status==='hashing'?'Checking '+current.filename:current.filename,'upload-current-file'));
    if(upload.phase==='paused'||upload.phase==='stopped') node.append(make('p','Accepted files are saved. Resume to send the remaining files into this same collection.','scope-note'));
    const issues=items.filter(item=>item.reason); el('upload-outcomes').hidden=!issues.length;
    el('upload-failures').replaceChildren(...issues.slice(0,20).map(item=>make('li',item.filename+' — '+item.reason)));
    if(issues.length>20) el('upload-failures').append(make('li','Showing the first 20 file notes. The collection keeps the full import record.'));
    el('pause-upload').hidden=upload.phase!=='uploading'; el('pause-upload').disabled=upload.pauseRequested;
    if(upload.pauseRequested && upload.phase==='uploading') el('pause-upload').textContent='Pausing after active files…'; else el('pause-upload').textContent='Pause upload';
  }
  async function sendUpload(id,item) {
    const parameters=new URLSearchParams({filename:item.filename,upload_key:item.key,source_sha256:item.digest});
    for(let attempt=0;attempt<3;attempt++) {
      try {return await api('/'+encodeURIComponent(id)+'/files?'+parameters,undefined,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:item.file,signal:AbortSignal.timeout(120000)});}
      catch(problem) {
        const retryable=!problem.status || problem.status===429 || problem.status===408 || problem.status>=500;
        if(!retryable || attempt===2 || state.upload.pauseRequested || state.upload.stopRequested) throw problem;
        await new Promise(resolve=>setTimeout(resolve,Math.min(8000,(problem.retryAfter||2**(attempt+1))*1000)));
      }
    }
  }
  const creationStorageKey=window.TissueLab?.storageKey('tissue-collection-creation:'+new URL('api/collections',document.baseURI).pathname) || 'tissue-collection-creation:'+new URL('api/collections',document.baseURI).pathname;
  function savedCreation() {try{return JSON.parse(localStorage.getItem(creationStorageKey)||'null');}catch{return null;}}
  function rememberCreation(value) {state.createRequest=value;try{localStorage.setItem(creationStorageKey,JSON.stringify(value));}catch{}}
  function clearCreation(id) {const saved=state.createRequest||savedCreation();if(!id||saved?.id===id){state.createRequest=null;try{localStorage.removeItem(creationStorageKey);}catch{}}}
  function creationIdentity(payload) {
    const encoded=JSON.stringify(payload),saved=state.createRequest||savedCreation();
    if(saved?.payload===encoded)return saved;
    const random=new Uint8Array(16);globalThis.crypto.getRandomValues(random);
    const next={payload:encoded,key:[...random].map(value=>value.toString(16).padStart(2,'0')).join(''),id:null};rememberCreation(next);return next;
  }
  async function importCollection(event) {
    event.preventDefault(); error('','import-error');
    const isUpload=state.source==='upload'; const limits=state.config?.limits||{};
    if(isUpload && !state.files.length) {error('Choose the image files or folder to upload.','import-error'); el('import-files').focus(); return;}
    if(isUpload && limits.max_images && state.files.length>limits.max_images) {error('This collection accepts up to '+fmt(limits.max_images)+' images. Choose a smaller batch.','import-error');return;}
    const oversized=isUpload ? state.files.filter(file=>file.size>(limits.max_file_bytes||Infinity)) : [];
    if(oversized.length) {error(fmt(oversized.length)+' files exceed the '+bytes(limits.max_file_bytes)+' per-image limit, including '+oversized[0].name+'. Choose image tiles within the limit.','import-error');return;}
    if(isUpload && limits.max_collection_bytes && state.files.reduce((sum,file)=>sum+file.size,0)>limits.max_collection_bytes) {error('The selected files exceed the '+bytes(limits.max_collection_bytes)+' limit. Choose a smaller batch.','import-error');return;}
    busy(true);stopPoll();el('import-progress').hidden=false;el('import-progress').replaceChildren(make('p',state.resume?'Opening the saved import…':'Creating your collection…'));let created=state.resume;
    try {
      if(created) {
        const latest=await api('/'+encodeURIComponent(created.id));
        if(latest.manifest_sha256) {el('import-dialog').close(); clearCreation(latest.id);state.upload=null;state.resume=null;await openCollection(latest.id); message('This collection has already finished importing.');return;}
        if(latest.status==='failed' && latest.import_resumable) created=await api('/'+encodeURIComponent(created.id)+'/resume-import',{});
      } else {
        const domain=el('import-domain').value;
        const payload={name:el('import-name').value.trim(),source_kind:state.source,...(!isUpload?{folder_path:el('import-path').value.trim()}:{}),declared_domain:domain,auto_scan:domain==='colorectal_he'&&state.config?.scan_available!==false};
        const identity=creationIdentity(payload);
        if(identity.uncertain && !state.config?.capabilities?.idempotent_create) throw new Error('The connection was interrupted while creating this collection. Close this form and check the collection list before creating another.');
        try{created=await api('',{...payload,...(state.config?.capabilities?.idempotent_create?{client_request_id:identity.key}:{})});}
        catch(problem){identity.uncertain=!problem.status;rememberCreation(identity);throw problem;}
        identity.id=created.id;identity.uncertain=false;rememberCreation(identity);state.resume=created;lockImportIdentity();
        if(created.manifest_sha256){el('import-dialog').close();clearCreation(created.id);state.resume=null;await openCollection(created.id);message('This collection has already finished importing.');return;}
      }
      if(!isUpload) {el('import-dialog').close();clearCreation(created.id);state.resume=null;await openCollection(created.id);message('Folder import started. You can leave this page while it continues.');return;}
      state.upload=state.upload?.id===created.id ? state.upload : {id:created.id,items:null};
      const upload=state.upload; upload.pauseRequested=false;upload.stopRequested=false;upload.phase='preparing';
      if(!upload.items) upload.items=state.files.map(file=>({file,filename:file.webkitRelativePath||file.name,status:'pending',digest:null,key:null,reason:null}));
      for(const item of upload.items) if(item.status==='failed') {item.status='pending';item.reason=null;}
      renderUploadProgress(true); const receipts=await uploadReceipts(created.id); upload.phase='uploading'; renderUploadProgress(true);
      let next=0;
      const work=async()=>{
        while(!upload.pauseRequested && !upload.stopRequested && next<upload.items.length) {
          const item=upload.items[next++]; if(['accepted','duplicate','rejected','retained'].includes(item.status)) continue;
          try {
            item.status='hashing';renderUploadProgress();
            if(!item.digest) item.digest=await digestBytes(await item.file.arrayBuffer());
            if(!item.key) item.key=await digestBytes(new TextEncoder().encode(item.filename+'\n'+item.digest));
            const receipt=receipts.get(item.key);
            if(receipt && receipt.source_sha256===item.digest) {item.status=receipt.status==='accepted'?'retained':receipt.status;item.reason=receipt.reason||null;renderUploadProgress();continue;}
            if(upload.pauseRequested||upload.stopRequested) {item.status='pending';break;}
            item.status='sending';renderUploadProgress(); const result=await sendUpload(created.id,item);
            item.status=result.accepted?'accepted':result.duplicate?'duplicate':'rejected';item.reason=result.reason||null;
          } catch(problem) {item.status='failed';item.reason=problem.message;upload.stopRequested=true;}
          renderUploadProgress();
        }
      };
      await Promise.all([work(),work()]);
      const pending=upload.items.some(item=>['pending','failed','hashing','sending'].includes(item.status));
      if(pending) {upload.phase=upload.pauseRequested?'paused':'stopped';renderUploadProgress(true);return;}
      upload.phase='finalizing';renderUploadProgress(true);
      const finished=await api('/'+encodeURIComponent(created.id)+'/finalize',{});
      el('import-dialog').close();clearCreation(created.id);state.upload=null;state.resume=null;el('import-form').reset();selectedFiles([]);
      await openCollection(finished.id);message('Import complete. Accepted images are preserved with their scan and import records.');
    } catch(problem) {
      error(problem.message,'import-error');
      if(state.upload) {state.upload.phase='stopped';renderUploadProgress(true);}
      if(created) {state.resume=created;message('This collection is saved. Resume its upload to continue, or inspect the accepted images.');}
    } finally {busy(false);}
  }

  function viewURL() {
    if(!state.collection) return;
    const url=new URL(location.href); url.searchParams.set('collection',state.collection.id);url.searchParams.delete('import');
    const values={q:state.query,label:state.label,review:state.review==='all'?'':state.review,page:state.offset?String(1+Math.floor(state.offset/state.limit)):'',size:state.limit===24?'':String(state.limit),prediction_version:versionId(),image:detailImageId,batch:batchActive?batch?.plan.id:null};
    for(const [key,value] of Object.entries(values)) {if(value) url.searchParams.set(key,value);else url.searchParams.delete(key);}
    history.replaceState({},'',url);
    displayedViewURL=url.href;
    rememberBrowseView();
  }
  async function openCollection(id, changeLocation=true) {
    closeImageForNavigation();
    overview=null;workflowRequest++;workflowLoading=true;batch=null;batchActive=false;batchRequestKey=null;pendingBatchId=null;
    stopPoll();stopTeachingPoll();state.teaching=null;state.teachingRequest++;el('collection-teaching').hidden=true;el('show-collection-teaching').hidden=true;el('collection-teaching').open=false;el('collection-version-bar').hidden=true; if(changeLocation) setLocation(id);
    window.TissueNavigation?.setCollection(id,new URLSearchParams(location.search).get('prediction_version'));
    state.semantic=null; ++state.semanticRequest;
    resetFindings(id);
    for(const key of ['search-collection','collection-search-tab','collection-findings-tab','collection-findings','collection-search-status']) el(key).hidden=true;
    const query=new URLSearchParams(location.search);state.version=query.get('prediction_version'); state.limit=[24,48,96].includes(Number(query.get('size')))?Number(query.get('size')):24;
    pendingBatchId=/^[a-f0-9]{32}$/.test(query.get('batch')||'')?query.get('batch'):null;
    restoreBrowseView=!['q','label','review','page','size','image'].some(key=>query.has(key));pendingImageId=validImageId(query.get('image'));detailImageId=null;browseNotice();
    state.offset=(Math.min(1000000,Math.max(1,Number.parseInt(query.get('page'),10)||1))-1)*state.limit;
    state.label=query.get('label')||'';state.review=['needs_review',...humanReviewFilters].includes(query.get('review'))?query.get('review'):'all';state.query=(query.get('q')||'').slice(0,250);state.imageSignature=null;
    el('filename-search').value=state.query;el('collections-home').hidden=true;el('collection-workspace').hidden=false;el('collection-gallery').replaceChildren();el('images-message').textContent='Loading images…';await refreshCollection(id,true);window.scrollTo({top:0,behavior:'instant'});
  }
  async function refreshCollection(id=state.collection?.id, loadImages=true) {
    if(!id) return; const request=++state.request;
    if(!state.config) {state.config=await api(); configureImport();}
    let collection=await api('/'+encodeURIComponent(id)+(state.version?'?prediction_version='+encodeURIComponent(state.version):'')); if(request!==state.request) return;
    if(restoreBrowseView) {
      restoreBrowseView=false;
      const saved=savedBrowseView(id,collection.selected_prediction_version?.id||state.version||'original');
      if(saved) {state.limit=saved.limit;state.offset=saved.offset;state.query=saved.query;state.label=saved.label;state.review=saved.review;pendingImageId=saved.image;el('filename-search').value=state.query;
        if(saved.offset||saved.query||saved.label||saved.review!=='all'||saved.limit!==24||saved.image)browseNotice('Resumed your previous view in this tab.');}
    }
    if(!state.config?.capabilities?.filename_search) {state.query='';el('filename-search').value='';}
    if(state.label && !state.config.classes.some(row=>row.code===state.label))state.label='';
    state.collection=collection;state.version=collection.selected_prediction_version?.id || state.version || 'original';renderCollection();loadTeaching(id); loadCollectionSearch(id); if(loadImages) await loadImagesPage();loadVisualWorkflows(id);
    stopPoll(); if(activeStatuses.has(collection.status)||collection.scan?.retry_blocked) state.poll=setTimeout(()=>act(()=>refreshCollection(id,true)),5000);
  }
  async function loadCollectionSearch(id) {
    const request=++state.semanticRequest;
    findings.abort?.abort();findings.request++;findings.busy=false;
    try {
      const summary=await api('/'+encodeURIComponent(id)+'/semantic');
      if(request!==state.semanticRequest || state.collection?.id!==id)return;
      if(summary.collection?.id!==id)throw new Error('Search status belongs to another collection.');
      state.semantic=summary;updateImageFindingLink();
      const search=withVersion('semantic?collection='+encodeURIComponent(id));
      for(const key of ['search-collection','collection-search-tab']) {el(key).href=search+'&panel=find';el(key).hidden=!summary.available;}
      el('collection-findings-tab').href=search+'&panel=describe';el('collection-findings-tab').hidden=!summary.available;
      el('collection-search-status').hidden=Boolean(summary.available);
      el('collection-search-status').textContent=state.config?.scan_available===false&&state.collection.status!=='complete'&&state.collection.declared_domain==='colorectal_he'?'You can browse these images. New scanning is paused, so tissue search is not available yet.':summary.reason || 'Search will be available when this collection’s image features are ready.';
      const saved=summary.descriptions || [];
      const sameHead=saved[0]?.id===findings.rows[0]?.id;
      const merged=new Map(saved.map(row=>[row.id,row]));
      for(const row of findings.rows)if(!merged.has(row.id))merged.set(row.id,row);
      findings.rows=[...merged.values()];findings.total=Math.max(Number(summary.descriptions_total)||saved.length,findings.rows.length);
      if(!sameHead||findings.nextOffset===null)findings.nextOffset=saved.length;
      if(findings.rows.length>=findings.total)findings.nextOffset=null;
      el('collection-findings-status').textContent='';renderFindings();
    } catch (_) {
      if(request!==state.semanticRequest || state.collection?.id!==id)return;
      state.semantic=null;updateImageFindingLink();
      for(const key of ['search-collection','collection-search-tab','collection-findings-tab']) el(key).hidden=true;
      el('collection-search-status').hidden=false;
      el('collection-search-status').textContent='Search status is unavailable. You can keep inspecting images; Refresh will check again.';
      renderFindings();
    }
  }
  function resetFindings(id=null) {
    findings.abort?.abort();findings.request++;
    findings={collectionId:id,rows:[],total:0,nextOffset:null,busy:false,request:0,abort:null};
    el('collection-findings-status').textContent='';renderFindings();
  }
  function renderFindings() {
    const id=findings.collectionId,search='semantic?collection='+encodeURIComponent(id||'');
    el('collection-findings').hidden=!findings.rows.length;
    el('collection-findings-list').replaceChildren(...findings.rows.map(record=>{
      const row=make('article',undefined,'collection-finding');
      const link=make('a',record.text || 'Saved finding');link.href=search+'&description='+encodeURIComponent(record.id)+'&prediction_version='+encodeURIComponent(record.prediction_version?.id||'original');
      const report=make('a','View report','secondary');report.href='report?kind=description&collection='+encodeURIComponent(id)+'&id='+encodeURIComponent(record.id);
      row.append(link,make('p',record.reviewer || 'Saved description','scope-note'),report);return row;
    }));
    const more=el('collection-findings-more');more.hidden=findings.nextOffset===null;more.disabled=findings.busy;
    more.textContent=findings.busy?'Loading older findings…':'Show older findings';
  }
  async function olderFindings() {
    const current=findings,id=current.collectionId;
    if(current.busy||current.nextOffset===null||state.collection?.id!==id)return;
    const request=++current.request,offset=current.nextOffset;
    current.busy=true;current.abort=new AbortController();el('collection-findings-status').textContent='';renderFindings();
    try {
      const page=await api('/'+encodeURIComponent(id)+'/semantic/descriptions?offset='+offset+'&limit=30',undefined,{signal:current.abort.signal});
      if(findings!==current||request!==current.request||state.collection?.id!==id)return;
      if(page.collection?.id!==id||!Array.isArray(page.descriptions))throw new Error('Older findings could not be read. Try again.');
      const merged=new Map(current.rows.map(row=>[row.id,row]));
      for(const row of page.descriptions)if(!merged.has(row.id))merged.set(row.id,row);
      current.rows=[...merged.values()];current.total=Math.max(Number(page.total)||0,current.rows.length);
      current.nextOffset=page.next_offset;
      el('collection-findings-status').textContent=current.rows.length+' of '+current.total+' saved versions shown.';
    } catch(problem) {
      if(findings===current&&request===current.request&&state.collection?.id===id&&problem.name!=='AbortError')el('collection-findings-status').textContent='Older findings could not be loaded. Your displayed findings are kept. Try again.';
    } finally {
      if(findings===current&&request===current.request){current.busy=false;current.abort=null;renderFindings();}
    }
  }
  function teachingPath(suffix='') {return '/'+encodeURIComponent(state.collection.id)+'/teaching'+suffix;}
  const candidateRows=()=>state.teaching?.candidates || [];
  const currentCandidate=()=>candidateRows().find(row=>row.id===state.teaching?.selectedCandidate) || candidateRows()[0];
  function rememberReviewer(value) {try{localStorage.setItem(reviewerStorageKey,value.trim());}catch{}}
  function reviewerField(id) {
    const field=make('label','Reviewer name'),input=make('input');field.htmlFor=id;input.id=id;input.required=true;input.maxLength=120;input.autocomplete='name';input.placeholder='Name or initials';input.value=reviewerName();input.oninput=()=>rememberReviewer(input.value);field.append(input);return field;
  }
  async function loadTeaching(id=state.collection?.id) {
    if(!id)return;const request=++state.teachingRequest;stopTeachingPoll();
    try {
      const data=await api('/'+encodeURIComponent(id)+'/teaching');
      if(request!==state.teachingRequest||state.collection?.id!==id)return;
      data.selectedCandidate=state.teaching?.selectedCandidate;state.teaching=data;renderTeaching();
      if(candidateRows().some(row=>['training','queued'].includes(row.status)))state.teachingPoll=setTimeout(()=>loadTeaching(id),2500);
    } catch(problem) {
      if(request!==state.teachingRequest||state.collection?.id!==id)return;
      el('collection-teaching').hidden=problem.status===404||!state.collection?.summary?.predicted_count;el('show-collection-teaching').hidden=el('collection-teaching').hidden;
      if(!el('collection-teaching').hidden){el('teaching-content').replaceChildren(make('p','Teaching status could not be loaded. Your collection and saved reviews are available.','scope-note'));}
    }
  }
  async function teachingAction(work) {
    if(state.teachingBusy)return;state.teachingBusy=true;error('','teaching-error');
    for(const node of el('teaching-content').querySelectorAll('button,input,select,textarea'))node.disabled=true;
    try {await work();}catch(problem){error(problem.message,'teaching-error');}
    finally {state.teachingBusy=false;renderTeaching();}
  }
  function renderTeaching() {
    const data=state.teaching;if(!data||!state.collection)return;
    const rows=candidateRows(), candidate=currentCandidate(), target=el('teaching-content');
    el('collection-teaching').hidden=!state.collection.summary?.predicted_count;el('show-collection-teaching').hidden=el('collection-teaching').hidden;
    const count=Number(data.development_count)||0,minimum=Number(data.minimum_development)||10;
    const classes=Array.isArray(data.development_classes)?data.development_classes.length:Number(data.development_classes)||0;
    el('teaching-summary').textContent=candidate?({training:'Making candidate…',queued:'Making candidate…',check_ready:'Ready to check',evaluated:'Check complete',applied:'Version available',failed:'Needs attention',closed:'Check stopped'}[candidate.status]||'Optional'):fmt(count)+' reviewed labels';
    const versions=rows.filter(row=>row.status==='applied'||row.applied_at||row.id===data.active_version?.id||row.id===data.active_version);
    const selected=versionId();
    el('collection-version-bar').hidden=!versions.length&&selected==='original';
    const versionSelect=el('prediction-version');versionSelect.replaceChildren(new Option('Original scan','original'),...versions.map(row=>new Option(row.name||'Taught version '+row.id.slice(0,8),row.id)));
    if(selected!=='original'&&!versions.some(row=>row.id===selected))versionSelect.append(new Option(state.collection.selected_prediction_version?.name||'Selected saved version',selected));
    versionSelect.value=selected;
    el('prediction-version-note').textContent=selected==='original'?'Original predictions and their review history.':'Counts and image predictions use this saved version. Original predictions remain available.';
    target.replaceChildren();
    if(rows.length>1){const chooser=make('label','Teaching record');chooser.htmlFor='teaching-candidate';const select=make('select');select.id='teaching-candidate';select.append(...rows.map(row=>new Option(row.name||row.id.slice(0,8),row.id)));select.value=candidate.id;select.onchange=()=>{data.selectedCandidate=select.value;renderTeaching();};chooser.append(select);target.append(chooser);}
    if(candidate){
      target.append(make('h3',candidate.name||'Collection candidate'));
      if(['training','queued'].includes(candidate.status)){target.append(make('p','Making a candidate from your saved labels…'),make('p','You can continue inspecting images while it runs.','scope-note'));return;}
      if(candidate.status==='closed'){
        target.append(make('p','This check was stopped before completion. Your current predictions are kept.','teaching-progress'),make('p','The saved labels and reserved images remain in this record. An incomplete check has no accuracy result.','scope-note'));
        const record=make('details');record.append(make('summary','Stopped check record'));
        record.append(make('p',fmt(candidate.check?.answered)+' of '+fmt(candidate.check?.planned)+' labels completed.','scope-note'));
        if(candidate.closure?.reviewer)record.append(make('p','Stopped by '+candidate.closure.reviewer,'scope-note'));
        if(candidate.closure?.note)record.append(make('p',candidate.closure.note));
        target.append(record);
        renderTeachingCreate(target,data,count,classes,minimum);
        if(data.available&&data.can_create===false&&count>=minimum&&classes>=2){
          const review=make('button','Review more images','primary');review.type='button';review.id='teaching-review-more';
          review.onclick=()=>act(async()=>{await filter('','unreviewed','');el('review-filter').focus();el('collection-gallery').scrollIntoView({block:'start'});});target.append(review);
        }
        return;
      }
      if(candidate.status==='failed'){target.append(make('p',candidate.failure||candidate.error||'This candidate could not be completed. Your original predictions and labels are preserved.','scope-note'));}
      else if(!candidate.result){
        const check=candidate.check||{};target.append(make('p',fmt(check.answered)+' of '+fmt(check.planned)+' separate images checked','teaching-progress'),make('p','Review this random sample without seeing either prediction. Its labels are reserved for the check.','scope-note'));
        const next=make('button',Number(check.remaining)===0?'Show check results':Number(check.answered)?'Continue image check':'Check candidate','primary');next.id='teaching-check';next.type='button';next.disabled=state.teachingBusy;
        next.onclick=()=>Number(check.remaining)===0?teachingAction(async()=>{await api(teachingPath('/candidates/'+candidate.id+'/evaluate'),{});await loadTeaching();}):openTeachingCheck(candidate.id);target.append(next);renderTeachingClose(target,candidate);return;
      }else{
        renderTeachingResult(target,candidate);
        if(candidate.status==='applied'||candidate.applied_at){const using=selected===candidate.id;target.append(make('p',using?'This version is shown in the collection.':'This checked version is available for inspection.','scope-note'));if(!using){const show=make('button','View this version','primary');show.type='button';show.onclick=()=>act(async()=>{state.version=candidate.id;await refreshCollection();});target.append(show);}}
        else if(candidate.can_apply||candidate.result.can_apply){
          const form=make('form',undefined,'teaching-form');form.append(reviewerField('teaching-apply-reviewer'));const use=make('button','Use this version','primary');use.id='teaching-apply';use.type='submit';use.disabled=state.teachingBusy;form.append(use);form.onsubmit=event=>{event.preventDefault();if(!form.reportValidity())return;const reviewer=el('teaching-apply-reviewer').value.trim();teachingAction(async()=>{await api(teachingPath('/candidates/'+candidate.id+'/apply'),{reviewer});state.version=candidate.id;el('teaching-status').textContent='This version is now used for collection predictions and new reports.';await refreshCollection();});};target.append(form);
        }else target.append(make('p',candidate.result.gate_reason||'This candidate is not eligible to apply. Keep the current version.','scope-note'));
      }
      const another=make('details',undefined,'teaching-another');another.append(make('summary','Make another candidate'));const inside=make('div');renderTeachingCreate(inside,data,count,classes,minimum);another.append(inside);target.append(another);return;
    }
    renderTeachingCreate(target,data,count,classes,minimum);
  }
  function renderTeachingClose(target,candidate) {
    if(!candidate.can_close)return;
    const draftKey=state.collection.id+':'+candidate.id,draft=teachingCloseDrafts.get(draftKey)||{};
    const details=make('details',undefined,'teaching-another');details.id='teaching-close-options';details.append(make('summary','Stop this check'));
    details.open=Boolean(draft.open);
    details.append(make('p','Stop if you cannot finish this candidate’s image check. Saved labels and reserved images are kept; the candidate will not be applied.','scope-note'),make('p','To teach again, review additional images or change a development label. Stopping cannot draw a different check for the same teaching labels.','scope-note'));
    const form=make('form',undefined,'teaching-form');form.append(reviewerField('teaching-close-reviewer'));
    const reviewerInput=form.querySelector('input');if(typeof draft.reviewer==='string')reviewerInput.value=draft.reviewer;
    const field=make('label','Reason (optional)'),note=make('textarea');field.htmlFor='teaching-close-note';note.id='teaching-close-note';note.rows=2;note.maxLength=2000;field.append(note);form.append(field);
    note.value=draft.note||'';
    const keepDraft=()=>teachingCloseDrafts.set(draftKey,{reviewer:reviewerInput.value,note:note.value,open:details.open});
    form.oninput=keepDraft;details.addEventListener('toggle',()=>{if(details.isConnected)keepDraft();});
    const stop=make('button','Stop check');stop.id='teaching-close';stop.type='submit';stop.disabled=state.teachingBusy;form.append(stop);
    form.onsubmit=event=>{event.preventDefault();if(!form.reportValidity())return;keepDraft();const id=state.collection.id,reviewer=reviewerInput.value.trim(),reason=note.value.trim();teachingAction(async()=>{
      await api('/'+encodeURIComponent(id)+'/teaching/candidates/'+candidate.id+'/close',{reviewer,note:reason});rememberReviewer(reviewer);
      teachingCloseDrafts.delete(draftKey);
      if(state.collection?.id===id){el('teaching-status').textContent='Check stopped. Your current predictions and saved reviews are kept.';await loadTeaching(id);}
    });};details.append(form);target.append(details);
  }
  function renderTeachingCreate(target,data,count,classes,minimum) {
    target.append(make('p',fmt(count)+' approved labels'+(classes?' across '+fmt(classes)+' tissue classes':''),'teaching-progress'));
    if(!data.available){target.append(make('p',data.reason||'Teaching needs a completed scan with cached image features.','scope-note'));return;}
    if(count<minimum||classes<2){
      target.append(make('p','First confirm or correct at least '+minimum+' images across two tissue classes. Those labels will be used for development.','scope-note'));
      const review=make('button','Review images','primary');review.type='button';review.onclick=async()=>{await filter('','unreviewed','');if(state.images.length)await act(()=>openImage(0));};target.append(review);return;
    }
    if(data.can_create===false){target.append(make('p',data.reason||'A new candidate cannot be created yet.','scope-note'));return;}
    const form=make('form',undefined,'teaching-form');form.append(reviewerField('teaching-reviewer'));
    const consent=make('label',undefined,'teaching-consent'),box=make('input');box.id='teaching-consent';box.type='checkbox';box.required=true;consent.append(box,document.createTextNode('Use these saved labels to train a candidate for this collection.'));form.append(consent);
    const button=make('button','Make candidate','primary');button.id='teaching-create';button.type='submit';button.disabled=state.teachingBusy;form.append(button,make('p','The next step checks up to 50 separate images before you choose a version.','scope-note'));
    form.onsubmit=event=>{event.preventDefault();if(!form.reportValidity())return;const reviewer=el('teaching-reviewer').value.trim();teachingAction(async()=>{const created=await api(teachingPath('/candidates'),{reviewer,consent:true,check_size:50});state.teaching.selectedCandidate=created.id||created.candidate?.id;el('teaching-status').textContent='Candidate requested. Your current predictions are kept while it is checked.';await loadTeaching();});};target.append(form);
  }
  function renderTeachingResult(target,candidate) {
    const result=candidate.result,n=Number(result.sample_size)||0;
    target.append(make('p','On '+fmt(n)+' randomly selected images','teaching-progress'));
    const metrics=make('div',undefined,'teaching-metrics');
    const previous=candidate.baseline_version&&candidate.baseline_version!=='original'?'Previous version':'Original scan';
    for(const [heading,value]of[[previous,result.original],['Candidate',result.candidate]]){const item=make('p');item.append(make('strong',fmt(value?.correct)+' / '+fmt(n)),make('span',heading));metrics.append(item);}target.append(metrics);
    target.append(make('p',fmt(result.paired?.fixed)+' mistakes fixed · '+fmt(result.paired?.introduced)+' new mistakes','teaching-paired'));
    target.append(make('p','This check describes the reserved images in this collection. Passing the check does not establish improvement on other images or patients.','scope-note'));
    const details=make('details',undefined,'teaching-method');details.append(make('summary','Check details'));
    details.append(make('p','Sampled from '+fmt(result.frame_size||candidate.check?.frame_size)+' eligible images, excluding development and previously checked images. Training predictions were frozen before labels were collected.','scope-note'));
    const ranges=make('p',undefined,'scope-note');for(const [heading,row]of[[previous,result.original],['Candidate',result.candidate]])if(Array.isArray(row?.interval))ranges.append(document.createTextNode(heading+' range: '+row.interval.map(score).join(' – ')+'. '));if(ranges.textContent)details.append(ranges);
    if(result.interval_scope)details.append(make('p',result.interval_scope,'scope-note'));
    details.append(make('p','These ranges describe sampling uncertainty assuming the entered reference labels are correct; they do not measure reviewer disagreement.','scope-note'));
    details.append(make('p','The intervals cover this fixed check and model pair; they do not cover selection across multiple candidate attempts.','scope-note'));
    details.append(make('p',result.gate_reason||'Applying requires a complete check with no more observed mistakes than the original scan.','scope-note'));target.append(details);
  }
  const checkDraftKey=(candidate,image)=>{const key='tissue-teaching-check:'+new URL('api/collections',document.baseURI).pathname+':'+state.collection.id+':'+candidate+':'+image;return window.TissueLab?.storageKey(key)||key;};
  async function openTeachingCheck(candidateId) {
    state.checkCandidate=candidateId;state.checkDirty=false;error('','teaching-check-error');el('teaching-check-content').replaceChildren(make('p','Loading the next check image…'));
    if(!el('teaching-check-dialog').open)el('teaching-check-dialog').showModal();
    try {await loadTeachingCheck();}catch(problem){error(problem.message,'teaching-check-error');}
  }
  async function loadTeachingCheck() {
    const candidate=state.checkCandidate,id=state.collection.id,request=++state.checkRequest;
    const current=()=>request===state.checkRequest&&candidate===state.checkCandidate&&id===state.collection?.id&&el('teaching-check-dialog').open;
    const path='/'+encodeURIComponent(id)+'/teaching/candidates/'+encodeURIComponent(candidate);
    let data;try{data=await api(path+'/check');}catch(problem){
      if(!current())return;
      const failed=make('div');failed.setAttribute('role','status');failed.append(make('p','The next check image could not be loaded. Your saved labels are kept.'),make('p',problem.message,'scope-note'));
      const retry=make('button','Retry loading check','secondary');retry.id='teaching-check-load-retry';retry.type='button';retry.onclick=()=>{if(current()){retry.disabled=true;loadTeachingCheck();}};failed.append(retry);el('teaching-check-content').replaceChildren(failed);return;
    }if(!current())return;
    el('teaching-check-progress').textContent=fmt(data.answered)+' of '+fmt(data.planned)+' labels saved';
    const target=el('teaching-check-content');target.replaceChildren();
    if(data.complete){target.append(make('p','All check labels are saved.'));const finish=make('button','Show check results','primary');finish.id='teaching-finish-check';finish.type='button';finish.onclick=async()=>{if(!current())return;finish.disabled=true;try{await api(path+'/evaluate',{});if(!current())return;state.checkDirty=false;el('teaching-check-dialog').close();await loadTeaching(id);}catch(problem){if(current()){error(problem.message,'teaching-check-error');finish.disabled=false;}}};target.append(finish);return;}
    const row=data.image,layout=make('div',undefined,'teaching-check-layout'),picture=make('img'),imagePanel=make('div',undefined,'image-detail-view');
    const imageState=reviewImage(picture,row.image_url,'Separate check image '+(data.position||Number(data.answered)+1),current,'teaching-check-image');
    imagePanel.append(picture,imageState.controls);layout.append(imagePanel);
    const form=make('form',undefined,'teaching-form');form.append(reviewerField('teaching-check-reviewer'));
    const field=make('label','Your tissue label');field.htmlFor='teaching-check-label';const select=make('select');select.id='teaching-check-label';select.required=true;select.append(new Option('Choose a tissue class',''),...(data.classes||state.config.classes||[]).map(item=>new Option(item.name,item.id||item.code)),new Option('Not sure yet','unsure'));field.append(select);form.append(field);
    const noteField=make('label','Note (optional)'),note=make('textarea');noteField.htmlFor='teaching-check-note';note.id='teaching-check-note';note.rows=2;note.maxLength=2000;noteField.append(note);form.append(noteField);
    const key=checkDraftKey(candidate,row.id);try{const draft=JSON.parse(localStorage.getItem(key)||'null');if(draft){select.value=draft.label||'';note.value=draft.note||'';}}catch{}
    const status=make('p','Your selections are kept on this device until submitted.','scope-note');status.setAttribute('role','status');
    const save=make('button','Save label & next','primary');save.id='teaching-check-save';save.type='submit';form.append(save,status);
    const refreshReady=()=>{const ready=imageState.ready();save.disabled=state.teachingBusy||!ready;if(ready&&status.textContent===readinessWarning)status.textContent=state.checkDirty?'Your entries are ready to submit.':'Select a tissue label when you are ready.';};form.refreshImageReadiness=refreshReady;imageState.subscribe(refreshReady);
    form.oninput=()=>{state.checkDirty=true;try{localStorage.setItem(key,JSON.stringify({label:select.value,note:note.value}));status.textContent='Draft saved on this device.';}catch{status.textContent='Draft has not been saved. Submit the label before leaving.';}};
    form.onsubmit=async event=>{event.preventDefault();if(!current()||state.teachingBusy)return;if(!imageState.ready()){status.textContent=readinessWarning;refreshReady();return;}if(!form.reportValidity())return;const payload={image_id:row.id,label:select.value,reviewer:form.querySelector('#teaching-check-reviewer').value.trim(),note:note.value.trim()};state.teachingBusy=true;error('','teaching-check-error');for(const input of form.querySelectorAll('input,select,textarea,button'))input.disabled=true;el('close-teaching-check').disabled=true;
      try{await api(path+'/check',payload);try{localStorage.removeItem(key);}catch{}if(!current())return;state.checkDirty=false;await loadTeachingCheck();if(payload.label==='unsure'&&id===state.collection?.id&&candidate===state.checkCandidate)error('This image still needs a tissue label to complete the check. You can pause and return later, or close this window and choose Stop this check in the teaching panel.','teaching-check-error');}
      catch(problem){if(current()){error(problem.message,'teaching-check-error');for(const input of form.querySelectorAll('input,select,textarea,button'))input.disabled=false;}}
      finally{state.teachingBusy=false;el('close-teaching-check').disabled=false;el('teaching-check-content').querySelector('form')?.refreshImageReadiness?.();}
    };layout.append(form);target.append(layout);imageState.start();
  }
  function closeTeachingCheck() {if(state.teachingBusy)return;state.checkRequest++;state.checkDirty=false;el('teaching-check-dialog').close();loadTeaching();}
  document.addEventListener('tissue:check-reservations-changed',()=>{if(state.collection)loadTeaching(state.collection.id);});
  function renderCollection() {
    const collection=state.collection; if(!collection) return;
    window.TissueScanCheck?.update(collection,versionId());
    el('add-collection').className='secondary';
    window.TissueNavigation?.setCollection(collection.id,versionId());
    el('collection-title').textContent=collection.name; el('collection-subtitle').textContent=collection.declared_domain==='colorectal_he' ? (collection.status==='complete' ? 'Browse tissue predictions and record your review.' : 'Browse your imported images. No scan results yet.') : 'Browse your imported images. This collection is outside the current classifier’s scope.';
    const back=make('a','Your collections'); back.href='collections'; back.onclick=event=>{event.preventDefault(); setLocation(null); act(home);}; el('collection-breadcrumb').replaceChildren(back,document.createTextNode(' / '+statusName(collection)));
    const facts=el('collection-facts'); facts.replaceChildren();
    [[collection.image_count,'images'],[collection.duplicate_count,'duplicates skipped'],[collection.rejected_count,'not imported']].forEach(([count,label],index)=>{if(index===0||Number(count)>0) {const item=make('span'); item.append(make('strong',fmt(count)),document.createTextNode(' '+label)); facts.append(item);}});
    el('start-scan').hidden=collection.declared_domain!=='colorectal_he' || !['ready','failed'].includes(collection.status) || !collection.image_count || !collection.manifest_sha256 || Boolean(collection.scan?.retry_blocked);
    el('start-scan').disabled=state.busy || state.config?.scan_available===false;
    el('start-scan').textContent=collection.status==='failed' ? 'Retry scan' : 'Scan images';
    el('finish-import').hidden=collection.status!=='importing' || collection.source_kind==='folder';
    el('finish-import').disabled=state.busy || !collection.image_count;
    el('resume-upload').hidden=collection.source_kind!=='upload' || !(collection.import_resumable ?? collection.status==='importing');
    el('resume-folder').hidden=collection.source_kind!=='folder' || collection.status!=='failed' || !collection.import_resumable;
    el('filename-form').hidden=!state.config?.capabilities?.filename_search;
    el('resume-folder').hidden=el('resume-folder').hidden || !state.config?.capabilities?.folder_resume;
    el('page-size').value=String(state.limit);
    el('download-collection').hidden=!(collection.summary?.predicted_count>0);
    el('download-collection').href=withVersion('api/collections/'+encodeURIComponent(collection.id)+'/export');
    el('download-collection').setAttribute('download','collection-'+collection.id+'.json');
    let reportLink=el('open-collection-report');if(!reportLink){reportLink=make('a','Open report','secondary');reportLink.id='open-collection-report';el('download-collection').before(reportLink);}reportLink.href=withVersion('report?kind=collection&id='+encodeURIComponent(collection.id));reportLink.hidden=!collection.summary?.predicted_count;
    let reviewDownload=el('download-reviews');if(!reviewDownload){reviewDownload=make('a','Export reviews','text-button');reviewDownload.id='download-reviews';el('download-collection').after(reviewDownload);}
    reviewDownload.href='api/collections/'+encodeURIComponent(collection.id)+'/reviews/export?format=csv';reviewDownload.hidden=!collection.review_summary?.reviewed;
    const reviewSelect=el('review-filter');reviewSelect.replaceChildren(...[['all','All images'],['needs_review','Flagged by model'],...(state.config?.capabilities?.image_reviews?[['unreviewed','Not yet reviewed'],['reviewed','Reviewed'],['confirmed','Saved confirmations'],['corrected','Saved corrections'],['unsure','Unsure']]:[])].map(([value,label])=>new Option(label,value)));
    renderProgress(); renderSummary(); renderIssues();renderBatchPanel();
    const select=el('class-filter'); select.replaceChildren(new Option('All classes',''),...(state.config?.classes||Object.keys(fallbackNames).map(code=>({code,name:name(code)}))).map(row=>new Option(row.name,row.code))); select.value=state.label; el('review-filter').value=state.review;
    el('clear-filters').hidden=!state.label && state.review==='all' && !state.query;
  }
  function renderProgress() {
    const collection=state.collection, scan=collection.scan||{}; const node=el('scan-progress'); node.replaceChildren(); node.hidden=!activeStatuses.has(collection.status) && collection.status!=='failed' && !(collection.status==='ready' && collection.declared_domain==='colorectal_he' && state.config?.scan_available===false);
    if(node.hidden) return;
    node.append(make('h2',statusName(collection)));
    if(collection.status==='importing') node.append(make('p',collection.source_kind==='folder' ? 'Checking files and creating thumbnails. Images appear as they are accepted.' : 'Accepted images are saved. Resume upload to send the remaining files, or finish using only the images already imported.'));
    if(collection.status==='queued') node.append(make('p','Waiting for a scheduled compute slot. You can browse the imported images while the scan is queued.'));
    if(collection.status==='scanning') {node.append(progressBar(scan.processed,scan.total||collection.image_count),make('p',fmt(scan.processed)+' of '+fmt(scan.total||collection.image_count)+' images processed. The report is available when the scan completes.'));}
    if(collection.status==='failed') {node.append(make('p',scan.error||collection.error||'The scan did not finish. Accepted images and completed predictions are preserved.'));if(collection.source_kind==='folder'&&collection.import_resumable&&collection.resume_note)node.append(make('p',collection.resume_note));}
    if(collection.status==='ready') node.append(make('p','Images are ready. Scanning is unavailable on this server right now.'));
    if(scan.job_id) node.append(make('p','Scheduled job '+scan.job_id));
  }
  async function loadVisualWorkflows(id) {
    const version=versionId(),request=++workflowRequest;
    const active=()=>request===workflowRequest&&state.collection?.id===id&&versionId()===version;
    if(!state.config?.capabilities?.visual_overview){workflowLoading=false;overview=null;renderSummary();return;}
    workflowLoading=true;overview=null;renderSummary();renderBatchPanel();
    const results=await Promise.allSettled([
      api('/'+encodeURIComponent(id)+'/overview?prediction_version='+encodeURIComponent(version)),
      api('/'+encodeURIComponent(id)+'/review-batches?prediction_version='+encodeURIComponent(version))]);
    if(!active())return;
    workflowLoading=false;
    if(results[0].status==='fulfilled'&&results[0].value.collection_id===id&&results[0].value.selected_prediction_version?.id===version)overview=results[0].value;
    renderSummary();
    const batches=results[1].status==='fulfilled'?results[1].value.batches:[];
    if(batch?.plan.collection_id!==id||batch?.plan.prediction_version.id!==version){batch=null;batchActive=false;}
    if(batch)batch=batches.find(row=>row.plan.id===batch.plan.id)||batch;
    else batch=batches.find(row=>!row.complete)||batches[0]||null;
    renderBatchPanel();
    if(pendingBatchId){const batchId=pendingBatchId,imageId=pendingImageId;pendingBatchId=null;pendingImageId=null;
      try {const restored=await api('/'+encodeURIComponent(id)+'/review-batches/'+batchId);if(!active())return;
        if(restored.plan.prediction_version.id!==version)throw new Error('This review batch belongs to another prediction version.');
        batch=restored;await resumeBatch(imageId);
      }catch(problem){if(active())error(problem.message);}
    }
  }
  function renderBatchPanel() {
    const panel=el('review-batch-panel');
    panel.hidden=!state.config?.capabilities?.review_batches||state.collection?.status!=='complete'||!state.collection?.summary?.predicted_count;
    if(panel.hidden)return;
    const current=batch?.plan.collection_id===state.collection.id&&batch?.plan.prediction_version.id===versionId()?batch:null;
    const button=el('start-review-batch'),description=el('review-batch-description'),status=el('review-batch-status'),download=el('export-review-batch');
    button.disabled=batchBusy||workflowLoading;button.textContent=batchBusy?'Preparing review…':workflowLoading?'Loading review…':current&&!current.complete?'Resume review':current?'Review another 20':'Review 20 images';
    description.textContent=current&&!current.complete?'Your fixed batch is saved. Continue where you left off.':'Inspect a varied set of up to 20 images. Save your assessments and continue later.';
    status.textContent=current?fmt(current.reviewed)+' of '+fmt(current.plan.sample.length)+' assessed'+(current.reserved?' · '+fmt(current.reserved)+' reserved for a separate check':'')+(current.complete?' · Batch finished':''):'';
    download.hidden=!current;if(current)download.href='api/collections/'+encodeURIComponent(state.collection.id)+'/review-batches/'+encodeURIComponent(current.plan.id)+'/export';
  }
  function batchImageIds() {return batch?.plan.sample.filter(row=>batch.progress.find(item=>item.image_id===row.image_id)?.status!=='reserved').map(row=>row.image_id)||[];}
  async function openBatchImage(imageId) {
    const item=batch?.plan.sample.find(row=>row.image_id===imageId);if(!item)return;
    batchActive=true;await openImage(-1,{id:imageId,filename:'Review image '+(batch.plan.sample.indexOf(item)+1)});
  }
  async function resumeBatch(imageId=null) {
    if(!batch)return;
    if(batch.plan.collection_id!==state.collection?.id||batch.plan.prediction_version.id!==versionId())throw new Error('This batch belongs to another collection or prediction version.');
    renderBatchPanel();
    if(batch.complete){message('Review batch finished. Your assessments are saved; collection accuracy has not been estimated.');return;}
    await openBatchImage(batchImageIds().includes(imageId)?imageId:batch.next_image_id);
  }
  async function startBatch() {
    if(batchBusy||workflowLoading||!canLeaveReview())return;
    const id=state.collection?.id,version=versionId();if(!id)return;
    batchBusy=true;renderBatchPanel();error('');
    try {
      let result;
      if(batch&&!batch.complete&&batch.plan.collection_id===id&&batch.plan.prediction_version.id===version)result=await api('/'+encodeURIComponent(id)+'/review-batches/'+batch.plan.id);
      else {batchRequestKey ||= freshRequestId();result=await api('/'+encodeURIComponent(id)+'/review-batches',{size:20,prediction_version:version,client_request_id:batchRequestKey});}
      if(state.collection?.id!==id||versionId()!==version)return;
      batch=result;batchRequestKey=null;await resumeBatch();
    }catch(problem){error(problem.message);}
    finally{batchBusy=false;renderBatchPanel();}
  }
  async function refreshBatch() {
    if(!batchActive||!batch)return;
    const id=batch.plan.collection_id,batchId=batch.plan.id;
    const refreshed=await api('/'+encodeURIComponent(id)+'/review-batches/'+batchId);
    if(batch?.plan.id===batchId&&state.collection?.id===id){batch=refreshed;renderBatchPanel();updateDetailNavigation();}
  }
  function renderSummary() {
    const collection=state.collection, summary=collection.summary||{}, counts=summary.counts||{}; const total=Number(summary.predicted_count)||Object.values(counts).reduce((sum,value)=>sum+Number(value||0),0);
    const classes=state.config?.classes||Object.keys(fallbackNames).map(code=>({code,name:name(code)}));
    const visual=overview?.collection_id===collection.id
      && overview.selected_prediction_version?.id===versionId()
      && overview.manifest_sha256===collection.manifest_sha256
      && overview.selected_prediction_version?.prediction_sha256===collection.selected_prediction_version?.prediction_sha256
      && Array.isArray(overview.classes)&&overview.classes.length===classes.length
      && new Set(overview.classes.map(row=>row.code)).size===classes.length
      && overview.classes.every(row=>classes.some(item=>item.code===row.code)
        && Number.isSafeInteger(row.count)&&row.count===Number(counts[row.code]||0)
        && Number.isSafeInteger(row.reviewed_current_version)&&row.reviewed_current_version>=0&&row.reviewed_current_version<=row.count)
      && overview.classes.reduce((sum,row)=>sum+row.count,0)===total ? overview : null;
    const metric=(id,value)=>{const node=el(id);node.textContent=value===null?'—':fmt(value);if(value===null)delete node.dataset.value;else node.dataset.value=String(value);};
    metric('scan-metric-scanned',total);
    el('scan-metric-coverage').textContent='of '+fmt(collection.image_count)+' imported';
    const flagged=Number.isSafeInteger(summary.uncertain_count)&&summary.uncertain_count>=0&&summary.uncertain_count<=total?summary.uncertain_count:total===0?0:null;
    metric('scan-metric-flagged',flagged);
    el('scan-filter-flagged').disabled=!flagged;
    el('scan-filter-flagged').setAttribute('aria-pressed',String(state.review==='needs_review'&&!state.label&&!state.query));
    const flagReasons=visual?.flag_reasons,flagNote=el('flag-reason-summary');
    flagNote.hidden=!flagged;
    if(flagged){
      const reasonNames={input_format:'input-format warnings',low_score:'low model scores',close_alternatives:'close alternatives',other:'other or unspecified reasons'};
      const valid=flagReasons?.flagged_count===flagged&&Object.keys(reasonNames).every(key=>Number.isSafeInteger(flagReasons.counts?.[key])&&flagReasons.counts[key]>=0&&flagReasons.counts[key]<=flagged);
      flagNote.textContent=valid?'Flag reasons: '+Object.entries(reasonNames).filter(([key])=>flagReasons.counts[key]).map(([key,label])=>fmt(flagReasons.counts[key])+' '+label).join(' · ')+'. Categories can overlap. Flags are not confirmed errors.':'Flags prompt inspection; they are not confirmed errors. Open an image to see its recorded reasons.';
    }
    metric('scan-metric-reviewed',visual?visual.classes.reduce((sum,row)=>sum+row.reviewed_current_version,0):null);
    el('scan-metric-review-note').textContent=visual?'Selected version · excludes random checks':workflowLoading?'Loading…':'Unavailable';
    el('scan-metric-reviewed').title='Images whose latest saved assessment belongs to this prediction version.';
    const target=el('class-summary'),focusedClass=target.contains(document.activeElement)?document.activeElement.closest('.class-summary-row')?.dataset.classCode:null;
    target.replaceChildren();
    if(!total) {target.append(make('p',collection.declared_domain!=='colorectal_he' ? 'No predictions: this collection is available for browsing.' : 'Predicted classes will appear here when the scan completes.','scope-note'));}
    else [...classes].sort((a,b)=>(counts[b.code]||0)-(counts[a.code]||0)).forEach(row=>{
      const count=Number(counts[row.code]||0),percent=100*count/total;
      const button=make('button',undefined,'class-summary-row');button.type='button';button.dataset.classCode=row.code;button.dataset.count=String(count);button.dataset.percent=String(percent);
      button.setAttribute('aria-pressed',String(state.label===row.code));button.setAttribute('aria-controls','collection-gallery');
      button.style.setProperty('--class-color',colors[row.code]||'#9aa8a5');
      const distribution=make('span',undefined,'class-distribution'),label=make('span',undefined,'class-label-row'),number=make('span',undefined,'class-count');
      number.append(make('strong',fmt(count)),make('span',percent.toFixed(1)+'%','class-percent'));label.append(make('span',row.name,'class-name'),number);
      const track=make('span',undefined,'class-bar-track'),fill=make('span',undefined,'class-bar-fill');track.setAttribute('aria-hidden','true');fill.style.width=percent+'%';track.append(fill);distribution.append(label,track);button.append(distribution);
      const examples=visual?.classes.find(item=>item.code===row.code)?.examples||[];
      if(examples.length){const strip=make('span',undefined,'class-example-strip');for(const example of examples.length>1?[examples[0],examples[examples.length-1]]:examples){
        const frame=make('span',undefined,'class-example'),image=make('img');
        loadImage(image,example.thumbnail_url,'Illustrative image predicted as '+row.name);
        frame.title=example.review?((example.review.prediction_version||'original')===versionId()?'Reviewed on this version':'Review belongs to an earlier version'):'Not reviewed';
        frame.append(image);strip.append(frame);
      }button.append(strip);}
      button.setAttribute('aria-label',row.name+': '+fmt(count)+' images, '+percent.toFixed(1)+' percent of scanned images. '+(state.label===row.code?'Selected. Activate to clear this class.':'Show this class and clear other filters.'));
      button.onclick=()=>filter(row.code===state.label?'':row.code,'all','');target.append(button);
    });
    const expanded=el('summary-expand').getAttribute('aria-expanded')==='true';
    el('summary-heading').textContent=expanded?'All predicted classes':'Top predicted classes';
    el('summary-expand').hidden=!total||classes.length<=4;
    el('summary-expand').textContent=expanded?'Show top 4 classes':'Show all '+classes.length+' tissue classes';
    el('summary-note').textContent='Percentage of scanned images in this collection (0–100% scale). Examples are illustrative.';
    el('summary-filter-note').hidden=!total;
    el('review-rule').textContent=total ? (flagged===total
      ? 'All images are flagged for inspection. Flags may reflect input-format differences, scores below 80%, or margins below 20 points; they are not confirmed errors.'
      : 'Flags prompt inspection, not confirmed errors. They use scores below 80%, margins below 20 points, or input-format differences. Unflagged images may also be wrong.') : '';
    let progress=el('review-progress');if(!progress){progress=make('details',undefined,'collection-review-progress');progress.id='review-progress';el('summary-expand').after(progress);}
    progress.replaceChildren();progress.hidden=!state.config?.capabilities?.image_reviews||!total;
    if(!progress.hidden){const reviewed=collection.review_summary||{};progress.append(make('summary','Review history'),make('p','Across all versions: '+fmt(reviewed.reviewed)+' of '+fmt(collection.image_count)+' images have a saved assessment','review-progress-count'));if(reviewed.reviewed)progress.append(make('p',fmt(reviewed.confirmed)+' confirmed · '+fmt(reviewed.corrected)+' corrected · '+fmt(reviewed.unsure)+' unsure','scope-note'));const start=make('button','Review unassessed images');start.type='button';start.onclick=async()=>{await filter('','unreviewed','');if(state.images.length)await act(()=>openImage(0));};progress.append(start,make('p','These counts include earlier prediction versions. The top metric counts latest assessments on the selected version only. Unsure counts as an assessment. Selected reviews do not measure collection-wide accuracy.','scope-note'));}
    const provenance=el('scan-provenance'); provenance.replaceChildren(); const list=make('dl');
    const model=collection.model; if(model) {if(typeof model==='string') list.append(make('dt','Model'),make('dd',model)); else for(const [key,value] of Object.entries(model)) {if(value!==null && typeof value!=='object') list.append(make('dt',key.replaceAll('_',' ')),make('dd',String(value)));}}
    list.append(make('dt','Collection ID'),make('dd',collection.id),make('dt','Declared image type'),make('dd',collection.declared_domain==='colorectal_he'?'Colorectal H&E':'Other / not sure'));
    provenance.append(list);
    if(focusedClass){const node=[...target.querySelectorAll('.class-summary-row')].find(item=>item.dataset.classCode===focusedClass);if(node&&getComputedStyle(node).display!=='none')node.focus({preventScroll:true});else if(!el('summary-expand').hidden)el('summary-expand').focus({preventScroll:true});}
  }
  function renderIssues() {
    const errors=state.collection.errors||[]; const count=Number(state.collection.rejected_count)||errors.length; el('collection-issues').hidden=!count && !errors.length; el('issues-heading').textContent=fmt(count)+' import or scan issues'; const list=make('ul'); for(const issue of errors) list.append(make('li',typeof issue==='string'?issue:[issue.filename,issue.reason||issue.error].filter(Boolean).join(' — '))); el('issues-list').replaceChildren(list); if(count>errors.length) el('issues-list').append(make('p','Showing '+fmt(errors.length)+' recorded issue details.','scope-note'));
  }
  async function filter(label,review,query=state.query) {browseNotice();state.label=label;state.review=review;state.query=query;state.offset=0;el('filename-search').value=query;renderCollection();await act(loadImagesPage);}
  async function loadImagesPage() {
    const id=state.collection?.id;if(!id) return;const request=++state.imageRequest;
    state.imageAbort?.abort();state.imageAbort=new AbortController();
    const query=new URLSearchParams({offset:state.offset,limit:state.limit,prediction_version:versionId()});if(state.label)query.set('label',state.label);if(state.review==='needs_review')query.set('uncertain','true');if(humanReviewFilters.includes(state.review))query.set('review',state.review);if(state.query)query.set('q',state.query);
    el('collection-gallery').setAttribute('aria-busy','true');el('collection-gallery').inert=true;if(!state.images.length&&!state.imageSignature)el('images-message').textContent='Loading images…';
    try {
      const data=await api('/'+encodeURIComponent(id)+'/images?'+query,undefined,{signal:state.imageAbort.signal});
      if(request!==state.imageRequest||state.collection?.id!==id)return;
      state.images=data.images||[];state.total=Number(data.total)||0;
      if(state.offset && !state.images.length && state.total<=state.offset) {state.offset=Math.max(0,Math.ceil(state.total/state.limit)-1)*state.limit;return loadImagesPage();}
      const signature=JSON.stringify([state.images,state.total,state.offset,state.limit,state.query,state.label,state.review]);
      if(signature!==state.imageSignature){renderImages();state.imageSignature=signature;}
      viewURL();
      if(pendingImageId && !pendingBatchId) {
        const imageId=pendingImageId;pendingImageId=null;
        const index=state.images.findIndex(row=>row.id===imageId);
        if(index>=0)await openImage(index);
        else browseNotice('The previous image is no longer on this page. Choose an image to continue.');
      }
    } catch(problem) {if(problem.name!=='AbortError'){state.images=[];state.imageSignature=null;el('collection-gallery').replaceChildren();el('images-message').textContent='Images could not be loaded. Refresh to try again.';throw problem;}}
    finally {if(request===state.imageRequest){el('collection-gallery').setAttribute('aria-busy','false');el('collection-gallery').inert=false;}}
  }
  async function goPage(page,scroll=true) {
    const pages=Math.max(1,Math.ceil(state.total/state.limit));state.offset=(Math.max(1,Math.min(pages,page))-1)*state.limit;
    await loadImagesPage();if(scroll)el('collection-gallery').scrollIntoView({block:'start'});
  }
  function renderImages() {
    const list=el('collection-gallery'); list.replaceChildren();
    const filtered=Boolean(state.label||state.review!=='all'||state.query);el('image-count').textContent=fmt(state.total)+(filtered?' matching images':' images');el('image-count').hidden=!filtered;
    el('images-message').textContent=state.images.length ? '' : state.collection.image_count ? 'No images match these filters.' : state.collection.status==='failed' ? 'No supported images are available. Check the import issues and add a new collection.' : 'Accepted images will appear here as the import progresses.';
    state.images.forEach((row,index)=>{
      const button=make('button',undefined,'collection-image'); button.type='button'; button.setAttribute('aria-label','Inspect '+imageName(row)+(row.predicted_label?', predicted '+name(row.predicted_label):''));
      if(row.thumbnail_url||row.image_url) {const image=make('img'); loadImage(image,row.thumbnail_url||row.image_url,imageName(row)); button.append(image);} else button.append(make('div','Preview unavailable','image-placeholder'));
      const text=make('span',undefined,'collection-image-text'); text.append(make('span',imageName(row),'collection-image-name'),make('span',imageFailed(row)?'Could not scan':name(row.predicted_label),'collection-image-label'));
      const detail=make('span',undefined,'collection-image-score'); if(row.predicted_label) detail.append(make('span',score(row.confidence)+' model score')); else detail.append(make('span',imageFailed(row)?'Open for details':'Awaiting scan'));
      if(imageNeedsReview(row)) detail.append(make('span','Model flag','review-flag')); text.append(detail); button.append(text); button.onclick=()=>act(()=>openImage(index)); list.append(button);
      if(row.review)text.append(make('span',row.review.label?'Researcher: '+name(row.review.label):reviewNames[row.review.disposition],'human-review-badge '+row.review.disposition));
    });
    el('collection-pagination').hidden=state.total===0; el('page-previous').disabled=state.offset===0; el('page-next').disabled=state.offset+state.limit>=state.total;
    el('page-position').textContent=(state.total?state.offset+1:0)+'–'+Math.min(state.offset+state.images.length,state.total)+' of '+fmt(state.total);
    const pages=Math.max(1,Math.ceil(state.total/state.limit));el('page-jump').max=String(pages);el('page-jump').value=String(1+Math.floor(state.offset/state.limit));el('page-total').textContent='of '+fmt(pages);el('page-size').value=String(state.limit);
  }
  function updateImageFindingLink() {
    const view=el('image-detail').querySelector('.image-detail-view');if(!view)return;
    let tools=view.explorationTools;
    if(!tools){
      tools=view;
      if(batchActive){tools=make('details',undefined,'image-extra-tools');tools.append(make('summary','Explore this image'));el('image-detail').querySelector('.image-detail-info').append(tools);}
      view.explorationTools=tools;
    }
    let use=tools.querySelector('[data-image-finding]');
    let similar=tools.querySelector('[data-image-similar]');
    const {collectionId,imageId,predictionVersion}=view.dataset;
    const current=el('image-dialog').open&&collectionId===state.collection?.id&&imageId===detailImageId&&predictionVersion===versionId();
    if(!current||state.semantic?.available!==true||state.semantic.collection?.id!==collectionId){use?.remove();similar?.remove();return;}
    if(state.semantic.image_search_available===true){
      if(!similar){similar=make('a','Find similar',batchActive?'secondary':'primary');similar.dataset.imageSimilar='';similar.onclick=event=>{if(!canLeaveReview())event.preventDefault();};tools.append(similar);}
      similar.href='semantic?collection='+encodeURIComponent(collectionId)+'&panel=find&similar='+encodeURIComponent(imageId)+'&prediction_version='+encodeURIComponent(predictionVersion);
    }else similar?.remove();
    if(!use){use=make('a','Use in a finding','secondary');use.dataset.imageFinding='';use.onclick=event=>{if(!canLeaveReview())event.preventDefault();};tools.append(use);}
    // Search availability concerns cached images; pin the prediction version
    // actually inspected, not the active version reported by its search index.
    use.href='semantic?collection='+encodeURIComponent(collectionId)+'&panel=describe&evidence='+encodeURIComponent(imageId)+'&prediction_version='+encodeURIComponent(predictionVersion);
    mountImageRegion(view);
  }
  function mountImageRegion(view) {
    const data=view.regionSource,{collectionId,imageId,predictionVersion}=view.dataset;
    if(view.regionSelector || !window.TissueRegionSelector || !data || state.semantic?.region_evidence_available!==true)return;
    const current=()=>view.isConnected && el('image-dialog').open && collectionId===state.collection?.id && imageId===detailImageId && predictionVersion===versionId();
    const host=make('div',undefined,'image-region-tools');(view.explorationTools||view).append(host);
    const save=async(bounds,search)=>{
      if(!current())throw new Error('Open the source image again before selecting its region.');
      if(!canLeaveReview())throw new Error('Save or discard the image review before continuing.');
      const record=await api('/'+encodeURIComponent(collectionId)+'/semantic/regions',{image_id:imageId,prediction_version:predictionVersion,bounds_px:bounds});
      if(record.collection_id!==collectionId || record.image_id!==imageId || record.prediction_version?.id!==predictionVersion || !/^[a-f0-9]{64}$/.test(record.id||'') || JSON.stringify(record.bounds_px)!==JSON.stringify(bounds))throw new Error('The saved region does not match this image and prediction version.');
      if(!current())return;
      if(!canLeaveReview())throw new Error('The region is saved. Save or discard your image review, then choose the region action again.');
      const query=new URLSearchParams({collection:collectionId,prediction_version:predictionVersion,panel:search?'find':'describe'});
      query.set(search?'region':'region_evidence',record.id);if(search)query.set('run_region','1');
      location.assign('semantic?'+query);
    };
    view.regionSelector=window.TissueRegionSelector.mount({host,image:view.querySelector('img'),width:data.width,height:data.height,minimum:state.semantic.region_min_size_px||32,maximum:state.semantic.region_max_size_px||4096,canLeave:canLeaveReview,
      onSearch:state.semantic.region_search_available===true?bounds=>save(bounds,true):null,onEvidence:bounds=>save(bounds,false)});
  }

  async function openImage(index, suppliedRow=null) {
    el('image-detail').querySelector('.image-detail-view')?.regionSelector?.destroy();
    reviewDirty=false;reviewChanged=false;pendingReviewRequest=null;
    const row=suppliedRow||state.images[index]; if(!row) return;if(!suppliedRow)batchActive=false; const request=++state.detailRequest;state.detailIndex=index;detailImageId=row.id;viewURL();el('image-detail-heading').textContent=imageName(row); el('image-detail').replaceChildren(make('p','Loading image prediction…')); if(!el('image-dialog').open) el('image-dialog').showModal();
    const collectionId=state.collection.id,predictionVersion=versionId();
    const currentDetail=()=>request===state.detailRequest&&el('image-dialog').open&&state.collection?.id===collectionId&&versionId()===predictionVersion&&detailImageId===row.id;
    updateDetailNavigation();let data;
    try {data=await api(withVersion('/'+encodeURIComponent(state.collection.id)+'/images/'+encodeURIComponent(row.id)));}
    catch(problem) {
      if(!currentDetail())return;
      const failed=make('div',undefined,'image-detail-error');failed.setAttribute('role','alert');failed.append(make('h3','Image details could not be loaded'),make('p',problem.message));
      const retry=make('button','Try again','primary');retry.type='button';retry.onclick=()=>act(()=>openImage(index,suppliedRow));failed.append(retry);el('image-detail').replaceChildren(failed);return;
    }
    if(!currentDetail())return;
    el('image-detail-heading').textContent=imageName(data)||imageName(row);
    const layout=make('div',undefined,'image-detail-layout'); const view=make('div',undefined,'image-detail-view'); const image=make('img');
    const imageState=reviewImage(image,data.image_url||row.image_url||row.thumbnail_url,imageName(data)||imageName(row),currentDetail,'review-image');
    Object.assign(imageState,{collectionId,predictionVersion});view.append(image,imageState.controls);layout.append(view);
    Object.assign(view.dataset,{collectionId,imageId:row.id,predictionVersion});view.regionSource=data;
    const info=make('div',undefined,'image-detail-info'); info.append(make('p',state.collection.selected_prediction_version?.name || (versionId()==='original'?'Original prediction':'Collection version prediction'),'detail-kicker'),make('h3',imageFailed(data)?'Could not scan':name(data.predicted_label)),make('p',data.predicted_label?score(data.confidence)+' model score':'No class prediction available.','detail-score'));
    if(batchActive){const reason=batch.plan.sample.find(item=>item.image_id===row.id)?.reason;const note=make('div',undefined,'batch-image-note');note.append(make('strong','Why this image'),make('p',reason||'Selected for this fixed inspection batch.'),make('p','This is targeted review. “Unsure” is a valid assessment.','scope-note'));info.append(note);}
    if(data.teaching_check_reserved)info.append(make('p','This image is reserved for a separate random check. Its check label is kept out of training.','detail-scope'));
    else if(state.config?.capabilities?.image_reviews && data.prediction)info.append(reviewForm(data,imageState));
    if(imageNeedsReview(data)) {info.append(make('span','Flagged by model','review-flag')); const reasons=make('ul',undefined,'detail-review-reasons'); (data.review_reasons||['Low model score, close alternatives or an input-format difference.']).forEach(reason=>reasons.append(make('li',reason))); info.append(reasons);}
    if(data.scope_warnings?.length) {const warnings=make('div',undefined,'detail-format-warning'); data.scope_warnings.forEach(warning=>warnings.append(make('p',warning))); info.append(warnings);}
    if(data.error) info.append(make('p',typeof data.error==='string'?data.error:JSON.stringify(data.error),'scope-note'));
    const scores=data.top_k||[]; if(scores.length&&!data.classification_scores) {const scoreDetails=make('details');scoreDetails.append(make('summary','Other class scores'));const ranked=make('ol',undefined,'top-scores');scores.forEach(item=>{const entry=make('li');entry.append(make('span',name(item.label||item.code)),make('span',score(item.score),'score-value'));ranked.append(entry);});scoreDetails.append(ranked);info.append(scoreDetails);}
    info.append(make('p','One image-level prediction. Model scores are not verified accuracy; they do not locate tissue boundaries or count cells.','detail-scope'));
    if(data.classification_scores) {const allScores=make('details'); allScores.append(make('summary','Compare all 9 class scores')); const list=make('ol',undefined,'top-scores'); Object.entries(data.classification_scores).sort((a,b)=>b[1]-a[1]).forEach(([label,value])=>{const item=make('li'); item.append(make('span',name(label)),make('span',score(value),'score-value')); list.append(item);}); allScores.append(list); info.append(allScores);}
    const details=make('details'); details.append(make('summary','Image & prediction record')); const metadata=make('dl'); [['Image ID',data.id||row.id],['Filename',imageName(data)||imageName(row)],['Dimensions',data.width&&data.height?data.width+' × '+data.height+' pixels':null],['Class margin',data.margin===undefined?null:score(data.margin)],['SHA-256',data.sha256||data.content_sha256]].forEach(([label,value])=>{if(value!==null&&value!==undefined) metadata.append(make('dt',label),make('dd',String(value)));}); details.append(metadata); info.append(details); layout.append(info); el('image-detail').replaceChildren(layout);imageState.start();updateImageFindingLink();
  }
  function reviewForm(data,imageState) {
    const form=make('form',undefined,'image-review-form');form.id='image-review-form';
    const {collectionId,predictionVersion}=imageState;
    const differentVersion=Boolean(data.review&&(data.review.prediction_version||'original')!==predictionVersion);
    const heading=make('h3','Your assessment');form.append(heading);
    if(data.review)form.append(make('p',reviewNames[data.review.disposition]+(data.review.label?' · '+name(data.review.label):'')+' · '+data.review.reviewer+((data.review.prediction_version||'original')!==versionId()?' · reviewed on another saved version':''),'saved-review-label'));
    const reviewerLabel=make('label','Reviewer name');reviewerLabel.htmlFor='reviewer-name';const reviewer=make('input');reviewer.id='reviewer-name';reviewer.name='reviewer';reviewer.maxLength=120;reviewer.required=true;reviewer.autocomplete='name';reviewer.placeholder='Name or initials';reviewer.value=reviewerName();reviewerLabel.append(reviewer);
    const decisionLabel=make('label','Assessment');decisionLabel.htmlFor='review-decision';const decision=make('select');decision.id='review-decision';decision.name='disposition';decision.required=true;decision.append(new Option('Choose an assessment',''),new Option('Confirm model prediction','confirmed'),new Option('Correct the tissue label','corrected'),new Option('Unsure / needs another opinion','unsure'));decision.value=differentVersion?'':data.review?.disposition||'';if(!data.predicted_label||data.error)decision.querySelector('[value="confirmed"]').disabled=true;decisionLabel.append(decision);
    const labelLabel=make('label','Your tissue label');labelLabel.htmlFor='review-label';const label=make('select');label.id='review-label';label.name='label';label.append(new Option('Select a tissue class',''),...(state.config?.classes||[]).filter(row=>row.code!==data.predicted_label).map(row=>new Option(row.name,row.code)));label.value=data.review?.disposition==='corrected'?data.review.label:'';labelLabel.append(label);
    const noteLabel=make('label','Note (optional)');noteLabel.htmlFor='review-note';const note=make('textarea');note.id='review-note';note.name='note';note.rows=2;note.maxLength=2000;note.placeholder='What did you notice?';note.value=data.review?.note||'';noteLabel.append(note);
    const status=make('p',data.review?'Saved revision '+data.review.revision+' · '+new Date(data.review.created_at).toLocaleString():'','image-review-status');status.id='image-review-status';status.setAttribute('role','status');status.setAttribute('aria-live','polite');status.tabIndex=-1;
    const controls=make('div',undefined,'review-actions');const save=make('button','Save review',batchActive?'secondary':'primary');save.id='save-image-review';save.type='submit';const next=make('button',batchActive&&batch.remaining<=1?'Save & finish':'Save & next',batchActive?'primary':'secondary');next.id='save-review-next';next.type='submit';next.value='next';next.disabled=!batchActive&&state.offset+state.detailIndex+1>=state.total;controls.append(...(batchActive?[next,save]:[save,next]));
    const refreshReady=()=>{const ready=imageState.ready(),blocked=reviewSaving||!ready;save.disabled=blocked;next.disabled=blocked||(!batchActive&&state.offset+state.detailIndex+1>=state.total);if(ready&&status.textContent===readinessWarning)status.textContent=reviewDirty?'Unsaved changes':'';};form.refreshImageReadiness=refreshReady;imageState.subscribe(refreshReady);
    const discard=make('button','Discard unsaved changes','text-button');discard.id='discard-review';discard.type='button';discard.hidden=true;discard.onclick=()=>{if(!imageState.current())return;reviewDirty=false;pendingReviewRequest=null;form.replaceWith(reviewForm(data,imageState));};
    const updateDecision=()=>{labelLabel.hidden=decision.value!=='corrected';label.disabled=labelLabel.hidden;label.required=!labelLabel.hidden;};decision.addEventListener('change',updateDecision);updateDecision();
    form.append(...(batchActive?[decisionLabel,labelLabel,reviewerLabel]:[reviewerLabel,decisionLabel,labelLabel]),noteLabel,controls,status,discard,make('p','Saved as a researcher observation. Model predictions stay visible and unchanged.','scope-note'));
    if(differentVersion)form.append(make('p','The saved assessment belongs to another prediction version. Choose an assessment for the version shown before saving.','scope-note'));
    if(data.review_history?.length){const history=make('details',undefined,'image-review-history');history.append(make('summary','Review history ('+data.review_history.length+')'));const list=make('ol');for(const row of [...data.review_history].reverse()){const item=make('li');item.append(make('strong','Revision '+row.revision+' · '+reviewNames[row.disposition]+(row.label?' · '+name(row.label):'')),make('span',row.reviewer+' · '+new Date(row.created_at).toLocaleString()));if(row.note)item.append(make('p',row.note));list.append(item);}history.append(list,make('p','Names are self-declared. The JSON report includes every saved revision and its image and prediction identifiers.','scope-note'));form.append(history);}
    form.addEventListener('input',()=>{reviewDirty=true;discard.hidden=false;status.textContent='Unsaved changes';});form.addEventListener('change',()=>{reviewDirty=true;discard.hidden=false;status.textContent='Unsaved changes';});
    form.onsubmit=async event=>{
      event.preventDefault();if(!imageState.current()||reviewSaving)return;if(!imageState.ready()){status.textContent=readinessWarning;refreshReady();return;}if(!form.reportValidity())return;
      const goNext=event.submitter?.value==='next';const fields={prediction_version:predictionVersion,reviewer:reviewer.value.trim(),disposition:decision.value,label:decision.value==='corrected'?label.value:null,note:note.value.trim(),expected_revision:data.review?.revision||0};const fingerprint=JSON.stringify(fields);if(pendingReviewRequest?.fingerprint!==fingerprint)pendingReviewRequest={fingerprint,key:freshRequestId()};
      reviewSaving=true;for(const node of form.querySelectorAll('button,input,select,textarea'))node.disabled=true;el('close-image').disabled=true;el('detail-previous').disabled=true;el('detail-next').disabled=true;status.textContent='Saving your review…';
      let savedThisTime=false;
      try {
        const result=await api('/'+encodeURIComponent(collectionId)+'/images/'+encodeURIComponent(data.id)+'/review',{...fields,client_request_id:pendingReviewRequest.key});
        try{localStorage.setItem(reviewerStorageKey,fields.reviewer);}catch{}
        if(!imageState.current())return;
        reviewDirty=false;reviewChanged=true;pendingReviewRequest=null;data.review=result.review;data.review_history=[...(data.review_history||[]).filter(row=>row.revision!==result.review.revision),result.review].sort((a,b)=>a.revision-b.revision);
        savedThisTime=true;
        if(batchActive){try{await refreshBatch();}catch(problem){message('Your review is saved. Batch progress could not refresh; Resume review will check it again.');}}
        state.collection.review_summary=result.review_summary;overview=null;workflowLoading=true;loadTeaching(state.collection.id);const row=state.images.find(row=>row.id===data.id);if(row)row.review=result.review;renderCollection();renderImages();form.replaceWith(reviewForm(data,imageState));loadVisualWorkflows(collectionId);
      } catch(problem){if(!imageState.current())return;status.textContent=problem.message;if(problem.status===409){const reload=make('button','Reload saved review');reload.type='button';reload.onclick=()=>{reviewDirty=false;act(()=>batchActive?openBatchImage(data.id):openImage(state.detailIndex));};status.append(document.createTextNode(' '),reload);}}
      finally {reviewSaving=false;for(const node of form.querySelectorAll('button,input,select,textarea'))node.disabled=false;updateDecision();el('close-image').disabled=false;updateDetailNavigation();el('image-review-form')?.refreshImageReadiness?.();}
      if(imageState.current()&&goNext&&savedThisTime&&!reviewDirty&&reviewChanged)await act(async()=>{if(batchActive){await refreshBatch();if(batch.complete){el('image-dialog').close();message('Review batch finished. Your assessments are saved.');}else await openBatchImage(batch.next_image_id);}else await adjacentImage(1);});
    };
    return form;
  }
  function updateDetailNavigation() {
    if(batchActive&&batch){const ids=batchImageIds(),position=ids.indexOf(detailImageId);el('detail-previous').disabled=position<=0;el('detail-next').disabled=position<0||position+1>=ids.length;el('detail-position').textContent='Batch image '+fmt(position+1)+' of '+fmt(ids.length)+' · '+fmt(batch.reviewed)+' assessed';el('close-image').textContent='Pause review';return;}
    el('close-image').textContent='Close';el('detail-previous').disabled=state.offset+state.detailIndex<=0;el('detail-next').disabled=state.offset+state.detailIndex+1>=state.total;el('detail-position').textContent='Image '+fmt(state.offset+state.detailIndex+1)+' of '+fmt(state.total);
  }
  async function adjacentImage(direction) {
    if(!canLeaveReview())return;
    if(batchActive){const ids=batchImageIds(),next=ids[ids.indexOf(detailImageId)+direction];if(next)await openBatchImage(next);return;}
    if(reviewChanged){const previousIndex=state.detailIndex,oldNext=state.images[previousIndex+direction]?.id,current=state.images[previousIndex];const stays=state.review==='all'||state.review==='needs_review'||state.review==='reviewed'||state.review===current?.review?.disposition;
      const position=state.offset+previousIndex+direction-(direction>0&&!stays?1:0);await loadImagesPage();reviewChanged=false;const found=state.images.findIndex(row=>row.id===oldNext);if(found>=0)return openImage(found);if(position>=0&&position<state.total){await goPage(Math.floor(position/state.limit)+1,false);return openImage(position-state.offset);}el('image-dialog').close();return;}
    const target=state.detailIndex+direction;
    if(target>=0&&target<state.images.length)return openImage(target);
    const position=state.offset+state.detailIndex+direction;if(position<0||position>=state.total)return;
    el('detail-previous').disabled=true;el('detail-next').disabled=true;
    await goPage(Math.floor(position/state.limit)+1,false);return openImage(position-state.offset);
  }

  el('add-collection').onclick=()=>openImport();el('close-import').onclick=closeImport;el('cancel-import').onclick=closeImport;el('import-dialog').addEventListener('cancel',event=>{if(state.busy)event.preventDefault();});
  el('scan-filter-flagged').onclick=()=>filter('',state.review==='needs_review'&&!state.label&&!state.query?'all':'needs_review','');
  el('start-review-batch').onclick=startBatch;
  el('upload-tab').onclick=()=>setSource('upload');el('folder-tab').onclick=()=>setSource('folder');el('import-domain').onchange=updateDomain;el('import-files').onchange=event=>selectedFiles(event.target.files);el('import-form').onsubmit=importCollection;
  el('choose-upload-directory').onclick=()=>el('import-directory').click();el('import-directory').onchange=event=>selectedFiles(event.target.files,true);
  document.addEventListener('click',event=>{const tools=el('collection-tools');if(tools.open&&!tools.contains(event.target))tools.open=false;});
  el('show-collection-teaching').onclick=()=>{el('collection-tools').open=false;el('collection-teaching').open=true;el('collection-teaching').scrollIntoView({block:'start'});el('collection-teaching').querySelector('summary').focus();};
  el('collection-tools').addEventListener('keydown',event=>{if(event.key==='Escape'){el('collection-tools').open=false;el('collection-tools').querySelector('summary').focus();}});
  el('pause-upload').onclick=()=>{if(state.upload){state.upload.pauseRequested=true;renderUploadProgress(true);}};
  const drop=el('upload-dropzone');drop.addEventListener('dragover',event=>{event.preventDefault();if(!state.busy)drop.classList.add('dragover');});drop.addEventListener('dragleave',()=>drop.classList.remove('dragover'));drop.addEventListener('drop',event=>{event.preventDefault();drop.classList.remove('dragover');if(!state.busy)selectedFiles(event.dataTransfer.files);});
  el('summary-expand').onclick=()=>{const expanded=el('summary-expand').getAttribute('aria-expanded')!=='true';el('summary-expand').setAttribute('aria-expanded',String(expanded));el('summary-expand').textContent=expanded?'Show top 4 classes':'Show all '+(state.config?.classes?.length||9)+' tissue classes';el('summary-heading').textContent=expanded?'All predicted classes':'Top predicted classes';el('class-summary').classList.toggle('summary-expanded',expanded);};
  el('refresh-collection').onclick=()=>act(()=>refreshCollection());
  el('collection-findings-more').onclick=olderFindings;
  el('resume-upload').onclick=()=>openImport(state.collection);
  el('resume-folder').onclick=()=>act(async()=>{busy(true);try{await api('/'+encodeURIComponent(state.collection.id)+'/resume-import',{});await refreshCollection();}finally{busy(false);}});
  el('start-scan').onclick=()=>act(async()=>{busy(true);try{await api('/'+encodeURIComponent(state.collection.id)+'/scan',{});await refreshCollection();}finally{busy(false);}});
  el('finish-import').onclick=()=>act(async()=>{busy(true);try{await api('/'+encodeURIComponent(state.collection.id)+'/finalize',{});if(state.upload?.id===state.collection.id){state.upload=null;state.resume=null;}await refreshCollection();}finally{busy(false);}});
  el('class-filter').onchange=()=>filter(el('class-filter').value,state.review);el('review-filter').onchange=()=>filter(state.label,el('review-filter').value);el('clear-filters').onclick=()=>filter('','all','');
  el('filename-form').onsubmit=event=>{event.preventDefault();filter(state.label,state.review,el('filename-search').value.trim());};
  el('filename-search').addEventListener('search',()=>{if(!el('filename-search').value)filter(state.label,state.review,'');});
  el('page-previous').onclick=()=>act(()=>goPage(Math.floor(state.offset/state.limit)));el('page-next').onclick=()=>act(()=>goPage(Math.floor(state.offset/state.limit)+2));
  el('page-jump-form').onsubmit=event=>{event.preventDefault();act(()=>goPage(Number(el('page-jump').value)||1));};
  el('page-size').onchange=()=>act(async()=>{state.limit=Number(el('page-size').value);state.offset=0;await loadImagesPage();});
  el('reset-collection-view').onclick=()=>act(async()=>{browseNotice();state.limit=24;pendingImageId=null;detailImageId=null;await filter('','all','');});
  el('close-image').onclick=()=>{if(canLeaveReview())el('image-dialog').close();};el('detail-previous').onclick=()=>act(()=>adjacentImage(-1));el('detail-next').onclick=()=>act(()=>adjacentImage(1));
  el('image-dialog').addEventListener('cancel',event=>{if(!canLeaveReview())event.preventDefault();});
  el('image-dialog').addEventListener('close',()=>{el('image-detail').querySelector('.image-detail-view')?.regionSelector?.destroy();if(skipDetailCloseRefresh){skipDetailCloseRefresh=false;return;}reviewDirty=false;reviewChanged=false;batchActive=false;detailImageId=null;state.detailRequest++;if(state.collection){viewURL();act(()=>refreshCollection());}});
  el('prediction-version').onchange=()=>act(async()=>{overview=null;workflowRequest++;workflowLoading=true;el('scan-metric-reviewed').textContent='—';delete el('scan-metric-reviewed').dataset.value;el('scan-metric-review-note').textContent='Loading…';batch=null;batchActive=false;pendingBatchId=null;batchRequestKey=null;state.version=el('prediction-version').value;state.offset=0;state.imageSignature=null;await refreshCollection();});
  el('close-teaching-check').onclick=closeTeachingCheck;
  el('teaching-check-dialog').addEventListener('cancel',event=>{event.preventDefault();closeTeachingCheck();});
  el('image-dialog').addEventListener('keydown',event=>{if(event.altKey||event.ctrlKey||event.metaKey||event.target.closest('input,textarea,select,button,a,summary'))return;if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();act(()=>adjacentImage(event.key==='ArrowLeft'?-1:1));}});
  window.addEventListener('popstate',()=>{
    if(el('image-dialog').open&&!canLeaveReview()){
      // Back has already moved the address. Replace that abandoned forward
      // entry with the displayed review; pushState does not fire popstate.
      history.pushState({},'',displayedViewURL);return;
    }
    act(async()=>{const id=new URLSearchParams(location.search).get('collection');if(id)await openCollection(id,false);else await home();});
  });
  window.addEventListener('beforeunload',event=>{if(state.busy||reviewDirty||reviewSaving||state.checkDirty||state.teachingBusy){event.preventDefault();event.returnValue='';}});
  act(async()=>{const query=new URLSearchParams(location.search),id=query.get('collection');if(id)await openCollection(id,false);else await home();if(query.get('import')==='1'&&state.config?.available)openImport();});
})();
