'use strict';
// Keep unavailable workspaces hidden, including links rendered after startup.
(() => {
  const marker = location.pathname.indexOf('/feedback-studies/');
  const base = marker >= 0 ? location.pathname.slice(0, marker + 1) : '';
  const route = location.pathname.split('/').filter(Boolean).pop() || '';
  const params = new URLSearchParams(location.search);
  let capabilities = null;
  let collection = params.get('collection') || (route==='report' && params.get('kind')==='collection' ? params.get('id') : null);
  const validVersion = value => typeof value==='string' && /^(original|[a-f0-9]{32})$/.test(value) ? value : null;
  let predictionVersion = collection ? validVersion(params.get('prediction_version')) : null;
  const setCollection = (id, version = undefined) => {
    // Pin the context actually displayed, including an explicitly selected
    // original scan when a taught version is active for the collection.
    predictionVersion = id ? (version===undefined && id===collection ? predictionVersion : validVersion(version)) : null;
    collection=id || null;
    const context = new URLSearchParams();
    if(collection) context.set('collection',collection);
    if(collection && predictionVersion) context.set('prediction_version',predictionVersion);
    for(const link of document.querySelectorAll('[data-workspace-nav]')) {
      const name=link.dataset.workspaceNav;
      if(name==='collections')link.href=base+'collections'+(context.size?'?'+context:'');
      if(name==='semantic'){
        link.href=context.size ? base+'semantic?panel=find&'+context
          : capabilities?.collections_available===false ? base+'semantic?panel=find' : base+'collections?next=search';
        link.textContent='Search images';
        link.dataset.featureLink=collection || capabilities?.collections_available!==false ? 'collections' : 'semantic';
      }
    }
  };
  const reportSection = route==='report' ? ({collection:'collections',description:'semantic',study:'research'}[params.get('kind')]) : null;
  const active = reportSection || (route === 'lab' ? 'lab' : route === 'collections' ? 'collections' : route === 'semantic' ? 'semantic' : route === 'research' || location.search.includes('study=') ? 'research' : ['classification','feedback','spatial'].includes(route) || marker >= 0 ? 'advanced' : 'workspace');
  const header = document.querySelector('.header-inner');
  if (header) {
    // Keep legacy data targets alive while every tool uses one stable shell.
    const retained = document.createElement('div'); retained.hidden = true;
    for (const id of ['collection-label','spatial-link']) { const element = document.getElementById(id); if (element) retained.append(element); }
    const nav = document.createElement('nav'); nav.className='workspace-navigation'; nav.setAttribute('aria-label','Main navigation');
    for (const [id,label,feature] of [['workspace','Workspace',null],['collections','Collections','collections'],['semantic','Search images','semantic'],['research','Example study','research']]) {
      const link=document.createElement('a'); link.href=base+id+(id==='research'?'?step=home':''); link.textContent=label;
      link.dataset.workspaceNav=id;
      if (id===active) link.setAttribute('aria-current','page');
      if (feature) {link.dataset.featureLink=feature;link.hidden=true;}
      nav.append(link);
    }
    const advanced=document.createElement('details'); advanced.className='optional-study-tools';
    if(active==='advanced')advanced.dataset.current='true';
    const summary=document.createElement('summary'); summary.textContent='Advanced'; advanced.append(summary);
    const menu=document.createElement('div'); menu.className='optional-tools-menu';
    for (const [path,label,feature] of [['semantic','Example image library','semantic'],['feedback','Example classifier teaching','feedback'],['classification','Tile proportion audit','classification'],['spatial','Spatial engineering pilot','spatial']]) {
      const link=document.createElement('a'); link.href=base+path; link.textContent=label;
      if (feature) {link.dataset.featureLink=feature;link.hidden=true;}
      menu.append(link);
    }
    advanced.append(menu);nav.append(advanced);
    document.addEventListener('click',event=>{if(advanced.open&&!advanced.contains(event.target))advanced.open=false;});
    document.addEventListener('keydown',event=>{
      if(event.key==='Escape'&&advanced.open&&!event.defaultPrevented){advanced.open=false;summary.focus();event.preventDefault();}
    });
    const labLink=document.createElement('a');labLink.href=base+'lab?view=team';
    labLink.textContent='Lab';labLink.hidden=true;labLink.className='lab-navigation-link';
    if(route==='lab') labLink.setAttribute('aria-current','page');
    nav.append(labLink);
    const showLab=value=>{
      labLink.hidden=!value?.configured;
      if(route==='lab') advanced.hidden=!value?.authenticated;
      if(value?.authenticated){
        labLink.title=value.workspace.name+' · '+value.user.name;
        const heading=document.querySelector('.workspace-heading h1');
        if(heading) heading.textContent=value.workspace.name;
        const savedNote=document.querySelector('.workspace-work .workspace-section-heading p');
        if(savedNote) savedNote.textContent='Collections, searches, findings and studies shared with your lab.';
      }
    };
    window.addEventListener('tissue-lab-session',event=>showLab(event.detail));
    window.TissueLab?.ready.then(showLab).catch(()=>{});
    const brand=document.createElement('a');brand.className='brand';brand.href=base+'workspace';brand.textContent='Tissology';
    header.replaceChildren(brand,nav,retained);
    setCollection(collection);
    if (!document.querySelector('link[href*="presentation.css"]')) { const css=document.createElement('link');css.rel='stylesheet';css.href=base+'static/presentation.css?v=10';document.head.append(css); }
  }
  const main=document.querySelector('main');
  if(main){main.id ||= 'main-content';const skip=document.createElement('a');skip.href='#'+main.id;skip.textContent='Skip to content';skip.className='skip-link';document.body.prepend(skip);}
  const refresh = (root = document) => {
    if (!capabilities) return;
    if (root.matches?.('[data-feature-link]')) root.hidden = !capabilities[root.dataset.featureLink + '_available'];
    root.querySelectorAll?.('[data-feature-link]').forEach(link => {
      link.hidden = !capabilities[link.dataset.featureLink + '_available'];
    });
  };
  const reload = async () => {
    try {
      const response = await fetch('api/health');
      if (!response.ok) return null;
      capabilities = await response.json();
      if (marker >= 0) {
        const parent = location.pathname.slice(0, marker + 1);
        const outer = await fetch(parent + 'api/health');
        if (outer.ok) Object.assign(capabilities, await outer.json());
      }
      setCollection(collection, predictionVersion);
      refresh();
      return capabilities;
    } catch (_) { return null; }
  };
  const ready = route==='lab' && window.TissueLab ? window.TissueLab.ready.then(value=>value.authenticated ? reload() : null).catch(()=>null) : reload();
  window.TissueNavigation = {refresh, ready, reload, setCollection};
  const observer = new MutationObserver(records => {
    for (const record of records) for (const node of record.addedNodes) {
      if (node.nodeType === 1) refresh(node);
    }
  });
  observer.observe(document.body, {childList:true, subtree:true});
})();
