'use strict';
(() => {
  // Collection relevance choices are handled in the current search. The
  // prepared library's saved guidance must never cross into another catalog.
  if (new URLSearchParams(location.search).has('collection')) return;
  let definition = null, applied = null, beforeApply = null, positives = [], negatives = [], dirty = true, available = false;
  const field = id => document.getElementById('guide-'+id);
  const indicator=()=>{field('applied-indicator').textContent=applied?'· applied':'· optional';};
  const status = text => { field('status').textContent = text; };
  const remember = () => semanticDraft.schedule();
  const same = (a,b) => JSON.stringify([...a].sort()) === JSON.stringify([...b].sort());
  async function request(path='',body) {
    const response = await fetch('api/guidance'+path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const value = await response.json();
    if (!response.ok) {
      const names={goal:'Goal',reviewer:'Researcher / team',inclusion_criteria:'Appearance',exclusions:'Exclusions',note:'Notes'};
      throw new Error(typeof value.detail === 'string' ? value.detail : Array.isArray(value.detail)
        ? value.detail.map(item=>(names[item.loc?.at(-1)] || 'Guidance')+': '+item.msg).join('; ')
        : 'Check the guidance fields and try again.');
    }
    return value;
  }
  function matches() {
    return applied && element('search-text').value === applied.effective_query && same(state.positive,applied.positive_ids) && same(state.negative,applied.negative_ids);
  }
  function sync() {
    if (applied && !matches()) {
      applied = null; beforeApply = null; field('stop').disabled = true;indicator();
      status('Your query or examples changed. Previous guidance is no longer applied; review and apply it again if wanted.');
    }
  }
  window.TaskGuidance = {
    sync,
    clearApplied() {
      const wasApplied=Boolean(applied);
      applied=null; beforeApply=null;
      field('stop').disabled=true; indicator();
      if(wasApplied)status('Previous guidance stopped for this new search. Its draft and saved versions are kept.');
      remember();
      return wasApplied;
    },
    draft() {
      return {...body(),definition_id:definition?.id || null,panel_open:element('guide-panel').open};
    },
    async restoreDraft(draft) {
      if (!draft) return;
      const names=[['goal','goal',300],['inclusion','inclusion_criteria',280],['exclusions','exclusions',600],['reviewer','reviewer',100],['note','note',2000],['scope','scope',30]];
      const ids = values => Array.isArray(values) && values.length<=40 && values.every(id=>typeof id==='string' && /^[A-Za-z0-9_-]{1,200}$/.test(id));
      if (!names.every(([,key,max])=>typeof draft[key]==='string' && draft[key].length<=max) ||
          !['current_task','saved_definition'].includes(draft.scope) || !ids(draft.positive_ids) || !ids(draft.negative_ids)) {
        status('The guidance recovery copy could not be read. Your other draft fields remain available.'); return;
      }
      definition=null; applied=null; beforeApply=null; dirty=true;
      let previous='';
      if (draft.definition_id && /^[A-Za-z0-9_-]{1,200}$/.test(draft.definition_id)) {
        try {
          const saved=await request('/definitions/'+encodeURIComponent(draft.definition_id));
          if (saved.status==='active') definition=saved;
          else previous=' The previous definition is no longer active; this is a new guidance draft.';
        } catch (_) { previous=' The previous definition is unavailable; this is a new guidance draft.'; }
      }
      for (const [name,key] of names) field(name).value=draft[key];
      positives=[...draft.positive_ids]; negatives=[...draft.negative_ids];
      field('stop').disabled=true;field('save').disabled=false;field('apply').disabled=false;field('retract').disabled=!definition;
      indicator(); preview();
      element('guide-panel').open=Boolean(draft.panel_open); syncPanel();
      status('Guidance draft recovered. It has not been applied; review it and choose Apply and search when ready.'+previous);
    },
    current() { sync(); return applied?.id || null; },
    loadedSearch(result) {
      if (!applied) { beforeApply=null; return; }
      if (applied && result.guidance_id !== applied.id) {
        applied = null; beforeApply = null; field('stop').disabled = true;indicator();
        status('Opened a search without applying the current guidance.');
      }
    }
  };
  function preview() {
    const goal=field('goal').value.trim(), inclusion=field('inclusion').value.trim();
    field('preview').textContent='Visible query when applied: '+goal+(inclusion ? '. '+inclusion : '');
    field('examples').textContent=positives.length+' relevant and '+negatives.length+' not-a-match examples attached. Their tissue labels are unchanged.';
    field('example-preview').replaceChildren(...[...positives.map(id=>[id,true]),...negatives.map(id=>[id,false])].slice(0,10).map(([id,positive])=>{
      const figure=make('figure',undefined,positive?'positive-example':'negative-example');
      const image=make('img');image.src=imageFor(id);image.alt=(positive?'Relevant':'Not a match')+' guidance example';
      figure.append(image,make('figcaption',positive?'Relevant':'Not a match'));return figure;
    }));
  }
  function body() {
    return {goal:field('goal').value,inclusion_criteria:field('inclusion').value,exclusions:field('exclusions').value,
      reviewer:field('reviewer').value,note:field('note').value,scope:field('scope').value,
      positive_ids:positives,negative_ids:negatives,parent_id:definition?.id || null};
  }
  function load(record) {
    definition=record; dirty=false;
    for (const [name,key] of [['goal','goal'],['inclusion','inclusion_criteria'],['exclusions','exclusions'],['reviewer','reviewer'],['note','note'],['scope','scope']]) field(name).value=record[key];
    positives=[...record.positive_ids];negatives=[...record.negative_ids];preview();
    const active=record.status === 'active';
    field('save').disabled=!active;field('apply').disabled=!active;field('retract').disabled=!active;
    status('Opened guidance '+record.id.slice(0,8)+' · '+record.status+'. Opening it does not apply it.');
    remember();
  }
  async function refresh() {
    const value=await request();available=value.available;
    element('guide-panel').hidden=!available;
    field('toggle').hidden=!available;
    field('saved').replaceChildren();
    for (const record of value.definitions || []) {
      const row=make('div',undefined,'saved-item');
      row.append(make('p',record.goal+' · '+record.status+' · '+(record.scope==='saved_definition'?'reusable definition':'task guidance')));
      const open=make('button','Open guidance');open.type='button';open.onclick=()=>load(record);row.append(open);field('saved').append(row);
    }
  }
  async function save() {
    if (!field('goal').value.trim()) {
      field('goal').focus();
      throw new Error('Describe what you want useful matches to show before saving or applying guidance.');
    }
    if (!field('reviewer').value.trim()) {
      field('reviewer').closest('details').open=true;
      field('reviewer').focus();
      throw new Error('Enter a researcher or team name before saving or applying guidance.');
    }
    if (!definition || dirty) {
      const revisedId=definition?.id;
      definition=await request('/definitions',body());dirty=false;
      if (applied?.definition_id===revisedId) { applied=null;field('stop').disabled=true;indicator(); }
      field('retract').disabled=false;
    }
    await refresh();
    remember();
    return definition;
  }
  const panel=element('guide-panel');
  const syncPanel=()=>{
    element('search-workspace').classList.toggle('guidance-open',panel.open);
    field('toggle').setAttribute('aria-expanded',String(panel.open));
    remember();
  };
  panel.ontoggle=syncPanel;
  field('toggle').onclick=()=>{
    panel.open=!panel.open;syncPanel();
    if (panel.open) {
      if (!field('goal').value.trim() && !definition) { field('goal').value=element('search-text').value;dirty=true;preview(); }
      if (window.innerWidth<=700) panel.scrollIntoView({block:'start'});
    }
  };
  field('close').onclick=()=>{panel.open=false;syncPanel();field('toggle').focus();};
  for (const name of ['goal','inclusion','exclusions','reviewer','note','scope']) field(name).oninput=()=>{dirty=true;preview();remember();};
  field('copy-examples').onclick=()=>{positives=[...state.positive];negatives=[...state.negative];dirty=true;preview();remember();status('Examples copied into this guidance draft. Apply explicitly to use the new definition.');};
  field('save').onclick=()=>act(async()=>{await save();status('Guidance saved without applying it.');});
  field('apply').onclick=()=>act(async()=>{
    const record=await save();
    if (!beforeApply) beforeApply={text:element('search-text').value};
    applied=await request('/definitions/'+record.id+'/apply',{});
    element('search-text').value=applied.effective_query;
    state.positive=new Set(applied.positive_ids);state.negative=new Set(applied.negative_ids);
    field('stop').disabled=false;indicator();renderChoices();renderResults();
    status('Guidance applied to this search only. Check the search notices for how examples and written exclusions were used.');
    await search();
  });
  field('stop').onclick=()=>{
    applied=null;field('stop').disabled=true;indicator();
    if (beforeApply) element('search-text').value=beforeApply.text;
    beforeApply=null;state.positive.clear();state.negative.clear();renderChoices();renderResults();
    status('Guidance stopped and image choices cleared. Search again to update the displayed results.');
  };
  field('new').onclick=()=>{
    definition=null;dirty=true;positives=[];negatives=[];
    for (const name of ['goal','inclusion','exclusions','note']) field(name).value='';
    field('scope').value='current_task';field('save').disabled=false;field('apply').disabled=false;field('retract').disabled=true;preview();
    status('New guidance draft. Existing applied guidance stays in effect until you stop or replace it.');
    remember();
  };
  field('retract').onclick=()=>act(async()=>{
    if (!definition) return;
    await request('/definitions/'+definition.id+'/retract',{reviewer:field('reviewer').value,note:field('note').value});
    applied=null;field('stop').disabled=true;indicator();
    if (beforeApply) element('search-text').value=beforeApply.text;
    beforeApply=null;state.positive.clear();state.negative.clear();renderChoices();renderResults();
    field('retract').disabled=true;field('save').disabled=true;field('apply').disabled=true;
    await refresh();status('Guidance retracted. Future searches cannot use it; saved searches and previous versions remain unchanged.');
  });
  // Old running servers do not expose this feature. Avoid probing a missing API.
  (async()=>{
    try {
      const response=await fetch('api/semantic');const summary=await response.json();
      if (!summary.guidance_available) return;
      await refresh();preview();
      if (summary.exclusion_text_supported===false) field('exclusion-note').textContent='To exclude a pattern, mark images Not a match. Written exclusions are saved as notes; they do not filter results.';
      const health=await (await fetch('api/health')).json();
      field('classifier-link').hidden=!health.feedback_available;
    } catch (error) { status(error.message); }
  })();
})();
