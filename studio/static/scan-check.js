'use strict';

// A fixed, blinded check of one saved scan. This module never starts training.
(() => {
  const $ = id => document.getElementById(id);
  const node = (tag, text, cls) => {const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el;};
  const fmt = value => Number(value).toLocaleString();
  const pct = value => (100*value).toFixed(1)+'%';
  const state = {key:null,collection:null,version:null,data:null,request:0,unitRequest:0,busy:false,reviewer:'',size:50,check:null};
  const panel=$('scan-check-panel'), content=$('scan-check-content');
  const toolbarButton=node('button','Check this scan','secondary');toolbarButton.id='show-scan-check';toolbarButton.type='button';toolbarButton.hidden=true;
  $('open-collection-report').before(toolbarButton);
  const dialog=node('dialog');dialog.id='scan-check-dialog';dialog.setAttribute('aria-labelledby','scan-check-heading');
  const heading=node('div',undefined,'dialog-heading'), title=node('div');
  title.append(node('p','Random image check','workspace-breadcrumb'));const h2=node('h2','What tissue is shown?');h2.id='scan-check-heading';title.append(h2);
  const progress=node('p');progress.id='scan-check-progress';title.append(progress);
  const pause=node('button','Pause check');pause.id='pause-scan-check';pause.type='button';heading.append(title,pause);
  const unitBody=node('div');unitBody.id='scan-check-unit';const unitError=node('p');unitError.id='scan-check-unit-error';unitError.setAttribute('role','alert');unitError.hidden=true;
  dialog.append(heading,node('p','Predictions are hidden. Choose your tissue label; these answers stay out of training.','scope-note'),unitBody,unitError);document.body.append(dialog);
  const storageKey='tissue-scan-check:'+new URL('api/collections',document.baseURI).href;
  const readLocal=key=>{try{return localStorage.getItem(storageKey+key)||'';}catch{return '';}};
  state.reviewer=readLocal(':reviewer');
  function notice(text='',unit=false){const el=unit?unitError:$('scan-check-error');el.textContent=text;el.hidden=!text;}
  function endpoint(suffix=''){return 'api/collections/'+encodeURIComponent(state.collection)+'/checks'+suffix;}
  async function request(url, body){
    const response=await fetch(url,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    let value;try{value=await response.json();}catch{throw new Error('The check service did not return a readable response. Your saved answers are preserved.');}
    if(!response.ok)throw new Error(typeof value.detail==='string'?value.detail:'The check could not be saved. Try again.');return value;
  }
  function button(text, id, action, cls='secondary'){const el=node('button',text,cls);el.type='button';if(id)el.id=id;el.onclick=action;return el;}
  function link(text, href){const el=node('a',text,'secondary');el.href=href;return el;}
  function reportURL(){return 'report?kind=collection&id='+encodeURIComponent(state.collection)+'&prediction_version='+encodeURIComponent(state.version);}
  function field(label, input){const el=node('label',label);el.append(input);return el;}
  function reviewerInput(id){const input=node('input');input.id=id;input.required=true;input.maxLength=120;input.autocomplete='name';input.value=state.reviewer;input.oninput=()=>{state.reviewer=input.value;try{localStorage.setItem(storageKey+':reviewer',input.value);}catch{}};return input;}
  function refreshTeaching(){document.dispatchEvent(new CustomEvent('tissue:check-reservations-changed'));}
  async function action(work){
    if(state.busy)return;state.busy=true;notice();const key=state.key;
    for(const el of panel.querySelectorAll('input,select,button'))el.disabled=true;
    try{await work();if(key===state.key){await load();refreshTeaching();}}
    catch(error){if(key===state.key)notice(error.message);}
    finally{state.busy=false;if(key===state.key){render();if(state.check?.status==='evaluated')panel.scrollIntoView({block:'start',behavior:'smooth'});}}
  }
  function render(){
    if(!state.data)return;const data=state.data, check=data.checks?.[0];state.check=check||null;
    content.replaceChildren();$('scan-check-status').textContent='';
    const name=data.selected_version?.name||'Selected scan';
    content.append(node('p',name+' · Review a random sample to estimate how often this scan agrees with your labels.','scan-check-intro'));
    if(!check){
      $('scan-check-summary').textContent=data.can_create?'Not checked':'Not available';
      if(!data.can_create){content.append(node('p',data.reason||'A completed, supported scan is required.','scope-note'));return;}
      const form=node('form');form.id='scan-check-start-form';form.className='scan-check-start';
      const reviewer=reviewerInput('scan-check-reviewer'), size=node('select');size.id='scan-check-size';
      for(const n of [20,50,100,200])size.append(new Option(String(n)+' images',String(n)));size.value=String(state.size);
      const count=node('p',undefined,'scope-note');count.id='scan-check-frame';
      const describe=()=>{state.size=Number(size.value);const n=Math.min(state.size,data.frame_size);count.textContent=fmt(n)+' images will be checked from '+fmt(data.frame_size)+' eligible images'+(n===data.frame_size?' — every eligible image.':'.');};size.onchange=describe;describe();
      form.append(field('Reviewer name',reviewer),field('Review budget',size));
      const submit=node('button','Start random check','primary');submit.id='start-scan-check';submit.type='submit';submit.disabled=state.busy;form.append(submit);
      form.onsubmit=event=>{event.preventDefault();if(!form.reportValidity())return;action(async()=>{const created=await request(endpoint(),{reviewer:reviewer.value.trim(),check_size:state.size,prediction_version:state.version});state.check=created;panel.open=true;await openUnit(created.id);});};
      content.append(count,form,node('p','Images matching recorded reviews, fitting examples or reserved checks are excluded. The result describes the eligible group, not automatically the whole collection.','scope-note'));
      content.append(node('p','The sample is fixed when you start. You can pause and resume; it cannot be redrawn for this prediction version.','scope-note'));
      return;
    }
    const p=check.progress;
    if(check.status==='evaluated'){
      const result=check.result;$('scan-check-summary').textContent=pct(result.accuracy)+' estimated accuracy';
      const card=node('div',undefined,'scan-check-result');card.id='scan-check-result';
      const metric=node('div');metric.append(node('strong',pct(result.accuracy)),node('span','Estimated accuracy'));
      const explanation=node('div');explanation.append(node('p',fmt(result.correct)+' of '+fmt(result.sample_size)+' checked images agreed with your labels.'));
      explanation.append(node('p','95% interval: '+pct(result.interval[0])+'–'+pct(result.interval[1])));
      explanation.append(node('p','Applies to '+fmt(result.frame_size)+' eligible images in this saved version.','scope-note'));card.append(metric,explanation);content.append(card);
      const actions=node('div',undefined,'actions'), report=link('Open report',reportURL());report.className='primary';actions.append(report);
      if(!$('collection-teaching').hidden)actions.append(button('Teach this collection','scan-check-teach',()=>{const teach=$('collection-teaching');if(teach.hidden){notice('Teaching is currently unavailable. Your check and report remain available.');return;}teach.open=true;teach.scrollIntoView({block:'start'});teach.querySelector('summary').focus();}));content.append(actions);
      const detail=node('details',undefined,'scan-check-method');detail.append(node('summary','What this result covers'),node('p',result.scope,'scope-note'));
      if(check.identity_scope)detail.append(node('p',check.identity_scope,'scope-note'));
      detail.append(node('p','The 95% interval describes random sampling uncertainty. It does not account for errors in reviewer labels. Teaching uses separate inspection labels.','scope-note'),link('Download check record',endpoint('/'+check.id+'/export')));content.append(detail);
    }else if(check.status==='closed'){
      $('scan-check-summary').textContent='Stopped · no accuracy estimate';
      content.append(node('p','This check was stopped with '+fmt(p.answered)+' of '+fmt(p.planned)+' images labeled. Its records are preserved, and the same version cannot draw another sample.'));
      content.append(link('Download incomplete check record',endpoint('/'+check.id+'/export')));
    }else{
      $('scan-check-summary').textContent=fmt(p.answered)+' / '+fmt(p.planned)+' checked';
      content.append(node('p',fmt(p.answered)+' of '+fmt(p.planned)+' images labeled · '+fmt(p.frame_size)+' eligible images.'));
      const bar=node('progress');bar.max=p.planned;bar.value=p.answered;bar.setAttribute('aria-label','Random check progress');content.append(bar);
      const actions=node('div',undefined,'actions');
      if(p.remaining)actions.append(button(p.answered?'Resume check':'Label sampled images','resume-scan-check',()=>openUnit(check.id),'primary'));
      else actions.append(button('View result','evaluate-scan-check',()=>action(()=>request(endpoint('/'+check.id+'/evaluate'),{})),'primary'));
      content.append(actions,node('p','Label every sampled image before calculating accuracy. Unsure answers stay unresolved; they are never replaced with easier images.','scope-note'));
      if(check.can_close){
        const stop=node('details',undefined,'scan-check-stop');stop.append(node('summary','Stop this check'));
        stop.append(node('p','Stopping keeps your answers and reserves the sample. No accuracy estimate will be reported, and this version cannot start another check.','scope-note'));
        const form=node('form');const reviewer=reviewerInput('scan-check-stop-reviewer'), note=node('input');note.maxLength=2000;
        form.append(field('Reviewer name',reviewer),field('Reason (optional)',note));const submit=node('button','Stop and keep records');submit.type='submit';submit.id='stop-scan-check';form.append(submit);
        form.onsubmit=event=>{event.preventDefault();if(form.reportValidity())action(()=>request(endpoint('/'+check.id+'/close'),{reviewer:reviewer.value.trim(),note:note.value}));};stop.append(form);content.append(stop);
      }
    }
    if(state.busy)for(const el of panel.querySelectorAll('input,select,button'))el.disabled=true;
  }
  async function load(){
    if(!state.key)return;const current=++state.request;
    try{const data=await request(endpoint('?prediction_version='+encodeURIComponent(state.version)));if(current!==state.request)return;state.data=data;render();}
    catch(error){if(current===state.request){notice(error.message);$('scan-check-summary').textContent='Check unavailable';content.replaceChildren(button('Retry loading check','retry-scan-check',load));}}
  }
  function draftKey(id,image){return storageKey+':draft:'+state.collection+':'+id+':'+image;}
  async function openUnit(id){
    const current=++state.unitRequest;notice('',true);unitBody.replaceChildren(node('p','Loading the next sampled image…'));progress.textContent='';if(!dialog.open)dialog.showModal();
    try{
      const unit=await request(endpoint('/'+id+'/unit'));if(current!==state.unitRequest)return;
      unitBody.replaceChildren();progress.textContent=fmt(unit.answered)+' of '+fmt(unit.planned)+' images labeled';
      if(unit.complete){unitBody.append(node('p','Every sampled image has a label.'),button('View result','finish-scan-check',()=>{pauseDialog();action(()=>request(endpoint('/'+id+'/evaluate'),{}));},'primary'));return;}
      const image=node('img');image.alt='Sampled tissue image '+unit.position;image.id='scan-check-image';
      const url=new URL(unit.image.image_url,document.baseURI);if(url.origin!==location.origin||!['http:','https:'].includes(url.protocol))throw new Error('The sampled image URL is invalid.');
      const imageStatus=node('p','Loading image…','scope-note');imageStatus.setAttribute('role','status');
      const form=node('form');form.id='scan-check-label-form';const select=node('select');select.id='scan-check-label';select.required=true;
      select.append(new Option('Choose a tissue category',''));for(const item of unit.classes)select.append(new Option(item.name,item.id));select.append(new Option('Unsure — needs another look','unsure'));
      const reviewer=reviewerInput('scan-check-label-reviewer'), note=node('textarea');note.id='scan-check-label-note';note.rows=2;note.maxLength=2000;
      const key=draftKey(id,unit.image.id);try{const saved=JSON.parse(localStorage.getItem(key)||'null');if(saved){select.value=saved.label||'';note.value=saved.note||'';}}catch{}
      const draft=node('p',undefined,'scope-note');draft.setAttribute('role','status');
      const saveDraft=()=>{try{localStorage.setItem(key,JSON.stringify({label:select.value,note:note.value}));draft.textContent='Draft saved in this browser. Submit to record your label.';}catch{draft.textContent='Draft cannot be saved in this browser. Submit your label before leaving.';}};
      select.onchange=saveDraft;note.oninput=saveDraft;
      form.append(field('Tissue category',select),field('Reviewer name',reviewer));const notes=node('details');notes.append(node('summary','Add a note'),field('Note (optional)',note));form.append(notes,draft);
      const submit=node('button','Save label and continue','primary');submit.id='submit-scan-check-label';submit.type='submit';submit.disabled=true;form.append(submit);
      let loaded=false;
      image.onload=()=>{if(current===state.unitRequest){loaded=!!image.naturalWidth;submit.disabled=!loaded;imageStatus.textContent=loaded?'':'Image could not be decoded.';}};
      image.onerror=()=>{submit.disabled=true;loaded=false;imageStatus.replaceChildren(node('span','Image unavailable. Labeling is paused. '),button('Retry image',null,()=>openUnit(id)));};image.src=url.href;
      form.onsubmit=async event=>{
        event.preventDefault();if(state.busy||!loaded||!form.reportValidity())return;state.busy=true;pause.disabled=true;notice('',true);
        for(const el of form.querySelectorAll('input,select,textarea,button'))el.disabled=true;
        const label=select.value;
        try{await request(endpoint('/'+id+'/labels'),{image_id:unit.image.id,label,reviewer:reviewer.value.trim(),note:note.value});try{localStorage.removeItem(key);}catch{}await openUnit(id);if(label==='unsure')notice('Your Unsure response is saved. This image still needs a tissue label; you may pause or stop this check.',true);await load();refreshTeaching();}
        catch(error){notice(error.message,true);for(const el of form.querySelectorAll('input,select,textarea,button'))el.disabled=false;}
        finally{state.busy=false;pause.disabled=false;render();}
      };
      const grid=node('div',undefined,'scan-check-label-layout'), picture=node('div');picture.append(image,imageStatus);grid.append(picture,form);unitBody.append(grid);
    }catch(error){notice(error.message,true);unitBody.replaceChildren(button('Retry next image','retry-check-unit',()=>openUnit(id)));}
  }
  function pauseDialog(){if(state.busy)return;++state.unitRequest;dialog.close();load();}
  pause.onclick=pauseDialog;dialog.addEventListener('cancel',event=>{event.preventDefault();pauseDialog();});
  toolbarButton.onclick=()=>{panel.open=true;panel.scrollIntoView({block:'start',behavior:'smooth'});panel.querySelector('summary').focus();load();};
  window.addEventListener('beforeunload',event=>{if(state.busy){event.preventDefault();event.returnValue='';}});
  window.TissueScanCheck={
    clear(){++state.request;++state.unitRequest;state.key=null;state.reviewSignature=null;state.data=null;panel.hidden=true;toolbarButton.hidden=true;if(dialog.open)dialog.close();},
    update(collection,version){
      if(collection.status!=='complete'||collection.declared_domain!=='colorectal_he'||!collection.summary?.predicted_count){this.clear();return;}
      const key=collection.id+':'+version, changed=state.key!==key, signature=JSON.stringify(collection.review_summary||{}), reviewed=state.reviewSignature!==signature;
      state.reviewSignature=signature;state.collection=collection.id;state.version=version;state.key=key;panel.hidden=false;toolbarButton.hidden=false;
      if(changed){state.data=null;state.check=null;panel.open=false;content.replaceChildren(node('p','Loading check status…'));$('scan-check-summary').textContent='Estimate accuracy on your images';notice();if(dialog.open)dialog.close();++state.unitRequest;}
      if(!state.busy&&(changed||reviewed))load();
    }
  };
})();
