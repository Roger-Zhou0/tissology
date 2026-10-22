'use strict';
(() => {
  const $=id=>document.getElementById(id), MAX_PANELS=12, CAPTION_LIMIT=280;
  const assessments={unresolved:'Assessment unresolved',supported_by_reviewer:'Supported · researcher assessment',contradicted_by_reviewer:'Contradicted · researcher assessment'};
  let source=null, settings=null, panels=[], loading=false, exporting=false, ready=false, revision=0, loadEpoch=0;
  const cached=new Map();
  const selected=()=>panels.filter(panel=>panel.selected);
  const make=(tag,text,cls)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;};
  function safeImagePath(path){
    return typeof path==='string'&&(/^api\/(?:semantic\/images\/|collections\/[a-f0-9]{32}\/semantic\/images\/)[A-Za-z0-9_-]+$/.test(path)||/^api\/collections\/[a-f0-9]{32}\/semantic\/regions\/[a-f0-9]{64}\/image$/.test(path))?path:null;
  }
  const panelIdentity=panel=>panel.region_id?'Region '+panel.region_id.slice(0,8)+' · image '+panel.image_id+' · x '+panel.bounds_px[0]+'–'+panel.bounds_px[2]+', y '+panel.bounds_px[1]+'–'+panel.bounds_px[3]+' px':'Image '+panel.image_id;
  function status(text){$('figure-status').textContent=text;}
  function error(text){$('figure-error').hidden=!text;$('figure-error').textContent=text||'';}
  function changed(){revision++;error('');renderPreview();}
  function assessment(panel){const labels=[...new Set(panel.claims.map(claim=>assessments[claim.assessment]||'Assessment unresolved'))];return labels.length===1?labels[0]:'Mixed researcher assessments · see data record';}
  function setup(){
    if($('evidence-figure-dialog'))return;
    const open=make('button','Create figure');open.id='report-create-figure';open.type='button';open.hidden=true;
    document.querySelector('.report-toolbar .actions').prepend(open);
    const dialog=make('dialog');dialog.id='evidence-figure-dialog';dialog.setAttribute('aria-labelledby','figure-heading');dialog.setAttribute('aria-describedby','figure-help');
    dialog.innerHTML='<div class="figure-heading"><div><h2 id="figure-heading">Create an evidence figure</h2><p id="figure-help">Arrange saved images and add your own captions. Your saved finding stays unchanged.</p></div><button id="figure-close" type="button">Close</button></div><div class="figure-editor"><div class="figure-controls"><label for="figure-title">Figure title <span class="figure-note">(optional)</span><input id="figure-title" type="text" maxlength="120" autocomplete="off"></label><label for="figure-columns">Images per row<select id="figure-columns"><option value="1">1</option><option value="2" selected>2</option><option value="3">3</option></select></label><p id="figure-selection" class="figure-note"></p><div id="figure-panels"></div></div><div class="figure-preview-area"><h3>Preview</h3><div class="figure-preview-frame"><canvas id="figure-preview" hidden aria-label="Evidence figure preview"></canvas><p id="figure-empty">Loading selected images…</p></div><p id="figure-status" role="status" aria-live="polite"></p><p id="figure-error" role="alert" hidden></p><button id="figure-retry" type="button" hidden>Retry missing images</button><div class="figure-downloads"><button id="figure-download-png" class="primary" type="button" disabled>Download PNG</button><button id="figure-download-json" type="button" disabled>Download data record</button></div><p class="figure-note">Captions are researcher-authored. Image IDs and recorded assessments stay attached. These selected examples do not estimate accuracy or represent the whole collection. Download the data record with your figure to retain its full source history.</p></div></div>';
    document.body.append(dialog);
    open.onclick=()=>{if(exporting)return;dialog.showModal();$('figure-title').focus();renderPanels();loadImages();};
    $('figure-close').onclick=()=>dialog.close();
    $('figure-title').oninput=changed;$('figure-columns').onchange=changed;
    $('figure-retry').onclick=()=>loadImages();
    $('figure-download-png').onclick=()=>download('png');$('figure-download-json').onclick=()=>download('json');
  }
  function configure(record,options){
    setup();source=JSON.parse(JSON.stringify(record));settings={...options};panels=[];for(const item of cached.values())URL.revokeObjectURL(item.url);cached.clear();revision++;loadEpoch++;
    for(const [claimIndex,claim]of(source.claims||[]).entries())for(const evidence of(claim.evidence?.length?claim.evidence:(claim.image_ids||[]).map(image_id=>({image_id})))){
      const region=evidence.region_id, imageId=evidence.image_id, id=region?'region:'+region:imageId;
      let panel=panels.find(item=>item.id===id);
      if(!panel){
        if(region&&(!/^[a-f0-9]{64}$/.test(region)||!Array.isArray(evidence.bounds_px)||evidence.bounds_px.length!==4||!evidence.bounds_px.every(Number.isInteger)))continue;
        const path=safeImagePath(region?options.base+'/regions/'+region+'/image':options.base+'/images/'+encodeURIComponent(imageId));
        if(!path)continue;
        panel={id,path,image_id:imageId,image_sha256:evidence.image_sha256||null,region_id:region||null,region_sha256:evidence.region_sha256||null,crop_sha256:evidence.crop_sha256||null,bounds_px:evidence.bounds_px||null,source_size_px:evidence.source_size_px||null,annotation_extent:evidence.annotation_extent||null,caption:'',selected:panels.length<MAX_PANELS,claims:[]};panels.push(panel);
      }
      panel.claims.push({statement_number:claimIndex+1,text:claim.text,assessment:claim.assessment,note:claim.note||'',assessment_source:claim.assessment_source||'researcher judgment; not automatically verified'});
    }
    $('report-create-figure').hidden=!panels.length;
    $('figure-title').value='';$('figure-columns').value=panels.length===1?'1':'2';ready=false;
  }
  function renderPanels(focus){
    const host=$('figure-panels'),scroll=host.scrollTop,active=document.activeElement;
    const restoreCaption=host.contains(active)&&active.matches('[data-figure-caption]')?{id:active.dataset.figureCaption,start:active.selectionStart,end:active.selectionEnd}:null;
    host.replaceChildren();
    const chosen=selected();$('figure-selection').textContent=chosen.length+' of '+panels.length+' saved images selected. Choose up to '+MAX_PANELS+'; captions and order stay here when you close this editor.';
    panels.forEach((panel,index)=>{
      const card=make('section',undefined,'figure-panel');card.dataset.imageId=panel.id;
      const heading=make('div',undefined,'figure-panel-heading'),pickLabel=make('label'),pick=make('input');pick.type='checkbox';pick.checked=panel.selected;pick.disabled=exporting;pick.dataset.figureSelect=panel.id;
      const position=chosen.indexOf(panel);pick.setAttribute('aria-label','Include image '+panel.id);pickLabel.append(pick,make('span',position<0?'Not included':'Panel '+String.fromCharCode(65+position)));
      const picture=cached.get(panel.id)?.image;if(picture){const thumb=picture.cloneNode();thumb.alt='Saved evidence image '+panel.id;heading.append(thumb);}
      heading.append(pickLabel);card.append(heading,make('p',panelIdentity(panel),'figure-panel-id'),make('p',assessment(panel),'figure-panel-assessment'));
      const captionLabel=make('label','Your caption (optional)'),caption=make('textarea');caption.rows=2;caption.maxLength=CAPTION_LIMIT;caption.value=panel.caption;caption.disabled=exporting;caption.dataset.figureCaption=panel.id;caption.setAttribute('aria-label','Caption for image '+panel.id);captionLabel.append(caption);card.append(captionLabel);
      caption.oninput=()=>{panel.caption=caption.value;changed();};
      pick.onchange=()=>{if(pick.checked&&selected().length>=MAX_PANELS){pick.checked=false;error('Choose at most '+MAX_PANELS+' images for one figure. Uncheck an image before adding another.');return;}panel.selected=pick.checked;revision++;renderPanels({id:panel.id,kind:'select'});loadImages();};
      const order=make('div',undefined,'figure-order');
      for(const [direction,offset]of[['up',-1],['down',1]]){
        const button=make('button',direction==='up'?'Move earlier':'Move later');button.type='button';button.dataset.figureMove=direction;button.disabled=exporting||index+offset<0||index+offset>=panels.length;button.setAttribute('aria-label',(direction==='up'?'Move earlier: ':'Move later: ')+panel.id);
        button.onclick=()=>{[panels[index],panels[index+offset]]=[panels[index+offset],panels[index]];revision++;renderPanels({id:panel.id,kind:direction});renderPreview();};order.append(button);
      }
      card.append(order);host.append(card);
    });
    host.scrollTop=scroll;
    if(focus){const card=[...host.children].find(node=>node.dataset.imageId===focus.id),preferred=card?.querySelector(focus.kind==='select'?'[data-figure-select]':'[data-figure-move="'+focus.kind+'"]');(preferred&&!preferred.disabled?preferred:card?.querySelector('[data-figure-move]:not(:disabled),[data-figure-select]:not(:disabled)'))?.focus({preventScroll:true});}
    else if(restoreCaption){const field=[...host.querySelectorAll('[data-figure-caption]')].find(node=>node.dataset.figureCaption===restoreCaption.id);field?.focus({preventScroll:true});field?.setSelectionRange(restoreCaption.start,restoreCaption.end);}
  }
  async function fetchImage(panel){
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);
    let url,decodeTimer;
    try{
      const response=await fetch(panel.path,{signal:controller.signal,credentials:'same-origin',cache:'no-cache'});
      if(!response.ok)throw new Error('Image unavailable');
      const blob=await response.blob();if(!blob.type.startsWith('image/'))throw new Error('Unreadable image');
      url=URL.createObjectURL(blob);const image=new Image();image.src=url;
      await Promise.race([image.decode(),new Promise((_,reject)=>{decodeTimer=setTimeout(()=>reject(new Error('Image decode timed out')),15000);})]);
      if(!image.naturalWidth||!image.naturalHeight)throw new Error('Empty image');
      const servedHash=await hash(await blob.arrayBuffer());
      if(panel.crop_sha256&&servedHash&&servedHash!==panel.crop_sha256)throw new Error('Saved region image differs from its recorded pixels');
      return {image,url,served_image_sha256:servedHash};
    }catch(cause){if(url)URL.revokeObjectURL(url);throw cause;}
    finally{clearTimeout(timer);clearTimeout(decodeTimer);}
  }
  async function loadImages(){
    // A later selection owns the preview; already successful images can be reused.
    const epoch=++loadEpoch;loading=true;error('');renderPreview();
    const pending=selected().filter(panel=>!cached.has(panel.id));
    const failures=[];let next=0;
    await Promise.all(Array.from({length:Math.min(3,pending.length)},async()=>{
      while(next<pending.length){const panel=pending[next++];try{const value=await fetchImage(panel);if(epoch===loadEpoch){cached.set(panel.id,value);}else URL.revokeObjectURL(value.url);}catch(_){failures.push(panel.id);}}
    }));
    if(epoch!==loadEpoch)return;
    loading=false;renderPanels();renderPreview();
    if(failures.length)error(failures.length+' selected image'+(failures.length===1?' is':'s are')+' unavailable. Retry the images, or uncheck them to make a smaller figure. Your captions and layout are retained.');
  }
  function lines(ctx,text,width){
    const output=[];
    for(const paragraph of String(text).replace(/\r/g,'').split('\n')){
      let line='';
      for(const word of paragraph.split(/\s+/).filter(Boolean)){
        if(ctx.measureText(line?(line+' '+word):word).width<=width){line+=(line?' ':'')+word;continue;}
        if(line){output.push(line);line='';}
        for(const character of Array.from(word)){if(line&&ctx.measureText(line+character).width>width){output.push(line);line='';}line+=character;}
      }
      output.push(line);
    }
    return output;
  }
  function textBlock(ctx,text,x,y,width,fontSize=22,color='#342c28',weight='normal'){
    ctx.font=weight+' '+fontSize+'px Arial, sans-serif';ctx.fillStyle=color;ctx.textBaseline='top';
    const wrapped=lines(ctx,text,width),height=fontSize*1.4;
    wrapped.forEach((line,index)=>ctx.fillText(line,x,y+index*height));return wrapped.length*height;
  }
  function renderPreview(){
    ready=false;const chosen=selected(),missing=chosen.filter(panel=>!cached.has(panel.id));
    $('figure-download-png').disabled=true;$('figure-download-json').disabled=true;
    $('figure-retry').hidden=loading||!missing.length;$('figure-retry').disabled=exporting;
    if(!chosen.length||missing.length||loading){
      $('figure-preview').hidden=true;$('figure-empty').hidden=false;
      $('figure-empty').textContent=!chosen.length?'Choose at least one saved image for this figure.':loading?'Loading '+chosen.length+' selected image'+(chosen.length===1?'…':'s…'):'Waiting for '+missing.length+' selected image'+(missing.length===1?'.':'s.');
      status(loading?'Loading image evidence. You can keep editing your title and captions.':!chosen.length?'No images selected.':'Export will be available when every selected image has loaded.');return;
    }
    const canvas=$('figure-preview'),ctx=canvas.getContext('2d'),columns=Math.min(Number($('figure-columns').value),chosen.length),cell=480,pad=36,gap=28,imageHeight=360,width=pad*2+columns*cell+(columns-1)*gap;
    const title=$('figure-title').value.trim();ctx.font='bold 32px Arial, sans-serif';
    const titleHeight=title?lines(ctx,title,width-pad*2).length*45+22:0;
    const measurements=chosen.map(panel=>{
      ctx.font='normal 18px Arial, sans-serif';const idHeight=lines(ctx,panelIdentity(panel),cell).length*25.2;
      const assessmentHeight=lines(ctx,assessment(panel),cell).length*25.2;
      ctx.font='normal 23px Arial, sans-serif';const captionHeight=panel.caption.trim()?lines(ctx,panel.caption.trim(),cell).length*32.2:0;
      return {idHeight,assessmentHeight,captionHeight,height:38+imageHeight+14+idHeight+6+assessmentHeight+(captionHeight?12+captionHeight:0)};
    });
    const rowHeights=[];for(let index=0;index<chosen.length;index+=columns)rowHeights.push(Math.max(...measurements.slice(index,index+columns).map(item=>item.height)));
    const sourceName=source.collection?.name||'Prepared colorectal development library';
    const version=source.prediction_version?.name||source.prediction_version?.id||'No collection prediction version';
    const footer=['Selected evidence · '+sourceName,'Finding '+source.id+' · '+version,'Researcher '+source.reviewer+' · captions are researcher-authored','Researcher assessments; not independently verified. Selected images do not represent the whole collection.'];
    ctx.font='normal 18px Arial, sans-serif';const footerHeight=footer.reduce((sum,line)=>sum+lines(ctx,line,width-pad*2).length*25.2+5,0);
    const height=Math.ceil(pad*2+titleHeight+rowHeights.reduce((sum,value)=>sum+value,0)+(rowHeights.length-1)*gap+30+footerHeight);
    if(height>18000){canvas.hidden=true;$('figure-empty').hidden=false;$('figure-empty').textContent='This figure is too tall. Shorten captions or use more columns.';error('The figure exceeds the export size limit. No content has been clipped.');return;}
    canvas.width=width;canvas.height=height;ctx.fillStyle='#fffdf9';ctx.fillRect(0,0,width,height);
    if(title)textBlock(ctx,title,pad,pad,width-pad*2,32,'#342c28','bold');
    let rowY=pad+titleHeight;
    chosen.forEach((panel,index)=>{
      const column=index%columns;if(index&&column===0)rowY+=rowHeights[Math.floor(index/columns)-1]+gap;
      const x=pad+column*(cell+gap),item=cached.get(panel.id),image=item.image;
      textBlock(ctx,String.fromCharCode(65+index),x,rowY,cell,26,'#342c28','bold');
      const imageY=rowY+38,scale=Math.min(cell/image.naturalWidth,imageHeight/image.naturalHeight),w=image.naturalWidth*scale,h=image.naturalHeight*scale;
      ctx.fillStyle='#f2eee8';ctx.fillRect(x,imageY,cell,imageHeight);ctx.drawImage(image,x+(cell-w)/2,imageY+(imageHeight-h)/2,w,h);
      let y=imageY+imageHeight+14;y+=textBlock(ctx,panelIdentity(panel),x,y,cell,18,'#645d55');y+=6+textBlock(ctx,assessment(panel),x,y+6,cell,18,'#645d55');
      if(panel.caption.trim())textBlock(ctx,panel.caption.trim(),x,y+12,cell,23);
    });
    let footerY=pad+titleHeight+rowHeights.reduce((sum,value)=>sum+value,0)+(rowHeights.length-1)*gap+30;
    for(const line of footer)footerY+=textBlock(ctx,line,pad,footerY,width-pad*2,18,'#645d55')+5;
    canvas.hidden=false;$('figure-empty').hidden=true;ready=true;
    $('figure-download-png').disabled=exporting;$('figure-download-json').disabled=exporting;
    status(chosen.length+' image'+(chosen.length===1?'':'s')+' · '+width+' × '+height+' pixels. Original image proportions retained; no scale bars added.');
  }
  async function hash(bytes){
    if(!globalThis.crypto?.subtle)return null;
    return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),byte=>byte.toString(16).padStart(2,'0')).join('');
  }
  function save(blob,filename){const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(url),30000);}
  async function download(kind){
    if(!ready||loading||exporting)return;
    exporting=true;error('');$('report-create-figure').disabled=true;const frozenRevision=revision;
    const inputs=[...$('evidence-figure-dialog').querySelectorAll('input,textarea,select,button:not(#figure-close)')];inputs.forEach(input=>input.disabled=true);
    try{
      const canvas=$('figure-preview'),blob=await new Promise((resolve,reject)=>canvas.toBlob(value=>value?resolve(value):reject(new Error('Your browser could not create this image. Use fewer panels or shorter captions and retry.')),'image/png'));
      const pngHash=await hash(await blob.arrayBuffer());if(revision!==frozenRevision)throw new Error('The figure changed during export. Review the preview and download again.');
      const payload={schema:'tissue-evidence-figure-v1',exported_at:new Date().toISOString(),title:$('figure-title').value.trim(),caption_authorship:'researcher-authored; not independently verified',scope:'Selected saved evidence; not a representative sample or an accuracy estimate',source:{finding_id:source.id,finding_sha256:source.sha256,parent_id:source.parent_id||null,collection_id:settings.collectionId||null,catalog_sha256:source.catalog_sha256,collection:source.collection||null,prediction_version:source.prediction_version||null,reviewer:source.reviewer,record:source},layout:{requested_columns:Number($('figure-columns').value),columns:Math.min(Number($('figure-columns').value),selected().length),image_fit:'contain; saved image or explicitly recorded region; no additional crop',scale_bars:false,width:canvas.width,height:canvas.height,png_sha256:pngHash},panels:selected().map((panel,index)=>({order:index+1,panel_label:String.fromCharCode(65+index),image_id:panel.image_id,image_sha256:panel.image_sha256,region_id:panel.region_id,region_sha256:panel.region_sha256,crop_sha256:panel.crop_sha256,bounds_px:panel.bounds_px,source_size_px:panel.source_size_px,coordinate_convention:panel.region_id?'integer pixels; top-left origin; half-open right and bottom':null,served_image_sha256:cached.get(panel.id).served_image_sha256,source_url:panel.path,annotation_extent:panel.annotation_extent,caption:panel.caption.trim(),assessment_label:assessment(panel),claim_assessments:panel.claims,natural_width:cached.get(panel.id).image.naturalWidth,natural_height:cached.get(panel.id).image.naturalHeight}))};
      const filename='tissue-figure-'+source.id.slice(0,8);
      save(kind==='png'?blob:new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),filename+'.'+kind);
      status(kind==='png'?'Figure downloaded. Download its data record too to retain captions, image identities and the saved finding.':'Data record downloaded with the exact figure order, captions, image identities and saved finding.');
    }catch(cause){error(cause.message||'The figure could not be downloaded. Your edits are retained; please retry.');}
    finally{exporting=false;$('report-create-figure').disabled=false;inputs.forEach(input=>input.disabled=false);renderPanels();$('figure-download-png').disabled=!ready;$('figure-download-json').disabled=!ready;}
  }
  window.TissueEvidenceFigure={configure};
})();
