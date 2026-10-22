'use strict';

// A rectangle in the decoded source image's pixel coordinates. This tool only
// records a user's selection; it never supplies segmentation or localization.
window.TissueRegionSelector = (() => {
  let sequence=0;
  const make=(tag,text,className)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;};
  function mount({host,image,width,height,onSearch,onEvidence,minimum=32,maximum=4096,initialBounds=null,canLeave=()=>true}) {
    if(!host||!image||!Number.isInteger(width)||!Number.isInteger(height)||width<minimum||height<minimum)return null;
    const identity='region-selector-'+(++sequence);
    let bounds=null,drag=null,busy=false,disposed=false,previewReady=false,previewFailed=false,originalHidden=image.hidden;
    const root=make('section',undefined,'region-selector');root.dataset.regionSelector='';
    const toggle=make('button','Select region','region-toggle');toggle.type='button';toggle.setAttribute('aria-expanded','false');toggle.setAttribute('aria-controls',identity);
    const editor=make('div',undefined,'region-editor');editor.id=identity;editor.hidden=true;
    const hint=make('p','Drag a rectangle over the image, or enter its pixel bounds below.','region-hint');hint.id=identity+'-hint';
    const tools=make('div',undefined,'region-view-tools');
    const zoomLabel=make('label','Zoom ');const zoom=make('select');zoom.setAttribute('aria-label','Region image zoom');for(const [value,label]of [['1','Fit image'],['2','2×'],['4','4×']])zoom.append(new Option(label,value));zoomLabel.append(zoom);
    const clear=make('button','Clear selection');clear.type='button';clear.dataset.regionClear='';tools.append(zoomLabel,clear);
    const viewport=make('div',undefined,'region-viewport');viewport.dataset.regionViewport='';
    const stage=make('div',undefined,'region-stage');stage.dataset.regionStage='';stage.setAttribute('aria-label','Draw region on '+(image.alt||'source image'));stage.setAttribute('aria-describedby',hint.id);
    const picture=make('img');picture.src=image.src;picture.alt=image.alt||'Source image';picture.draggable=false;
    const rectangle=make('div',undefined,'region-rectangle');rectangle.hidden=true;rectangle.setAttribute('aria-hidden','true');stage.append(picture,rectangle);viewport.append(stage);
    const fields=make('div',undefined,'region-bounds');fields.setAttribute('aria-label','Region bounds in source pixels');
    const inputs={};for(const [key,label,min,max]of [['x','Left',0,width-minimum],['y','Top',0,height-minimum],['width','Width',minimum,Math.min(width,maximum)],['height','Height',minimum,Math.min(height,maximum)]]){
      const labelNode=make('label',label);const input=make('input');input.type='number';input.inputMode='numeric';input.step='1';input.min=String(min);input.max=String(max);input.dataset.regionBound=key;input.setAttribute('aria-label',label+' in pixels');input.setAttribute('aria-describedby',identity+'-dimensions');labelNode.append(input);fields.append(labelNode);inputs[key]=input;
    }
    const dimensions=make('p',width+' × '+height+' source pixels · Minimum '+minimum+' × '+minimum+' pixels','region-dimensions');dimensions.id=identity+'-dimensions';
    const status=make('p','Loading source image…','region-status');status.dataset.regionStatus='';status.setAttribute('role','status');status.setAttribute('aria-live','polite');
    const actions=make('div',undefined,'region-actions');
    const search=make('button','Search this region','primary');search.type='button';search.dataset.regionSearch='';search.hidden=!onSearch;
    const evidence=make('button','Use in a finding');evidence.type='button';evidence.dataset.regionEvidence='';evidence.hidden=!onEvidence;actions.append(search,evidence);
    const scope=make('p','Search compares this crop with whole images. It does not locate matching regions inside the results.','region-scope');scope.hidden=!onSearch;
    editor.append(hint,tools,viewport,fields,dimensions,status,actions,scope);root.append(toggle,editor);host.replaceChildren(root);
    const valid=value=>Array.isArray(value)&&value.length===4&&value.every(Number.isInteger)&&value[0]>=0&&value[1]>=0&&value[2]<=width&&value[3]<=height&&value[2]-value[0]>=minimum&&value[3]-value[1]>=minimum&&value[2]-value[0]<=maximum&&value[3]-value[1]<=maximum;
    function draw() {
      rectangle.hidden=!bounds;
      if(bounds){const[x0,y0,x1,y1]=bounds;Object.assign(rectangle.style,{left:100*x0/width+'%',top:100*y0/height+'%',width:100*(x1-x0)/width+'%',height:100*(y1-y0)/height+'%'});}
      search.disabled=busy||!previewReady||!valid(bounds);evidence.disabled=busy||!previewReady||!valid(bounds);clear.disabled=busy||!bounds;
      for(const input of Object.values(inputs))input.disabled=busy||!previewReady;
    }
    function setBounds(value,writeFields=true) {
      bounds=value?value.map(Math.round):null;
      if(writeFields){const v=bounds?[bounds[0],bounds[1],bounds[2]-bounds[0],bounds[3]-bounds[1]]:['','','',''];Object.keys(inputs).forEach((key,i)=>{inputs[key].value=v[i];});}
      draw();
      status.textContent=previewFailed?'The source preview could not be verified. Close the inspector and retry.':!previewReady?'Loading source image…':!bounds?'No region selected.':valid(bounds)?'Selected '+(bounds[2]-bounds[0])+' × '+(bounds[3]-bounds[1])+' pixels at ('+bounds[0]+', '+bounds[1]+').':'Keep the region inside the image, with each side between '+minimum+' and '+maximum+' pixels.';
    }
    function applyZoom() {
      if(editor.hidden)return;
      const available=viewport.clientWidth;
      const cap=window.matchMedia('(max-width:600px)').matches?330:Math.min(430,window.innerHeight*.48);
      stage.style.width=Math.max(1,Math.min(available,cap*width/height))*Number(zoom.value)+'px';
    }
    function setOpen(open) {
      if(busy)return;editor.hidden=!open;toggle.setAttribute('aria-expanded',String(open));toggle.textContent=open?'Close region selection':'Select region';image.hidden=open?true:originalHidden;
      if(open){viewport.scrollTo(0,0);requestAnimationFrame(applyZoom);}
    }
    toggle.onclick=()=>setOpen(editor.hidden);
    zoom.onchange=applyZoom;
    picture.addEventListener('load',applyZoom);
    const resize=new ResizeObserver(()=>applyZoom());resize.observe(viewport);
    clear.onclick=()=>setBounds(null);
    for(const input of Object.values(inputs))input.addEventListener('input',()=>{
      if(busy||!previewReady)return;const values=Object.values(inputs).map(node=>node.value===''?NaN:Number(node.value));
      const[x,y,w,h]=values;setBounds(values.every(Number.isInteger)?[x,y,x+w,y+h]:null,false);
      if(!values.every(Number.isInteger))status.textContent='Enter all four bounds as whole pixels.';
    });
    const point=event=>{const box=stage.getBoundingClientRect();return[Math.max(0,Math.min(width,(event.clientX-box.left)*width/box.width)),Math.max(0,Math.min(height,(event.clientY-box.top)*height/box.height))];};
    stage.addEventListener('pointerdown',event=>{if(busy||!previewReady||event.button!==0)return;event.preventDefault();drag=point(event);stage.setPointerCapture(event.pointerId);setBounds([Math.floor(drag[0]),Math.floor(drag[1]),Math.ceil(drag[0]),Math.ceil(drag[1])]);});
    stage.addEventListener('pointermove',event=>{if(!drag||busy)return;const end=point(event);setBounds([Math.floor(Math.min(drag[0],end[0])),Math.floor(Math.min(drag[1],end[1])),Math.ceil(Math.max(drag[0],end[0])),Math.ceil(Math.max(drag[1],end[1]))]);});
    stage.addEventListener('pointerup',event=>{drag=null;if(stage.hasPointerCapture(event.pointerId))stage.releasePointerCapture(event.pointerId);});
    stage.addEventListener('pointercancel',()=>{drag=null;});
    function previewLoaded(){
      if(disposed)return;previewReady=picture.naturalWidth===width&&picture.naturalHeight===height;previewFailed=!previewReady;
      status.setAttribute('role',previewFailed?'alert':'status');setBounds(bounds,false);applyZoom();
    }
    picture.onload=previewLoaded;
    picture.onerror=()=>{if(disposed)return;previewReady=false;previewFailed=true;status.setAttribute('role','alert');setBounds(bounds,false);};
    async function submit(callback,kind) {
      if(busy||!previewReady||!valid(bounds)||!canLeave())return;
      busy=true;drag=null;status.setAttribute('role','status');editor.setAttribute('aria-busy','true');for(const node of root.querySelectorAll('button,input,select'))node.disabled=true;
      status.textContent=kind==='search'?'Saving the region and preparing its search…':'Saving selected region…';
      try{await callback([...bounds]);if(!disposed)status.textContent=kind==='search'?'Region ready for search.':'Region added to your finding.';}
      catch(error){if(!disposed){status.textContent=error.message||'The region could not be saved. Your selection is kept; try again.';status.setAttribute('role','alert');}}
      finally{busy=false;if(!disposed){editor.setAttribute('aria-busy','false');for(const node of root.querySelectorAll('button,input,select'))node.disabled=false;draw();}}
    }
    search.onclick=()=>submit(onSearch,'search');evidence.onclick=()=>submit(onEvidence,'evidence');
    draw();if(picture.complete&&picture.naturalWidth)previewLoaded();if(valid(initialBounds)){setBounds(initialBounds);setOpen(true);}
    return {destroy(){disposed=true;resize.disconnect();image.hidden=originalHidden;root.remove();},open(value){if(valid(value))setBounds(value);setOpen(true);},get busy(){return busy;},get bounds(){return bounds&&[...bounds];}};
  }
  return {mount};
})();
