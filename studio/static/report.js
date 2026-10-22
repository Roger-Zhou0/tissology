'use strict';
(() => {
  const $=id=>document.getElementById(id), query=new URLSearchParams(location.search);
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=value=>typeof value==='number'&&Number.isFinite(value)?(100*value).toFixed(2)+'%':'Undefined';
  const range=value=>Array.isArray(value)&&value.length===2?value.map(pct).join(' – '):'Unresolved';
  const label={ADI:'Adipose tissue',BACK:'Background',DEB:'Debris / necrotic tissue',LYM:'Lymphocyte-rich tissue',MUC:'Mucus',MUS:'Smooth muscle',NORM:'Normal colorectal mucosa',STR:'Stroma',TUM:'Colorectal tumor epithelium pattern'};
  const classColors={ADI:'#d6b867',BACK:'#c8cdd1',DEB:'#a48772',LYM:'#896aa8',MUC:'#79b7bd',MUS:'#b67489',NORM:'#c18da9',STR:'#8eafa0',TUM:'#9a718d'};
  const generated=new Date().toISOString();let title='Tissue research report',dataRecord=null,evidenceReady=false,evidenceChecking=false,downloading=false;
  const safePath=path=>typeof path==='string'&&/^api\/(research|collections|semantic)\//.test(path)&&!path.includes('..')&&!path.includes('\\')?path:'';
  const figure=(url,caption)=>safePath(url)?`<figure><img src="${esc(url)}" alt="${esc(caption)}"><figcaption>${esc(caption)}</figcaption></figure>`:'';
  function findingEvidence(claim,base){
    const rows=claim.evidence?.length?claim.evidence:(claim.image_ids||[]).map(image=>({image_id:image}));
    return rows.map((item,index)=>{
      if(!item.region_id)return figure(base+'/images/'+encodeURIComponent(item.image_id),'Selected image '+(index+1));
      const bounds=item.bounds_px,size=item.source_size_px;
      if(!/^[a-f0-9]{64}$/.test(item.region_id)||!Array.isArray(bounds)||bounds.length!==4||!bounds.every(Number.isInteger)||
         !Array.isArray(size)||size.length!==2||!size.every(value=>Number.isInteger(value)&&value>0)||
         bounds[0]<0||bounds[1]<0||bounds[2]<=bounds[0]||bounds[3]<=bounds[1]||bounds[2]>size[0]||bounds[3]>size[1])throw new Error('A saved region has invalid coordinates. Its source record needs to be checked.');
      const crop=base+'/regions/'+item.region_id+'/image',source=base+'/images/'+encodeURIComponent(item.image_id);
      const label='Selected region '+(index+1)+' · '+(bounds[2]-bounds[0])+' × '+(bounds[3]-bounds[1])+' pixels';
      const coordinates='Source image '+item.image_id+' · x '+bounds[0]+'–'+bounds[2]+', y '+bounds[1]+'–'+bounds[3]+' px (right and bottom edges excluded).';
      const box='left:'+(100*bounds[0]/size[0])+'%;top:'+(100*bounds[1]/size[1])+'%;width:'+(100*(bounds[2]-bounds[0])/size[0])+'%;height:'+(100*(bounds[3]-bounds[1])/size[1])+'%;';
      return '<figure class="report-region" data-region-id="'+item.region_id+'"><img src="'+esc(crop)+'" alt="'+esc(label)+'"><figcaption>'+esc(label)+'<p>'+esc(coordinates)+'</p><details class="report-region-source"><summary>Source image and selected region</summary><div class="report-region-context"><img src="'+esc(source)+'" alt="Source image; rectangle marks the selected region"><span aria-hidden="true" class="report-region-outline" style="'+box+'"></span></div></details></figcaption></figure>';
    }).join('');
  }
  const table=(head,rows)=>`<div class="report-table" tabindex="0" role="region" aria-label="Data table; scroll horizontally if needed"><table><thead><tr>${head.map(x=>`<th>${esc(x)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(x=>`<td>${esc(x)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  function collectionDistribution(summary){
    const total=summary.predicted_count,counts=summary.counts||{};
    const rows=Object.keys(label).map(code=>({code,count:counts[code]??0}));
    if(!Number.isSafeInteger(total)||total<0||rows.some(row=>!Number.isSafeInteger(row.count)||row.count<0)||
       Object.keys(counts).some(code=>!(code in label))||rows.reduce((sum,row)=>sum+row.count,0)!==total){
      throw new Error('The saved class counts do not match the scanned-image total. The report cannot show a verified distribution.');
    }
    if(!total)return '<p>No classified images are available for a distribution yet.</p>';
    rows.sort((a,b)=>b.count-a.count||a.code.localeCompare(b.code));
    return '<figure class="report-distribution" aria-labelledby="report-distribution-title" data-scanned-count="'+total+'">'+
      '<figcaption id="report-distribution-title">Predicted class distribution</figcaption>'+
      '<p class="report-distribution-scope">Percentage of '+total.toLocaleString()+' scanned images. Each image counts once; these are image counts, not tissue areas or cell counts.</p>'+
      '<div class="report-distribution-axis" aria-hidden="true"><span>0%</span><span>100%</span></div>'+
      '<ol>'+rows.map(row=>{
        const percent=100*row.count/total;
        return '<li data-class-code="'+row.code+'" data-image-count="'+row.count+'" data-percentage="'+percent+'">'+
          '<span class="report-distribution-label">'+esc(label[row.code])+'</span>'+
          '<span class="report-distribution-track" aria-hidden="true"><span style="width:'+percent+'%;background:'+classColors[row.code]+'"></span></span>'+
          '<span class="report-distribution-value"><strong>'+row.count.toLocaleString()+'</strong> <span>images · '+percent.toFixed(1)+'%</span></span></li>';
      }).join('')+'</ol></figure>';
  }
  const notice=text=>`<div class="report-notice"><p>${esc(text)}</p></div>`;
  function confidenceSet(row){
    if(Array.isArray(row.confidence_set)){
      if(!row.confidence_set.length)return 'Unresolved / empty confidence set';
      const segments=row.confidence_set.map(segment=>Array.isArray(segment)&&segment.length===2&&segment.every(value=>typeof value==='number'&&Number.isFinite(value))?range(segment):'Unbounded / unresolved segment');
      return segments.join(' or ');
    }
    return range(row.interval);
  }
  function endpointWarnings(row){
    const messages=[...(row.warnings||[])];
    if(row.status&&!['bounded','exact_census'].includes(row.status))messages.push('Confidence-set status: '+row.status.replaceAll('_',' ')+'.');
    if(row.interval_status&&!['defined','valid'].includes(row.interval_status))messages.push(row.interval_note||'Interval status: '+row.interval_status.replaceAll('_',' ')+'.');
    if(typeof row.estimate==='number'&&(row.estimate<0||row.estimate>1))messages.push('Raw estimate lies outside 0–100%; it is shown without clipping.');
    const interval=row.interval;
    if(row.estimate_outside_interval||(Array.isArray(interval)&&interval.every(value=>typeof value==='number'&&Number.isFinite(value))&&typeof row.estimate==='number'&&(row.estimate<interval[0]||row.estimate>interval[1])))messages.push('The reported estimate lies outside its uncertainty range; this result is unresolved.');
    if(row.standard_error===0&&row.status!=='exact_census')messages.push('No observed variation; this approximate range may be unreliable.');
    if(Array.isArray(row.confidence_set)&&!row.confidence_set.length)messages.push('Empty confidence set: no interval conclusion is available.');
    return [...new Set(messages)];
  }
  const provenance=value=>`<details class="report-provenance"><summary>Source identity and reproducibility</summary><pre>${esc(JSON.stringify(value,null,2))}</pre></details>`;
  function modelIdentity(model){
    if(!model||typeof model!=='object')return model;
    return Object.fromEntries(['name','classifier_sha256','configuration_sha256','checkpoint_sha256','encoder_identity','code_revision','reference_labels_used','model_training'].filter(key=>key in model).map(key=>[key,model[key]]));
  }
  function measurementDefinitions(definitions){
    const fields=[['Numerator','numerator_definition'],['Denominator','denominator_definition'],['Eligible tissue','eligible_tissue'],['Excluded regions','excluded_regions'],['Physical scale','physical_scale'],['Aggregation','aggregation'],['Boundary rule','boundary_rule'],['Zero denominator','zero_denominator'],['Interpretation','interpretation']];
    return (definitions||[]).map(row=>'<h3>'+esc(row.name)+'</h3>'+table(['Definition','Rule'],fields.filter(([,key])=>row[key]).map(([label,key])=>[label,typeof row[key]==='string'?row[key]:JSON.stringify(row[key])]))).join('');
  }
  async function get(path){const response=await fetch(path);if(!response.ok)throw new Error(response.status===404?'This saved record is unavailable. Return to the workspace and choose an existing record.':'The report could not be loaded. Your saved records are unchanged.');return response.json();}
  function heading(name,source){title=name;document.title=name+' · Tissology';return `<p class="eyebrow">Tissology · Research report</p><h1>${esc(name)}</h1><p class="report-meta">${esc(source)}<br>Report generated ${esc(generated)}</p>`;}
  function links(back,record){
    $('report-back').href=back;$('report-back').textContent='← Return to this record';
    // Download the same loaded source snapshot embedded in the HTML report.
    // A live export could silently include reviews added in another tab.
    const url=URL.createObjectURL(new Blob([JSON.stringify(dataRecord,null,2)+'\n'],{type:'application/json'}));
    $('report-record').href=url;$('report-record').download='tissue-report-data-'+query.get('kind')+'-'+query.get('id').slice(0,8)+'.json';
    $('report-record').textContent='Download report data';$('report-record').title='The source records loaded for this report. Reload the report to include later changes.';$('report-record').hidden=false;
    // Keep this one URL for the document lifetime, including Back/Forward
    // cache restoration. The browser releases it when the document is gone.
  }
  function teachingEvidence(candidate){
    const result=candidate.result;if(!result)return notice('This prediction version has no completed image check.');
    const n=result.sample_size,paired=result.paired||{},previous=candidate.baseline_version&&candidate.baseline_version!=='original'?'Previous version':'Original scan';
    let html='<h2>Collection teaching</h2><p>'+esc(candidate.name||'Saved candidate')+' · '+esc(n)+' randomly selected images checked separately from training.</p>';
    html+=table([previous+' correct','Candidate correct','Mistakes fixed','New mistakes'],[[result.original?.correct+' / '+n,result.candidate?.correct+' / '+n,paired.fixed??'Unavailable',paired.introduced??'Unavailable']]);
    html+='<p>The check concerns the reserved eligible images in this collection. A complete check with no more observed mistakes permits application; this does not establish improvement on other collections or patients.</p>';
    const changes=result.count_changes;
    if(changes){
      const rows=Array.isArray(changes)?changes:Object.entries(changes).map(([code,row])=>typeof row==='object'?{code,...row}:{code,change:row});
      if(rows.length)html+='<details class="report-provenance"><summary>Changes in predicted image counts</summary>'+table(['Tissue class','Change from '+previous.toLowerCase()],rows.map(row=>[label[row.code||row.class_id||row.label]||row.code||row.class_id||row.label,row.change??row.delta??'Recorded in source data']))+'<p>One predicted class per image. A change in counts is not a measured change in tissue area.</p></details>';
    }
    html+='<details class="report-provenance"><summary>Image check and evidence</summary><p>Sample: '+esc(n)+' of '+esc(result.frame_size||candidate.check?.frame_size)+' eligible images. Development images and previously checked images were excluded. Both predictions were hidden while these labels were entered.</p>';
    if(Array.isArray(result.original?.interval)&&Array.isArray(result.candidate?.interval))html+=table(['Version','Correct on this check','Recorded uncertainty range'],[[previous,pct(result.original.accuracy),range(result.original.interval)],['Candidate',pct(result.candidate.accuracy),range(result.candidate.interval)]]);
    if(result.interval_scope)html+='<p>'+esc(result.interval_scope)+'</p>';
    html+='<p>These ranges describe sampling uncertainty assuming the entered reference labels are correct; they do not measure reviewer disagreement.</p>';
    html+='<p>The intervals cover this fixed check and model pair; they do not cover selection across multiple candidate attempts.</p>';
    html+='<p>'+esc(result.gate_reason||'Passing is an operational check, not proof of general improvement.')+'</p></details>';
    const measurements=result.measurements;
    if(measurements){
      html+='<details class="report-provenance"><summary>Estimated image composition</summary><p>'+esc(measurements.scope)+'</p><p>All nine class fractions use the same '+esc(measurements.sample_size)+' reviewed images from the '+esc(measurements.frame_size)+'-image eligible frame. Each image counts once; this is not tissue area.</p>';
      const estimate=row=>row.infeasible_point_estimate?'Unresolved (raw '+pct(row.estimated_fraction)+')':pct(row.estimated_fraction);
      for(const [key,title]of[['candidate','This version'],['original',previous]]){
        const rows=measurements.model_assisted?.[key]?.classes||[];
        html+='<h3>'+esc(title)+'</h3>'+table(['Image class','Predicted fraction','Audit-corrected fraction','Conservative simultaneous range'],rows.map(row=>[label[row.class]||row.class,pct(row.predicted_fraction),estimate(row),range(row.interval)]));
      }
      html+='<p>'+esc(measurements.uncertainty?.model_assisted_family)+'. '+esc(measurements.uncertainty?.dependence)+'</p><p>'+esc(measurements.uncertainty?.raw_estimates)+'</p>';
      const manual=measurements.manual_only?.classes||[];
      if(manual.length)html+='<h3>Reference labels alone</h3>'+table(['Image class','Sample estimate','Separate simultaneous range'],manual.map(row=>[label[row.class]||row.class,pct(row.estimated_fraction),range(row.interval)]))+'<p>'+esc(measurements.uncertainty?.manual_only_family)+'. This family is separate from the model-assisted family.</p>';
      html+='<h3>Measurement definition</h3>'+table(['Rule','Definition'],Object.entries(measurements.measurement||{}).map(([key,value])=>[key.replaceAll('_',' '),value]))+'</details>';
    }
    return html;
  }
  async function collection(id){
    const path='api/collections/'+encodeURIComponent(id),version=query.get('prediction_version')||'original';
    if(version!=='original'&&!/^[a-f0-9]{32}$/.test(version))throw new Error('Choose a saved prediction version from this collection.');
    const versionQuery='prediction_version='+encodeURIComponent(version);
    const [source,reviews,teaching,teachingExport,scanChecks]=await Promise.all([get(path+'?'+versionQuery),get(path+'/reviews/export'),version==='original'?Promise.resolve(null):get(path+'/teaching/candidates/'+encodeURIComponent(version)),version==='original'?Promise.resolve(null):get(path+'/teaching/versions/'+encodeURIComponent(version)+'/export'),get(path+'/checks?'+versionQuery)]);
    const checkExports=await Promise.all((scanChecks.checks||[]).map(check=>get(path+'/checks/'+encodeURIComponent(check.id)+'/export')));
    dataRecord={source,reviews,teaching,teaching_export:teachingExport,scan_checks:checkExports,prediction_version:version};
    window.TissueNavigation?.setCollection(id,version);
    links('collections?collection='+encodeURIComponent(id)+'&'+versionQuery,path+'/export?'+versionQuery);
    const summary=source.summary, latest=reviews.latest||[], assessed=reviews.summary||{};
    let html=heading(source.name,'Image collection · '+source.image_count+' accepted unique images · '+(source.declared_domain==='colorectal_he'?'colorectal H&E tiles':'organ / stain not verified'));
    html+=notice('Preliminary image-level inspection. Each image has at most one predicted class. These counts are not tissue areas, cell counts or diagnoses. Researcher-selected reviews do not establish collection-wide accuracy.');
    html+='<h2>Collection and scan</h2>'+table(['Accepted images','Predicted images','Flagged for inspection','Scan status'],[[source.image_count,summary?.predicted_count??0,summary?.uncertain_count??'Not available',source.status]]);
    html+='<p><strong>Prediction version:</strong> '+esc(source.selected_prediction_version?.name||(version==='original'?'Original scan':teaching?.name||version))+'</p>';
    if(summary)html+=collectionDistribution(summary);
    html+='<h2>Random scan check</h2>';
    if(!scanChecks.checks?.length)html+='<p>This prediction version has not had a random accuracy check. The benchmark and selected inspection reviews do not estimate accuracy on this collection.</p>';
    for(const check of scanChecks.checks||[]){
      const result=check.result;
      if(check.status==='evaluated'&&result){
        html+=table(['Correct / checked','Estimated accuracy','95% interval','Eligible images'],[[result.correct+' / '+result.sample_size,pct(result.accuracy),pct(result.interval[0])+'–'+pct(result.interval[1]),result.frame_size]]);
        html+='<p>'+esc(result.scope)+'</p><p>Uniform random sampling without replacement. Predictions were hidden during labeling. Check labels stay out of training; the model and saved predictions were not changed.</p>';
        html+='<p>Agreement is with the recorded reviewer labels. This is not evidence of patient-level generalization, tissue-area accuracy, or clinical effectiveness.</p>';
        const evidence=checkExports.find(item=>item.check_id===check.id)?.sample_evidence||[];
        if(evidence.length){
          html+='<details class="report-provenance"><summary>Images from the random check</summary><p>First six sampled images in recorded order. The data record contains every submitted label and prediction.</p><div class="report-evidence">';
          html+=evidence.slice(0,6).map(row=>figure(row.image_url,'Prediction: '+(label[row.predicted_label]||row.predicted_label)+' · Reviewer: '+(label[row.reference_label]||row.reference_label)+' · '+(row.correct?'Agree':'Differ'))).join('');
          html+='</div></details>';
        }
      }else html+='<p>'+esc(check.status==='closed'?'Stopped':'In progress')+' · '+esc(check.progress.answered)+' of '+esc(check.progress.planned)+' sampled images labeled. No accuracy estimate is available.</p>';
      html+=provenance({check_id:check.id,prediction_version:check.prediction_version,check_export_sha256:checkExports.find(item=>item.check_id===check.id||item.id===check.id||item.plan?.id===check.id)?.sha256});
    }
    if(teaching)html+=teachingEvidence(teaching);
    html+='<h2>Researcher inspection</h2>'+table(['Reviewed images','Confirmed','Corrected','Unsure'],[[assessed.reviewed||0,assessed.confirmed||0,assessed.corrected||0,assessed.unsure||0]]);
    html+='<p>Review counts include judgments on earlier prediction versions. Reviewers record their own judgments; their identities are self-declared. These selected observations are separate from a candidate’s reserved image check. Saving a review alone does not retrain the model.</p>';
    if(latest.length){html+='<details class="report-provenance"><summary>Reviewed image list ('+esc(latest.length)+')</summary>';html+=`<p>Latest judgment per image${latest.length>100?' · showing the first 100 of '+latest.length+'; all judgments and revisions are in the data export':''}.</p>`;
      html+=table(['Image','Prediction when reviewed','Reviewed version','Review','Researcher label','Reviewer / note'],latest.slice(0,100).map(row=>[row.filename,label[(row.reviewed_prediction||row.original_prediction)?.predicted_label]||(row.reviewed_prediction||row.original_prediction)?.predicted_label||'Unavailable',row.prediction_version||'original',row.disposition,label[row.label]||row.label||'Unresolved',row.reviewer+(row.note?' · '+row.note:'')]));
      html+='</details>';
      html+='<h2>Reviewed image evidence</h2><p>Up to six reviewed images, in record order; these are selected observations, not a representative sample.</p><div class="report-evidence">'+latest.slice(0,6).map(row=>figure(path+'/images/'+encodeURIComponent(row.image_id)+'/image',row.filename+' · '+row.disposition+' · '+row.reviewer)).join('')+'</div>';
    }else html+='<p>No human image reviews have been saved. Open an image in the collection to record a judgment.</p>';
    html+=provenance({collection_id:id,prediction_version:source.selected_prediction_version||version,teaching_export_sha256:teachingExport?.sha256,teaching_result_sha256:teachingExport?.result?.sha256,teaching_plan_sha256:teachingExport?.plan?.sha256,teaching_check_sha256:teachingExport?.check?.sha256,manifest_sha256:source.manifest_sha256,prediction_sha256:source.prediction_sha256,model:modelIdentity(source.model),model_provenance_sha256:reviews.model_provenance_sha256,review_export_sha256:reviews.sha256,review_revisions:assessed.revisions,scope:reviews.scope});return html;
  }
  async function study(id){
    const audit=query.get('audit');if(!audit)throw new Error('Open a report from a completed study audit.');
    const path='api/research/studies/'+encodeURIComponent(id), auditPath=path+'/audits/'+encodeURIComponent(audit);
    const [source,review]=await Promise.all([get(path),get(auditPath)]);if(!review.result)throw new Error('Complete the selected audit sample before creating a results report.');
    dataRecord=await get(auditPath+'/export');links('research?study='+encodeURIComponent(id)+'&audit='+encodeURIComponent(audit)+'&step=results',auditPath+'/export');
    const result=review.result, endpoints=result.endpoints||[],manual=review.plan.method==='manual';
    const confidence=typeof result.simultaneous_confidence==='number'?pct(result.simultaneous_confidence):typeof result.alpha==='number'?pct(1-result.alpha):'recorded';
    let html=heading(source.name,'Breast tissue area study · '+source.collection.roi_count+' selected regions · '+source.collection.slide_count+' slides');
    html+=notice('Simulated review using existing reference annotations. Results concern the finite collection of selected regions. They do not estimate a patient population, whole slides or cell counts. The predictor remains frozen.');
    html+='<h2>Audited measurements</h2>'+table(['Measurement','AI estimate',manual?'Manual sample estimate':'Corrected estimate','Approximate simultaneous range'],endpoints.map(row=>[row.name,pct(source.initial.find(item=>item.id===row.id)?.estimate),pct(row.estimate),confidenceSet(row)]));
    html+='<p>Nominal '+esc(confidence)+' simultaneous confidence family. Approximate Fieller ranges can under-cover; a nominal level is not a guarantee of coverage. '+(manual?'Numerator and denominator totals are estimated jointly from the sampled references.':'Numerator and denominator totals are corrected jointly using model residuals.')+' Ratio estimates are not exactly unbiased.</p>';
    for(const row of endpoints)for(const warning of endpointWarnings(row))html+=notice(row.name+': '+warning);
    for(const warning of result.warnings||[])html+=notice(warning);
    html+='<h2>Conservative finite-population bounds</h2>'+table(['Measurement','Conservative range'],endpoints.map(row=>[row.name,range(row.conservative_interval)]));
    html+='<p>These separate bounds may be wide at a small review budget. Do not substitute the approximate range for the conservative bound when drawing a conclusion.</p>';
    html+='<h2>Audit design</h2>'+table(['Method','Final sample','Separate pilot','Total annotation units'],[[review.plan.method,review.plan.budget,result.pilot_units??source.allocation_pilot_cost,result.total_review_units??review.plan.total_review_units]]);
    html+='<p>The image-window sample, selection probabilities, random seed and predictions were frozen before reference labels were revealed. The budget counts annotation units, not expert minutes. The same annotation contributes to each selected endpoint.</p>';
    html+='<h2>Measurement definitions</h2>'+measurementDefinitions(source.endpoint_definitions);
    html+='<h2>'+esc(result.evidence_title||(manual?'Reviewed contributions to the manual estimate':'Linked image evidence'))+'</h2><p>'+esc(result.evidence_description||(manual?'Windows ranked by inverse-probability-weighted reference contributions to the manual estimate.':'Windows ranked by their recorded contribution to the measurement correction.'))+' Up to six windows are shown. These illustrate the result; they are not an independent evaluation.</p><div class="report-evidence">'+(result.evidence||[]).slice(0,6).map(row=>figure(row.context_url,row.unit_id+' · outlined measurement window')).join('')+'</div>';
    html+=provenance({study_id:id,audit_id:audit,seed:review.plan.seed,study_sha256:source.sha256,prediction_sha256:source.prediction_sha256,plan_sha256:review.plan.sha256,result_sha256:result.sha256,export_sha256:dataRecord.sha256,source:source.collection.scope,model:source.collection.model,interval_method:result.interval_method,conservative_method:result.conservative_method});return html;
  }
  async function description(id){
    const collection=query.get('collection');
    if(query.has('collection')&&!/^[a-f0-9]{32}$/.test(collection || ''))throw new Error('Choose a saved finding from its collection.');
    const base=collection?'api/collections/'+encodeURIComponent(collection)+'/semantic':'api/semantic';
    const path=base+'/descriptions/'+encodeURIComponent(id), record=await get(path);dataRecord=record;
    window.TissueEvidenceFigure?.configure(record,{base,collectionId:collection});
    if(collection)window.TissueNavigation?.setCollection(collection,record.prediction_version?.id||'original');
    links('semantic?description='+encodeURIComponent(id)+(collection?'&collection='+encodeURIComponent(collection)+'&prediction_version='+encodeURIComponent(record.prediction_version?.id||'original'):''),path+'/export');
    let html=heading(collection?'Collection finding':'Image-grounded description',(collection?(record.collection?.name || 'Image collection'):'Prepared colorectal development library')+' · reviewer '+record.reviewer);
    const claims=record.claims||[];
    const simple=Boolean(collection&&claims.length===1&&claims[0].text===record.text&&claims[0].assessment==='unresolved'&&!claims[0].note);
    html+=notice(simple?'Not yet reviewed. This saved observation has not been independently verified. Selected images do not represent the whole collection.':'The researcher links images and records an assessment. The software does not independently verify a claim. Retrieved examples are not a representative study sample.');
    if(collection)html+='<p><strong>Prediction version:</strong> '+esc(record.prediction_version?.name||'Original scan')+'</p>';
    html+='<h2>'+(simple?'Finding':'Description')+'</h2><p style="white-space:pre-wrap">'+esc(record.text)+'</p>';
    if(simple){
      html+='<h2>Selected image evidence</h2><div class="report-evidence">'+findingEvidence(claims[0],base)+'</div>';
    }else for(const [index,claim]of(claims).entries()){html+='<section class="report-claim"><h2>Statement '+(index+1)+'</h2><p>'+esc(claim.text)+'</p><p><strong>'+esc(({unresolved:'Assessment unresolved',supported_by_reviewer:'Supported — reviewer assessment',contradicted_by_reviewer:'Contradicted — reviewer assessment'})[claim.assessment]||claim.assessment)+'</strong></p><p>'+esc(claim.note)+'</p><div class="report-evidence">'+findingEvidence(claim,base)+'</div></section>';}
    html+=provenance({id:record.id,parent_id:record.parent_id,prediction_version:record.prediction_version,collection_id:collection,collection:record.collection,reviewer:record.reviewer,created_at:record.created_at,sha256:record.sha256,scope:record.scope});return html;
  }
  function clearError(){ $('report-error').textContent='';$('report-error').hidden=true; }
  function imageReady(img){return img.complete&&img.naturalWidth>0;}
  function waitForImage(img,retry){
    if(imageReady(img))return Promise.resolve(true);
    if(img.complete&&!retry)return Promise.resolve(false);
    return new Promise(resolve=>{
      let timer;
      const finish=ok=>{clearTimeout(timer);img.removeEventListener('load',loaded);img.removeEventListener('error',failed);resolve(ok);};
      const loaded=()=>finish(imageReady(img)),failed=()=>finish(false);
      img.addEventListener('load',loaded);img.addEventListener('error',failed);
      timer=setTimeout(()=>finish(false),15000);
      if(retry){const url=new URL(img.src);url.searchParams.set('_report_retry',Date.now());img.src=url.href;}
      else if(img.complete)finish(imageReady(img));
    });
  }
  async function checkEvidence(retry=false){
    if(evidenceChecking)return;
    evidenceChecking=true;evidenceReady=false;$('report-print').disabled=true;
    $('report-retry-evidence').disabled=true;
    const status=$('report-evidence-status');status.hidden=false;status.textContent='Loading image evidence before printing…';
    const images=[...$('report-content').querySelectorAll('img')];
    const loaded=await Promise.all(images.map(img=>waitForImage(img,retry)));
    const missing=loaded.filter(ok=>!ok).length;
    evidenceReady=missing===0;evidenceChecking=false;
    $('report-print').disabled=!evidenceReady;
    $('report-retry-evidence').hidden=evidenceReady;$('report-retry-evidence').disabled=false;
    status.hidden=evidenceReady;
    status.textContent=missing?`${missing} of ${images.length} evidence images could not be loaded. Retry the images before printing. Downloads also require all image evidence.`:'';
  }
  async function download(){
    if(downloading)return;
    downloading=true;clearError();$('report-download').disabled=true;$('report-status').textContent='Preparing a standalone report with image evidence…';
    try{
      const article=$('report-content').cloneNode(true);
      await Promise.all([...article.querySelectorAll('img')].map(async img=>{
        const response=await fetch(img.src);
        if(!response.ok)throw new Error('An evidence image could not be embedded. Retry the download when it is available.');
        const blob=await response.blob();
        if(!blob.type.startsWith('image/'))throw new Error('An evidence image returned an unreadable response. Retry the download when it is available.');
        img.src=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error('An evidence image could not be embedded. Please retry the download.'));reader.readAsDataURL(blob);});
        try{await img.decode();}catch(_){throw new Error('An evidence image could not be read. Retry the download when it is available.');}
      }));
      const css=await Promise.all(['style.css','presentation.css','report.css'].map(async file=>{const response=await fetch('static/'+file);if(!response.ok)throw new Error('Report styles could not be loaded.');return response.text();}));
      const payload=JSON.stringify(dataRecord).replace(/</g,'\\u003c');
      const html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+esc(title)+'</title><style>'+css.join('\n')+'</style></head><body><main class="report-main">'+article.outerHTML+'</main><script type="application/json" id="source-record">'+payload+'</script></body></html>';
      const url=URL.createObjectURL(new Blob([html],{type:'text/html;charset=utf-8'})),link=document.createElement('a');link.href=url;link.download='tissue-report-'+query.get('kind')+'-'+query.get('id').slice(0,8)+'.html';link.click();setTimeout(()=>URL.revokeObjectURL(url),30000);
      $('report-status').textContent='Report downloaded with embedded images and a machine-readable source record.';
    }catch(error){$('report-error').textContent=error.message||'The report could not be downloaded. Please retry.';$('report-error').hidden=false;$('report-status').textContent='Download not completed. The report and saved records remain available; retry when the connection is ready.';}
    finally{downloading=false;$('report-download').disabled=false;}
  }
  $('report-print').onclick=()=>{
    if(!evidenceReady||[...$('report-content').querySelectorAll('img')].some(img=>!imageReady(img))){checkEvidence();return;}
    document.querySelectorAll('#report-content details').forEach(el=>el.open=true);window.print();
  };
  $('report-download').onclick=download;$('report-retry-evidence').onclick=()=>checkEvidence(true);
  (async()=>{const kind=query.get('kind'),id=query.get('id');if(!id||!/^[a-f0-9]{32}$/.test(id)||!['collection','study','description'].includes(kind))throw new Error('Choose a saved collection, completed study audit or description to create a report.');$('report-content').innerHTML=await({collection,study,description}[kind])(id);$('report-status').textContent='Read-only report · saved records and original predictions are unchanged.';$('report-download').disabled=false;await checkEvidence();})().catch(error=>{$('report-status').textContent='';$('report-error').textContent=error.message;$('report-error').hidden=false;});
})();
