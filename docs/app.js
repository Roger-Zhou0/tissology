'use strict';
const data = [
  { code: 'ADI', name: 'Adipose', support: 1338, correct: 1337 },
  { code: 'BACK', name: 'Background', support: 847, correct: 847 },
  { code: 'DEB', name: 'Debris', support: 339, correct: 338 },
  { code: 'LYM', name: 'Lymphocytes', support: 634, correct: 631 },
  { code: 'MUC', name: 'Mucus', support: 1035, correct: 1027 },
  { code: 'MUS', name: 'Smooth muscle', support: 592, correct: 451 },
  { code: 'NORM', name: 'Normal mucosa', support: 741, correct: 737 },
  { code: 'STR', name: 'Stroma', support: 421, correct: 381 },
  { code: 'TUM', name: 'Tumor epithelium', support: 1233, correct: 1156 }
];
const confusion = [
  [1337,1,0,0,0,0,0,0,0],
  [0,847,0,0,0,0,0,0,0],
  [0,0,338,0,0,0,0,1,0],
  [0,0,0,631,0,0,0,3,0],
  [6,0,1,0,1027,0,0,0,1],
  [0,0,0,0,0,451,0,141,0],
  [0,0,0,1,1,1,737,0,1],
  [0,0,6,0,2,29,0,381,3],
  [2,0,24,11,1,1,38,0,1156]
];
const $ = id => document.getElementById(id);
const fmt = n => n.toLocaleString('en-US');
const pct = n => (n * 100).toFixed(2) + '%';
const metrics = data.map((d,i) => {
  const predicted = confusion.reduce((sum,row)=>sum+row[i],0);
  const precision = d.correct/predicted, recall = d.correct/d.support;
  return {...d,index:i,predicted,precision,recall,f1:2*precision*recall/(precision+recall)};
});
const pairs = confusion.flatMap((row,i)=>row.map((count,j)=>({i,j,count}))).filter(p=>p.i!==p.j&&p.count).sort((a,b)=>b.count-a.count);
let selected = 5, pair = {i:5,j:7}, findings = [];
function el(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
function button(text,fn,cls='button'){const b=el('button',text,cls);b.type='button';b.addEventListener('click',fn);return b;}
function renderClasses(){
  const query=$('filter').value.trim().toLowerCase(), sort=$('sort').value;
  const rows=metrics.filter(d=>(d.name+' '+d.code).toLowerCase().includes(query)).sort((a,b)=>sort==='name'?a.name.localeCompare(b.name):sort==='support'?b.support-a.support:a[sort]-b[sort]);
  $('class-rows').replaceChildren(...rows.map(d=>{
    const tr=el('tr');tr.classList.toggle('selected',selected===d.index);
    const name=el('td'), b=button(d.name,()=>selectClass(d.index),'class-link');b.append(el('small',d.code));b.setAttribute('aria-pressed',String(selected===d.index));name.append(b);tr.append(name,el('td',fmt(d.support)),el('td',pct(d.precision)));
    const recall=el('td',pct(d.recall)),track=el('span',undefined,'recall'),fill=el('i');fill.style.width=pct(d.recall);track.append(fill);recall.append(track);tr.append(recall,el('td',pct(d.f1)));return tr;
  }));
  $('table-status').textContent=rows.length?`${rows.length} of 9 classes`:'No classes match your search.';
}
function selectClass(index){selected=index;renderClasses();renderInspector();}
function renderInspector(){
  const d=metrics[selected],root=$('inspector');root.replaceChildren(el('span',d.code+' / REFERENCE CLASS','code'),el('h3',d.name),el('p',`${fmt(d.support)} tiles · ${fmt(d.support-d.correct)} misclassified`,'description'),el('h2','Predicted distribution'));
  const dist=el('div',undefined,'distribution');
  confusion[selected].map((count,j)=>({count,j})).filter(x=>x.count).sort((a,b)=>b.count-a.count).forEach(({count,j})=>{
    const row=el('div',undefined,j===selected?'':'wrong'),label=el('p');label.append(el('span',data[j].name),el('span',fmt(count)+' · '+(100*count/d.support).toFixed(1)+'%'));
    const track=el('div',undefined,'track'),fill=el('i');fill.style.width=pct(count/d.support);track.append(fill);row.append(label,track);dist.append(row);
  });
  root.append(dist,button('Write a finding →',()=>writeFinding('class:'+selected)),el('p','Aggregate counts from the archived evaluation.','small muted'));
}
function renderMatrix(){
  const table=el('table',undefined,'matrix');table.setAttribute('aria-label','Confusion matrix: reference rows and prediction columns');
  const head=el('thead'),hr=el('tr');hr.append(el('th','Ref. ↓'));data.forEach(d=>{const th=el('th',d.code);th.scope='col';hr.append(th);});head.append(hr);table.append(head);
  const body=el('tbody');confusion.forEach((row,i)=>{const tr=el('tr'),th=el('th',data[i].code);th.scope='row';tr.append(th);row.forEach((count,j)=>{
    const cell=el('td'),b=button($('matrix-mode').value==='count'?(count?fmt(count):'–'):(count?(count/data[i].support*100).toFixed(1)+'%':'–'),()=>{pair={i,j};selectClass(i);renderMatrix();},'');
    b.style.background=`rgba(124,145,100,${.055+.55*count/data[i].support})`;b.classList.toggle('selected',pair.i===i&&pair.j===j);b.setAttribute('aria-pressed',String(pair.i===i&&pair.j===j));b.setAttribute('aria-label',`${data[i].name} predicted as ${data[j].name}: ${count} tiles, ${pct(count/data[i].support)}`);cell.append(b);tr.append(cell);
  });body.append(tr);});table.append(body);$('matrix').replaceChildren(table);
  $('pair-title').textContent=data[pair.i].name+' → '+data[pair.j].name;
  $('pair-value').textContent=`${fmt(confusion[pair.i][pair.j])} of ${fmt(data[pair.i].support)} reference tiles (${pct(confusion[pair.i][pair.j]/data[pair.i].support)}). ${pair.i===pair.j?'Correctly classified.':'Assigned a different label.'}`;
}
function renderErrors(){
  const top=pairs[0];$('error-summary').replaceChildren(el('strong',`${pct(top.count/275)} of all errors are ${data[top.i].name.toLowerCase()} → ${data[top.j].name.toLowerCase()}.`),el('p',`${top.count} tiles in this pair account for more than half of the 275 errors. Inspect both recall and precision before interpreting the overall agreement rate.`));
  $('error-rows').replaceChildren(...pairs.map(p=>{const tr=el('tr');tr.append(el('td',data[p.i].name),el('td',data[p.j].name),el('td',fmt(p.count)));const share=el('td'),bar=el('span',undefined,'share-bar');bar.style.width=(p.count/275*100)+'px';share.append(bar,document.createTextNode(pct(p.count/275)));const action=el('td');action.append(button('Inspect ↗',()=>{pair={i:p.i,j:p.j};selectClass(p.i);renderMatrix();location.hash='results';setTimeout(()=>$('matrix').scrollIntoView({block:'center',behavior:'smooth'}),0);}));tr.append(share,action);return tr;}));
}
function evidenceText(value){const [kind,a,b]=value.split(':'),i=Number(a);if(kind==='class'){const d=metrics[i];return `${d.name}: ${d.correct}/${d.support} correct; recall ${pct(d.recall)}, precision ${pct(d.precision)}, F1 ${pct(d.f1)}.`;}const j=Number(b);return `${data[i].name} → ${data[j].name}: ${confusion[i][j]}/${data[i].support} reference tiles (${pct(confusion[i][j]/data[i].support)}).`;}
function writeFinding(value){$('evidence').value=value;$('evidence-summary').textContent=evidenceText(value);location.hash='findings';setTimeout(()=>$('finding-title').focus(),0);}
function renderFindings(){
  $('note-count').textContent=findings.length;$('export').disabled=!findings.length;
  if(!findings.length){const empty=el('div',undefined,'empty');empty.append(el('h3','Start with an observation.'),el('p','Select a class or confusion pair, then record what the evidence suggests. Your findings will appear here with their source counts.'));$('findings-list').replaceChildren(empty);return;}
  $('findings-list').replaceChildren(...findings.map(f=>{const article=el('article',undefined,'finding');article.append(el('span',f.status,'status'),el('h3',f.title),el('p',f.evidence,'evidence-line'),el('p',f.note),button('Remove finding',()=>{findings=findings.filter(x=>x.id!==f.id);renderFindings();},'delete'));return article;}));
}
function route(){const key=location.hash.slice(1),view=['results','errors','findings','about'].includes(key)?key:'results';const titles={results:['Evaluation','Inspect the recorded results. Follow the errors. Keep your findings.'],errors:['Error analysis','Understand the mistakes behind the overall agreement rate.'],findings:['Findings','Keep your interpretation connected to the evidence.'],about:['About this workspace','Scope, provenance, and how to read the results.']};
  document.querySelectorAll('.view').forEach(node=>node.hidden=node.id!==view+'-view');document.querySelectorAll('[data-view]').forEach(a=>{a.classList.toggle('active',a.dataset.view===view);if(a.dataset.view===view)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});$('page-title').textContent=titles[view][0];$('page-description').textContent=titles[view][1];document.title='Tissology · '+titles[view][0];
}
data.forEach((d,i)=>{const o=el('option',d.name);o.value='class:'+i;$('evidence').append(o);});
confusion.forEach((row,i)=>row.forEach((count,j)=>{const o=el('option',`${data[i].code} → ${data[j].code} (${count} tiles)`);o.value=`pair:${i}:${j}`;$('evidence').append(o);}));
$('evidence').value='class:5';$('evidence-summary').textContent=evidenceText('class:5');
$('evidence').addEventListener('change',()=>{$('evidence-summary').textContent=evidenceText($('evidence').value);});
$('filter').addEventListener('input',renderClasses);$('sort').addEventListener('change',renderClasses);$('matrix-mode').addEventListener('change',renderMatrix);$('pair-note').addEventListener('click',()=>writeFinding(`pair:${pair.i}:${pair.j}`));
$('finding-form').addEventListener('submit',e=>{e.preventDefault();const title=$('finding-title').value.trim(),note=$('finding-note').value.trim();if(!title||!note){$('save-status').textContent='Add a title and interpretation before saving.';return;}findings.unshift({id:Date.now()+'-'+Math.random().toString(36).slice(2),title,note,status:$('finding-status').value,evidence_key:$('evidence').value,evidence:evidenceText($('evidence').value),created_at:new Date().toISOString()});$('finding-title').value='';$('finding-note').value='';$('save-status').textContent='Finding saved in this tab.';renderFindings();});
$('export').addEventListener('click',()=>{const blob=new Blob([JSON.stringify({workspace:'Tissology',release:'conch-retention-001',dataset:'CRC-VAL-HE-7K',scope:'User interpretations of archived aggregate evaluation; not clinical validation.',findings},null,2)+'\n'],{type:'application/json'});const url=URL.createObjectURL(blob),a=el('a');a.href=url;a.download='tissology-findings.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});
window.addEventListener('hashchange',route);
window.addEventListener('beforeunload',e=>{if(findings.length||$('finding-title').value||$('finding-note').value){e.preventDefault();e.returnValue='';}});
renderClasses();renderInspector();renderMatrix();renderErrors();renderFindings();route();
