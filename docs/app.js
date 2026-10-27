'use strict';
// All displayed evaluation metrics are transcribed from the repository's
// studio/static/collection-validation-conch-retention-001.json. No networking.
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
const get = id => document.getElementById(id);
const namespace = 'http://www.w3.org/2000/svg';
let current = 0;
const reviews = Object.create(null);

function illustration(index) {
  // These deterministic abstract SVGs are not microscopy, nor outputs from a model.
  const palettes = ['#e7bbb0','#cfc7b6','#debac1','#c8b7d9','#e2bbb6','#c6a9ba','#e8bcb8','#d7aabb','#d4aebf'];
  const svg = document.createElementNS(namespace, 'svg');
  svg.setAttribute('viewBox','0 0 280 180');
  svg.setAttribute('role','img');
  svg.setAttribute('aria-label','Synthetic abstract sample for ' + data[index].name + '. Not a tissue image.');
  const bg = document.createElementNS(namespace,'rect');
  bg.setAttribute('width','280'); bg.setAttribute('height','180'); bg.setAttribute('fill',palettes[index]); svg.append(bg);
  let seed = (index+1)*62417;
  function random(){seed=(seed*16807)%2147483647;return seed/2147483647;}
  for (let n=0;n<75;n++){
    const e=document.createElementNS(namespace,n%5===0?'ellipse':'circle');
    e.setAttribute('cx',String(Math.round(random()*280)));e.setAttribute('cy',String(Math.round(random()*180)));
    if(n%5===0){e.setAttribute('rx',String(6+random()*18));e.setAttribute('ry',String(4+random()*15));}
    else e.setAttribute('r',String(2+random()*11));
    e.setAttribute('fill',n%3===0?'#75366e':n%3===1?'#fce9e3':'#9c4f92');
    e.setAttribute('opacity',String(.15+random()*.42));svg.append(e);
  }
  return svg;
}
function element(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
function classBars(){const root=get('class-bars');root.replaceChildren();
  data.forEach((d,i)=>{
    const b=element('button','class-row'+(i===current?' active':''));b.type='button';
    b.setAttribute('aria-pressed',String(i===current));b.setAttribute('aria-label',d.name+' recall '+(100*d.correct/d.support).toFixed(1)+' percent');
    const bar=element('span','bar'),fill=element('i');fill.style.width=(d.correct/d.support*100).toFixed(2)+'%';bar.append(fill);
    b.append(element('span','class-name',d.name),bar,element('span','pct',(d.correct/d.support*100).toFixed(1)+'%'));
    b.addEventListener('click',()=>select(i));root.append(b);
  });
  const d=data[current],top=confusion[current].map((n,i)=>({n,i})).filter(x=>x.i!==current).sort((a,b)=>b.n-a.n)[0];
  get('class-detail').replaceChildren(element('strong','',d.name+': '+d.correct+' / '+d.support+' correct'),element('p','',(d.support-d.correct)+' incorrect. '+(top.n?'Most common confusion: '+data[top.i].name+' ('+top.n+' tiles).':'No errors in this class in the archived evaluation.')));
}
function matrix(){const root=get('matrix'),table=element('table','matrix');table.setAttribute('aria-label','Confusion matrix for nine tissue classes');
  const header=element('tr');header.append(element('th','', 'Ref ↓'));data.forEach(d=>header.append(element('th','',d.code)));table.append(header);
  confusion.forEach((row,i)=>{const tr=element('tr');tr.append(element('th','',data[i].code));
    row.forEach((count,j)=>{const td=element('td'),b=element('button','',count?String(count):'·');b.type='button';
      b.style.background='rgba(44,133,107,'+(.06+count/data[i].support*.80).toFixed(3)+')';
      b.title=data[i].name+' → '+data[j].name+': '+count;
      b.setAttribute('aria-label',b.title);b.addEventListener('click',()=>select(i));td.append(b);tr.append(td);
    });table.append(tr);
  });root.replaceChildren(table);
}
function samples(){const root=get('samples'),q=get('filter').value.toLowerCase().trim();root.replaceChildren();let shown=0;
  data.forEach((d,i)=>{if(!d.name.toLowerCase().includes(q)&&!d.code.toLowerCase().includes(q))return;shown++;
    const b=element('button','sample'+(current===i?' selected':''));b.type='button';
    b.setAttribute('aria-pressed',String(current===i));b.append(illustration(i),element('strong','',d.name),element('small','',d.code+' · Synthetic'));b.addEventListener('click',()=>select(i));root.append(b);
  });get('sample-count').textContent=shown+' illustrative categories shown.';
}
function select(index){current=index;const d=data[current];classBars();samples();get('detail-art').replaceChildren(illustration(current));
  get('chosen-name').textContent=d.name;get('chosen-code').textContent=d.code+' · illustration only';
  get('review-status').value=reviews[d.code]?.status||'Not reviewed';get('review-text').value=reviews[d.code]?.note||'';
  get('save-status').textContent='Session-only preview · nothing sent to a server.';
}
get('filter').addEventListener('input',samples);
get('save').addEventListener('click',()=>{const d=data[current];reviews[d.code]={status:get('review-status').value,note:get('review-text').value.trim()};get('save-status').textContent='Saved example note for '+d.name+' (this tab only).';});
get('export').addEventListener('click',()=>{const d=data[current],note=get('review-text').value.trim(),status=get('review-status').value;
  if(note||status!=='Not reviewed')reviews[d.code]={note,status};
  const obj={demo:'Tissology static public preview',images:'synthetic only',clinical_use:false,notes:reviews};
  const blob=new Blob([JSON.stringify(obj,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download='tissology-demo-notes.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),3000);
  get('save-status').textContent='Notes downloaded to your device; no data uploaded.';
});
get('hero-art').append(illustration(8));matrix();select(0);
