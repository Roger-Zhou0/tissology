"""Browser integration checks. Run against a disposable Chrome profile only.

Serve docs on 127.0.0.1:8765 and start headless Chrome with
--remote-debugging-port=9341 --remote-allow-origins=http://localhost:9341
--user-data-dir=/tmp/tissology-cpu-tests. Requires websocket-client.
Run from the repository root. Tests clear this test origin's sample work.
"""
import json,urllib.request,websocket,base64,time
from pathlib import Path
pages=json.load(urllib.request.urlopen('http://localhost:9341/json/list'))
w=websocket.create_connection(next(p['webSocketDebuggerUrl'] for p in pages if p['type']=='page'),origin='http://localhost:9341',timeout=120)
seq=0;errors=[];requests=[]
def call(method,params={}):
 global seq
 seq+=1;w.send(json.dumps({'id':seq,'method':method,'params':params}))
 while True:
  r=json.loads(w.recv())
  if r.get('method')=='Runtime.exceptionThrown':errors.append(r)
  if r.get('method')=='Network.requestWillBeSent':requests.append(r['params']['request'])
  if r.get('id')==seq:
   assert 'error' not in r,r
   return r.get('result',{})
def ev(js):
 r=call('Runtime.evaluate',{'expression':js,'returnByValue':True,'awaitPromise':True})
 assert 'exceptionDetails' not in r,r
 return r['result'].get('value')
call('Runtime.enable');call('Network.enable');call('Page.enable')
call('Page.navigate',{'url':'http://127.0.0.1:8765/'});time.sleep(1)
call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1100,'deviceScaleFactor':1,'mobile':False})
assert not [r for r in requests if '.onnx' in r['url'] or '.wasm' in r['url']]
print(ev('''(async()=>{
window.testWait=async f=>{for(let i=0;i<600;i++){if(f())return;await new Promise(r=>setTimeout(r,100));}throw Error('Timeout: '+$('analysis-status').textContent);};
window.testAssert=(v,m)=>{if(!v)throw Error(m);};
location.hash='images';
LocalAnalysis.enable();LocalAnalysis.stop();await new Promise(r=>setTimeout(r,200));testAssert(LocalAnalysis.state==='idle','cancel initialization');
LocalAnalysis.enable();await testWait(()=>LocalAnalysis.state==='ready');
await LocalAnalysis.run();testAssert(LocalAnalysis.predictions.size===20,'batch cap');
await LocalAnalysis.run();testAssert(LocalAnalysis.predictions.size===36,'resume skips finished');
return {state:LocalAnalysis.state,count:LocalAnalysis.predictions.size};
})()'''))
ref=json.loads(Path('docs/models/reference-outputs.json').read_text())
results=ev('[...LocalAnalysis.predictions.values()]')
max_error=0
for p in results:
 r=next(x for x in ref if x['id']==p['image_id']);labels=['BACK','NORM','DEB','TUM','ADI','MUC','MUS','STR','LYM']
 assert p['predicted_label']==r['predicted_label']
 error=max(abs(p['scores'][label]-r['scores'][i]) for i,label in enumerate(labels));max_error=max(max_error,error);assert error<1e-5
print('PASS: all 36 browser outputs match Python ONNX; max score error',max_error)
print(ev('''(async()=>{
openImage(sample[0].id);testAssert($('local-prediction').textContent.includes('Local model prediction'),'detail scores');closeImage();
route();route();testAssert(document.querySelectorAll('.model-label').length===36,'no duplicated badges');
selected.add(sample[0].id);$('finding-title').value='CPU test';$('finding-body').value='Testing local prediction evidence.';$('finding-form').requestSubmit();testAssert(findings[0].local_predictions[sample[0].id].model_sha256,'finding provenance');
let exported;const original=URL.createObjectURL;URL.createObjectURL=b=>{exported=b;return original(b);};$('export-predictions').click();testAssert(JSON.parse(await exported.text()).predictions.length===36,'prediction export');
exportReviews();testAssert(Object.keys(JSON.parse(await exported.text()).local_predictions).length===36,'review export');
await report(findings[0]);testAssert((await exported.text()).includes('tia-mobilenet-v3-small-kather-onnx-v1'),'report model');
const image=await loadImage(sample[0]),canvas=document.createElement('canvas');canvas.width=canvas.height=224;canvas.getContext('2d').drawImage(image,0,0);const blob=await new Promise(r=>canvas.toBlob(r));await importFiles([new File([blob],'cpu-local.png',{type:'image/png'})]);
await LocalAnalysis.run();testAssert(!LocalAnalysis.predictions.has(imported[0].id),'domain gate');$('confirm-domain').checked=true;await LocalAnalysis.run();testAssert(LocalAnalysis.predictions.has(imported[0].id),'imported inference');
canvas.width=400;canvas.height=200;const rect=await new Promise(r=>canvas.toBlob(r));await importFiles([new File([rect],'rect.png',{type:'image/png'})]);await LocalAnalysis.run([imported[1]]);testAssert(LocalAnalysis.state==='idle'&&$('analysis-status').textContent.includes('square'),'reject rectangular');
LocalAnalysis.enable();await testWait(()=>LocalAnalysis.state==='ready');LocalAnalysis.predictions.delete(sample[0].id);LocalAnalysis.run([sample[0]]);LocalAnalysis.stop();await new Promise(r=>setTimeout(r,200));testAssert(LocalAnalysis.state==='idle','cancel inference');
const cache=await caches.open('tissology-local-model-v1');await cache.put(new URL('models/mobilenet-v3-small-kather.onnx',location.href),new Response('corrupt'));
LocalAnalysis.enable();await testWait(()=>LocalAnalysis.state==='ready');testAssert((await (await cache.match(new URL('models/mobilenet-v3-small-kather.onnx',location.href))).arrayBuffer()).byteLength===3741332,'corrupt cache repaired');
LocalAnalysis.stop();$('forget-model').click();await new Promise(r=>setTimeout(r,200));testAssert(!(await caches.keys()).includes('tissology-local-model-v1'),'forget cache');
collection='sample';$('collection-select').value='sample';clearFilters();location.hash='images';return 'PASS: detail scores, badges, prediction/review/report export, finding provenance, local import inference, domain gate, shape rejection, cancellation, cache corruption recovery and removal';
})()'''))
Path('/tmp/tissology-cpu-desktop.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',{'format':'png'})['data']))
call('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':True});time.sleep(.2)
assert ev('document.documentElement.scrollWidth<=innerWidth'),'mobile overflow'
Path('/tmp/tissology-cpu-mobile.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',{'format':'png'})['data']))
assert not errors,errors
print('PASS: mobile layout and no runtime exceptions')
print('Browser CPU inference milliseconds:',{'min':min(x['elapsed_ms'] for x in results),'max':max(x['elapsed_ms'] for x in results),'mean':sum(x['elapsed_ms'] for x in results)/len(results)})
# Leave this isolated browser with no test work.
ev("imported=[];findings=[];reviews={};selected.clear();persist();$('finding-title').value='';$('finding-body').value='';")
w.close()
