"""Export the pinned TIA model; run with the versions in docs/models/README.md."""
from pathlib import Path
import hashlib,json,requests,torch,torchvision,onnx,onnxruntime as ort,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
REV='ea909bdcce3aa409bf436c09b4495cca90a8e99b'
URL=f'https://huggingface.co/TIACentre/TIAToolbox_pretrained_weights/resolve/{REV}/mobilenet_v3_small-kather100k.pth'
CACHE=Path('/tmp/tissology-mobilenet-small.pth')
if not CACHE.exists():
 r=requests.get(URL,timeout=120);r.raise_for_status();CACHE.write_bytes(r.content)
assert hashlib.sha256(CACHE.read_bytes()).hexdigest() == 'd5ac6a7d63db2100f0a6ab2ea8669525b28616c629e208583ec7386c509a3bf5', 'Source checkpoint checksum mismatch'
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.feat_extract=torchvision.models.mobilenet_v3_small(weights=None).features;self.pool=torch.nn.AdaptiveAvgPool2d((1,1));self.classifier=torch.nn.Linear(576,9)
 def forward(self,x):return torch.softmax(self.classifier(torch.flatten(self.pool(self.feat_extract(x)),1)),dim=-1)
torch.set_num_threads(2)
model=Model().eval();state=torch.load(CACHE,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
out=ROOT/'docs/models/mobilenet-v3-small-kather.onnx'
torch.onnx.export(model,torch.zeros(1,3,224,224),str(out),input_names=['pixels'],output_names=['scores'],opset_version=17,dynamo=False)
onnx.checker.check_model(str(out));session=ort.InferenceSession(str(out),providers=['CPUExecutionProvider'])
labels=['BACK','NORM','DEB','TUM','ADI','MUC','MUS','STR','LYM'];rows=[];max_error=0
manifest=json.loads((ROOT/'docs/samples.json').read_text())
with torch.inference_mode():
 for image in manifest['images']:
  a=np.array(Image.open(ROOT/'docs'/image['src']).convert('RGB'),dtype=np.float32).transpose(2,0,1)[None]/255
  ref=model(torch.from_numpy(a)).numpy();actual=session.run(None,{'pixels':a})[0];error=float(np.max(np.abs(ref-actual)));max_error=max(max_error,error)
  assert np.argmax(ref)==np.argmax(actual) and error<1e-5,(image['id'],error)
  rows.append({'id':image['id'],'reference_label':image['label'],'predicted_label':labels[int(actual.argmax())],'scores':actual[0].tolist()})
metadata={'id':'tia-mobilenet-v3-small-kather-onnx-v1','name':'TIA MobileNetV3-Small · Kather100K','source':URL,'source_revision':REV,'source_sha256':hashlib.sha256(CACHE.read_bytes()).hexdigest(),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'labels':labels,'license':'CC-BY-4.0','input':{'name':'pixels','shape':[1,3,224,224],'dtype':'float32','preprocessing':'RGB, resize square tiles to 224 x 224, channels-first, divide each channel by 255; no mean/std normalization'},'output':{'name':'scores','shape':[1,9],'meaning':'Softmax scores, not calibrated confidence'},'opset':17,'conversion':{'torch':torch.__version__,'torchvision':torchvision.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__},'validation':{'sample_count':len(rows),'max_absolute_score_error':max_error,'top1_matches':len(rows),'scope':'Numerical conversion check on the 36 bundled samples; not an independent accuracy evaluation.'}}
(ROOT/'docs/models/manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
(ROOT/'docs/models/reference-outputs.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
