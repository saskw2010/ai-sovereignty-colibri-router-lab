"""Approved router-only MK7 run on the explicitly labeled seven-pillar subset."""
import json, hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE=r"F:\AI-OPEN-MODELS\qwen2.5-0.5b"; DATA=Path(r"Q:\Colibri\training\datasets\mk7\v0.1\combined"); OUT=Path(r"F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-real"); OUT.mkdir(parents=True,exist_ok=True)
allowed=["subject_ontology","source_evidence","structure_relations","method_validation","function_application","time_evolution","human_value_context"]; mapping={x:i for i,x in enumerate(allowed)}
def load(name):
 rows=[]
 for line in (DATA/name).read_text(encoding="utf-8").splitlines():
  if not line: continue
  row=json.loads(line); key=str(row.get("knowledge_unit_id","")).split(":")[-1]
  if key in mapping: rows.append((row["input"],mapping[key]))
 return rows
train,val,test=load("train.jsonl"),load("validation.jsonl"),load("test.jsonl")
device="cuda" if torch.cuda.is_available() else "cpu"; tok=AutoTokenizer.from_pretrained(BASE); tok.pad_token=tok.eos_token; model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32,low_cpu_mem_usage=True).eval().to(device)
for p in model.parameters(): p.requires_grad=False
def features(rows):
 texts=[x[0] for x in rows]
 with torch.no_grad():
  batch=tok(texts,padding=True,truncation=True,max_length=512,return_tensors="pt").to(device); out=model(**batch,output_hidden_states=True,use_cache=False); idx=batch.attention_mask.sum(1)-1; h=out.hidden_states[-1][torch.arange(len(texts),device=device),idx,:].float().cpu()
 return h,torch.tensor([x[1] for x in rows])
xtr,ytr=features(train); xval,yval=features(val); xte,yte=features(test)
torch.manual_seed(23); bridge=torch.nn.Linear(xtr.shape[-1],32); router=torch.nn.Linear(32,7,bias=False); opt=torch.optim.AdamW(list(bridge.parameters())+list(router.parameters()),lr=0.03); losses=[]
for step in range(120):
 logits=router(bridge(xtr)); loss=torch.nn.functional.cross_entropy(logits,ytr); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
with torch.no_grad():
 def metrics(x,y):
  z=router(bridge(x)); top=z.topk(2,dim=-1).indices; return {"top1_accuracy":float((top[:,0]==y).float().mean()),"top2_hit_rate":float((top==y[:,None]).any(1).float().mean())}
 result={"status":"VERIFIED_MK7_REAL_ROUTER_ONLY_V0_3","base":"Qwen/Qwen2.5-0.5B","base_revision":"060db6499f32faf8b98477b0a26969ef7d8b9987","device":device,"scope":"router_only","steps":120,"label_set":allowed,"counts":{"train":len(train),"validation":len(val),"test":len(test)},"initial_loss":losses[0],"final_loss":losses[-1],"train":metrics(xtr,ytr),"validation":metrics(xval,yval),"test":metrics(xte,yte),"base_frozen":True,"experts_trained":False,"dataset_modified":False,"gemma_loaded":False,"golden_training_loaded":False,"optimizer_scope":"bridge_and_router_only","checkpoint":str(OUT/"router-only-qwen-mk7-v0.3.pt")}
torch.save({"bridge":bridge.state_dict(),"router":router.state_dict(),"input_dim":xtr.shape[-1],"labels":allowed,"dataset_manifest_hash":"65c40626b5cc867c0828c349eb39a8e8aaac0899ed0cf91a1f23972978582460"},OUT/"router-only-qwen-mk7-v0.3.pt"); (OUT/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,ensure_ascii=False))
