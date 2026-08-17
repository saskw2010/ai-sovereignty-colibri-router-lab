"""Reproducible telemetry for the frozen-base MK7 v1.0 router."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE=Path(r"F:\AI-OPEN-MODELS\qwen2.5-0.5b")
DATA=Path(r"Q:\Colibri\training\datasets\mk7\v0.1\combined")
CKPT=Path(r"F:\AI-OPEN-MODELS\mk7-versions\router-v1.0.0-real\router-only-qwen-mk7-v1.0-domain.pt")
OUT=CKPT.parent/"telemetry-v1.0.json"
LABELS=["subject_ontology","source_evidence","structure_relations","method_validation","function_application","time_evolution","human_value_context"]
MP={x:i for i,x in enumerate(LABELS)}
CHALLENGE=[("كيف نعرّف نطاق الكيان الذي تدور حوله هذه المعرفة؟",0),("ما المفاهيم التي يجب وضعها في قاموس الموضوع؟",0),("ما المرجع الذي يمكن الرجوع إليه للتحقق من النتيجة؟",1),("كيف نميز الدليل القوي من الادعاء غير الموثق؟",1),("كيف تمثل الروابط بين الوحدات داخل النظام؟",2),("ما الذي يفسر اعتماد هذا الجزء على ذاك؟",2),("ما الاختبار المناسب للتأكد من سلامة الإجراءات؟",3),("أي فرضية قد تجعل الاستنتاج غير موثوق؟",3),("كيف يستفيد فريق العمل من هذه المعرفة عمليًا؟",4),("ما الخطوة التي تحول الفكرة إلى خدمة قابلة للتنفيذ؟",4),("ما التغيرات التي مرت بها هذه القاعدة؟",5),("كيف نعرف أن الإصدار الحالي أحدث من السابق؟",5),("كيف يتأثر المستخدمون بهذا الاختيار؟",6),("ما الاعتبارات الأخلاقية عند تطبيق الحل؟",6)]
def load(name):
 rows=[]
 for line in (DATA/name).read_text(encoding="utf-8").splitlines():
  if line:
   r=json.loads(line); key=str(r.get("knowledge_unit_id","")).split(":")[-1]
   if key in MP: rows.append((r["input"],MP[key]))
 return rows
def main():
 device="cuda" if torch.cuda.is_available() else "cpu"
 tok=AutoTokenizer.from_pretrained(str(BASE)); tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(str(BASE),torch_dtype=torch.float32,low_cpu_mem_usage=True).eval().to(device)
 for p in model.parameters(): p.requires_grad=False
 state=torch.load(CKPT,map_location="cpu",weights_only=True)
 router=torch.nn.Sequential(torch.nn.Linear(state["input_dim"],128),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(128,7))
 router.load_state_dict(state["router"]); router.eval()
 def features(rows):
  out=[]
  with torch.no_grad():
   for start in range(0,len(rows),16):
    part=rows[start:start+16]; b=tok([x[0] for x in part],padding=True,truncation=True,max_length=512,return_tensors="pt").to(device)
    h=model(**b,output_hidden_states=True,use_cache=False).hidden_states[-1]; idx=b.attention_mask.sum(1)-1
    out.append(h[torch.arange(len(part),device=device),idx,:].float().cpu())
  return torch.cat(out)
 def report(rows):
  x=features(rows); y=torch.tensor([r[1] for r in rows]); logits=router(x); probs=logits.softmax(-1); pred=logits.argmax(-1)
  cm=[[0]*7 for _ in range(7)]
  for actual,guess in zip(y.tolist(),pred.tolist()): cm[actual][guess]+=1
  counts=[int((pred==i).sum()) for i in range(7)]
  ent=-(probs*probs.clamp_min(1e-12).log()).sum(-1)
  return {"samples":len(rows),"top1":float((pred==y).float().mean()),"top2":float((logits.topk(2,-1).indices==y[:,None]).any(1).float().mean()),"mean_entropy_nats":float(ent.mean()),"mean_entropy_normalized":float((ent/torch.log(torch.tensor(7.))).mean()),"predicted_expert_counts":dict(zip(LABELS,counts)),"confusion_matrix":cm}
 result={"status":"VERIFIED_MK7_ROUTER_V1_TELEMETRY","checkpoint":str(CKPT),"base_frozen":True,"router_only":True,"experts_trained":False,"datasets_untouched":True,"splits":{n:report(load(n)) for n in ["validation.jsonl","test.jsonl"]},"challenge":report(CHALLENGE),"labels":LABELS}
 OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__": main()
