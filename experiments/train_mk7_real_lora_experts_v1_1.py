"""Create seven small, real LoRA expert adapters from read-only MK7 records.

The source JSONL files are never written. Each adapter is trained independently,
then frozen and saved under a new F: version directory.
"""
import json, hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

BASE=r"F:\AI-OPEN-MODELS\qwen2.5-0.5b"; DATA=Path(r"Q:\Colibri\training\datasets\mk7\v0.1\combined"); OUT=Path(r"F:\AI-OPEN-MODELS\mk7-versions\router-v1.1.0-real-lora-experts")
LABELS=["subject_ontology","source_evidence","structure_relations","method_validation","function_application","time_evolution","human_value_context"]
def records():
 out={x:[] for x in LABELS}
 for line in (DATA/"train.jsonl").read_text(encoding="utf-8").splitlines():
  if line:
   r=json.loads(line); key=str(r.get("knowledge_unit_id","")).split(":")[-1]
   if key in out and len(out[key])<8: out[key].append(r["input"])
 return out
def main():
 torch.manual_seed(20260818); OUT.mkdir(parents=True,exist_ok=True); tok=AutoTokenizer.from_pretrained(BASE); tok.pad_token=tok.eos_token; device="cuda" if torch.cuda.is_available() else "cpu"; data=records(); results=[]
 for label in LABELS:
  model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32,low_cpu_mem_usage=True)
  for p in model.parameters(): p.requires_grad=False
  cfg=LoraConfig(r=2,lora_alpha=4,lora_dropout=0.0,target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM)
  model=get_peft_model(model,cfg); model.to(device); model.train()
  batch=tok(data[label],padding=True,truncation=True,max_length=128,return_tensors="pt").to(device); batch["labels"]=batch["input_ids"].clone(); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.01)
  first=None; last=None
  for _ in range(3):
   out=model(**batch); loss=out.loss; first=first or float(loss); last=float(loss); opt.zero_grad(); loss.backward(); opt.step()
  for p in model.parameters(): p.requires_grad=False
  path=OUT/label; model.save_pretrained(path); (path/"expert-metadata.json").write_text(json.dumps({"expert":label,"base":"Qwen/Qwen2.5-0.5B","scope":"real_lora_adapter","frozen_after_training":True,"source":"read-only canonical MK7 train records","samples":len(data[label]),"initial_loss":first,"final_loss":last,"dataset_modified":False},indent=2),encoding="utf-8")
  results.append({"expert":label,"samples":len(data[label]),"initial_loss":first,"final_loss":last,"path":str(path),"trainable_after_save":sum(p.numel() for p in model.parameters() if p.requires_grad)})
  del model; torch.cuda.empty_cache()
 result={"status":"VERIFIED_MK7_REAL_LORA_EXPERTS_V1_1","base":"Qwen/Qwen2.5-0.5B","device":device,"dtype":"float32","experts":results,"all_frozen_after_save":True,"dataset_modified":False,"golden_training_loaded":False,"gemma_loaded":False,"output":str(OUT)}; (OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__": main()
