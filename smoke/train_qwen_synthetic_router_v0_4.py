"""Expanded synthetic router experiment with held-out paraphrases."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"F:\AI-OPEN-MODELS\qwen2.5-0.5b"; OUT = Path(r"F:\AI-OPEN-MODELS\mk7-versions\router-v0.4.0-qwen-synthetic"); OUT.mkdir(parents=True, exist_ok=True)
groups = [
 ["Define the entities and concepts in this subject.","What are the main concepts and objects involved here?","Identify the subject's core entities.","Build an ontology for this topic.","Classify the things discussed in this domain."],
 ["List the evidence and sources supporting this claim.","Which sources substantiate this statement?","Trace this assertion to its references.","Separate supported facts from unsupported claims.","Cite the evidence behind the conclusion."],
 ["Explain the relations between these entities.","How do the components connect to each other?","Map the dependencies among the parts.","Describe how these concepts interact.","Show the structure linking the entities."],
 ["Validate the method and check its assumptions.","Check whether this procedure is sound and justified.","Test the reliability of the proposed method.","Review the assumptions and methodology.","Determine whether the approach is valid."],
 ["Apply the function to a practical workflow.","Use this capability in an operational scenario.","Turn the concept into an actionable process.","Demonstrate how this works in practice.","Design an application of the method."],
 ["Describe how this changed over time.","Trace the development of this idea across different periods.","Explain the historical evolution of the subject.","Compare the earlier and later states.","Identify the timeline and turning points."],
 ["Explain human values and consequences in this case.","Discuss the effects on people and the ethical implications.","Assess the social and human impact.","What values and responsibilities are involved?","Evaluate the consequences for affected communities."],
]
device="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(17); tok=AutoTokenizer.from_pretrained(BASE); tok.pad_token=tok.eos_token; model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32,low_cpu_mem_usage=True).eval().to(device)
for p in model.parameters(): p.requires_grad=False
texts=[t for g in groups for t in g]; labels=torch.tensor([i for i in range(7) for _ in range(5)])
with torch.no_grad():
 batch=tok(texts,padding=True,return_tensors="pt").to(device); out=model(**batch,output_hidden_states=True,use_cache=False); lengths=batch.attention_mask.sum(dim=1)-1; x=out.hidden_states[-1][torch.arange(len(texts),device=device),lengths,:].float().cpu()
train_idx=torch.tensor([i for i in range(len(texts)) if i%5<4]); val_idx=torch.tensor([i for i in range(len(texts)) if i%5==4]); y=labels
bridge=torch.nn.Linear(x.shape[-1],32); router=torch.nn.Linear(32,7,bias=False); opt=torch.optim.AdamW(list(bridge.parameters())+list(router.parameters()),lr=0.05); losses=[]
for step in range(1000):
 logits=router(bridge(x[train_idx])); loss=torch.nn.functional.cross_entropy(logits,y[train_idx]); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
with torch.no_grad():
 train_pred=router(bridge(x[train_idx])).argmax(-1); val_logits=router(bridge(x[val_idx])); val_pred=val_logits.argmax(-1); val_top2=val_logits.topk(2,dim=-1).indices
result={"status":"VERIFIED_QWEN_EXPANDED_SYNTHETIC_ROUTER_V0_4","version":"0.4.0-qwen-synthetic","device":device,"samples":len(texts),"train_samples":len(train_idx),"validation_samples":len(val_idx),"steps":1000,"initial_loss":losses[0],"final_loss":losses[-1],"train_accuracy":float((train_pred==y[train_idx]).float().mean()),"validation_top1_accuracy":float((val_pred==y[val_idx]).float().mean()),"validation_top2_hit_rate":float((val_top2==y[val_idx,None]).any(dim=1).float().mean()),"base_frozen":True,"real_mk7_dataset_loaded":False,"gemma_loaded":False,"golden_training_loaded":False,"checkpoint":str(OUT/"qwen-router-v0.4-synthetic.pt")}
torch.save({"bridge":bridge.state_dict(),"router":router.state_dict(),"input_dim":x.shape[-1],"output_dim":7},OUT/"qwen-router-v0.4-synthetic.pt"); (OUT/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,ensure_ascii=False))
