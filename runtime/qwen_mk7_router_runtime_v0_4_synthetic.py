"""Runtime smoke for expanded synthetic Qwen router v0.4."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
BASE=r"F:\AI-OPEN-MODELS\qwen2.5-0.5b"; CKPT=r"F:\AI-OPEN-MODELS\mk7-versions\router-v0.4.0-qwen-synthetic\qwen-router-v0.4-synthetic.pt"; P=["O_subject_ontology","E_source_evidence","R_structure_relations","M_method_validation","A_function_application","T_time_evolution","H_human_value_context"]
device="cuda" if torch.cuda.is_available() else "cpu"; tok=AutoTokenizer.from_pretrained(BASE); base=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32,low_cpu_mem_usage=True); cfg=LoraConfig(r=2,lora_alpha=4,lora_dropout=0,target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM); model=get_peft_model(base,cfg,adapter_name=P[0])
for n in P[1:]: model.add_adapter(n,cfg)
for p in model.parameters(): p.requires_grad=False
model.eval().to(device); s=torch.load(CKPT,map_location="cpu",weights_only=True); bridge=torch.nn.Linear(s["input_dim"],32); bridge.load_state_dict(s["bridge"]); router=torch.nn.Linear(32,7,bias=False); router.load_state_dict(s["router"]); bridge.eval().to(device); router.eval().to(device)
inputs=tok("Review the assumptions of this method and apply it responsibly.",return_tensors="pt").to(device)
with torch.no_grad(), model.disable_adapter():
 h=model(**inputs,output_hidden_states=True,use_cache=False).hidden_states[-1][:,-1,:]; probs=torch.softmax(router(bridge(h)),dim=-1); values,indices=torch.topk(probs,2,dim=-1); selected=[P[i] for i in indices[0].tolist()]; outs=[]
 for name in selected:
  model.set_adapter(name); [setattr(p,"requires_grad",False) for p in model.parameters()]; outs.append(model(**inputs,use_cache=False).logits)
 mixed=sum(float(values[0,i])*outs[i] for i in range(2))
for p in list(model.parameters())+list(bridge.parameters())+list(router.parameters()): p.requires_grad=False
print(json.dumps({"status":"VERIFIED_QWEN_MK7_SYNTHETIC_ROUTER_V0_4_RUNTIME","checkpoint":CKPT,"device":device,"selected":selected,"weights":[float(x) for x in values[0]],"logits_shape":list(mixed.shape),"trainable_parameters":sum(p.numel() for p in list(model.parameters())+list(bridge.parameters())+list(router.parameters()) if p.requires_grad),"training_started":False,"dataset_loaded":False},ensure_ascii=False))
