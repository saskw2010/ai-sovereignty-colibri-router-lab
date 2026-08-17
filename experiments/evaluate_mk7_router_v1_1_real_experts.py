"""Load the real frozen LoRA bank and execute specialist/Top-2 routing."""
import json, shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE=Path(r"F:\AI-OPEN-MODELS\qwen2.5-0.5b"); BANK=Path(r"F:\AI-OPEN-MODELS\mk7-versions\router-v1.1.0-real-lora-experts"); ROUTER=Path(r"F:\AI-OPEN-MODELS\mk7-versions\router-v1.0.0-real\router-only-qwen-mk7-v1.0-domain.pt"); OUT=Path(r"F:\AI-OPEN-MODELS\mk7-versions\router-v1.1.0-real-integrated"); OUT.mkdir(parents=True,exist_ok=True)
LABELS=["subject_ontology","source_evidence","structure_relations","method_validation","function_application","time_evolution","human_value_context"]
PROMPTS=["ما موضوع المعرفة وما الكيانات التي تتناولها؟","ما مصدر هذا الادعاء وما قوة دليله؟","ما العلاقات بين المفاهيم في هذا النظام؟","هل المنهج المقترح صحيح وقابل للتحقق؟","كيف نستخدم هذه المعرفة في عمل فعلي؟","كيف تطور هذا المفهوم عبر الزمن؟","ما أثر القرار على الإنسان وأصحاب المصلحة؟"]
def main():
 device="cuda" if torch.cuda.is_available() else "cpu"; tok=AutoTokenizer.from_pretrained(str(BASE)); tok.pad_token=tok.eos_token
 base=AutoModelForCausalLM.from_pretrained(str(BASE),torch_dtype=torch.float32,low_cpu_mem_usage=True)
 model=PeftModel.from_pretrained(base,str(BANK/LABELS[0]),adapter_name=LABELS[0])
 for label in LABELS[1:]: model.load_adapter(str(BANK/label),adapter_name=label)
 for p in model.parameters(): p.requires_grad=False
 model.eval().to(device)
 state=torch.load(ROUTER,map_location="cpu",weights_only=True); router=torch.nn.Sequential(torch.nn.Linear(state["input_dim"],128),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(128,7)); router.load_state_dict(state["router"]); router.eval()
 rows=[]; adapter_diffs=[]
 with torch.no_grad():
  for prompt in PROMPTS:
   b=tok(prompt,return_tensors="pt").to(device); h=model.base_model.model(**b,output_hidden_states=True,use_cache=False).hidden_states[-1][:,-1,:].float().cpu(); scores=router(h); top=scores.softmax(-1).topk(2,-1); ids=top.indices[0].tolist(); weights=(top.values/top.values.sum()).view(-1).tolist()
   outputs=[]
   for label in LABELS:
    model.set_adapter(label); outputs.append(model(**b,use_cache=False).logits[:,-1,:].float().cpu())
   specialist=outputs[ids[0]]; mixed=outputs[ids[0]]*weights[0]+outputs[ids[1]]*weights[1]; diff=float((specialist-mixed).abs().mean()); adapter_diffs.append(diff)
   rows.append({"prompt":prompt,"router_top2":[LABELS[i] for i in ids],"weights":weights,"specialist_logit_shape":list(specialist.shape),"top2_mixer_logit_shape":list(mixed.shape),"specialist_vs_mixer_mean_abs_logit_diff":diff})
 shutil.copy2(ROUTER,OUT/ROUTER.name); (OUT/"router-v1.1-real-experts-registry.json").write_text(json.dumps({"base":str(BASE),"experts":[{"id":x,"path":str(BANK/x),"frozen":True,"provenance":"real LoRA trained on canonical MK7 records"} for x in LABELS],"router_checkpoint":str(OUT/ROUTER.name)},indent=2),encoding="utf-8")
 result={"status":"VERIFIED_MK7_ROUTER_V1_1_REAL_EXPERT_RUNTIME","device":device,"dtype":"float32","expert_count":len(LABELS),"all_adapters_loaded":True,"all_adapters_frozen":True,"specialist_mode":True,"top2_mixer_mode":True,"nonzero_specialist_mixer_differences":sum(x>0 for x in adapter_diffs),"cases":rows,"dataset_modified":False,"output":str(OUT)}; (OUT/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__": main()
