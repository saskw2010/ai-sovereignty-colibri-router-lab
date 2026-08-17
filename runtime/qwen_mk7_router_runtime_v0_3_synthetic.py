"""Runtime smoke for the trained synthetic Qwen v0.3 bridge/router checkpoint."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

BASE = r"F:\AI-OPEN-MODELS\qwen2.5-0.5b"
CKPT = r"F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-qwen-synthetic\qwen-router-v0.3-synthetic.pt"
PILLARS = ["O_subject_ontology", "E_source_evidence", "R_structure_relations", "M_method_validation", "A_function_application", "T_time_evolution", "H_human_value_context"]
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True)
cfg = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
model = get_peft_model(base, cfg, adapter_name=PILLARS[0])
for name in PILLARS[1:]: model.add_adapter(name, cfg)
for p in model.parameters(): p.requires_grad = False
model.eval().to(device)
state = torch.load(CKPT, map_location="cpu", weights_only=True)
bridge = torch.nn.Linear(state["input_dim"], 32); bridge.load_state_dict(state["bridge"])
router = torch.nn.Linear(32, 7, bias=False); router.load_state_dict(state["router"])
bridge.eval().to(device); router.eval().to(device)
for p in list(bridge.parameters()) + list(router.parameters()): p.requires_grad = False
inputs = tok("Apply the method and explain its human consequences.", return_tensors="pt").to(device)
with torch.no_grad(), model.disable_adapter():
    probe = model(**inputs, output_hidden_states=True, use_cache=False)
    scores = router(bridge(probe.hidden_states[-1][:, -1, :]))
    weights = torch.softmax(scores, dim=-1)
    values, indices = torch.topk(weights, k=2, dim=-1)
    selected = [PILLARS[i] for i in indices[0].tolist()]
    outputs = []
    for adapter in selected:
        model.set_adapter(adapter)
        for p in model.parameters(): p.requires_grad = False
        outputs.append(model(**inputs, use_cache=False).logits)
    mixed = sum(float(values[0, i]) * outputs[i] for i in range(2))
for p in list(model.parameters()) + list(bridge.parameters()) + list(router.parameters()):
    p.requires_grad = False
result = {"status":"VERIFIED_QWEN_MK7_SYNTHETIC_ROUTER_V0_3_RUNTIME","checkpoint":CKPT,"device":device,"selected":selected,"weights":[float(x) for x in values[0]],"top_k":2,"logits_shape":list(mixed.shape),"trainable_parameters":sum(p.numel() for p in list(model.parameters())+list(bridge.parameters())+list(router.parameters()) if p.requires_grad),"training_started":False,"dataset_loaded":False,"real_mk7_dataset_loaded":False,"interpretation":"Runtime use of a synthetic-trained bridge/router checkpoint; no semantic quality claim."}
print(json.dumps(result, ensure_ascii=False))
