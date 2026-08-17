"""Read-only multi-probe telemetry for the Qwen/MK7 frozen router runtime."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

BASE = r"F:\AI-OPEN-MODELS\qwen2.5-0.5b"
ROUTER = r"F:\AI-OPEN-MODELS\mk7-versions\router-v0.1.0\weights\router-seven-pillars.pt"
PILLARS = ["O_subject_ontology", "E_source_evidence", "R_structure_relations", "M_method_validation", "A_function_application", "T_time_evolution", "H_human_value_context"]
PROBES = {
    "english": "Explain the relation between two entities in a structured system.",
    "arabic": "اشرح العلاقة بين دليل ومعلومة في سياق معرفي.",
    "code": "Write a Python function that validates a JSON schema.",
    "erp": "Create an ERP workflow for an invoice approval process.",
    "contract": "Summarize the obligations and evidence required by this contract.",
    "history": "Describe how this policy changed over time and why.",
    "human_values": "Explain the human impact and ethical tradeoffs of this decision.",
}
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True)
cfg = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
model = get_peft_model(base, cfg, adapter_name=PILLARS[0])
for name in PILLARS[1:]: model.add_adapter(name, cfg)
for p in model.parameters(): p.requires_grad = False
model.eval().to(device)
router = torch.nn.Linear(32, 7, bias=False)
router.load_state_dict(torch.load(ROUTER, map_location="cpu", weights_only=True))
router.eval().to(device)
for p in router.parameters(): p.requires_grad = False

rows = []
for probe_name, text in PROBES.items():
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad(), model.disable_adapter():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    scores = router(out.hidden_states[-1][:, -1, :32])
    weights = torch.softmax(scores, dim=-1)[0]
    values, indices = torch.topk(weights, k=2)
    rows.append({"probe": probe_name, "selected": [PILLARS[i] for i in indices.tolist()], "weights": [float(x) for x in values.tolist()], "entropy": float(-(weights * (weights + 1e-12).log()).sum())})

print(json.dumps({"status":"VERIFIED_QWEN_MK7_ROUTER_PROBE_SUITE","device":device,"router_version":"router-v0.1.0","top_k":2,"rows":rows,"training_started":False,"dataset_loaded":False},ensure_ascii=False))
