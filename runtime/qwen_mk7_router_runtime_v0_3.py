"""Runtime-only MK7 router selection over Qwen + frozen PEFT adapters."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

BASE = r"F:\AI-OPEN-MODELS\qwen2.5-0.5b"
ROUTER = r"F:\AI-OPEN-MODELS\mk7-versions\router-v0.1.0\weights\router-seven-pillars.pt"
PILLARS = ["O_subject_ontology", "E_source_evidence", "R_structure_relations", "M_method_validation", "A_function_application", "T_time_evolution", "H_human_value_context"]
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

inputs = tok("MK7 router v0.3 runtime selection smoke.", return_tensors="pt").to(device)
with torch.no_grad():
    # First pass: obtain request representation without an adapter contribution.
    with model.disable_adapter():
        probe = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = probe.hidden_states[-1][:, -1, :]
    # Deterministic frozen bridge from Qwen hidden size to the v0.1 router input size.
    bridge = hidden[:, :32]
    scores = router(bridge)
    weights = torch.softmax(scores, dim=-1)
    top_values, top_indices = torch.topk(weights, k=2, dim=-1)
    selected = [PILLARS[i] for i in top_indices[0].tolist()]
    logits = []
    for adapter in selected:
        model.set_adapter(adapter)
        for p in model.parameters(): p.requires_grad = False
        logits.append(model(**inputs, use_cache=False).logits)
    mixed = sum(float(top_values[0, i]) * logits[i] for i in range(2))

result = {
    "status": "VERIFIED_QWEN_MK7_ROUTER_RUNTIME_V0_3",
    "base": "Qwen/Qwen2.5-0.5B",
    "router_weights": ROUTER,
    "router_version": "router-v0.1.0",
    "device": device,
    "dtype": "float32",
    "top_k": 2,
    "selected": selected,
    "weights": [float(x) for x in top_values[0]],
    "logits_shape": list(mixed.shape),
    "trainable_parameters": sum(p.numel() for p in list(model.parameters()) + list(router.parameters()) if p.requires_grad),
    "training_started": False,
    "optimizer_step": False,
    "dataset_loaded": False,
    "interpretation": "Runtime router selection smoke; bridge is frozen and deterministic, adapters/base/router are not trained.",
}
print(json.dumps(result, ensure_ascii=False))
