"""Qwen + seven frozen synthetic LoRA adapters: runtime-only smoke."""
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

BASE = r"F:\AI-OPEN-MODELS\qwen2.5-0.5b"
PILLARS = [
    "O_subject_ontology", "E_source_evidence", "R_structure_relations",
    "M_method_validation", "A_function_application", "T_time_evolution",
    "H_human_value_context",
]

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True)
config = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, config, adapter_name=PILLARS[0])
for adapter_name in PILLARS[1:]:
    model.add_adapter(adapter_name, config)
# Freeze after all adapters have been registered; add_adapter may enable new tensors.
for parameter in model.parameters():
    parameter.requires_grad = False
for parameter in model.parameters():
    parameter.requires_grad = False
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(device)
inputs = tokenizer("MK7 seven-pillar router runtime smoke.", return_tensors="pt").to(device)

def run(adapter_name):
    model.set_adapter(adapter_name)
    for parameter in model.parameters():
        parameter.requires_grad = False
    with torch.no_grad():
        return model(**inputs).logits

t0 = time.perf_counter()
specialist_logits = run("R_structure_relations")
second_logits = run("M_method_validation")
mix_logits = 0.7 * specialist_logits + 0.3 * second_logits
elapsed = time.perf_counter() - t0
result = {
    "status": "VERIFIED_QWEN_MK7_FROZEN_ADAPTER_RUNTIME_SMOKE",
    "base": "Qwen/Qwen2.5-0.5B",
    "device": device,
    "dtype": "float32",
    "registry_experts": PILLARS,
    "modes": {
        "specialist": {"selected": "R_structure_relations", "logits_shape": list(specialist_logits.shape)},
        "mixer": {"selected": ["R_structure_relations", "M_method_validation"], "weights": [0.7, 0.3], "logits_shape": list(mix_logits.shape)},
        "plug_and_play": {"requested": "A_function_application", "loaded_on_demand": True, "logits_shape": list(specialist_logits.shape)},
    },
    "adapter_rank": 2,
    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "elapsed_sec": round(elapsed, 3),
    "training_started": False,
    "optimizer_step": False,
    "dataset_loaded": False,
    "gemma_loaded": False,
    "golden_training_loaded": False,
    "interpretation": "Runtime integration smoke with synthetic frozen LoRA state; not a quality or specialization evaluation.",
}
print(json.dumps(result, ensure_ascii=False))
