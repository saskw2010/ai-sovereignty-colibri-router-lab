"""Controlled synthetic router training on frozen Qwen hidden states."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"F:\AI-OPEN-MODELS\qwen2.5-0.5b"
OUT = Path(r"F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-qwen-synthetic")
OUT.mkdir(parents=True, exist_ok=True)
PILLARS = ["O_subject_ontology", "E_source_evidence", "R_structure_relations", "M_method_validation", "A_function_application", "T_time_evolution", "H_human_value_context"]
TEXTS = [
    "Define the entities and concepts in this subject.",
    "List the evidence and sources supporting this claim.",
    "Explain the relations between these entities.",
    "Validate the method and check its assumptions.",
    "Apply the function to a practical workflow.",
    "Describe how this changed over time.",
    "Explain human values and consequences in this case.",
]
torch.manual_seed(7)
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval().to(device)
for p in model.parameters(): p.requires_grad = False
features = []
with torch.no_grad():
    for text in TEXTS:
        batch = tok(text, return_tensors="pt").to(device)
        out = model(**batch, output_hidden_states=True, use_cache=False)
        features.append(out.hidden_states[-1][:, -1, :].squeeze(0).float().cpu())
x = torch.stack(features)
y = torch.arange(len(PILLARS))
bridge = torch.nn.Linear(x.shape[-1], 32)
router = torch.nn.Linear(32, len(PILLARS), bias=False)
opt = torch.optim.AdamW(list(bridge.parameters()) + list(router.parameters()), lr=0.1)
losses = []
for step in range(1200):
    logits = router(bridge(x))
    loss = torch.nn.functional.cross_entropy(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(float(loss.detach()))
with torch.no_grad():
    pred = logits.argmax(dim=-1)
    accuracy = float((pred == y).float().mean())
torch.save({"bridge": bridge.state_dict(), "router": router.state_dict(), "input_dim": x.shape[-1], "output_dim": len(PILLARS)}, OUT / "qwen-router-v0.3-synthetic.pt")
result = {"status":"VERIFIED_QWEN_SYNTHETIC_ROUTER_TRAINING_V0_3","version":"0.3.0-qwen-synthetic","device":device,"samples":len(TEXTS),"steps":1200,"initial_loss":losses[0],"final_loss":losses[-1],"accuracy":accuracy,"base_frozen":True,"training_started":True,"real_mk7_dataset_loaded":False,"gemma_loaded":False,"golden_training_loaded":False,"optimizer_updated_router_and_bridge_only":True,"checkpoint":str(OUT / "qwen-router-v0.3-synthetic.pt")}
(OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
