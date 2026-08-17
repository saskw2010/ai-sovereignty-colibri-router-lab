"""Closed-set evaluation for the synthetic Qwen router checkpoint."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"F:\\AI-OPEN-MODELS\\qwen2.5-0.5b"
CKPT = r"F:\\AI-OPEN-MODELS\\mk7-versions\\router-v0.3.0-qwen-synthetic\\qwen-router-v0.3-synthetic.pt"
texts = [
    "Define the entities and concepts in this subject.",
    "List the evidence and sources supporting this claim.",
    "Explain the relations between these entities.",
    "Validate the method and check its assumptions.",
    "Apply the function to a practical workflow.",
    "Describe how this changed over time.",
    "Explain human values and consequences in this case.",
]
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval().to(device)
state = torch.load(CKPT, map_location="cpu", weights_only=True)
bridge = torch.nn.Linear(state["input_dim"], 32); bridge.load_state_dict(state["bridge"])
router = torch.nn.Linear(32, 7, bias=False); router.load_state_dict(state["router"])
bridge.eval().to(device); router.eval().to(device)
rows = []
with torch.no_grad():
    for label, text in enumerate(texts):
        batch = tok(text, return_tensors="pt").to(device)
        hidden = model(**batch, output_hidden_states=True, use_cache=False).hidden_states[-1][:, -1, :]
        probs = torch.softmax(router(bridge(hidden)), dim=-1)[0]
        values, indices = torch.topk(probs, 2)
        rows.append({"label": label, "top1": int(indices[0]), "top2": indices.tolist(), "top1_weight": float(values[0])})
result = {
    "status": "VERIFIED_QWEN_SYNTHETIC_ROUTER_CLOSED_SET_EVAL",
    "samples": len(rows),
    "top1_accuracy": sum(row["top1"] == row["label"] for row in rows) / len(rows),
    "top2_hit_rate": sum(row["label"] in row["top2"] for row in rows) / len(rows),
    "rows": rows,
    "device": device,
    "training_started": False,
    "dataset_loaded": False,
}
print(json.dumps(result, ensure_ascii=False))
