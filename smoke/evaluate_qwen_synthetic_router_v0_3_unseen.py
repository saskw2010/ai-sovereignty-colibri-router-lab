"""Unseen paraphrase evaluation for the synthetic Qwen router."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"F:\\AI-OPEN-MODELS\\qwen2.5-0.5b"
CKPT = r"F:\\AI-OPEN-MODELS\\mk7-versions\\router-v0.3.0-qwen-synthetic\\qwen-router-v0.3-synthetic.pt"
texts = [
    "What are the main concepts and objects involved here?",
    "Which sources substantiate this statement?",
    "How do the components connect to each other?",
    "Check whether this procedure is sound and justified.",
    "Use this capability in an operational scenario.",
    "Trace the development of this idea across different periods.",
    "Discuss the effects on people and the ethical implications.",
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
    "status": "VERIFIED_QWEN_SYNTHETIC_ROUTER_UNSEEN_EVAL",
    "samples": len(rows),
    "top1_accuracy": sum(x["top1"] == x["label"] for x in rows) / len(rows),
    "top2_hit_rate": sum(x["label"] in x["top2"] for x in rows) / len(rows),
    "rows": rows,
    "device": device,
    "training_started": False,
    "dataset_loaded": False,
}
print(json.dumps(result, ensure_ascii=False))
