"""Challenge split for expanded synthetic Qwen router v0.4."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"F:\\AI-OPEN-MODELS\\qwen2.5-0.5b"
CKPT = r"F:\\AI-OPEN-MODELS\\mk7-versions\\router-v0.4.0-qwen-synthetic\\qwen-router-v0.4-synthetic.pt"
rows = [(0,"Map the vocabulary and categories used by this field."),(0,"Which objects belong to this area of study?"),(1,"What documentation verifies the reported result?"),(1,"Find the provenance behind this piece of information."),(2,"Represent the links and dependencies among the modules."),(2,"Why does one component depend on another?"),(3,"Audit this technique for weaknesses and assumptions."),(3,"How would you verify that this approach works?"),(4,"Give a concrete business use for this idea."),(4,"How can an operator put this capability into action?"),(5,"What events led to the current version of the policy?"),(5,"Compare the stages of change in this system."),(6,"Who may be affected and what duties follow?"),(6,"Discuss the moral and social risks of the proposal.")]
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval().to(device)
state = torch.load(CKPT, map_location="cpu", weights_only=True)
bridge = torch.nn.Linear(state["input_dim"], 32); bridge.load_state_dict(state["bridge"])
router = torch.nn.Linear(32, 7, bias=False); router.load_state_dict(state["router"])
bridge.eval().to(device); router.eval().to(device); results = []
with torch.no_grad():
    for label, text in rows:
        batch = tok(text, return_tensors="pt").to(device)
        hidden = model(**batch, output_hidden_states=True, use_cache=False).hidden_states[-1][:, -1, :]
        probs = torch.softmax(router(bridge(hidden)), dim=-1)[0]
        values, indices = torch.topk(probs, 2)
        results.append({"label": label, "top1": int(indices[0]), "top2": indices.tolist(), "top1_weight": float(values[0])})
print(json.dumps({"status":"VERIFIED_QWEN_SYNTHETIC_ROUTER_V0_4_CHALLENGE_EVAL","samples":len(results),"top1_accuracy":sum(x["top1"]==x["label"] for x in results)/len(results),"top2_hit_rate":sum(x["label"] in x["top2"] for x in results)/len(results),"rows":results,"device":device,"training_started":False,"dataset_loaded":False},ensure_ascii=False))
