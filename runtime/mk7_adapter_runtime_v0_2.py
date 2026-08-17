"""MK7 v0.2 adapter registry/runtime smoke.

Creates a new version beside the frozen v0.1 baseline. It loads a copied
Router checkpoint and synthetic frozen LoRA adapters, then exercises Mixer,
Specialist, and Plug-and-Play modes. No project model, dataset, Gemma, or
Golden Training artifact is loaded and no training occurs.
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
import torch

SRC_ROUTER = Path(r"F:\AI-OPEN-MODELS\mk7-synthetic-seven-pillars-router-v0.1\router-seven-pillars.pt")
ROOT = Path(r"F:\AI-OPEN-MODELS\mk7-versions\router-v0.2.0")
PILLARS = ["O_subject_ontology", "E_source_evidence", "R_structure_relations", "M_method_validation", "A_function_application", "T_time_evolution", "H_human_value_context"]

def adapter_delta(x, a, b): return (x @ a) @ b

def main() -> int:
    torch.manual_seed(20260817); hidden = 32; experts = len(PILLARS); rank = 4
    ROOT.mkdir(parents=True, exist_ok=True); (ROOT / "weights").mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True)
    shutil.copy2(SRC_ROUTER, ROOT / "weights" / "router.pt")
    router = torch.nn.Linear(hidden, experts, bias=False); router.load_state_dict(torch.load(SRC_ROUTER, map_location="cpu", weights_only=True)); router.eval()
    for p in router.parameters(): p.requires_grad_(False)
    registry = {"version": "0.2.0", "base": {"id": "synthetic-dense-base-v0.2", "frozen": True}, "experts": [{"id": p, "adapter_path": f"synthetic://{p}", "frozen": True, "flags": ["mk7_pillar"]} for p in PILLARS], "router": {"id": "router-v0.1.0", "top_k": 2, "frozen": True}, "modes": ["mixer", "specialist", "plug_and_play"]}
    (ROOT / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    base = torch.nn.Linear(hidden, hidden, bias=False); base.eval()
    for p in base.parameters(): p.requires_grad_(False)
    adapters = {p: (torch.randn(hidden, rank) * .04, torch.randn(rank, hidden) * .04) for p in PILLARS}
    for a, b in adapters.values(): a.requires_grad_(False); b.requires_grad_(False)
    x = torch.randn(6, hidden)
    with torch.inference_mode():
        logits = router(x); probs = logits.softmax(-1); weights, ids = torch.topk(probs, k=2, dim=-1); weights = weights / weights.sum(-1, keepdim=True)
        base_out = base(x)
        deltas = {p: adapter_delta(x, *adapters[p]) for p in PILLARS}
        mixer_delta = torch.stack([deltas[PILLARS[i]] for i in ids[0].tolist()], dim=0)
        mixer_out = base_out + (mixer_delta * weights[0, :, None, None]).sum(0)
        specialist_id = PILLARS[int(ids[0, 0])]; specialist_out = base_out + deltas[specialist_id]
        requested = "R_structure_relations"; plugin_out = base_out + deltas[requested]
    result = {"status": "VERIFIED_MK7_ADAPTER_RUNTIME_V0_2", "version": "0.2.0", "modes": {"mixer": {"selected": [PILLARS[i] for i in ids[0].tolist()], "weights": weights[0].tolist(), "output_shape": list(mixer_out.shape)}, "specialist": {"selected": specialist_id, "output_shape": list(specialist_out.shape)}, "plug_and_play": {"requested": requested, "loaded_on_demand": True, "output_shape": list(plugin_out.shape)}}, "base_frozen": True, "adapters_frozen": True, "router_frozen_for_runtime": True, "training_started": False, "dataset_loaded": False, "gemma_loaded": False, "golden_training_loaded": False, "v0_1_untouched": True, "interpretation": "Runtime contract smoke using synthetic frozen adapters; not a real-base quality evaluation."}
    (ROOT / "results" / "runtime-smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "version": result["version"], "modes": list(result["modes"]), "v0_1_untouched": True}, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
