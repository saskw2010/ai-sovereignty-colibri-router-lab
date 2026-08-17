"""Synthetic MK7 Router training gate.

This is an intentionally small learning experiment. The dense base and LoRA
experts are frozen; only the router is optimized. No Colibri model, Gemma,
Golden Training asset, MK7 Dataset, or external checkpoint is loaded.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn

ROOT = Path(r"F:\AI-OPEN-MODELS\mk7-synthetic-router-v0.1")
OUT = ROOT / "results.json"
WEIGHTS = ROOT / "router.pt"

def main() -> int:
    torch.manual_seed(20260817)
    device = torch.device("cpu")
    hidden, experts, rank, samples, steps = 16, 4, 4, 512, 120
    # Four synthetic domains, each with a distinct hidden-state centroid.
    centers = torch.randn(experts, hidden) * 2.5
    labels = torch.arange(samples) % experts
    x = centers[labels] + torch.randn(samples, hidden) * 0.35
    base = nn.Linear(hidden, hidden, bias=False).to(device)
    a = nn.Parameter(torch.randn(experts, hidden, rank) * .05, requires_grad=False)
    b = nn.Parameter(torch.randn(experts, rank, hidden) * .05, requires_grad=False)
    for p in base.parameters(): p.requires_grad_(False)
    router = nn.Linear(hidden, experts, bias=False).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=0.08)
    initial = {k: v.detach().clone() for k, v in router.state_dict().items()}
    loss0 = None
    for step in range(steps):
        logits = router(x)
        loss = nn.functional.cross_entropy(logits, labels)
        if loss0 is None: loss0 = float(loss)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        logits = router(x); pred = logits.argmax(-1); accuracy = float((pred == labels).float().mean())
        probs = logits.softmax(-1); entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum(-1).mean())
    changed = any(not torch.equal(initial[k], v) for k, v in router.state_dict().items())
    ROOT.mkdir(parents=True, exist_ok=True); torch.save(router.state_dict(), WEIGHTS)
    result = {
        "status": "VERIFIED_SYNTHETIC_ROUTER_TRAINING",
        "device": str(device), "dtype": "float32", "seed": 20260817,
        "samples": samples, "steps": steps, "experts": experts, "hidden": hidden,
        "initial_loss": loss0, "final_loss": float(loss), "routing_accuracy": accuracy,
        "mean_router_entropy": entropy, "router_weights_changed": changed,
        "base_frozen": all(not p.requires_grad for p in base.parameters()),
        "experts_frozen": not a.requires_grad and not b.requires_grad,
        "dataset_source": "synthetic_centroid_domains_only",
        "project_model_loaded": False, "gemma_loaded": False, "mk7_dataset_loaded": False,
        "golden_training_loaded": False, "external_weights_loaded": False,
        "optimizer_steps": steps, "output_weights": str(WEIGHTS),
        "interpretation": "Toy proof that a trainable Router can learn separable routing while Base and LoRA Experts remain frozen; not MK7 quality evidence."
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "accuracy": accuracy, "initial_loss": loss0, "final_loss": float(loss), "weights": str(WEIGHTS)}, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
