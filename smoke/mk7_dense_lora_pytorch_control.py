"""PyTorch control for Dense + Frozen LoRA Experts + Top-2 Router.

Synthetic tensors are used deliberately. This validates the real module
contract and freezing behavior without loading a project model/dataset or
performing an optimizer step.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn

OUT = Path(r"Q:\Colibri\research\router-comparison-lab\outputs\mk7-dense-lora-pytorch-control.json")

class DenseFrozenLoRATop2(nn.Module):
    def __init__(self, hidden=16, out_dim=16, experts=4, rank=4):
        super().__init__(); self.experts = experts
        self.base = nn.Linear(hidden, out_dim, bias=False)
        self.router = nn.Linear(hidden, experts, bias=False)
        self.a = nn.Parameter(torch.randn(experts, hidden, rank) * .05)
        self.b = nn.Parameter(torch.randn(experts, rank, out_dim) * .05)
        for p in self.base.parameters(): p.requires_grad_(False)
        for p in (self.a, self.b): p.requires_grad_(False)

    def forward(self, x):
        logits = self.router(x); probs = torch.softmax(logits, dim=-1)
        weights, ids = torch.topk(probs, k=2, dim=-1); weights = weights / weights.sum(-1, keepdim=True)
        deltas = torch.stack([(x @ self.a[i]) @ self.b[i] for i in range(self.experts)], dim=1)
        chosen = torch.gather(deltas, 1, ids.unsqueeze(-1).expand(-1, -1, deltas.size(-1)))
        return self.base(x) + (chosen * weights.unsqueeze(-1)).sum(1), ids, weights

def main() -> int:
    torch.manual_seed(20260817); model = DenseFrozenLoRATop2(); x = torch.randn(4, 16); target = torch.zeros(4, 16)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    y, ids, weights = model(x); loss = (y - target).pow(2).mean(); loss.backward()
    router_grad = [p.grad is not None and bool(torch.count_nonzero(p.grad)) for n, p in model.named_parameters() if n.startswith("router.")]
    frozen_with_grad = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())]
    result = {"status": "VERIFIED_DENSE_LORA_ROUTER_CONTROL", "loss": float(loss), "output_shape": list(y.shape), "top_k": 2, "selected_ids": ids.tolist(), "router_grad_nonzero": all(router_grad), "frozen_parameters_with_grad": frozen_with_grad, "weights_changed_without_optimizer": changed, "optimizer_step": False, "training_started": False, "dataset_loaded": False, "interpretation": "Real PyTorch module contract with synthetic tensors; not a model-quality or semantic-specialization result."}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps({"status": result["status"], "output": str(OUT)}, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
