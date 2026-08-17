"""No-update diagnostic for QMoE frozen experts and trainable routers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer


def main() -> int:
    root = Path(__file__).resolve().parents[0].parent
    model_dir = root / "safetensors"
    sys.path.insert(0, str(root))
    from source.configuration_qmoe import QMoEConfig
    from source.modeling_qmoe import QMoEForCausalLM

    config = QMoEConfig.from_pretrained(model_dir)
    model = QMoEForCausalLM(config).float().cpu()
    model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device="cpu"), strict=True)
    model.train()

    for parameter in model.parameters():
        parameter.requires_grad = False
    router_parameters = []
    for block in model.blocks:
        for parameter in block.moe.router.gate.parameters():
            parameter.requires_grad = True
            router_parameters.append(parameter)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoded = tokenizer("router gradient diagnostic", return_tensors="pt")
    input_ids = encoded["input_ids"]
    outputs = model(input_ids=input_ids, labels=input_ids)
    outputs.loss.backward()

    router_nonzero = sum(1 for parameter in router_parameters if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0)
    router_total = len(router_parameters)
    expert_parameters = [parameter for block in model.blocks for parameter in block.moe.experts.parameters()]
    expert_with_grad = sum(1 for parameter in expert_parameters if parameter.grad is not None)
    all_frozen_except_router = all((not parameter.requires_grad) for name, parameter in model.named_parameters() if ".moe.router.gate." not in name)

    result = {
        "status": "VERIFIED_ROUTER_ONLY_GRADIENT_PATH",
        "loss": float(outputs.loss.detach()),
        "router_parameter_tensors": router_total,
        "router_parameter_tensors_with_nonzero_grad": router_nonzero,
        "expert_parameter_tensors_with_grad": expert_with_grad,
        "all_non_router_parameters_frozen": all_frozen_except_router,
        "backward": True,
        "optimizer_step": False,
        "weights_updated": False,
        "training_started": False,
    }
    path = root / "smoke" / "qmoe_router_freeze_diagnostic.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(path), "router_nonzero": f"{router_nonzero}/{router_total}", "expert_with_grad": expert_with_grad}, ensure_ascii=False))
    return 0 if router_nonzero == router_total and expert_with_grad == 0 and all_frozen_except_router else 1


if __name__ == "__main__":
    raise SystemExit(main())
