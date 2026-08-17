"""QMoE-400 FP32 inference and router telemetry smoke.

This script intentionally performs no backward pass, optimizer step, or weight update.
It imports the pinned local source package and loads the verified Safetensors file directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[0].parent)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-input-tokens", type=int, default=64)
    args = parser.parse_args()

    root = args.root.resolve()
    model_dir = root / "safetensors"
    source_dir = root / "source"
    output_dir = root / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    sys.path.insert(0, str(root))
    from source.configuration_qmoe import QMoEConfig
    from source.modeling_qmoe import QMoEForCausalLM

    device = torch.device(args.device)
    config = QMoEConfig.from_pretrained(model_dir)
    model = QMoEForCausalLM(config).to(device=device, dtype=torch.float32)
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    del state
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompt = "Router smoke test. Explain why load balancing matters in a mixture of experts model."
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    router_outputs: dict[str, torch.Tensor] = {}
    hooks = []
    for layer_idx, block in enumerate(model.blocks):
        gate = block.moe.router.gate
        hooks.append(gate.register_forward_hook(lambda _m, _i, out, idx=layer_idx: router_outputs.__setitem__(str(idx), out.detach().cpu())))

    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    elapsed = time.perf_counter() - started
    for hook in hooks:
        hook.remove()

    telemetry = []
    for layer_idx in range(len(model.blocks)):
        logits = router_outputs[str(layer_idx)]
        probs = torch.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(probs, k=config.moe_top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        counts = torch.bincount(top_indices.reshape(-1), minlength=config.num_experts)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        telemetry.append(
            {
                "layer": layer_idx,
                "tokens": int(logits.shape[1]),
                "num_experts": config.num_experts,
                "top_k": config.moe_top_k,
                "top_indices": top_indices.tolist(),
                "top_weights": top_values.tolist(),
                "mean_entropy": float(entropy.mean()),
                "expert_counts_topk": counts.tolist(),
                "max_expert_share": float(counts.max() / counts.sum()),
            }
        )

    result = {
        "status": "VERIFIED_FORWARD_SMOKE",
        "device": str(device),
        "dtype": "float32",
        "prompt": prompt,
        "input_tokens": int(input_ids.shape[1]),
        "elapsed_seconds": elapsed,
        "logits_shape": list(outputs.logits.shape),
        "config": {
            "num_layers": config.num_layers,
            "num_experts": config.num_experts,
            "top_k": config.moe_top_k,
            "d_model": config.d_model,
            "vocab_size": config.vocab_size,
        },
        "telemetry": telemetry,
        "training": {"backward": False, "optimizer_step": False, "weights_updated": False},
    }
    output_path = output_dir / "qmoe_fp32_smoke_result.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output_path), "elapsed_seconds": elapsed, "logits_shape": result["logits_shape"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
