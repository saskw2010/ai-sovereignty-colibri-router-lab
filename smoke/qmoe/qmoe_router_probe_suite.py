"""Read-only multi-probe router telemetry for QMoE-400.

No backward pass, optimizer, or weight update is performed. The probes are
measurement prompts only; they cannot establish semantic expert ownership.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

PROBES = {
    "english_general": "Explain load balancing in a mixture of experts language model.",
    "arabic_general": "اشرح أهمية توزيع الرموز على الخبراء في نموذج لغوي متعدد الخبراء.",
    "code": "Write a Python function that validates a JSON payload and returns clear errors.",
    "erp_like": "Create an ERP inventory report showing stock levels, reorder points, and suppliers.",
    "contract_like": "Summarize the obligations, renewal clause, and risks in a commercial contract.",
}

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[0].parent)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--max-input-tokens", type=int, default=48)
    args = p.parse_args()
    root = args.root.resolve(); model_dir = root / "safetensors"; sys.path.insert(0, str(root))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    from source.configuration_qmoe import QMoEConfig
    from source.modeling_qmoe import QMoEForCausalLM
    config = QMoEConfig.from_pretrained(model_dir)
    device = torch.device(args.device)
    model = QMoEForCausalLM(config).to(device=device, dtype=torch.float32)
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected: raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    del state; model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    all_results = {}; started = time.perf_counter()
    for name, prompt in PROBES.items():
        captured = {}; hooks = []
        for i, block in enumerate(model.blocks):
            hooks.append(block.moe.router.gate.register_forward_hook(lambda _m, _i, out, idx=i: captured.__setitem__(idx, out.detach().cpu())))
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
        ids = encoded["input_ids"].to(device); mask = encoded.get("attention_mask")
        if mask is not None: mask = mask.to(device)
        with torch.inference_mode(): model(input_ids=ids, attention_mask=mask)
        for h in hooks: h.remove()
        layers = []
        for i in range(len(model.blocks)):
            logits = captured[i]; probs = torch.softmax(logits, dim=-1)
            _, top = torch.topk(probs, k=config.moe_top_k, dim=-1)
            counts = torch.bincount(top.reshape(-1), minlength=config.num_experts)
            entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
            layers.append({"layer": i, "mean_entropy": float(entropy.mean()), "max_expert_share": float(counts.max()/counts.sum()), "expert_counts_topk": counts.tolist()})
        all_results[name] = {"input_tokens": int(ids.shape[1]), "layers": layers}
    result = {"status": "VERIFIED_MULTI_PROBE_TELEMETRY", "device": str(device), "dtype": "float32", "probe_count": len(PROBES), "probes": all_results, "elapsed_seconds": time.perf_counter()-started, "training": {"backward": False, "optimizer_step": False, "weights_updated": False}}
    out = root / "smoke" / "qmoe_router_probe_suite_result.json"; out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(out), "probe_count": len(PROBES), "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False))
    return 0
if __name__ == "__main__": raise SystemExit(main())
