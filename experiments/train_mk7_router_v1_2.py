#!/usr/bin/env python3
"""MK7 Router v1.2 — real router-only training on the approved mk7-v0.2-real
Mathematics batch (343 records), following the v1.0 domain-aug methodology.

Protocol (mirrors experiments/train_mk7_router_real_qwen_v1_0_domain_aug.py):
  - Base model frozen (requires_grad=False everywhere), router head only trains.
  - Deterministic seed; best checkpoint selected on validation; loss curve kept.
  - The UNTRAINED router's validation accuracy is recorded as the frozen baseline
    inside the same run, so trained-vs-baseline is one artifact.

Governance fields are written into result.json: base_frozen, experts_trained,
dataset_modified, and the owner authorization note for this pilot run.

Usage:
  python train_mk7_router_v1_2.py --dataset <mk7-v0.2-real-dataset.jsonl> \
      --out <dir> [--base Qwen/Qwen2.5-0.5B] [--steps 500] [--seed 20260905]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REGISTRY = [
    "O_subject_ontology", "E_source_evidence", "R_structure_relations",
    "M_method_validation", "A_function_application", "T_time_evolution",
    "H_human_value_context",
]
LABELS = [e.split("_", 1)[1] for e in REGISTRY]
LABEL_TO_IDX = {e: i for i, e in enumerate(REGISTRY)}

# v1.0 challenge set kept verbatim for cross-version comparability.
CHALLENGE = [
    ("كيف نعرّف نطاق الكيان الذي تدور حوله هذه المعرفة؟", 0),
    ("ما المفاهيم التي يجب وضعها في قاموس الموضوع؟", 0),
    ("ما المرجع الذي يمكن الرجوع إليه للتحقق من النتيجة؟", 1),
    ("كيف نميز الدليل القوي من الادعاء غير الموثق؟", 1),
    ("كيف تمثل الروابط بين الوحدات داخل النظام؟", 2),
    ("ما الذي يفسر اعتماد هذا الجزء على ذاك؟", 2),
    ("ما الاختبار المناسب للتأكد من سلامة الإجراءات؟", 3),
    ("أي فرضية قد تجعل الاستنتاج غير موثوق؟", 3),
    ("كيف يستفيد فريق العمل من هذه المعرفة عمليًا؟", 4),
    ("ما الخطوة التي تحول الفكرة إلى خدمة قابلة للتنفيذ؟", 4),
    ("ما التغيرات التي مرت بها هذه القاعدة؟", 5),
    ("كيف نعرف أن الإصدار الحالي أحدث من السابق؟", 5),
    ("كيف يتأثر المستخدمون بهذا الاختيار؟", 6),
    ("ما الاعتبارات الأخلاقية عند تطبيق الحل؟", 6),
]


def _confine(path: Path, base: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise SystemExit(f"Path escapes the working directory: {resolved}") from exc
    return resolved


def load_split(path: Path) -> dict:
    split = {"train": [], "validation": []}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["decidability"] == "decisive" and r["target_expert"] in LABEL_TO_IDX \
                    and r.get("split") in split:
                split[r["split"]].append((r["input"], LABEL_TO_IDX[r["target_expert"]]))
    return split


def main():
    ap = argparse.ArgumentParser(description="Router-only training on mk7-v0.2-real (frozen base).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260905)
    args = ap.parse_args()

    base_dir = Path.cwd().resolve()
    dataset_path = _confine(Path(args.dataset), base_dir)
    out_dir = _confine(Path(args.out), base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_split(dataset_path)
    if not data["train"] or not data["validation"]:
        raise SystemExit("dataset must contain decisive train and validation records")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False  # frozen base — invariant of the contract

    def features(rows):
        hidden, labels = [], []
        with torch.no_grad():
            for start in range(0, len(rows), 16):
                part = rows[start:start + 16]
                batch = tok([x[0] for x in part], padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
                out = model(**batch, output_hidden_states=True, use_cache=False)
                idx = batch.attention_mask.sum(1) - 1
                hidden.append(out.hidden_states[-1][torch.arange(len(part), device=device), idx, :].float().cpu())
                labels.extend(x[1] for x in part)
        return torch.cat(hidden), torch.tensor(labels)

    x, y = features(data["train"])
    xv, yv = features(data["validation"])
    xc, yc = features([(t, i) for t, i in CHALLENGE])

    torch.manual_seed(args.seed)
    router = torch.nn.Sequential(
        torch.nn.Linear(x.shape[-1], 128), torch.nn.ReLU(),
        torch.nn.Dropout(0.1), torch.nn.Linear(128, len(REGISTRY)))

    def acc(scores, targets):
        return float((scores.argmax(-1) == targets).float().mean())

    with torch.no_grad():
        router.eval()
        untrained_val_acc = acc(router(xv), yv)  # frozen baseline inside the same run

    opt = torch.optim.AdamW(router.parameters(), lr=0.005, weight_decay=0.01)
    best_acc, best_state, best_step, losses = -1.0, None, 0, []
    for step in range(1, args.steps + 1):
        router.train()
        loss = torch.nn.functional.cross_entropy(router(x), y, label_smoothing=0.05)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        router.eval()
        val_acc = acc(router(xv), yv)
        if val_acc > best_acc:
            best_acc, best_step = val_acc, step
            best_state = {k: v.detach().clone() for k, v in router.state_dict().items()}
    router.load_state_dict(best_state)

    def metrics(scores, targets):
        top2 = scores.topk(2, dim=-1).indices
        return {"top1_accuracy": acc(scores, targets),
                "top2_hit_rate": float((top2 == targets[:, None]).any(1).float().mean())}

    with torch.no_grad():
        router.eval()
        result = {
            "status": "VERIFIED_MK7_ROUTER_V1_2_REAL_DATASET",
            "base": args.base,
            "scope": "router_only",
            "seed": args.seed,
            "steps": args.steps,
            "best_step": best_step,
            "dataset_sha256": "sha256:" + hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "counts": {k: len(v) for k, v in data.items()},
            "challenge_count": len(CHALLENGE),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "untrained_router_validation_top1": untrained_val_acc,
            "train": metrics(router(x), y),
            "validation": metrics(router(xv), yv),
            "challenge": metrics(router(xc), yc),
            "base_frozen": True,
            "experts_trained": False,
            "dataset_modified": False,
            "held_out_used": False,
            "owner_authorization": "router_only pilot directed by owner in session 2026-09-05",
            "checkpoint": str(out_dir / "router-v1.2.0-real.pt"),
        }
    torch.save({"router": router.state_dict(), "input_dim": x.shape[-1],
                "labels": LABELS, "seed": args.seed},
               out_dir / "router-v1.2.0-real.pt")
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "untrained_router_validation_top1", "train", "validation", "challenge",
        "best_step", "initial_loss", "final_loss")}, ensure_ascii=False, indent=1))
    print(f"checkpoint: {result['checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
