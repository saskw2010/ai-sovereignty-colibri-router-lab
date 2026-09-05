#!/usr/bin/env python3
"""Frozen Router baseline evaluator — MK7 track, step 1 of the training order (roadmap 3.5).

READ-ONLY with respect to training: this tool never updates any weight.
It measures how well routing decisions can be recovered from the approved
mk7-v0.2-real Mathematics batch before any optimizer step exists.

Modes:
  keyword          Deterministic lexical floor over lens keywords. A pipeline
                   check only — NOT a capability claim (questions are templated).
  untrained_router Real frozen baseline: frozen base + frozen LoRA experts with
                   the initial (untrained) router head. Requires torch,
                   transformers and an adapters directory on the host.

Outputs a baseline report JSON (deterministic, hashed to the dataset file).

Usage:
  python frozen_router_baseline.py --dataset <mk7-v0.2-real-dataset.jsonl> \
      --mode keyword --out baseline-report.json
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REGISTRY = [
    "O_subject_ontology", "E_source_evidence", "R_structure_relations",
    "M_method_validation", "A_function_application", "T_time_evolution",
    "H_human_value_context",
]

KEYWORDS = {
    "O_subject_ontology": ["موضوع", "كيانات", "أساسية", "ontology", "تعريف"],
    "E_source_evidence": ["مراجع", "أدلة", "مصدر", "دليل", "evidence", "يستند"],
    "R_structure_relations": ["يرتبط", "علاقات", "بنية", "مجاور", "حدود", "relations", "يرتبط"],
    "M_method_validation": ["صحة", "مناهج", "براهين", "تحقق", "proof", "معايير"],
    "A_function_application": ["نطبق", "تطبيق", "مسألة", "خطوات", "application", "عملي"],
    "T_time_evolution": ["تطورت", "تاريخيًا", "زمن", "تطور", "time", "المحطات"],
    "H_human_value_context": ["أثر", "إنسان", "أصحاب المصلحة", "value", "الحياة", "الصناعة"],
}


def _confine(path: Path, base: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise SystemExit(f"Path escapes the working directory: {resolved}") from exc
    return resolved


def load_records(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def keyword_route(text: str):
    """Return (top1, top2, fallback_used) by lexical score. Deterministic."""
    scores = {}
    for expert, kws in KEYWORDS.items():
        scores[expert] = sum(text.count(k) for k in kws)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if ranked[0][1] == 0:
        return None, [], True
    return ranked[0][0], [e for e, s in ranked[:2] if s > 0], False


def evaluate(records: list, mode: str) -> dict:
    n = len(records)
    top1_hits = 0
    top2_hits = 0
    fallbacks = 0
    per_expert = {e: {"n": 0, "top1": 0} for e in REGISTRY}
    confusion = {gold: {pred: 0 for pred in REGISTRY + ["fallback"]} for gold in REGISTRY}
    for r in records:
        gold = r["target_expert"]
        if gold not in REGISTRY:  # ambiguous_top2 / unknown rows: scored separately
            continue
        per_expert[gold]["n"] += 1
        if mode == "keyword":
            pred, top2, fallback = keyword_route(r["input"])
        else:
            raise SystemExit(f"mode '{mode}' requires the host model runtime (see untrained_router docs)")
        if fallback:
            fallbacks += 1
            confusion[gold]["fallback"] += 1
            continue
        if pred == gold:
            top1_hits += 1
            per_expert[gold]["top1"] += 1
        if gold in top2:
            top2_hits += 1
        confusion[gold][pred] += 1
    scored = sum(v["n"] for v in per_expert.values())
    return {
        "mode": mode,
        "records_total": n,
        "records_scored": scored,
        "fallback_rate": round(fallbacks / scored, 4) if scored else None,
        "top1_accuracy": round(top1_hits / scored, 4) if scored else None,
        "top2_hit_rate": round(top2_hits / scored, 4) if scored else None,
        "per_expert_top1": {e: round(v["top1"] / v["n"], 4) if v["n"] else None for e, v in per_expert.items()},
        "confusion_matrix": confusion,
        "note": "lexical floor only; templated questions inflate it — not a capability claim",
    }


def main():
    ap = argparse.ArgumentParser(description="Frozen router baseline over mk7-v0.2-real (no training).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", choices=["keyword", "untrained_router"], default="keyword")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = Path.cwd().resolve()
    dataset_path = _confine(Path(args.dataset), base)
    out_path = _confine(Path(args.out), base)

    records = load_records(dataset_path)
    report = {
        "artifact": "frozen_router_baseline_report",
        "dataset_sha256": "sha256:" + hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "training_performed": False,
        "results": {},
    }
    for split in ("train", "validation"):
        subset = [r for r in records if r.get("split") == split]
        if subset:
            report["results"][split] = evaluate(subset, args.mode)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for split, res in report["results"].items():
        print(f"[{split}] scored={res['records_scored']} top1={res['top1_accuracy']} "
              f"top2={res['top2_hit_rate']} fallback={res['fallback_rate']}")
    print(f"output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
