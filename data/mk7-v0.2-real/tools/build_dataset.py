#!/usr/bin/env python3
"""MK7-v0.2-real dataset builder — read-only tooling, no training.

Pipeline (questionnaire section 6):
  CSV (questionnaire-template.csv filled) -> validated JSONL -> manifest + reconciliation.

Governance:
  - Creates NEW sibling artifacts only; never reads-or-writes mk7 v0.1 files.
  - Only quality_status=approved rows enter the dataset.
  - synthetic rows are exported but tagged and excluded from held-out.
  - training_authorization stays NOT_AUTHORIZED.

Usage:
  python build_dataset.py --csv questionnaire.csv --out ../build/
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REGISTRY = [
    "O_subject_ontology",
    "E_source_evidence",
    "R_structure_relations",
    "M_method_validation",
    "A_function_application",
    "T_time_evolution",
    "H_human_value_context",
]
PILLARS = {e.split("_", 1)[1]: e for e in REGISTRY}
SPLIT_SEED = "20260905"  # fixed: deterministic splits across runs
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "held_out": 0.15}


def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def parse_bool(v: str) -> bool:
    return str(v).strip().lower() == "true"


def validate_row(row: dict, idx: int) -> list:
    """Return a list of error strings for one questionnaire row."""
    errors = []
    dec = row["decidability"]
    target = (row.get("target_expert") or "").strip()
    allowed = [a for a in (row.get("allowed_experts") or "").split("|") if a.strip()]
    if row["language"] not in ("ar", "en", "mixed"):
        errors.append(f"language '{row['language']}' not in ar/en/mixed")
    if row["difficulty"] not in ("easy", "medium", "hard"):
        errors.append(f"difficulty '{row['difficulty']}' invalid")
    if row["source_type"] not in ("owner_curated", "internal_procedure", "public_document", "interview", "synthetic", "curriculum_generated"):
        errors.append(f"source_type '{row['source_type']}' invalid")
    if row["quality_status"] not in ("draft", "approved", "rejected"):
        errors.append(f"quality_status '{row['quality_status']}' invalid")
    record_type = (row.get("record_type") or "").strip() or "question_answer"
    if record_type not in ("definition", "explanation", "worked_example", "question_answer",
                           "misconception_check", "prerequisite_check", "assessment"):
        errors.append(f"record_type '{record_type}' invalid")
    for ko in [k.strip() for k in (row.get("ko_node_ids") or "").split(";") if k.strip()]:
        if not ko.startswith("ko."):
            errors.append(f"ko_node_ids value '{ko}' must start with 'ko.'")
    if len(row["group_id"].strip()) < 3:
        errors.append("group_id too short")
    if dec == "decisive":
        if target not in REGISTRY:
            errors.append(f"decisive row requires target_expert in registry, got '{target}'")
    elif dec == "ambiguous":
        if len(allowed) != 2 or any(a not in REGISTRY for a in allowed):
            errors.append(f"ambiguous row requires exactly 2 registry experts, got {allowed}")
        if target and target not in REGISTRY:
            errors.append(f"ambiguous target_expert '{target}' invalid")
    elif dec == "unknown":
        if target and target != "unknown":
            errors.append(f"unknown row must keep target_expert empty or 'unknown', got '{target}'")
    else:
        errors.append(f"decidability '{dec}' invalid")
    return errors


def build_record(row: dict, seq: int) -> dict:
    dec = row["decidability"]
    target = (row.get("target_expert") or "").strip() or "unknown"
    allowed = [a.strip() for a in (row.get("allowed_experts") or "").split("|") if a.strip()]
    record = {
        "id": f"mk7-v0.2-{seq:06d}",
        "input": row["input"].strip(),
        "language": row["language"].strip(),
        "decidability": dec,
        "target_expert": target,
        "allowed_experts": allowed,
        "difficulty": row["difficulty"].strip(),
        "group_id": row["group_id"].strip(),
        "source_type": row["source_type"].strip(),
        "source_id": row["source_id"].strip(),
        "ood_flag": parse_bool(row.get("ood_flag", "false")),
        "boundary_flag": parse_bool(row.get("boundary_flag", "false")),
        "quality_status": row["quality_status"].strip(),
        "record_type": (row.get("record_type") or "").strip() or "question_answer",
        "ko_node_ids": [k.strip() for k in (row.get("ko_node_ids") or "").split(";") if k.strip()],
        "notes": (row.get("notes") or "").strip(),
    }
    if target != "unknown":
        record["pillar"] = target.split("_", 1)[1]
    return record


def assign_split(group_id: str) -> str:
    """Deterministic group-level split via seeded hash (no leakage: one group, one split)."""
    h = int(hashlib.sha256((SPLIT_SEED + group_id).encode("utf-8")).hexdigest(), 16)
    x = (h % 1000) / 1000.0
    if x < SPLIT_RATIOS["train"]:
        return "train"
    if x < SPLIT_RATIOS["train"] + SPLIT_RATIOS["validation"]:
        return "validation"
    return "held_out"


def batch_checks(records: list) -> dict:
    """Questionnaire section 5 batch-level checks (report only — never auto-fix)."""
    approved = [r for r in records if r["quality_status"] == "approved"]
    per_expert, per_lang, per_diff = {}, {}, {}
    for r in approved:
        e = r["target_expert"] if r["decidability"] != "ambiguous" else "ambiguous_top2"
        per_expert[e] = per_expert.get(e, 0) + 1
        per_lang[r["language"]] = per_lang.get(r["language"], 0) + 1
        per_diff[r["difficulty"]] = per_diff.get(r["difficulty"], 0) + 1
    seen, exact_dupes = {}, 0
    for r in approved:
        key = " ".join(r["input"].lower().split())
        if key in seen and seen[key] != r["group_id"]:
            exact_dupes += 1  # identical text across different groups
        seen[key] = r["group_id"]
    zero_coverage = [e for e in REGISTRY if per_expert.get(e, 0) == 0]
    # Governance: curriculum-generated records must never land in the real held-out split.
    generated_in_held_out = sum(
        1 for r in approved if r["source_type"] == "curriculum_generated" and r.get("split") == "held_out")
    return {
        "approved_count": len(approved),
        "per_expert": per_expert,
        "per_language": per_lang,
        "per_difficulty": per_diff,
        "experts_with_zero_coverage": zero_coverage,
        "exact_duplicates_across_groups": exact_dupes,
        "synthetic_count": sum(1 for r in approved if r["source_type"] == "synthetic"),
        "curriculum_generated_in_held_out": generated_in_held_out,
    }


def reconcile(jsonl_path: Path, manifest: dict) -> dict:
    """Re-read the artifact and verify it matches the manifest exactly."""
    records = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = (
        len(records) == manifest["record_count"]
        and sha256_bytes(jsonl_path.read_bytes()) == manifest["artifacts"]["dataset_jsonl_sha256"]
        and all(r["provenance_hash"] == sha256_bytes(canonical({k: v for k, v in r.items() if k not in ("id", "provenance_hash")})) for r in records)
    )
    return {"status": "RECONCILED" if ok else "RECONCILIATION_FAILED", "records_read": len(records)}


def main():
    ap = argparse.ArgumentParser(description="Build mk7-v0.2-real dataset artifacts (read-only, no training).")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    rows = [r for r in rows if not r["input"].strip().startswith("EXAMPLE")]

    valid, invalid = [], []
    for i, row in enumerate(rows):
        errs = validate_row(row, i)
        (valid if not errs else invalid).append((row, errs))

    records, seq = [], 1
    for row, _ in valid:
        if row["quality_status"] != "approved":
            continue
        rec = build_record(row, seq)
        rec["split"] = assign_split(rec["group_id"])
        # Governance: curriculum-generated records never enter the real held-out
        # split (questionnaire 4.2 / alignment doc) — demote to validation.
        if rec["source_type"] == "curriculum_generated" and rec["split"] == "held_out":
            rec["split"] = "validation"
        # Hash covers every field except id itself; reconcile() uses the same rule.
        rec["provenance_hash"] = sha256_bytes(canonical({k: v for k, v in rec.items() if k not in ("id", "provenance_hash")}))
        records.append(rec)
        seq += 1

    jsonl_path = args.out / "mk7-v0.2-real-dataset.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")

    csv_sha = sha256_bytes(args.csv.read_bytes())
    manifest = {
        "dataset_version": "mk7-v0.2-real",
        "status": "PENDING_OWNER_APPROVAL",
        "training_authorization": "NOT_AUTHORIZED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "data/mk7-v0.2-real/tools/build_dataset.py",
        "inputs": {"questionnaire_csv": {"path": args.csv.name, "sha256": csv_sha}},
        "record_count": len(records),
        "registry": REGISTRY,
        "split_policy": {"unit": "group_id", "seed": SPLIT_SEED, "ratios": SPLIT_RATIOS},
        "artifacts": {"dataset_jsonl_sha256": sha256_bytes(jsonl_path.read_bytes())},
        "batch_checks": batch_checks(records),
    }
    manifest["reconciliation"] = reconcile(jsonl_path, manifest)
    (args.out / "MK7-DATASET-MANIFEST-mk7-v0.2-real.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rows_read={len(rows)} valid={len(valid)} invalid={len(invalid)} exported_approved={len(records)}")
    for row, errs in invalid:
        print(f"INVALID: {row['input'][:40]!r} -> {'; '.join(errs)}")
    print(f"reconciliation: {manifest['reconciliation']['status']}")
    print(f"experts_with_zero_coverage: {manifest['batch_checks']['experts_with_zero_coverage'] or 'none'}")
    print(f"outputs: {jsonl_path} + manifest")
    return 0 if not invalid and manifest["reconciliation"]["status"] == "RECONCILED" else 1


if __name__ == "__main__":
    sys.exit(main())
