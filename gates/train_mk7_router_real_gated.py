"""Guarded entry point for future real MK7 Router training.

This file intentionally refuses to train unless an owner-approved scope file
and explicit --start-real-training flag are present. It is a gate, not a
training run, and currently performs only preflight validation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

DATASET = Path(r"Q:\Colibri\training\datasets\mk7\v0.1")
PREFLIGHT = Path(r"F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-preflight\preflight-manifest.json")
REGISTRY_DIR = PREFLIGHT.parent

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--start-real-training", action="store_true"); p.add_argument("--approval-file", type=Path); args = p.parse_args()
    manifest = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    checks = {
        "dataset_exists": DATASET.exists(),
        "preflight_verified": manifest.get("status") == "PREFLIGHT_VERIFIED_TRAINING_NOT_STARTED",
        "owner_approval": False,
        "base_selected": False,
        "registry_selected": False,
        "explicit_start_flag": args.start_real_training,
    }
    if args.approval_file and args.approval_file.exists():
        approval = json.loads(args.approval_file.read_text(encoding="utf-8"))
        required = ["owner", "approved_at", "base_model", "base_revision_or_hash", "registry_version", "canonical_dataset_path", "dataset_manifest_hash", "allowed_training_scope", "proposed_max_steps", "checkpoint_destination", "evaluation_contract"]
        fields_complete = all(bool(approval.get(key)) for key in required)
        registry_path = REGISTRY_DIR / "registry-v0.3-qwen-candidate.json"
        registry_valid = registry_path.exists() and json.loads(registry_path.read_text(encoding="utf-8")).get("status") == "OWNER_SELECTED_PENDING_TRAINING"
        eval_valid = Path(approval.get("evaluation_contract", "")).exists()
        checks["owner_approval"] = approval.get("approved") is True and fields_complete
        checks["base_selected"] = approval.get("base_model") == "Qwen/Qwen2.5-0.5B" and bool(approval.get("base_revision_or_hash"))
        checks["registry_selected"] = registry_valid and approval.get("registry_version") == "registry-v0.3-qwen-candidate"
        checks["evaluation_contract_valid"] = eval_valid
        checks["canonical_dataset_path_valid"] = approval.get("canonical_dataset_path", "").endswith(r"mk7\v0.1\combined")
        checks["scope_router_only"] = approval.get("allowed_training_scope") == "router_only"
        checks["step_limit_positive"] = int(approval.get("proposed_max_steps", 0)) > 0
    result = {"status": "READY_FOR_OWNER_REVIEW" if not all(checks.values()) else "AUTHORIZED_GATE_PASSED", "checks": checks, "training_started": False, "dataset_modified": False, "optimizer_step": False, "reason": "No real training is started by this gate script."}
    out = PREFLIGHT.parent / "real-training-gate-result.json"; out.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
