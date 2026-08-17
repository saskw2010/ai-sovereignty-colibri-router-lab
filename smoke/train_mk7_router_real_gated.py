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
        approval = json.loads(args.approval_file.read_text(encoding="utf-8")); checks["owner_approval"] = approval.get("approved") is True; checks["base_selected"] = bool(approval.get("base_model")); checks["registry_selected"] = bool(approval.get("registry_version"))
    result = {"status": "READY_FOR_OWNER_REVIEW" if not all(checks.values()) else "AUTHORIZED_GATE_PASSED", "checks": checks, "training_started": False, "dataset_modified": False, "optimizer_step": False, "reason": "No real training is started by this gate script."}
    out = PREFLIGHT.parent / "real-training-gate-result.json"; out.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
