"""Read-only validation of the MK7 v0.3 preflight contract."""
import hashlib
import json
from pathlib import Path

ROOT = Path(r"F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-preflight")
DATA = Path(r"Q:\Colibri\training\datasets\mk7\v0.1\combined")
manifest = json.loads((ROOT / "canonical-dataset-manifest.json").read_text(encoding="utf-8"))
contract = json.loads((ROOT / "evaluation-contract-v0.3.json").read_text(encoding="utf-8"))

checks = {}
for name, spec in manifest["files"].items():
    path = DATA / name
    data = path.read_bytes()
    checks[f"exists:{name}"] = path.exists()
    checks[f"sha256:{name}"] = hashlib.sha256(data).hexdigest() == spec["sha256"]
    records = sum(1 for line in data.splitlines() if line.strip())
    checks[f"records:{name}"] = records == spec["records"]

checks["manifest_status_is_unapproved"] = manifest["status"] == "CANONICAL_VIEW_PROPOSED_NOT_APPROVED"
checks["contract_is_draft"] = contract["status"] == "DRAFT_FOR_OWNER_REVIEW"
checks["training_not_started"] = manifest["training_started"] is False
checks["dataset_not_modified"] = manifest["dataset_modified"] is False
checks["router_only_scope"] = contract["scope"] == "router_only"
checks["test_unseen_required"] = contract["acceptance_rules"]["test_split_must_remain_unseen"] is True

result = {
    "status": "VERIFIED_V03_PREFLIGHT_READ_ONLY" if all(checks.values()) else "FAILED_PREFLIGHT",
    "checks": checks,
    "training_started": False,
    "dataset_modified": False,
}
out = ROOT / "v03-preflight-validation.json"
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
