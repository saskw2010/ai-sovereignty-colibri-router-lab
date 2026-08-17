"""Read-only audit of the complete Qwen owner-review packet."""
import json
from pathlib import Path

p = Path(r"F:\\AI-OPEN-MODELS\\mk7-versions\\router-v0.3.0-preflight\\owner-review-packet-qwen-candidate.json")
packet = json.loads(p.read_text(encoding="utf-8"))
required = ["base_model", "base_revision_or_hash", "registry_version", "dataset_id", "canonical_dataset_path", "dataset_manifest_hash", "allowed_training_scope", "proposed_max_steps", "checkpoint_destination", "evaluation_contract"]
checks = {key: bool(packet.get(key)) for key in required}
checks["approval_is_explicitly_closed"] = packet["approved"] is False
checks["safety_training_not_started"] = packet["safety"]["training_started"] is False
checks["safety_dataset_unchanged"] = packet["safety"]["dataset_modified"] is False
result = {"status": "READY_FOR_OWNER_REVIEW" if all(checks.values()) else "INCOMPLETE_OWNER_PACKET", "checks": checks, "training_started": False, "reason": "The packet is technically complete but not an authorization while approved=false."}
print(json.dumps(result, ensure_ascii=False))
