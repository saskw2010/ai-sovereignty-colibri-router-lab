"""Read-only validation of the proposed MK7 Qwen registry."""
import json
from pathlib import Path

p = Path(r"F:\\AI-OPEN-MODELS\\mk7-versions\\router-v0.3.0-preflight\\registry-v0.3-qwen-candidate.json")
registry = json.loads(p.read_text(encoding="utf-8"))
expected = ["O_subject_ontology", "E_source_evidence", "R_structure_relations", "M_method_validation", "A_function_application", "T_time_evolution", "H_human_value_context"]
checks = {
    "candidate_not_selected": registry["status"] == "CANDIDATE_NOT_SELECTED",
    "base_is_qwen": registry["base"]["id"] == "Qwen/Qwen2.5-0.5B",
    "revision_pinned": registry["base"]["revision"] == "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "targets_pinned": registry["base"]["adapter_targets"] == ["q_proj", "v_proj"],
    "seven_experts": [x["id"] for x in registry["experts"]] == expected,
    "paths_not_falsely_claimed": all(x["adapter_path"] == "TO_BE_APPROVED" for x in registry["experts"]),
    "top_k_two": registry["router"]["top_k"] == 2,
    "training_not_authorized": registry["real_training_authorized"] is False,
}
result = {"status": "VERIFIED_REGISTRY_CANDIDATE_READ_ONLY" if all(checks.values()) else "FAILED_REGISTRY_VALIDATION", "checks": checks, "training_started": False}
print(json.dumps(result, ensure_ascii=False))
