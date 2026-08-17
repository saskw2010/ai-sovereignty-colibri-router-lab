"""Read-only reconciliation of MK7 v0.1 JSONL files."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

SRC = Path(r"Q:\Colibri\training\datasets\mk7\v0.1")
OUT = Path(r"F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-preflight\dataset-reconciliation.json")

def main() -> int:
    files = []; ids = Counter(); splits = Counter(); batches = Counter(); rows = 0
    for path in sorted(SRC.rglob("*.jsonl")):
        count = 0; file_splits = Counter(); file_ids = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row = json.loads(line); count += 1; rows += 1
            rid = row.get("id", ""); ids[rid] += 1; split = row.get("split", ""); file_splits[split] += 1; splits[split] += 1; batches[row.get("batch", "")] += 1; file_ids.add(rid)
        files.append({"path": str(path), "records": count, "splits": dict(file_splits), "unique_ids": len(file_ids)})
    duplicate_ids = sum(1 for v in ids.values() if v > 1)
    result = {"status": "VERIFIED_RECONCILIATION_READ_ONLY", "jsonl_files": len(files), "records_scanned": rows, "split_counts": dict(splits), "batch_counts": dict(batches), "unique_ids": len(ids), "duplicate_id_values": duplicate_ids, "manifest_expected_total": 1000, "manifest_reconciles": rows == 1000, "dataset_modified": False, "training_started": False, "files": files}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"); print(json.dumps({k: result[k] for k in ["status", "jsonl_files", "records_scanned", "split_counts", "batch_counts", "unique_ids", "duplicate_id_values", "manifest_reconciles"]}, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
