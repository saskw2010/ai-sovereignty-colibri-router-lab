# MK7 Dataset Reconciliation — 2026-08-17

## Status

**VERIFIED — READ-ONLY RECONCILIATION**

The reconciliation scanned the existing JSONL files without modifying the dataset and without starting training.

## Evidence

| Check | Result |
|---|---:|
| JSONL files scanned | 18 |
| Records scanned | 2,000 |
| Unique IDs | 1,000 |
| Duplicate ID values | 1,000 |
| Manifest expected total | 1,000 |
| Manifest reconciles as currently scanned | **false** |
| Dataset modified | false |
| Training started | false |

The 2,000 scanned records consist of the five `batch-*` trees plus the `combined` tree. A second read-only comparison hashed every logical record and verified:

- 1,000 unique IDs in the batch view and 1,000 in the combined view;
- identical ID sets;
- identical canonical record content for every ID;
- zero batch-only IDs, combined-only IDs, or content mismatches.

Therefore, the extra 1,000 scanned rows are exact duplicates caused by enumerating both views. This is a source-selection issue, not evidence of 2,000 unique examples.

## Decision gate

Real MK7 router training remains **BLOCKED pending owner review**, because the canonical source must be explicitly selected and the approval gate must contain:

- canonical dataset path and manifest hash;
- exact base model and revision/hash;
- registry version;
- approved scope (`router_only` or another scope);
- maximum steps and checkpoint destination on `F:`;
- held-out evaluation contract;
- owner identity, timestamp, and explicit approval.

The gated runner is not permitted to infer these values and will not train while they are absent.

## Safe next action

Recommended canonical view: `combined`, because it is the already assembled single view and has been proven byte-equivalent at the logical-record level to the five batch trees. The owner must still record this choice and regenerate/sign the manifest. Until that decision is recorded, only read-only audits and synthetic/versioned smoke tests may continue.

## Related evidence

- `F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-preflight\dataset-reconciliation.json`
- `F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-preflight\real-training-gate-result.json`
- `F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-preflight\real-training-approval-template.json`
- `Q:\Colibri\research\router-comparison-lab\mk7_dataset_reconciliation_readonly.py`
