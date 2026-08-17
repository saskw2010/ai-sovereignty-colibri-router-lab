# MK7 Real Training Approval Gate

## Current state

`READY_FOR_OWNER_REVIEW` — no real training has started.

## Required fields before `approved=true`

- Owner identity and approval timestamp.
- Exact Base model and revision/hash.
- Registry version.
- Dataset ID/path and manifest hash.
- Explicit scope: `router_only` or another approved scope.
- Maximum steps and checkpoint destination on F.
- Evaluation contract and held-out split policy.

Template:

`F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-preflight\real-training-approval-template.json`

The gated runner will not infer missing values and will not train while any field is absent.
