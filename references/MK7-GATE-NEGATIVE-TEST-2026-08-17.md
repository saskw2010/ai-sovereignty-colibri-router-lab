# MK7 Gate Negative Test — 2026-08-17

## Status

**VERIFIED — unsafe start request refused**

The hardened gate was invoked with `--start-real-training` and the current Qwen candidate packet. It returned `READY_FOR_OWNER_REVIEW` because:

- `owner_approval=false`;
- `registry_selected=false` because the registry is still a candidate;
- no training started;
- no optimizer step occurred;
- the Dataset was not modified.

This is a safety regression test: the explicit start flag alone cannot bypass owner approval and registry selection.
