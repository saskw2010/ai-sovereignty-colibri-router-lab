# MK7 Registry Candidate v0.3 — 2026-08-17

## Status

**PARTIAL — candidate metadata prepared; registry not selected**

The candidate registry describes seven MK7 pillar adapters for the verified Qwen base:

- Base: `Qwen/Qwen2.5-0.5B` at revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- LoRA targets: `q_proj`, `v_proj`.
- Rank/alpha: `2 / 4`.
- Router mode set: Mixer, Specialist, Plug-and-Play.
- Top-k: `2`.

Every adapter path is intentionally `TO_BE_APPROVED`; no real adapter weights are claimed by this file. The registry is therefore safe metadata for owner review, not a training authorization.

## Missing before selection

- Approved adapter provenance and dataset manifest per expert.
- Exact adapter output/checkpoint destinations on `F:`.
- Owner selection of this registry version.
- Explicit training approval.
