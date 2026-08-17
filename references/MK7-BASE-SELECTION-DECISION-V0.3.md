# MK7 Base Selection Decision — v0.3

## Decision status

**PLANNED / BASE NOT SELECTED**

## Separation of responsibilities

| Candidate | Role | v0.3 decision |
|---|---|---|
| QMoE-400 | Native sparse-MoE router and telemetry laboratory | Keep as research reference; do not use as the Dense+LoRA MK7 base |
| Synthetic dense control | Contract and routing-path test | Already verified; not a product base |
| Approved dense causal LM | MK7 base for frozen LoRA experts | Required next selection; exact ID/revision/hash still missing |

## Why QMoE is not the MK7 base

QMoE has its own native expert tensors and routing architecture. The MK7 runtime contract under test is a different composition:

`Dense frozen base → frozen LoRA adapters → learned router`

Using QMoE weights as if they were a Dense LoRA-compatible base would conflate two experiments and invalidate the adapter compatibility claim.

## Required base-selection evidence

Before real training, the owner must record:

1. Exact model repository and revision/hash.
2. License and redistribution status.
3. Architecture compatibility with the selected LoRA target modules.
4. Local precision plan (FP32 only on the current P2000 path, unless a separate remote approval is recorded).
5. Expected weight size and destination on `F:`.
6. A no-training forward smoke using the exact base and one frozen adapter.

## Current safety state

- QMoE smoke: VERIFIED.
- Dense + Frozen LoRA control: VERIFIED.
- v0.1.0 and v0.2.0: frozen/tagged.
- Real MK7 training: NOT STARTED.
- Base selection: UNRESOLVED pending owner decision.
- No Gemma, Golden Training, or MK7 Dataset modification in this lane.
