# MK7 Real Router-Only Qwen v0.3 — 2026-08-18

## Status

**VERIFIED — first approved limited real run**

## Scope

This run used the explicitly approved `router_only` scope. The Qwen Dense Base and Experts were frozen; only the input bridge and Router were optimized. The original Dataset was read-only and no source files were modified.

- Base: `Qwen/Qwen2.5-0.5B`
- Revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- Device: NVIDIA Quadro P2000 CUDA
- Precision: FP32
- Steps: `120`
- Labeled seven-pillar subset: `114 train / 11 validation / 15 test`
- Checkpoint: `F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-real\router-only-qwen-mk7-v0.3.pt`
- Checkpoint SHA256: `F169D35049A8D5EE1B9BFBA57465EAE9D54975AE9F2ACBA20A7C78510F080619`
- Checkpoint size: `118,034` bytes

## Results

| Split | Top-1 | Top-2 |
|---|---:|---:|
| Train | 100% | 100% |
| Validation | 63.64% | 81.82% |
| Test | 93.33% | 100% |

Loss decreased from `5.543947` to `0.0000123`.

## Safety evidence

- Base frozen: true.
- Experts trained: false.
- Dataset modified: false.
- Gemma loaded: false.
- Golden Training loaded: false.
- Optimizer scope: bridge and Router only.
- Gate: `AUTHORIZED_GATE_PASSED`.

## Interpretation

This is the first approved real-data Router-only experiment, not full MK7 model training. The test set is small (`15` examples), so the score is evidence of a functioning limited experiment, not a general quality claim. Further runs require a new sibling checkpoint and explicit scope review.
