# MK7 Synthetic Router Training — 2026-08-17

## Scope

This is the first actual training step in the new Router lane, using only synthetic centroid-domain tensors. It is not training on Gemma, Golden Training, MK7 Dataset, or any external model.

## Result

- Status: `VERIFIED_SYNTHETIC_ROUTER_TRAINING`.
- 512 synthetic samples, 4 experts, hidden size 16, 120 AdamW steps on CPU FP32.
- Initial loss: `1.9640766`.
- Final loss: `0.0000316`.
- Synthetic routing accuracy: `100%`.
- Base parameters frozen: `true`.
- LoRA expert parameters frozen: `true`.
- Router weights changed: `true`.
- Gemma/MK7 Dataset/Golden Training/external weights loaded: `false`.

## Artifacts

- Router weights: `F:\AI-OPEN-MODELS\mk7-synthetic-router-v0.1\router.pt`
- Result manifest: `F:\AI-OPEN-MODELS\mk7-synthetic-router-v0.1\results.json`
- Source: `Q:\Colibri\research\router-comparison-lab\train_synthetic_mk7_router.py`

## Interpretation

The Router can learn separable routing while the Base and Experts remain frozen. This validates the training mechanics and gate only. The result is **not** evidence of MK7 quality, real-domain specialization, or readiness to train on project data.

## Next gate

Before using any real Base, Adapter, or Dataset, require an explicit training scope, approved data manifest, frozen-base check, held-out evaluation contract, and checkpoint destination.
