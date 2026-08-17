# Qwen Frozen LoRA Compatibility — 2026-08-17

## Status

**VERIFIED — compatibility smoke only; no training**

## Test

- Base: `Qwen/Qwen2.5-0.5B` from `F:\AI-OPEN-MODELS\qwen2.5-0.5b`
- LoRA library: PEFT in `venv-py311-clean`
- Target modules: `q_proj`, `v_proj`
- Rank: `2`
- Alpha: `4`
- Device: NVIDIA Quadro P2000 CUDA
- Dtype: FP32
- Trainable parameters: `0` (all Base and LoRA tensors explicitly frozen)
- Output logits: `[1, 5, 151936]`
- Base target tensor max absolute difference before/after forward: `0.0`

## Safety evidence

- Optimizer step: false
- Dataset loaded: false
- MK7 real data loaded: false
- Gemma loaded: false
- Golden Training loaded: false

## Interpretation

Qwen2.5-0.5B is compatible with the planned Dense + Frozen LoRA runtime shape for a controlled router experiment. This does not prove adapter quality, specialization, or real MK7 training readiness; those remain separate gates.
