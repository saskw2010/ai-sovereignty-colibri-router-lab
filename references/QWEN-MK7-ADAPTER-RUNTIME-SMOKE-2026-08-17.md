# Qwen + MK7 Adapter Runtime Smoke — 2026-08-17

## Status

**VERIFIED — runtime integration only; no training**

## Verified path

`Qwen2.5-0.5B Dense Base → seven named frozen LoRA adapters → runtime mode selection`

- Device: NVIDIA Quadro P2000 CUDA
- Dtype: FP32
- LoRA targets: `q_proj`, `v_proj`
- LoRA rank: `2`
- Registry entries: all seven MK7 pillars (`O/E/R/M/A/T/H`)
- Trainable parameters after adapter registration and mode switching: `0`
- Optimizer step: false
- Dataset loaded: false
- Gemma/Golden Training loaded: false

## Modes

- `specialist`: selected `R_structure_relations`
- `mixer`: combined `R_structure_relations` and `M_method_validation` with weights `0.7 / 0.3`
- `plug_and_play`: requested `A_function_application`, loaded on demand
- Logits shape: `[1, 9, 151936]`
- Runtime for smoke passes: approximately `0.529 s`

## Interpretation

The actual Qwen runtime can register and execute seven separate frozen LoRA adapters and expose the three planned modes. This proves integration mechanics only. It does not prove expert specialization, adapter quality, or real MK7 training readiness.
