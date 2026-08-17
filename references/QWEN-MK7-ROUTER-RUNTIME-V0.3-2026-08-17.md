# Qwen + MK7 Router Runtime v0.3 — 2026-08-17

## Status

**VERIFIED — learned-router selection runtime; no training**

## Runtime path

`Qwen Dense Base → frozen hidden-state bridge → router-v0.1.0 → Top-2 frozen LoRA adapters → mixed logits`

- Base: `Qwen/Qwen2.5-0.5B`
- Router checkpoint: `F:\AI-OPEN-MODELS\mk7-versions\router-v0.1.0\weights\router-seven-pillars.pt`
- Router input/output: `32 → 7`
- Top-k: `2`
- Device: NVIDIA Quadro P2000 CUDA
- Dtype: FP32
- Trainable parameters: `0`
- Logits shape: `[1, 11, 151936]`

## Observed selection

- Selected experts: `E_source_evidence`, `H_human_value_context`
- Weights: approximately `0.9333 / 0.0658`

The selection is a runtime observation from the frozen checkpoint and deterministic bridge. It is not evidence that those experts are semantically specialized; the bridge is a controlled compatibility layer and no real MK7 data was used.

## Safety evidence

- Training started: false
- Optimizer step: false
- Dataset loaded: false
- Base and adapters frozen: true
- Router frozen: true
- v0.1.0 checkpoint unchanged: true
