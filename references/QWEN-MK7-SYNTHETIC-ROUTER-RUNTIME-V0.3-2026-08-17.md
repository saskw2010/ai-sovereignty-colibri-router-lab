# Qwen MK7 Synthetic Router Runtime v0.3 — 2026-08-17

## Status

**VERIFIED — v0.3 synthetic checkpoint executed; no training**

## Evidence

- Checkpoint: `F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-qwen-synthetic\qwen-router-v0.3-synthetic.pt`
- Base: `Qwen/Qwen2.5-0.5B`
- Device: NVIDIA Quadro P2000 CUDA
- Dtype: FP32
- Top-k: `2`
- Selected adapters: `A_function_application`, `O_subject_ontology`
- Weights: `1.0 / 0.0`
- Logits shape: `[1, 9, 151936]`
- Trainable parameters at runtime: `0`

## Boundary

This is execution of the synthetic-trained v0.3 bridge/router checkpoint over seven frozen synthetic LoRA adapters. It proves checkpoint loading and routing integration only; it does not prove semantic specialization and does not use the real MK7 Dataset.
