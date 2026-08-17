# Qwen Synthetic Router Closed-Set Evaluation — 2026-08-17

## Status

**VERIFIED — closed-set checkpoint evaluation**

- Checkpoint: `router-v0.3.0-qwen-synthetic`
- Samples: 7 synthetic pillar prompts
- Top-1 accuracy: `100%`
- Top-2 hit rate: `100%`
- Device: NVIDIA Quadro P2000 CUDA
- Training during evaluation: false
- Dataset loaded: false

Every synthetic prompt routed to its intended label as Top-1. This is a closed-set evaluation on the same controlled prompt family used for the synthetic training run; it does not measure generalization, semantic specialization, or real MK7 quality.
