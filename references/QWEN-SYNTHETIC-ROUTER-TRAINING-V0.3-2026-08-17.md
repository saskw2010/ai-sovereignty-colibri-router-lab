# Qwen Synthetic Router Training v0.3 — 2026-08-17

## Status

**VERIFIED — controlled synthetic training; real MK7 training not started**

## Run

- Base: `Qwen/Qwen2.5-0.5B`
- Device: NVIDIA Quadro P2000 CUDA
- Precision: FP32
- Synthetic samples: 7, one per MK7 pillar
- Steps: 1,200
- Initial loss: `4.167990`
- Final loss: `0.0`
- Accuracy: `100%`
- Trainable scope: bridge + router only
- Qwen Base: frozen
- LoRA/real adapters: not trained

## Output

`F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-qwen-synthetic\qwen-router-v0.3-synthetic.pt`

## Safety boundary

- Real MK7 Dataset loaded: false
- Gemma loaded: false
- Golden Training loaded: false
- This is a sibling synthetic version and does not modify `v0.1.0` or `v0.2.0`.

## Interpretation

The result proves that a small router/bridge can learn a controlled seven-label mapping from real Qwen hidden states while the Base remains frozen. It does not prove semantic specialization or authorize real MK7 training.
