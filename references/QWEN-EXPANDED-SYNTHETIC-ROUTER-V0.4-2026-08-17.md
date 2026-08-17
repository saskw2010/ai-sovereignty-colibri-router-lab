# Qwen Expanded Synthetic Router v0.4 — 2026-08-17

## Status

**VERIFIED — improved held-out synthetic evaluation; not real MK7 training**

## Run

- Base: `Qwen/Qwen2.5-0.5B`
- Synthetic examples: `35` (five paraphrases per MK7 pillar)
- Train split: `28`
- Validation split: `7`
- Steps: `1,000`
- Device: NVIDIA Quadro P2000 CUDA, FP32
- Qwen Base: frozen
- Trainable scope: bridge + router only

## Results

- Train accuracy: `100%`
- Validation Top-1 accuracy: `71.43%`
- Validation Top-2 hit rate: `100%`
- Initial loss: `5.140216`
- Final loss: `0.000000196`

## Interpretation

The expanded paraphrase set improves held-out behavior over v0.3's unseen Top-1 result (`42.86%`), while confirming that Top-2 routing is more robust in this controlled setup. It remains a synthetic mechanism experiment: no real MK7 Dataset, Gemma, or Golden Training was loaded, and semantic expert specialization is not established.

## Checkpoint

`F:\AI-OPEN-MODELS\mk7-versions\router-v0.4.0-qwen-synthetic\qwen-router-v0.4-synthetic.pt`
