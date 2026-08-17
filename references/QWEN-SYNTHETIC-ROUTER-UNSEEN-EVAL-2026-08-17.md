# Qwen Synthetic Router Unseen Evaluation — 2026-08-17

## Status

**VERIFIED — generalization gap identified**

The `v0.3.0-qwen-synthetic` checkpoint was evaluated on seven paraphrased prompts that were not used during training.

- Samples: 7
- Top-1 accuracy: `42.86%`
- Top-2 hit rate: `71.43%`
- Device: NVIDIA Quadro P2000 CUDA
- Training during evaluation: false
- Dataset loaded: false

## Interpretation

The checkpoint memorizes the controlled seven-example mapping well enough for the closed-set test (`100%`), but its unseen paraphrase performance is not sufficient to claim semantic generalization or expert specialization. Therefore `v0.3.0-qwen-synthetic` remains a **mechanism proof / research checkpoint**, not a production or real-MK7 candidate.

## Next design implication

The next router experiment should use more varied synthetic paraphrases per pillar, explicit held-out validation, and a less brittle representation/bridge. The real MK7 Dataset remains gated and must not be substituted silently.
