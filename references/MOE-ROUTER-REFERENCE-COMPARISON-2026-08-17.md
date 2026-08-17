# Colibri Router Reference Comparison — 2026-08-17

## Purpose

This is a code-and-local-evidence comparison for the MK7 design. It does not authorize training or imply that a reference checkpoint is a Colibri model.

## Current evidence matrix

| Reference | What is locally evidenced | Router/expert lesson | Status | MK7 role |
|---|---|---|---|---|
| QMoE-400 | Verified local FP32 forward, 8 experts, Top-2, 12 layers, telemetry and Router-only gradient path | Native sparse expert routing; Router can be isolated while experts remain frozen | VERIFIED | First runtime/router smoke |
| Tiny-MoE | Official HF model card and linked GitHub identified: 14-layer decoder-only, 8 routed + 1 shared expert, Top-2, Apache-2.0; not downloaded locally | Strong code/model reference, but base model is English and not instruction tuned | VERIFIED REMOTE / NOT LOCAL | Code reference candidate |
| Switch-Base-8 | Existing research decision identifies Transformers router-logit API and Top-1 Switch behavior | Useful minimal Top-1 routing/loss reference; encoder-decoder semantics differ from MK7 | PARTIAL | API/loss reference only |
| MixLoRA / MoE-PEFT | Local source, tests, config, routing strategy, and Llama-2-7B adapter documentation are present | Frozen dense base + multiple LoRA experts + learned Top-k router; adapter experts are code/config-defined and weights are base-specific | VERIFIED (reference) | Dense + Frozen LoRA control/reference |
| MoST | Paper identifies MAMoE with modality-specific groups, shared experts, modality mask, and Top-2-style routing; local learned checkpoint/runtime not present | Directly informs modality-aware masks and shared fallback, but requires speech/audio stack and training data outside this smoke | VERIFIED PAPER / NOT LOCAL | Router policy reference only |

## Architectural separation

There are two different experiments:

1. **Native MoE:** the model contains dense expert weights and an internal Router. QMoE is the active smoke baseline.
2. **MK7 candidate:** a frozen dense base contains externally registered Frozen LoRA Experts, and a learned Router selects/admixtures adapters. MixLoRA informs the adapter injection and routing contract; it does not replace the MK7 design.

The Router selecting native FFN experts is not equivalent to a Router selecting LoRA adapters. Telemetry fields and evaluation must keep these paths separate.

## Guardrails

- No Gemma or Golden Training assets are touched.
- No MK7 Dataset is used in the smoke or reference comparison.
- No MixLoRA adapter is paired with QMoE or Gemma; the documented adapter remains paired with its Llama-2-7B base.
- No training is started. The QMoE backward result is a no-update gradient diagnostic only.

## Decision from current evidence

Keep QMoE-400 as the Native-MoE runtime baseline. Use the local MixLoRA source as the first Dense+Frozen-LoRA reference. Tiny-MoE is now verified as a remote Apache-2.0 code/model reference but remains outside F until separately approved for download. MoST is verified at paper/design level, not as a local runtime. The next MK7 artifact is the framework-neutral Router Contract and Expert Registry, followed by the completed dry-run control harness—not training.

## Control harness result

The deterministic control harness completed at:

`Q:\Colibri\research\router-comparison-lab\outputs\mk7-control-harness.json`

It verifies tensor flow for both paths with synthetic hidden states:

- Native MoE: internal Router → Top-2 full expert projections.
- MK7 candidate: frozen dense projection + Top-2 weighted LoRA residual experts.
- Same batch/output shape and explicit frozen-state flags.
- `training_started=false`, `weights_updated=false`, `dataset_loaded=false`.

**Classification:** `VERIFIED` as a forward/contract control; `UNRESOLVED` for quality, specialization, and real-model equivalence.

## Remote source evidence

- Tiny-MoE model card: `https://huggingface.co/AbdelrhmanEbied/Tiny-MoE`
- Tiny-MoE source: `https://github.com/AbdelrhmanEbied/Tiny-MoE`
- MoST paper: `https://arxiv.org/abs/2601.10272`
- MoST source link cited by the paper: `https://github.com/NUS-HPC-AI-Lab/MoST`

Remote evidence is recorded for study only; it does not authorize downloading weights, datasets, or starting training.
