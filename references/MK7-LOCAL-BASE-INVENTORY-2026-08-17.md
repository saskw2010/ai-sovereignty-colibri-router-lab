# MK7 Local Base Inventory — 2026-08-17

## Status

**VERIFIED — F: INVENTORY ONLY; NO DOWNLOAD**

The active model workspace `F:\AI-OPEN-MODELS` was inspected for an existing Dense causal-language-model base by directory and local `config.json` evidence.

## Current local model families

| Path | Current classification | MK7 Dense+LoRA base? |
|---|---|---|
| `F:\AI-OPEN-MODELS\qmoe-400` | Native sparse MoE, 8 experts, Top-2 | No — router research reference |
| `F:\AI-OPEN-MODELS\OLMoE` | Native MoE reference | No — architecture/router reference |
| `F:\AI-OPEN-MODELS\MoST` | Router/code research reference | Not verified as a compatible Dense checkpoint |
| `F:\AI-OPEN-MODELS\Muse-Glimmer` | Reference/runtime material | No compatible Dense LoRA base verified |
| `F:\AI-OPEN-MODELS\Aria` | Reference/model material | No compatible Dense LoRA base verified |

Only QMoE exposed a local model `config.json` in the scoped inspection, and its known architecture is Native MoE. No existing local Dense causal LM with a verified LoRA-compatible architecture was found.

## Consequence

The MK7 Dense+Frozen-LoRA path remains **UNRESOLVED for base selection**. No model is selected by assumption, and no download is authorized by this inventory step.

The next legitimate action is an explicit candidate decision (repository, revision/hash, license, size, and destination on `F:`), followed by a no-training forward smoke.

## Safety

- No files were moved, deleted, or replaced.
- No model was downloaded.
- No training started.
- QMoE remains separate from the MK7 Dense+LoRA path.
