# MK7 Router v1.1 — Real LoRA Experts Integration

Date: 2026-08-18  
Status: **VERIFIED runtime / PARTIAL semantic specialization**

## What succeeded

- Seven real LoRA adapters were trained independently from read-only canonical MK7 training records.
- Each adapter was saved under `F:\AI-OPEN-MODELS\mk7-versions\router-v1.1.0-real-lora-experts`.
- All seven adapters loaded into one Qwen runtime.
- All adapter parameters were frozen after training.
- Specialist mode executed successfully.
- Top-2 Mixer mode executed successfully.
- All seven test prompts produced a non-zero Specialist-vs-Mixer logit difference.
- A new integrated package was written under `F:\AI-OPEN-MODELS\mk7-versions\router-v1.1.0-real-integrated`.

## Router telemetry

Telemetry is stored in:

`F:\AI-OPEN-MODELS\mk7-versions\router-v1.0.0-real\telemetry-v1.0.json`

It includes validation/test/challenge Top-1 and Top-2, entropy in nats and normalized form, predicted expert counts, and full confusion matrices.

The v1.0 router remains strong on the fixed sets: Validation 100%, Test 100%, Challenge 85.71%. Challenge entropy is higher than validation entropy, which correctly indicates lower confidence on unseen phrasing.

## Runtime evidence

Runtime result:

`F:\AI-OPEN-MODELS\mk7-versions\router-v1.1.0-real-integrated\result.json`

Verified fields:

- `expert_count`: 7
- `all_adapters_loaded`: true
- `all_adapters_frozen`: true
- `specialist_mode`: true
- `top2_mixer_mode`: true
- `nonzero_specialist_mixer_differences`: 7
- dtype: float32
- device: CUDA
- original Dataset modified: false

## Interpretation boundary

This proves the complete plug-and-play mechanics: a frozen base can execute one selected LoRA specialist or combine the Router's Top-2 adapters. It does not yet prove that each adapter is a high-quality domain specialist; the training run is intentionally small (8 records per pillar and 3 steps) and is a controlled integration checkpoint. Semantic specialization requires a larger separately governed training/evaluation phase.

## Decision

**VERIFIED:** telemetry, real adapter loading, freezing, specialist execution, Top-2 mixing, and independent checkpoint packaging.  
**PARTIAL:** semantic quality and expert specialization.

Scripts:

- `Q:\Colibri\research\router-comparison-lab\telemetry_mk7_router_v1_0.py`
- `Q:\Colibri\research\router-comparison-lab\train_mk7_real_lora_experts_v1_1.py`
- `Q:\Colibri\research\router-comparison-lab\evaluate_mk7_router_v1_1_real_experts.py`
