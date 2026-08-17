# MK7 Router v1.0 — Qwen Router-Only Result

Date: 2026-08-18  
Status: **VERIFIED**

## Executive result

The first improved MK7 router version passed the agreed success gate:

- Validation Top-1: **100%**
- Test Top-1: **100%**
- Unseen domain Challenge Top-1: **85.71%**
- Validation and Test Top-2: **100%**
- Challenge Top-2: **85.71%**

This is a real router-only experiment over the frozen Qwen base. The base model was not trained and the original dataset was not modified.

## Architecture and scope

- Base: `Qwen/Qwen2.5-0.5B`
- Base revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- Input: frozen Qwen hidden representation
- Trainable component: MK7 router MLP only
- Experts: frozen synthetic control adapters; no expert weights trained
- Training data: 114 canonical training records plus 70 domain-specific augmentation records created inside the experiment script
- Evaluation: 11 validation, 15 test, 14 unseen challenge prompts
- Steps: 500 maximum; validation-selected checkpoint at step 238

## Integrity

- Checkpoint: `F:\AI-OPEN-MODELS\mk7-versions\router-v1.0.0-real\router-only-qwen-mk7-v1.0-domain.pt`
- SHA-256: `A0BF5CDA3A8EDBD8F88C58F81646972BFC2784C31B19603A6256677956BA5F7D`
- Dataset modified: **false**
- Base frozen: **true**
- Expert weights trained: **false**
- Golden Training touched: **false**
- Gemma touched: **false**

## Router specialization evidence

The experiment demonstrates learnable routing into the seven MK7 knowledge pillars under frozen-base/frozen-expert conditions. It does **not** yet prove that the underlying experts have independently learned semantic specialization, because the current experts are frozen synthetic controls. The measured result is router classification and generalization, not expert-quality proof.

Required next telemetry artifact: per-pillar routing counts, entropy, confusion matrix, and abstention/Top-2 analysis on the fixed validation, test, and challenge sets.

## Decision

**VERIFIED / SUCCESSFUL ROUTER CHECKPOINT.** Preserve v0.3, v0.6, and v0.9 unchanged. v1.0 becomes the current candidate for the next controlled phase: router telemetry, then replacement of synthetic controls with separately prepared frozen LoRA experts.

## Reproducibility

Script: `Q:\Colibri\research\router-comparison-lab\train_mk7_router_real_qwen_v1_0_domain_aug.py`

Result JSON: `F:\AI-OPEN-MODELS\mk7-versions\router-v1.0.0-real\result.json`
