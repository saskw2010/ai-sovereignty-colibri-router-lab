# Colibri Router Lab

Research-only evidence package for the Colibri/MK7 Router path.

## Scope

- QMoE-400 Native-MoE FP32 smoke and read-only Router telemetry.
- MK7 Router Contract and Expert Registry.
- Comparison notes for Tiny-MoE, Switch, MixLoRA, and MoST.
- Synthetic forward control for Native MoE versus Dense + Frozen LoRA Experts.

## Explicit exclusions

This repository intentionally contains no model weights, datasets, Gemma assets, Golden Training assets, or MK7 Dataset. It contains no training launcher. The diagnostic backward pass has no optimizer step and no weight update.

## Status

`LOCAL_GITHUB_STAGING_ONLY` — this directory has not been pushed. External publication requires an owner-approved GitHub repository and visibility decision.

## Evidence labels

`VERIFIED` means the local artifact or command result was observed. `PARTIAL`, `UNRESOLVED`, and `PENDING` are preserved rather than promoted to success.
