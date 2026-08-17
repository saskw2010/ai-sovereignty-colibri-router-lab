# Qwen Dense Base Smoke — 2026-08-17

## Status

**VERIFIED — downloaded to F and forward-smoked; no training**

## Artifact

- Model: `Qwen/Qwen2.5-0.5B`
- Local path: `F:\AI-OPEN-MODELS\qwen2.5-0.5b`
- Format: Safetensors, separated from GGUF and source code
- Weight size: `988,097,824` bytes
- Weight SHA256: `88C142557820CCAD55BB59756BFCFCF891DE9CC6202816BD346445188A0ED342`
- Config SHA256: `479DCF0C5286339E41AD3992CD08AE88A467C4187587936248E2B7C96283484B`

## Forward smoke

- Device: NVIDIA Quadro P2000
- Dtype: FP32
- Logits shape: `[1, 6, 151936]`
- Forward time: approximately `0.34 s`
- Training started: false
- Optimizer step: false
- LoRA loaded: false
- Dataset loaded: false

## Interpretation

Qwen2.5-0.5B is now a verified local Dense Base candidate for the next no-training adapter compatibility smoke. It is not yet the approved base for real MK7 training. The next safe test is to attach a tiny synthetic frozen LoRA control and verify the adapter target modules, while preserving v0.1.0 and v0.2.0.
