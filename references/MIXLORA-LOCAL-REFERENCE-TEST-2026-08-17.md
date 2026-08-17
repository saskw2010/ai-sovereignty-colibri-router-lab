# MixLoRA Local Reference Test — 2026-08-17

## Scope

Read-only code reference validation. No Llama-2-7B weights, MixLoRA adapter, dataset, or training run was started.

## Command

`Q:\Colibri\training\venv-py311-clean\Scripts\python.exe -m unittest Q:\Colibri\research\external\MixLoRA\tests\test_moe_layer.py -v`

with local source on `PYTHONPATH`.

## Result

`VERIFIED_REFERENCE_RUNTIME`

- `test_llama_forward`: passed.
- `test_phi_forward`: passed.
- `test_phi3_forward`: passed.
- 3 tests passed in about 0.10 seconds.

## Interpretation

The local MixLoRA implementation can construct and forward Top-2 sparse LoRA-MoE layers for the tested Llama/Phi/Phi-3 variants. This verifies code-path shape/forward behavior only. It does not verify adapter quality, Llama-2 checkpoint compatibility in this environment, semantic expert specialization, or MK7 readiness.

The MixLoRA adapter remains paired only with its documented Llama-2 base and is not used with QMoE or Gemma.
