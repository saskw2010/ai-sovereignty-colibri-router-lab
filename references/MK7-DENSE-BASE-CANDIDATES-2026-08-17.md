# MK7 Dense Base Candidates — 2026-08-17

## Status

**PARTIAL — RESEARCHED, NOT DOWNLOADED, NOT SELECTED**

The current `F:` inventory has no compatible Dense base. Official Hugging Face model pages were checked for small, pretrained, causal Dense candidates.

| Candidate | Evidence | Fit for first MK7 LoRA control | Decision |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B` | Apache-2.0; causal text-generation model; repository page lists a ~988 MB safetensors file and multilingual support including Arabic | Best first candidate for a small multilingual Dense + LoRA forward smoke | **CANDIDATE #1** |
| `HuggingFaceTB/SmolLM-135M` | Apache-2.0; 135M causal LM; model page lists ~538 MB safetensors; primarily English | Smallest practical control, but weaker Arabic relevance | **CANDIDATE #2** |
| `openai-community/gpt2` | MIT; 124M causal LM | Very small and compatible in principle, but English/older tokenizer and weaker MK7 relevance | **REFERENCE ONLY** |

## Recommendation

Use `Qwen/Qwen2.5-0.5B` as the proposed first Dense Base for a no-training forward smoke, pending owner approval and explicit download authorization. Its approximate FP32 weight payload is under 1 GB by the model-page file listing, but runtime memory must include activations, optimizer state only if training is later authorized, and LoRA/router tensors.

SmolLM-135M remains the fallback if the Qwen smoke exceeds local memory or latency limits.

## Required before download

- Confirm the exact immutable revision/hash.
- Confirm license and intended publication scope.
- Confirm destination under `F:\AI-OPEN-MODELS`.
- Record expected files and hashes.
- Obtain explicit authorization for the download.
- Run forward smoke only; do not start training.

## Sources

- https://huggingface.co/Qwen/Qwen2.5-0.5B
- https://huggingface.co/HuggingFaceTB/SmolLM-135M
- https://huggingface.co/openai-community/gpt2
