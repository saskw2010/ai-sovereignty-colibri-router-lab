# Qwen MK7 Router Probe Suite — 2026-08-17

## Status

**VERIFIED — telemetry suite; specialization remains unresolved**

Seven frozen probes were routed through Qwen + `router-v0.1.0` using the deterministic 896→32 bridge and Top-2 selection.

| Probe | Top-2 selection | Top weight | Entropy |
|---|---|---:|---:|
| English | A / H | 0.9865 | 0.0714 |
| Arabic | E / R | 0.5602 | 0.8652 |
| Code | O / E | 1.0000 | ~0 |
| ERP | O / A | 1.0000 | ~0 |
| Contract | A / O | 0.9947 | 0.0359 |
| History | H / A | 1.0000 | ~0 |
| Human values | H / O | 0.9605 | 0.1946 |

## Interpretation

The router is operational and produces different selections and entropy levels across probes. This is telemetry evidence only. It does **not** establish semantic expert specialization because the bridge is a controlled compatibility layer and the LoRA adapters are synthetic and frozen.

## Safety

- Router checkpoint unchanged.
- Base and adapters frozen.
- Training started: false.
- Dataset loaded: false.
