# Colibri / MK7 Versioning and Checkpoint Policy

## قاعدة عدم التعديل

بعد إغلاق أي نسخة Router أو Adapter، لا نعدّل ملفاتها في مكانها. أي تجربة لاحقة تبدأ من نسخة محفوظة جديدة.

## Current baseline

- Version: `router-v0.1.0`
- Scope: contracts, QMoE evidence, MixLoRA reference tests, synthetic Router and synthetic seven-pillar Router proof.
- Real project Base/Adapters/Datasets: not loaded.
- Training on project data: not authorized by this baseline.

## Checkpoint layout

Each checkpoint must preserve:

- Router weights.
- Adapter Registry.
- Base identity and revision.
- Dataset manifest and split hashes, if a dataset is approved.
- Training configuration and seed.
- Telemetry and evaluation results.
- SHA256 and status.

Suggested F layout:

`F:\AI-OPEN-MODELS\mk7-versions\router-vX.Y.Z\`

## Change rules

- Patch: documentation or non-behavioral metadata only.
- Minor: new adapter, router mode, or compatible telemetry field.
- Major: incompatible Router Contract, Registry schema, or checkpoint format.
- Never overwrite a released checkpoint; create a sibling version.
