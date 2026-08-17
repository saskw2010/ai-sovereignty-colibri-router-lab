# MK7 Adapter Runtime v0.2 — 2026-08-17

تم بناء نسخة جديدة بجانب `router-v0.1.0` دون تعديل النسخة الأولى.

## الأوضاع التي تم اختبارها

- `Mixer`: اختيار Top-2 ودمج LoRA Expertين بأوزان Router.
- `Specialist`: اختيار Expert واحد من الـRegistry.
- `Plug-and-Play`: تحميل Adapter واحد عند الطلب.

## النتيجة

- الحالة: `VERIFIED_MK7_ADAPTER_RUNTIME_V0_2`.
- Base مجمد.
- كل Adapters مجمدة.
- Router مجمد أثناء Runtime smoke.
- لا Dataset ولا Gemma ولا Golden Training.
- `v0_1_untouched=true`.

## Artifacts

- Runtime: `Q:\Colibri\research\router-comparison-lab\mk7_adapter_runtime_v0_2.py`
- Registry: `F:\AI-OPEN-MODELS\mk7-versions\router-v0.2.0\registry.json`
- Results: `F:\AI-OPEN-MODELS\mk7-versions\router-v0.2.0\results\runtime-smoke.json`
- Copied Router checkpoint: `F:\AI-OPEN-MODELS\mk7-versions\router-v0.2.0\weights\router.pt`

هذه خطوة Runtime contract باستخدام Adapters صناعية مجمدة، وليست تقييم جودة على Base حقيقي.
