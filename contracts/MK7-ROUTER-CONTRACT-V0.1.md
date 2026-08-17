# MK7 Router Contract v0.1

## القرار

MK7 لا يُبنى الآن داخل QMoE. التصميم المرشح هو **Dense frozen base + Frozen LoRA Experts + learned Top-2 Router**. QMoE يظل مختبر Native MoE منفصلًا.

## حدود العقد

كل قرار Router يجب أن يعيد:

- `expert_ids`: خبراء Top-2 لكل token.
- `weights`: أوزان الدمج بعد التطبيع.
- `entropy`: عدم يقين القرار.
- `fallback_used`: هل استُخدم Shared/Unknown fallback.
- هوية الـBase وهوية كل Adapter وإصدار registry.

لا يُسمح بإخفاء قرار Router داخل prompt أو دمج Intent وMK7 تلقائيًا.

## Registry الأولي

الـRegistry يعرّف Shared وERP وArabic/MENA وCode/Data كأدوار مقترحة فقط. الاسم لا يثبت التخصص؛ إثباته يحتاج held-out probes مستقلة وparaphrase/seed comparisons.

## بوابات قبل التدريب

1. تحقق shape وbase/adapter identity.
2. تحقق أن الـBase والأدابتورات مجمدة في smoke.
3. telemetry قابلة لإعادة الإنتاج.
4. held-out domain probes وقياس fallback/abstention.
5. مقارنة baseline: Base-only مقابل Adapter ثابت مقابل Router Top-2.
6. تقرير ذاكرة/زمن وmanifest/hash/license.
7. موافقة مستقلة صريحة قبل أي optimizer step أو Dataset.

الـJSON الموازي هو المصدر الآلي للعقد: `MK7-ROUTER-CONTRACT-V0.1.json`.
