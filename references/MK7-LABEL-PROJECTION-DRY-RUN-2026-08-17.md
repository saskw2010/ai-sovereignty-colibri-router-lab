# MK7 Label Projection Dry-Run — 2026-08-17

تم فحص ملفات JSONL قراءة فقط لإعداد labels للـRouter.

## Result

- Status: `VERIFIED_READ_ONLY_LABEL_PROJECTION`.
- Source files scanned: 18.
- Records scanned from JSONL: `2000`.
- Manifest-reported total: `1000`.
- Explicit seven-pillar labels: `280` (`40` لكل عمود).
- Non-pillar or missing labels: `1720`.
- Dataset modified: `false`.
- Training started: `false`.

## Gate

يوجد تعارض يجب حسمه قبل التدريب الحقيقي: manifest يقول 1000 سجل، بينما ملفات JSONL تعطي 2000 عند القراءة الحالية. كذلك أغلب السجلات لا تحتوي `knowledge_unit_id` مطابقًا لأحد الأعمدة السبعة. لا يتم إسقاط أو إعادة تصنيف هذه السجلات تلقائيًا.

## Reconciliation result

التدقيق التفصيلي أثبت أن الـ2000 ليست 2000 معرفًا مستقلًا: يوجد `1000` ID فريد، وكل ID مكرر مرة لأن مجلدات `batch-*` و`combined` كلاهما داخل نطاق القراءة. العد الصحيح يعتمد على اختيار مصدر canonical واحد، وليس جمع الاثنين. النتيجة المجمعة محفوظة في:

`F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-preflight\dataset-reconciliation.json`

التصنيف: `VERIFIED_READ_ONLY_RECONCILIATION`، مع بقاء قرار المصدر canonical وlabel policy مفتوحًا.

النتيجة المجمعة محفوظة في:

`F:\AI-OPEN-MODELS\mk7-versions\router-v0.3.0-preflight\label-projection-dry-run.json`
