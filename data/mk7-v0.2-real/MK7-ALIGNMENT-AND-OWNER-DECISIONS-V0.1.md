# MK-7 المواءمة وقرارات المالك المعلقة — V0.1

**Status:** APPROVED_BY_OWNER — 2026-09-05 (جلسة ZCode: المالك وافق على التوصيات D1–D3 كما هي)
**Date:** 2026-09-05
**Scope:** توحيد إطار العدسات MK-7 بين `Knowledge-OS` و`ai-sovereignty-colibri-router-lab`، وحسم التعارضات الثلاثة المكتشفة في فحص 2026-09-05، ومواءمة كويشنير `mk7-v0.2-real` مع `mk7-view.schema.json`.
**غير مسموح:** هذه الوثيقة لا تعتمد أي داتاسيت ولا تفعّل أي تدريب — كل بند ينتظر قرار المالك الصريح.

## 1. تصحيح مرجعي: MK-7 سبع عدسات لا ثماني

الإطار الموثق في المستودعين هو **"Mustafian Knowledge — Seven-Lens Framework (MK-7)"**:

| # | العدسة | الحرف |
|---|---|---|
| 1 | subject_ontology — موضوع المعرفة والكيانات | O |
| 2 | source_evidence — المصدر وقوة الدليل | E |
| 3 | structure_relations — العلاقات والبنية | R |
| 4 | method_validation — صحة المنهج وقابليته للتحقق | M |
| 5 | function_application — التطبيق العملي | A |
| 6 | time_evolution — التطور عبر الزمن | T |
| 7 | human_value_context — أثر القرار على الإنسان | H |

الزاوية "الثامنة" في الذاكرة هي **`shared` fallback** — دور تشغيلي في عقد الـRouter (مخرج `fallback_used`) وليست عدسة تحليل، ولا تظهر كقيمة وسم في الداتاسيت. إذا أراد المالك ترقيتها لعدسة ثامنة رسمية، فهذا إصدار registry جديد عبر مسار الـgates — لا تعديل صامت.

## 2. قرارات المالك الثلاثة — الأدلة والتوصية

### قرار D1: الـBase Model الكانوني

**التعارض:** `manifests/manifest.json` يقول **QMoE-400** (VERIFIED، Apache-2.0، 8 خانات خبراء، smoke gates ناجحة، `training: NOT_STARTED`) بينما `MK7-OWNER-REVIEW-PACKET-V0.3` يقول **Qwen/Qwen2.5-0.5B** (revision مثبت، وكل نتائج v1.0/v1.1 الحقيقية نُفذت على Qwen).

**التوصية:** سياسة قاعدتين معلنتين — **Qwen2.5-0.5B قاعدة تطوير** لكل تجارب Router (وهي المسار المثبت فعليًا بالنتائج)، و**QMoE-400 قاعدة إنتاج** تنتقل إليها بوابة معلنة بعد نجاح بوابة Qwen المستقلة (وهو المتوافق مع معمارية 8-experts ومسار التحويل). يُوثق ذلك في إصدار manifest جديد يذكر القاعدتين بأدوارهما، وتُعلَّم الحزمة V0.3 بأنها superseded في بند الـbase فقط.

### قرار D2: إصدار الـRegistry

**التعارض:** الحزمة تشير لـ`registry_version: router-v0.2.0`، بينما آخر registry مُتحقق منه هو السبعة (نتيجة `router-v1.1-real-experts-result.json`، `all_adapters_frozen: true`).

**التوصية:** اعتماد الـregistry السباعي المُتحقق منه كإصدار immutable باسم `router-v1.1.0` (sibling — بدون لمس v0.2.0)، وتحديث مرجع الحزم المستقبلية إليه. كل وسم داتاسيت (الكويشنير) يلتزم بقائمة السبعة حصراً — كما هو مطبق فعلاً في `build_dataset.py` والـschema.

### قرار D3: مصير داتاسيت `sky365-mustafian-knowledge-v0.1`

**الوضع:** 1000 سجل (800/100/100) بهاشات مثبتة وحزمة مراجعة `READY_FOR_OWNER_REVIEW` منذ إنشائها، `approved: false`.

**التوصية:** اعتمادها **لغرض واحد محدود: تقييمات الـFrozen Router baseline والمقارنات** (لا يُستخدم في أي tuning — حفاظاً على نقاء الـheld-out)، مع إبقاء كل البيانات الجديدة الواقعية تعبر مسار `mk7-v0.2-real` (الكويشنير + الباني + بوابته). هكذا نفتح باب الـbaseline بدون المساس بقاعدة "held-out قبل الضبط".

## 3. مواءمة الحقول مع `mk7-view.schema.json` (Knowledge-OS)

الجدول المرجعي الملزم لأي أداة توليد أو تحويل مستقبلية:

| كويشنير mk7-v0.2-real | mk7-view.schema.json | قاعدة المواءمة |
|---|---|---|
| `quality_status` (draft/approved/rejected) | `review_state` (unreviewed/machine_validated/human_reviewed/approved/rejected) | `review_state` هو دورة الحياة الكانونية الخمسية؛ `quality_status` اختصار تشغيلي: draft→unreviewed/machine_validated، approved→approved، rejected→rejected |
| (غير موجود) | `record_type` (definition/explanation/worked_example/question_answer/misconception_check/prerequisite_check/assessment) | **أُضيف للكويشنير V0.2** — الافتراضي `question_answer` |
| `split` (train/validation/held_out) + `ood_flag`/`boundary_flag` | `split` (development/validation/test/challenge/release) | train→development، validation→validation، held_out→test، boundary_flag→challenge، ood_flag→release. الأعلام تبقى موجودة لأنها أسبق زمنياً في مسارنا |
| `group_id` | `parent_path_ids` + `source_node_ids` | التكامل لا الإلغاء: كل سجل يستطيع أن يحمل `ko_node_ids` اختيارية تشير لعقد منهج Knowledge-OS |
| `provenance_hash` | طبقة Provenance/Evidence | نفس الفلسفة — الهاش للسجل، وprovenance للمرجع الأصلي |
| `mk7-v0.2-NNNNNN` | `ko.mk7.v0.1.*` | معرفان مستقلان بنطاقين — يربطهما `ko_node_ids` |

**القاعدة الحاكمة:** التوليد المستقبلي من شجرة منهج Knowledge-OS يقرأ العقد من `curriculum.schema.json` ويكتب سجلات متوافقة مع الجدول أعلاه — لا يخترع مخططاً ثالثاً.

## 4. ما بعد قرار المالك

عند توقيع D1–D3 (بأي رأي أو تعديل على التوصيات):

1. تُنشأ نسخ manifest الـsibling المطلوبة (registry v1.1.0، قاعدتان معلنتان).
2. تُرفع ملفات `data/mk7-v0.2-real/` (الكويشنير V0.2 + الباني + هذه الوثيقة) على فرع للمراجعة.
3. تفتح تعبئة الدفعة الأولى الحقيقية — والـHeld-out/OOD/Boundary تظل يدوية بالكامل وفق الكويشنير.
