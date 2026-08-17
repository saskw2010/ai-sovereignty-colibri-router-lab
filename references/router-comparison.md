# مقارنة Router Versions

الحالة: **VERIFIED**. المقارنة بين الآليات وعقد الـTelemetry، وليست مقارنة بين أرقام Experts.

| النسخة | القرار | Experts / Top-k | الدليل |
|---|---|---:|---|
| MoST modality Top-2 mini v0.1.0 | hard modality mask before Top-2 | 8 / 2 | VERIFIED / CODE_ONLY |
| MoST modality Top-2 with Unknown fallback v0.2.0 | modality mask; Unknown uses explicit OPEN_POOL_TOP2 | 8 / 2 | VERIFIED / CODE_ONLY |
| Kimi-style Top-2 mini v0.1.0 | learned-style correction bias affects choice only | 8 / 2 | VERIFIED / CODE_ONLY |
| Kimi-style Top-2 correction-bias sweep v0.2.0 | frozen gate with correction-bias scales 0.00 to 2.00 | 8 / 2 | VERIFIED / CODE_ONLY |
| OLMoE native Top-8 analysis v0.1.0 | learned native Router with auxiliary balancing loss | 64 / 8 | VERIFIED methodology / PARTIAL local raw telemetry |
| OLMoE Top-8 telemetry contract v0.2.0 | sanitized entropy, load CV, max-share and heuristic flags | 64 / 8 | VERIFIED methodology / PARTIAL local 0125 |

## ما تعلمناه

- MoST يفرض Candidate Pool صريحًا حسب الـmodality، ثم يختار Top-2.
- MoST v0.2.0 تعامل مع 2 Unknown Tokens عبر OPEN_POOL_TOP2 بلا violations.
- Kimi-style correction bias غيّر مجموعة Experts في 1 من 6 Tokens في التشغيل الاصطناعي الحالي.
- Kimi v0.2.0: تغيرت المجموعات عند المقاييس 0.00/0.25/0.50/1.00/2.00 بعدد 0/0/0/1/4 Tokens.
- OLMoE يضيف Router حقيقيًا داخل كل طبقة مع Expert execution وload-balancing، لكنه Top-8 وليس Top-2.
- OLMoE v0.2.0 صدّر 18 صف Telemetry، والـflags بحثية وليست معيارًا رسميًا.
- لا يجوز اعتبار Expert ID متطابقًا بين النماذج أو اعتبار routing affinity إثباتًا للتخصص.

## النسخة التالية المقترحة

اكتملت مجموعة v0.1/v0.2 المتفق عليها. أي v0.3 أو v1.0 يحتاج سؤال بحث جديد أو عقد MK7 معتمد.
