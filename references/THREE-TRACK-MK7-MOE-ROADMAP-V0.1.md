# خطة المسارات الثلاثة: MK7 Training، colibri Runtime، وmoe-stream Benchmark

**الإصدار:** v0.1

**الحالة:** DRAFT — OWNER REVIEW REQUIRED

**الغرض:** توحيد خطة العمل الخاصة بإعداد MK7 Dataset وHeld-out Evaluation والتدريب الحقيقي، وتطوير مسار colibri للتحويل والتشغيل المحلي، وبناء مقارنة مستقلة مع moe-stream والمشاريع العالمية ذات الصلة.

> هذه الوثيقة خطة تنفيذ فقط. لا تمنح موافقة على بدء تدريب حقيقي، ولا تسمح بتعديل MK7 Dataset أو Base Model أو Golden Training أو أوزان سابقة.

## 1. القرار المعماري

سننفذ ثلاثة مسارات مترابطة مع فصل واضح للمسؤوليات:

| المسار | المستودع أو المرجع | المسؤولية | الناتج الأساسي |
|---|---|---|---|
| MK7 Training & Evaluation | هذا المستودع `ai-sovereignty-colibri-router-lab` | البيانات، العقود، Router training، التقييم، والـgates | Router checkpoints وتقارير تقييم قابلة للمراجعة |
| Local Conversion & Runtime | `saskw2010/colibri` | تحويل checkpoints وتشغيل MoE محليًا مع expert streaming | Converted model artifacts وruntime benchmarks |
| Comparative Benchmark | `GOBA-AI-Labs/moe-stream` مع مصادر عالمية | مقارنة SSD streaming وCPU/GPU offload والكمون والذاكرة | Benchmark matrix وتقرير قرار تقني |

لا تُخزَّن الأوزان أو datasets كبيرة داخل Git. يُحفظ داخل المستودع manifest وSHA-256 وprovenance ومسارات التخزين الخارجية فقط.

## 2. قواعد السلامة والحوكمة

تبقى الإصدارات السابقة immutable. لا يجوز تعديل `MK7-CANONICAL-DATASET-MANIFEST-V0.1.json` أو استبدال النتائج السابقة. أي تغيير ينتج sibling version جديدًا، مثل `mk7-v0.2-real` أو `router-v0.5.0-real`.

لا يبدأ أي تدريب حقيقي قبل تحقق الشروط التالية:

1. اعتماد Dataset Manifest من المالك.
2. اكتمال provenance وSHA-256 لكل split.
3. إثبات عدم وجود duplicate أو semantic leakage بين splits.
4. اعتماد Held-out وOOD وBoundary sets خارج مسار الضبط.
5. نجاح Frozen Router baseline قبل أي optimizer step.
6. إثبات أن Base وAdapters لم تتغير وأن Router فقط قابل للتحديث.
7. تحديد checkpoint destination وseed وhyperparameters قبل التشغيل.

## 3. المسار الأول: MK7 Dataset وHeld-out Evaluation والتدريب

### 3.1 الهدف

المرحلة الأولى هي **Router-only training**. يبقى Base Model وLoRA Experts مجمّدين، ويتعلم Router اختيار الخبير أو الخبراء المناسبين لكل سجل. لا ننتقل إلى تحديث LoRA أو Base إلا بعد نجاح بوابة Router المستقلة.

### 3.2 إصدار البيانات

ننشئ إصدارًا جديدًا باسم `mk7-v0.2-real`، مع إبقاء `v0.1` دون تعديل. يجب أن يحتوي كل سجل على معرف ثابت ومصدر ووسم خبير ومجموعة دلالية تمنع تسرب إعادة الصياغة.

الحد الأدنى المقترح للسجل:

```json
{
  "id": "mk7-v0.2-000001",
  "input": "النص أو السؤال الأصلي",
  "target_expert": "H_human_value_context",
  "allowed_experts": ["H_human_value_context", "D_decision_support"],
  "pillar": "human_value_context",
  "difficulty": "medium",
  "source_id": "source-opaque-id-001",
  "source_type": "owner_curated",
  "language": "ar",
  "quality_status": "approved",
  "group_id": "topic-group-17",
  "provenance_hash": "sha256:..."
}
```

`target_expert` هو الاختيار الأساسي، بينما `allowed_experts` يسمح بتقييم Top-2 عندما يكون المثال صالحًا لأكثر من خبير. الحالات غير القابلة للحسم تُسجل كـ`ambiguous` أو `unknown` بدل فرض label خاطئ.

### 3.3 تقسيم البيانات

يكون التقسيم على مستوى `group_id` أو العائلة الدلالية، لا على مستوى السطر. كل paraphrases أو أمثلة الموضوع نفسه يجب أن تبقى داخل split واحد.

| Split | النسبة الإرشادية | الاستخدام |
|---|---:|---|
| Train | 70% | تحديث Router فقط |
| Validation | 15% | اختيار checkpoint وhyperparameters |
| Held-out Test | 15% | تقييم نهائي لا يُستخدم أثناء الضبط |
| OOD | 100–300 سجل أو حسب المصدر | مواضيع أو صياغات غير مرئية |
| Boundary/Challenge | 100–300 سجل أو حسب المصدر | الالتباس، التداخل، unknown، والتوازن |

الأرقام النهائية تعتمد على حجم المصدر الحقيقي. لا تُنشأ سجلات صناعية لتغطية نقص البيانات دون وسمها بوضوح كـsynthetic وعدم خلطها مع الاختبار الواقعي.

### 3.4 Held-out Evaluation Contract

يجب أن يحافظ التقييم على عقد `MK7-EVALUATION-CONTRACT-V0.3.json`، مع إضافة المقاييس التالية:

| الفئة | المقاييس |
|---|---|
| Accuracy | overall، per-pillar، Top-1، Top-2 hit rate |
| Routing health | entropy، expert utilization، load-balance coefficient |
| Generalization | paraphrase/OOD accuracy، worst-pillar accuracy، boundary accuracy |
| Reliability | abstention/unknown rate، calibration، confidence distribution |
| Diagnostics | confusion matrix، per-length، per-language، per-difficulty |

تُقارن النتائج على نفس الأمثلة بين Frozen Router baseline وTrained Router وSpecialist mode وMixer mode وPlug-and-Play mode.

### 3.5 ترتيب التدريب

يبدأ التنفيذ بـread-only reconciliation ثم baseline frozen، ثم Router-only training محدود الخطوات، ثم تقييم validation، ثم held-out مرة واحدة بعد تثبيت checkpoint. كل تشغيل ينتج manifest للبيئة، seed، commit SHA، dataset hash، model hash، metrics، وcheckpoint hash.

## 4. المسار الثاني: colibri Conversion وLocal Runtime

### 4.1 الهدف

اختبار colibri كطبقة تحويل وتشغيل مستقلة، وليس كبديل لمسار MK7 training. يظل Python مستخدمًا في التحويل فقط، بينما يكون inference في المحرك المحلي حسب تصميم المشروع.

### 4.2 مراحل التنفيذ

| المرحلة | العمل | معيار النجاح |
|---|---|---|
| C0 | تثبيت commit وبيئة reproducible | commit وtoolchain موثقين |
| C1 | تحويل نموذج صغير أو tiny oracle | container يمر باختبار القراءة والهاش |
| C2 | FP8→int4 shard-by-shard | لا يلزم وجود checkpoint كامل أثناء التحويل |
| C3 | تحقق معماري مقابل oracle | teacher-forcing وgreedy ضمن tolerance معلن |
| C4 | اختبار نموذج MoE متوسط | RAM، disk، cache، وlatency موثقة |
| C5 | تشغيل GLM-5.2 عند توفر التخزين والعتاد | تقرير full-runtime مع caveats واضحة |

### 4.3 المقاييس

يجب تسجيل حجم artifacts، زمن التحويل، Peak RSS، cold/warm latency، throughput، expert cache hit rate، قراءات القرص لكل token، نسبة MTP acceptance، ودقة النص مقابل oracle. نجاح التحويل وحده لا يثبت صحة inference، ونجاح inference لا يثبت صلاحية التدريب.

## 5. المسار الثالث: moe-stream والمقارنة العالمية

### 5.1 الهدف

بناء baseline مستقل لفكرة SSD expert streaming، ثم مقارنة moe-stream مع colibri على workload ونموذج مشترك عندما تسمح التوافقية.

### 5.2 محاور المقارنة

| المحور | القياس |
|---|---|
| Model support | architecture، total params، active params، experts/layer |
| Storage | format، quantization، disk footprint، conversion time |
| Memory | resident RAM، VRAM، peak RSS، cache size |
| Speed | cold start، warm decode، prefill، tokens/s |
| I/O | GB/token، expert loads، readahead، cache hit rate |
| Correctness | oracle agreement، deterministic replay، sampling behavior |
| Operations | OS support، GPU support، dependency surface، API |
| Reproducibility | pinned commit، public artifact، command، logs |

تُفصل المقارنة بين أحدث نموذج عالمي وبين أحدث runtime قابل لإعادة التشغيل. لا نعلن أن نموذجًا عالميًا يستخدم تقنية colibri أو converter الخاص به دون دليل أولي.

## 6. الاعتماديات بين المسارات

المسار الأول يحدد عقود البيانات والتقييم التي يجب أن تظل مستقلة عن runtime. المسار الثاني يمكنه استخدام نفس فلسفة manifests والهاشات، لكنه لا يقرأ MK7 Dataset تلقائيًا. المسار الثالث يزوّد قرارًا مقارنًا ولا يغيّر training contract.

| الاعتمادية | النتيجة |
|---|---|
| MK7 Dataset قبل التدريب | لا training run قبل approval |
| Held-out قبل hyperparameter tuning | يمنع test leakage |
| colibri oracle قبل full model | يمنع تفسير أخطاء converter كأخطاء نموذج |
| moe-stream workload مشترك | يمنع مقارنة غير عادلة |
| artifacts خارج Git | يحمي المستودع ويضمن reproducibility عبر manifests |

## 7. خارطة الإصدارات

| الإصدار | الحالة المقصودة |
|---|---|
| `mk7-v0.1` | الإصدار الحالي المقترح؛ لا يُعدل |
| `mk7-v0.2-real` | أول Dataset واقعي معتمد |
| `router-v0.5.0-real` | أول Router checkpoint بعد gate |
| `colibri-conversion-v0.1` | أول pipeline تحويل reproducible |
| `benchmark-moe-v0.1` | أول مصفوفة مقارنة colibri/moe-stream |
| `decision-v0.1` | تقرير القرار النهائي والقيود |

## 8. Definition of Done

تُعتبر الخطة مكتملة عندما يكون لدينا Dataset Manifest معتمد، Held-out report غير قابل لإعادة الكتابة، Router checkpoint موثق، converter benchmark قابل للتكرار، مقارنة moe-stream عادلة، وتقرير موحد يوضح ما تم إثباته وما بقي غير محسوم.

## 9. الحالة الحالية والإجراء التالي

**الحالة الحالية:** الخطة موثقة، ولا توجد موافقة على التدريب الحقيقي ضمن هذه الوثيقة.

**الإجراء التالي المقترح:** مراجعة المالك لمصدر MK7 الواقعي، ثم إنشاء `mk7-v0.2-real` manifest وread-only reconciliation دون تشغيل optimizer أو تعديل أي إصدار سابق.

## References

1. [`MK7-EVALUATION-CONTRACT-V0.3.json`](./MK7-EVALUATION-CONTRACT-V0.3.json)
2. [`MK7-CANONICAL-DATASET-MANIFEST-V0.1.json`](./MK7-CANONICAL-DATASET-MANIFEST-V0.1.json)
3. [`MK7-REAL-TRAINING-APPROVAL-GATE-2026-08-17.md`](./MK7-REAL-TRAINING-APPROVAL-GATE-2026-08-17.md)
4. [`ai-sovereignty-colibri-router-lab`](https://github.com/saskw2010/ai-sovereignty-colibri-router-lab)
5. [`colibri`](https://github.com/saskw2010/colibri)
6. [`moe-stream`](https://github.com/GOBA-AI-Labs/moe-stream)
7. [`DeepSeek-V4 Preview Release`](https://api-docs.deepseek.com/news/news260424/)
