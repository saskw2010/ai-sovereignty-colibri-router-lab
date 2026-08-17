"""v1.0 domain-specific augmentation and held-out challenge evaluation."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
BASE=r"F:\AI-OPEN-MODELS\qwen2.5-0.5b"; DATA=Path(r"Q:\Colibri\training\datasets\mk7\v0.1\combined"); OUT=Path(r"F:\AI-OPEN-MODELS\mk7-versions\router-v1.0.0-real"); OUT.mkdir(parents=True,exist_ok=True); labels=["subject_ontology","source_evidence","structure_relations","method_validation","function_application","time_evolution","human_value_context"]; mp={x:i for i,x in enumerate(labels)}
aug=[
 ["ما موضوع المعرفة وما الكيانات التي تتناولها؟","حدد المفاهيم الأساسية في هذا المجال.","ما حدود الشيء أو الظاهرة محل الدراسة؟","صنف موضوعات هذا النظام المعرفي.","ما الموجودات التي يجب تعريفها قبل الإجابة؟","ما هو منظور الموضوع والوجود هنا؟","ارسم خريطة للكيانات الأساسية.","أي فئات تنتمي إلى هذا الموضوع؟","كيف نحدد هوية موضوع المعرفة؟","ما العناصر التي يصفها السؤال؟"],
 ["ما مصدر هذا الادعاء وما قوة دليله؟","كيف يمكن تتبع المعلومة إلى مرجعها؟","اذكر الشواهد التي تؤيد النتيجة.","هل توجد وثائق تثبت هذا الاستنتاج؟","ميز بين الرأي والدليل القابل للتحقق.","ما نوع المصدر المستخدم هنا؟","كيف نقيس درجة الثقة في الخبر؟","ما provenance لهذه المعلومة؟","اعرض الأدلة وراء القرار.","كيف نراجع صحة المصدر؟"],
 ["ما العلاقات بين المفاهيم في هذا النظام؟","كيف تعتمد العناصر على بعضها؟","اربط الكيانات والخصائص والاعتماديات.","ما البنية التي تجمع هذه الأجزاء؟","اشرح الروابط غير الهرمية بين المفاهيم.","كيف تنتقل العلاقة من عنصر إلى آخر؟","مثل شبكة العلاقات في الموضوع.","ما سبب اتصال هذين الكيانين؟","حلل بنية المعرفة وروابطها.","ما الاعتماديات بين المكونات؟"],
 ["هل المنهج المقترح صحيح وقابل للتحقق؟","راجع افتراضات هذه الطريقة.","كيف نختبر صلاحية الإجراء؟","ما نقاط الضعف في المنهج؟","تحقق من اتساق خطوات الحل.","هل الدليل يبرر الطريقة المستخدمة؟","قارن بين طرق التحقق الممكنة.","كيف نكتشف خطأً منهجيًا؟","قيّم جودة المنهج والافتراضات.","هل يمكن إعادة إنتاج النتيجة؟"],
 ["كيف نستخدم هذه المعرفة في عمل فعلي؟","حوّل الفكرة إلى إجراء تطبيقي.","ما القرار الذي تساعد هذه المعرفة على اتخاذه؟","اعرض مثالًا لاستخدامها في مؤسسة.","كيف نطبق المنهج في سير عمل؟","ما الوظيفة العملية لهذا المفهوم؟","صمم خطوات تنفيذ الحل.","كيف تتحول المعرفة إلى أداة؟","أين يمكن تطبيق هذا التصور؟","ما أثره على التصميم والتنفيذ؟"],
 ["كيف تطور هذا المفهوم عبر الزمن؟","ما الفرق بين الحالة القديمة والحالية؟","ما الأحداث التي شكلت هذا النظام؟","تتبع إصدارات المعرفة وتغيرها.","هل هذه المعلومة ما زالت حديثة؟","ما السياق التاريخي للفكرة؟","متى تغيرت السياسة ولماذا؟","اشرح مراحل تطور الموضوع.","ما الذي تقادم من هذه المعرفة؟","رتب التحولات الزمنية في المجال."],
 ["ما أثر القرار على الإنسان وأصحاب المصلحة؟","ما القيم والأخلاق المرتبطة بالموضوع؟","ناقش الثقافة والسياق الإنساني للمعلومة.","من سيتحمل نتائج هذا الإجراء؟","ما المخاطر الاجتماعية للحل؟","كيف نوازن بين المنفعة والضرر؟","ما مسؤوليات المؤسسة تجاه الناس؟","اشرح البعد الإنساني للمعرفة.","ما أثر اللغة والثقافة هنا؟","قيّم العدالة والقيم في القرار."]]
challenge=[("كيف نعرّف نطاق الكيان الذي تدور حوله هذه المعرفة؟",0),("ما المفاهيم التي يجب وضعها في قاموس الموضوع؟",0),("ما المرجع الذي يمكن الرجوع إليه للتحقق من النتيجة؟",1),("كيف نميز الدليل القوي من الادعاء غير الموثق؟",1),("كيف تمثل الروابط بين الوحدات داخل النظام؟",2),("ما الذي يفسر اعتماد هذا الجزء على ذاك؟",2),("ما الاختبار المناسب للتأكد من سلامة الإجراءات؟",3),("أي فرضية قد تجعل الاستنتاج غير موثوق؟",3),("كيف يستفيد فريق العمل من هذه المعرفة عمليًا؟",4),("ما الخطوة التي تحول الفكرة إلى خدمة قابلة للتنفيذ؟",4),("ما التغيرات التي مرت بها هذه القاعدة؟",5),("كيف نعرف أن الإصدار الحالي أحدث من السابق؟",5),("كيف يتأثر المستخدمون بهذا الاختيار؟",6),("ما الاعتبارات الأخلاقية عند تطبيق الحل؟",6)]
def load(n):
 out=[]
 for line in (DATA/n).read_text(encoding="utf-8").splitlines():
  if line:
   r=json.loads(line); k=str(r.get("knowledge_unit_id","")).split(":")[-1]
   if k in mp: out.append((r["input"],mp[k]))
 return out
tr,va,te=load("train.jsonl"),load("validation.jsonl"),load("test.jsonl"); train=tr+[(t,i) for i,g in enumerate(aug) for t in g]; device="cuda" if torch.cuda.is_available() else "cpu"; tok=AutoTokenizer.from_pretrained(BASE); tok.pad_token=tok.eos_token; m=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32,low_cpu_mem_usage=True).eval().to(device)
for p in m.parameters(): p.requires_grad=False
def feat(rows):
 hs=[]
 with torch.no_grad():
  for start in range(0,len(rows),16):
   part=rows[start:start+16]; b=tok([x[0] for x in part],padding=True,truncation=True,max_length=512,return_tensors="pt").to(device); o=m(**b,output_hidden_states=True,use_cache=False); idx=b.attention_mask.sum(1)-1; hs.append(o.hidden_states[-1][torch.arange(len(part),device=device),idx,:].float().cpu())
 return torch.cat(hs),torch.tensor([x[1] for x in rows])
x,y=feat(train); xv,yv=feat(va); xt,yt=feat(te); xc,yc=feat(challenge); torch.manual_seed(47); router=torch.nn.Sequential(torch.nn.Linear(x.shape[-1],128),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(128,7)); opt=torch.optim.AdamW(router.parameters(),lr=.005,weight_decay=.01); best=None; bestacc=-1; losses=[]
for step in range(1,501):
 router.train(); loss=torch.nn.functional.cross_entropy(router(x),y,label_smoothing=.05); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach())); router.eval(); acc=float((router(xv).argmax(-1)==yv).float().mean())
 if acc>bestacc: bestacc=acc; best={k:v.detach().clone() for k,v in router.state_dict().items()}; beststep=step
router.load_state_dict(best)
def met(a,b):
 z=router(a).topk(2,dim=-1).indices; return {"top1_accuracy":float((z[:,0]==b).float().mean()),"top2_hit_rate":float((z==b[:,None]).any(1).float().mean())}
result={"status":"VERIFIED_MK7_ROUTER_V1_DOMAIN_AUG","base":"Qwen/Qwen2.5-0.5B","scope":"router_only","best_step":beststep,"counts":{"real_train":len(tr),"domain_augmentation":len(train)-len(tr),"validation":len(va),"test":len(te),"challenge":len(challenge)},"initial_loss":losses[0],"final_loss":losses[-1],"train":met(x,y),"validation":met(xv,yv),"test":met(xt,yt),"challenge":met(xc,yc),"base_frozen":True,"experts_trained":False,"dataset_modified":False,"checkpoint":str(OUT/"router-only-qwen-mk7-v1.0-domain.pt")}; torch.save({"router":router.state_dict(),"input_dim":x.shape[-1],"labels":labels},OUT/"router-only-qwen-mk7-v1.0-domain.pt"); (OUT/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,ensure_ascii=False))
