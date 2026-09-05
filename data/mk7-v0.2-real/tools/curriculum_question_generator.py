#!/usr/bin/env python3
"""MK-7 curriculum question generator — Mathematics V0.1 pilot.

Reads the approved Knowledge-OS curriculum tree (knowledge-tree-v0.1.json +
atomic-disciplines-v0.1.json) and emits one questionnaire CSV row per
(node x lens), anchored to the node's Arabic name and definition.

Governance:
  - All rows are `curriculum_generated` + `draft` (hypotheses until owner review).
  - Rows carry ko_node_ids back-links; group_id = node id (anti-leakage unit).
  - Misconception rows (from atomic boundary_test) are boundary_flag=true.

Usage:
  python curriculum_question_generator.py \
      --tree <Knowledge-OS>/data/knowledge-tree-v0.1.json \
      --atomic <Knowledge-OS>/data/atomic-disciplines-v0.1.json \
      --out curriculum-questions.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

CSV_COLUMNS = ["input", "language", "decidability", "target_expert", "allowed_experts",
               "difficulty", "group_id", "source_type", "source_id", "record_type",
               "ko_node_ids", "ood_flag", "boundary_flag", "quality_status", "notes"]

KO_PREFIX = "ko.math.v0.1."

TEMPLATES = {
    "O_subject_ontology": "ما موضوع «{name}» وما الكيانات والمفاهيم الأساسية التي يتناولها؟ ({definition})",
    "E_source_evidence": "عند دراسة «{name}»: ما المراجع والأدلة الأساسية التي يستند إليها هذا الموضوع، وكيف نقيّم قوة كل دليل؟",
    "R_structure_relations": "كيف يرتبط «{name}» بموضوعه الأب «{parent}»؟ وما المفاهيم المجاورة التي تندرج معه تحت التخصص نفسه وما الحدود الفاصلة بينها؟",
    "M_method_validation": "ما معايير صحة المناهج والبراهين المستخدمة في «{name}»؟ وكيف نتحقق من صحة نتيجة قبل قبولها في هذا المجال؟",
    "A_function_application": "كيف نطبق مفاهيم «{name}» في مسألة عملية؟ اشرح خطوات الحل بمثال متدرج.",
    "T_time_evolution": "كيف تطورت «{name}» تاريخيًا؟ وما المشكلات التي أحدثت ولادتها وأبرز المحطات في تطورها؟",
    "H_human_value_context": "ما أثر «{name}» على الإنسان وأصحاب المصلحة؟ وأين تظهر تطبيقاته العملية في الحياة والصناعة؟",
}

MISCONCEPTION_TEMPLATE = ("من الأخطاء الشائعة عند دراسة «{name}»: ما الفروق الدقيقة التي يخلط فيها "
                          "المتعلمون بينه وبين المفاهيم المجاورة؟ ({boundary})")


def _confine(path: Path, base: Path) -> Path:
    """Path-traversal guard: resolve and confine to the working tree."""
    resolved = path.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise SystemExit(f"Path escapes the working directory: {resolved}") from exc
    return resolved


def node_depth(node: dict, by_id: dict) -> int:
    depth, seen = 0, set()
    cur = node
    while cur.get("parent_ids") and cur["parent_ids"][0] in by_id and cur["concept_id"] not in seen:
        seen.add(cur["concept_id"])
        cur = by_id[cur["parent_ids"][0]]
        depth += 1
    return depth


def difficulty_for(node: dict, by_id: dict, is_atomic: bool) -> str:
    if node.get("concept_type") == "domain":
        return "easy"
    if is_atomic:
        return "hard"
    return "easy" if node_depth(node, by_id) <= 1 else "medium"


def rows_for_node(node: dict, parent_name: str, is_atomic: bool, by_id: dict) -> list:
    name = node["name_ar"]
    definition = (node.get("definition") or "")[:180]
    gid = node["concept_id"]
    ko_id = KO_PREFIX + gid
    diff = difficulty_for(node, by_id, is_atomic)
    rows = []
    for expert, template in TEMPLATES.items():
        rows.append({
            "input": template.format(name=name, parent=parent_name, definition=definition),
            "language": "mixed" if expert.startswith("O_") else "ar",
            "decidability": "decisive",
            "target_expert": expert,
            "allowed_experts": "",
            "difficulty": diff,
            "group_id": gid,
            "source_type": "curriculum_generated",
            "source_id": "src-ko-curriculum-v0.1",
            "record_type": "question_answer",
            "ko_node_ids": ko_id,
            "ood_flag": "false",
            "boundary_flag": "false",
            "quality_status": "draft",
            "notes": f"curriculum_generated from {ko_id}; lens={expert.split('_', 1)[0]}",
        })
    boundary = (node.get("boundary_test") or "").strip()
    if is_atomic and boundary:
        rows.append({
            "input": MISCONCEPTION_TEMPLATE.format(name=name, boundary=boundary[:160]),
            "language": "mixed",
            "decidability": "decisive",
            "target_expert": "R_structure_relations",
            "allowed_experts": "",
            "difficulty": "hard",
            "group_id": gid,
            "source_type": "curriculum_generated",
            "source_id": "src-ko-curriculum-v0.1",
            "record_type": "misconception_check",
            "ko_node_ids": ko_id,
            "ood_flag": "false",
            "boundary_flag": "true",
            "quality_status": "draft",
            "notes": f"curriculum_generated from {ko_id}; lens=R; boundary_test",
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Generate MK-7 questions from a Knowledge-OS curriculum tree.")
    ap.add_argument("--tree", required=True)
    ap.add_argument("--atomic", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Path-traversal guard: inputs and output must stay inside the working tree.
    base = Path.cwd().resolve()
    tree_path = _confine(Path(args.tree), base)
    atomic_path = _confine(Path(args.atomic), base)
    out_path = _confine(Path(args.out), base)

    with tree_path.open(encoding="utf-8") as f:
        tree = json.load(f)
    with atomic_path.open(encoding="utf-8") as f:
        atomic = json.load(f)

    tree_nodes = tree["nodes"]
    atomic_records = next(v for v in atomic.values() if isinstance(v, list) and v and isinstance(v[0], dict))
    by_id = {n["concept_id"]: n for n in tree_nodes}
    atomic_ids = {n["concept_id"] for n in atomic_records}
    merged = dict(by_id)
    for n in atomic_records:
        merged[n["concept_id"]] = n  # atomic metadata (boundary_test) overrides the tree copy

    all_rows, seen = [], set()
    for node in tree_nodes + [n for n in atomic_records if n["concept_id"] not in by_id]:
        cid = node["concept_id"]
        if cid in seen:
            continue
        seen.add(cid)
        node = merged.get(cid, node)  # prefer atomic metadata (boundary_test, atomicity)
        parent = by_id.get((node.get("parent_ids") or [""])[0])
        parent_name = (parent or {}).get("name_ar") or "المستوى الأعلى"
        all_rows.extend(rows_for_node(node, parent_name, cid in atomic_ids, by_id))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(all_rows)

    per_expert = {}
    for r in all_rows:
        per_expert[r["target_expert"]] = per_expert.get(r["target_expert"], 0) + 1
    print(f"nodes_covered={len(seen)} rows={len(all_rows)} (7 per node + boundary extras)")
    print("per_expert:", per_expert)
    print(f"output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
