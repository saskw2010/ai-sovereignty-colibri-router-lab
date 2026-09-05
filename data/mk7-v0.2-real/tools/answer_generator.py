#!/usr/bin/env python3
"""Reference-anchored answer generator for mk7-v0.2-real records.

For each question, retrieves the most relevant passages from a local
references directory (stdlib TF-IDF), then asks a chat model to write the
answer STRICTLY from those passages. Every produced answer carries
provenance: source file + sha256 of the retrieved passages.

Governance:
  - Generated answers are `draft` until owner review (never auto-approved).
  - If no reference passage supports a question, the tool records
    `unanswered_no_evidence` instead of hallucinating an answer.

Backends:
  --backend dry_run   No model call: prints the retrieved evidence (pipeline test).
  --backend openai    Any OpenAI-compatible chat endpoint (e.g. a local server).

Usage:
  python answer_generator.py --dataset <dataset.jsonl> --references <dir> \
      --out answers.jsonl --backend dry_run [--top-k 3]
"""
import argparse
import hashlib
import json
import math
import socket
import sys
import urllib.parse
import ipaddress
import urllib.request
from collections import Counter
from pathlib import Path

CHUNK_SIZE = 900


def _confine(path: Path, base: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise SystemExit(f"Path escapes the working directory: {resolved}") from exc
    return resolved


def tokenize(text: str) -> list:
    return [w.lower() for w in "".join(c if c.isalnum() or c.isspace() else " " for c in text).split() if len(w) > 1]


def load_reference_chunks(ref_dir: Path) -> list:
    chunks = []
    for p in sorted(ref_dir.rglob("*")):
        if p.suffix.lower() in (".txt", ".md"):
            text = p.read_text(encoding="utf-8", errors="replace")
            file_sha = "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
            for i in range(0, len(text), CHUNK_SIZE):
                part = text[i:i + CHUNK_SIZE].strip()
                if len(part) > 80:
                    chunks.append({"file": p.name, "file_sha256": file_sha, "offset": i, "text": part})
        elif p.suffix.lower() == ".pdf":
            import fitz  # PyMuPDF — page-level chunks with page provenance
            file_sha = "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
            doc = fitz.open(p)
            with doc:
                for page_no, page in enumerate(doc, start=1):
                    part = page.get_text().strip()
                    if len(part) > 80:
                        chunks.append({"file": p.name, "file_sha256": file_sha,
                                       "offset": page_no, "page": page_no, "text": part})
    for ch in chunks:  # tokenize once, reuse across all questions
        ch["tokens"] = Counter(tokenize(ch["text"]))
    return chunks


# Cross-lingual retrieval bridge: Arabic lens terms -> English reference terms.
# Transparent heuristic (not machine translation) — keeps TF-IDF stdlib-only.
AR_EN_LEXICON = {
    "موضوع": ["subject", "concept", "topics"], "كيانات": ["entities", "objects", "sets"],
    "مراجع": ["references", "sources"], "أدلة": ["evidence", "proof", "theorems"],
    "يرتبط": ["related", "relation", "between"], "علاقات": ["relations", "relationships"],
    "بنية": ["structure", "structural"], "مجاور": ["adjacent", "neighboring"],
    "صحة": ["valid", "validity", "correct"], "مناهج": ["methods", "method"],
    "براهين": ["proof", "proofs", "prove"], "تحقق": ["verify", "verification", "check"],
    "نطبق": ["apply", "application"], "تطبيق": ["application", "apply", "use"],
    "مسألة": ["problem", "example", "exercise"], "خطوات": ["steps", "solution"],
    "تطورت": ["history", "development", "evolution"], "تاريخيًا": ["history", "historical"],
    "أثر": ["impact", "effect", "applications"], "إنسان": ["people", "human", "society"],
    "أصحاب المصلحة": ["stakeholders"], "الصناعة": ["industry", "engineering"],
    "أساسية": ["fundamental", "basic", "core"], "معايير": ["criteria", "standards"],
    "زمن": ["time"], "تطور": ["evolution", "development"], "تعريف": ["definition"],
    "الحياة": ["life", "real", "world"], "المفاهيم": ["concepts"], "الحدود": ["boundaries", "limits"],
    "قوة": ["strength", "quality"], "يستند": ["based", "reference"],
}


def expand_query(q_tokens: Counter) -> Counter:
    expanded = Counter(q_tokens)
    for text, weight in q_tokens.items():
        for en in AR_EN_LEXICON.get(text, []):
            expanded[en] += weight
    return expanded


def retrieve(question: str, chunks: list, top_k: int) -> list:
    q_tokens = expand_query(Counter(tokenize(question)))
    if not q_tokens:
        return []
    scored = []
    for ch in chunks:
        c_tokens = ch["tokens"]
        common = set(q_tokens) & set(c_tokens)
        if not common:
            continue
        score = sum(q_tokens[t] * (1 + math.log(1 + c_tokens[t])) for t in common)
        score /= math.sqrt(len(c_tokens))
        scored.append((score, ch))
    scored.sort(key=lambda sc: -sc[0])
    return [c for s, c in scored[:top_k] if s > 0]


def validate_endpoint(url: str, allow_remote: bool) -> str:
    """Validate scheme/host before any request; loopback allowed, remote only with flag."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise SystemExit(f"Unsupported endpoint URL: {url!r}")
    infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    addresses = sorted({info[4][0] for info in infos})
    if not addresses:
        raise SystemExit(f"Endpoint resolved to no addresses: {parsed.hostname!r}")
    loopback = all(ipaddress.ip_address(a).is_loopback for a in addresses)
    if not loopback and not allow_remote:
        raise SystemExit(f"Refusing non-loopback endpoint without --allow-remote: {parsed.hostname!r}")
    return url


SYSTEM_PROMPT = (
    "أنت مساعد دقيق. أجب عن السؤال بالعربية اعتمادًا حصريًا على المقاطع المرجعية "
    "المرفقة. إن لم تكفِ المقاطع، قل «الأدلة غير كافية» ولا تخترع معلومة."
)


def call_openai(endpoint: str, api_key: str, question: str, passages: list) -> str:
    evidence = "\n\n---\n\n".join(f"[{p['file']}]\n{p['text']}" for p in passages)
    body = json.dumps({
        "model": "local",
        "temperature": 0.2,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"المقاطع المرجعية:\n{evidence}\n\nالسؤال: {question}"},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # no redirects followed by Request by default policy
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser(description="Reference-anchored answer generation (drafts only).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--references", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["dry_run", "openai"], default="dry_run")
    ap.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N records (0 = all)")
    args = ap.parse_args()

    base = Path.cwd().resolve()
    dataset_path = _confine(Path(args.dataset), base)
    ref_dir = _confine(Path(args.references), base)
    out_path = _confine(Path(args.out), base)

    if args.backend == "openai":
        validate_endpoint(args.endpoint, args.allow_remote)

    chunks = load_reference_chunks(ref_dir)
    print(f"reference chunks loaded: {len(chunks)}")

    with dataset_path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        records = records[:args.limit]

    answered = unanswered = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            passages = retrieve(r["input"], chunks, args.top_k)
            item = {
                "record_id": r["id"],
                "question": r["input"],
                "target_expert": r["target_expert"],
                "status": "draft",
                "quality_status": "draft",
            }
            if not passages:
                item["answer_status"] = "unanswered_no_evidence"
                unanswered += 1
            else:
                evidence_sha = "sha256:" + hashlib.sha256(
                    json.dumps(passages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
                item["evidence"] = [{"file": p["file"], "file_sha256": p["file_sha256"],
                                     "offset": p["offset"]} for p in passages]
                item["evidence_sha256"] = evidence_sha
                if args.backend == "dry_run":
                    item["answer_status"] = "pending_model_call"
                    item["preview_passage"] = passages[0]["text"][:120]
                else:
                    item["answer"] = call_openai(args.endpoint, args.api_key, r["input"], passages)
                    item["answer_status"] = "generated"
                answered += 1
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"records={len(records)} with_evidence={answered} unanswered_no_evidence={unanswered}")
    print(f"output: {out_path} (all rows draft — owner review required)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
