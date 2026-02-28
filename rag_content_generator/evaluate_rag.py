import argparse
import json
import re
from pathlib import Path

from rag_local import run_query


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1(pred: str, gold: str) -> float:
    p = _norm(pred).split()
    g = _norm(gold).split()
    if not p or not g:
        return 0.0
    p_set, g_set = set(p), set(g)
    inter = len(p_set & g_set)
    if inter == 0:
        return 0.0
    precision = inter / max(1, len(p_set))
    recall = inter / max(1, len(g_set))
    return 2 * precision * recall / max(1e-9, (precision + recall))


def contains_expected(pred: str, expected_keywords):
    t = _norm(pred)
    hits = 0
    for k in expected_keywords:
        if _norm(k) in t:
            hits += 1
    return hits / max(1, len(expected_keywords))


def evaluate(store_dir: str, eval_file: str, top_k: int, fast_mode: bool, max_tokens: int, llm: str):
    rows = [json.loads(x) for x in Path(eval_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    results = []

    for i, row in enumerate(rows, 1):
        question = row["question"]
        gold = row.get("answer", "")
        keywords = row.get("keywords", [])
        content_type = row.get("content_type", "general")

        out = run_query(
            question,
            store_dir=store_dir,
            llm=llm,
            top_k=top_k,
            content_type=content_type,
            max_tokens=max_tokens,
            fast_mode=fast_mode,
        )
        pred = out.get("answer", "")

        f1 = token_f1(pred, gold) if gold else None
        kw = contains_expected(pred, keywords) if keywords else None

        results.append({
            "id": row.get("id", i),
            "question": question,
            "pred": pred,
            "gold": gold,
            "token_f1": f1,
            "keyword_hit_rate": kw,
        })

    f1_vals = [r["token_f1"] for r in results if r["token_f1"] is not None]
    kw_vals = [r["keyword_hit_rate"] for r in results if r["keyword_hit_rate"] is not None]

    summary = {
        "samples": len(results),
        "avg_token_f1": round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else None,
        "avg_keyword_hit_rate": round(sum(kw_vals) / len(kw_vals), 4) if kw_vals else None,
    }

    return summary, results


def main():
    ap = argparse.ArgumentParser(description="Evaluate RAG quality on a small QA set (JSONL).")
    ap.add_argument("--store", required=True, help="Path to rag_store")
    ap.add_argument("--eval", required=True, help="Path to eval_questions.jsonl")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fast-mode", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=128, help="Lower = faster eval")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct", help="Local HF model name")
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    summary, rows = evaluate(args.store, args.eval, args.top_k, args.fast_mode, args.max_tokens, args.llm)

    report = {"summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== RAG Eval Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Saved report: {args.out}")


if __name__ == "__main__":
    main()
