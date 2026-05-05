# Fine-tune Steps Applied (Quality-first)

## Changes applied in code
- Default local LLM upgraded:
  - `Qwen/Qwen2.5-0.5B-Instruct` → `Qwen/Qwen2.5-1.5B-Instruct`
- Chunking tuned for better retrieval precision:
  - `chunk_size=240`
  - `overlap=100`

## Run now
```bash
python rag_local.py ingest --docs . --store rag_store
python evaluate_rag.py --store rag_store --eval eval_questions_real.jsonl --top-k 10 --max-tokens 96 --out eval_report.json
python -m streamlit run app.py
```

## If runtime is too slow
- use `--max-tokens 64` in eval
- temporarily use `Qwen/Qwen2.5-0.5B-Instruct`
- keep reranker ON for quality
