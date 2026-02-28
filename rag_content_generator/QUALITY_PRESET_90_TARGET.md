# 90%-Target Quality Preset (Applied)

This preset is tuned for higher quality (not speed):

- Embedding model upgraded to `BAAI/bge-base-en-v1.5`
- Retrieval defaults raised to `top_k=10`
- Candidate pool increased (`candidate_multiplier=8`)
- Streamlit default mode set to quality (Fast Mode OFF)
- Streamlit default retrieval depth set to 10 (when quality mode)
- Chunking already tuned to ~260 words with overlap 90

## Important
To use the new embedding model, rebuild the store:

```bash
python rag_local.py ingest --docs . --store rag_store
```

Then run app:

```bash
python -m streamlit run app.py
```

For validation later:

```bash
python evaluate_rag.py --store rag_store --eval eval_questions_real.template.jsonl --top-k 10 --max-tokens 96 --out eval_report.json
```
