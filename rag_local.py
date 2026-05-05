"""
Minimal, API-free RAG pipeline for local document-grounded generation.

Features:
- Ingest text/PDF files from a folder
- Chunk text, embed with SentenceTransformers
- Store/retrieve with FAISS
- Generate grounded answers with a local HF causal LM

Usage (examples):
  python rag_local.py ingest --docs ./docs --store ./rag_store
  python rag_local.py query  --question "Summarize the paper" --store ./rag_store
  python rag_local.py query  --question "Give me a LinkedIn post" --store ./rag_store --max_tokens 256
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import List, Tuple

# Disable TensorFlow integrations in transformers to avoid tf-keras / Keras 3 issues
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Light defaults; swap with larger local models if you have GPU/VRAM
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
DEFAULT_LLM = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_IMAGE_MODEL = "stabilityai/sd-turbo"

# Lightweight in-process caches to avoid repeated model loading/log spam
_RERANKER_CACHE = {}
_GENERATOR_CACHE = {}

# Content type templates for different generation tasks
CONTENT_TEMPLATES = {
    "summary": (
        "Based on the following document context, write a SHORT summary in 2-4 sentences only.\n"
        "- Describe what the document is and its purpose.\n"
        "- Mention the key sections or topics covered (e.g., introduction, purpose, sample content, conclusion).\n"
        "- Do NOT include source names, chunk numbers, [Source:...], or citation notes in your summary.\n"
        "- Do NOT repeat information or add generic phrases like 'Additionally', 'Overall', 'By adhering to these guidelines'.\n"
        "- Write in a clear, professional tone. Be concise like a document abstract.\n\n"
        "Context:\n{context}\n\nSummary:"
    ),
    "blog_intro": (
        "Write an engaging and compelling blog post introduction based on the provided context.\n"
        "Make it hook the reader, set up the topic, and create interest.\n"
        "Keep it professional yet accessible.\n\n"
        "Context:\n{context}\n\nBlog Introduction:"
    ),
    "linkedin_post": (
        "Create a professional LinkedIn post based on the document context.\n"
        "Keep it concise (2-3 paragraphs), professional, engaging, and shareable.\n"
        "Include relevant insights and actionable takeaways.\n"
        "Add appropriate professional hashtags at the end.\n\n"
        "Context:\n{context}\n\nLinkedIn Post:"
    ),
    "general": (
        "You are a grounded assistant. Use ONLY the provided context to answer.\n"
        "If context is insufficient or ambiguous, reply exactly: 'I don't know based on the provided documents.'\n"
        "Do not invent facts, names, numbers, citations, or commands.\n"
        "Prefer exact terms from context (file names, model names, commands, frameworks) when available.\n"
        "Answer in 1-3 concise sentences.\n"
        "Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    )
}


def load_text_from_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            import pypdf  # lazy import to keep core deps small
        except ImportError as e:
            raise ImportError("Install pypdf to read PDFs: pip install pypdf") from e
        with path.open("rb") as f:
            reader = pypdf.PdfReader(f)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
    else:
        return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 90) -> List[str]:
    # Paragraph-aware chunking with sliding overlap for better retrieval quality
    MAX_TEXT_LENGTH = 500000  # ~500K characters max
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
        print(f"Warning: Text truncated to {MAX_TEXT_LENGTH} characters to prevent memory issues")

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    words: List[str] = []
    chunks: List[str] = []
    MAX_CHUNKS = 1000

    for para in paragraphs:
        p_words = para.split()
        if not p_words:
            continue

        # If a single paragraph is too large, split it safely
        while len(p_words) > chunk_size:
            head = p_words[:chunk_size]
            p_words = p_words[chunk_size - overlap:]
            chunk = " ".join(head).strip()
            if chunk:
                chunks.append(chunk)
                if len(chunks) >= MAX_CHUNKS:
                    print(f"Warning: Limited to {MAX_CHUNKS} chunks to prevent memory issues")
                    return chunks

        # Accumulate paragraphs into chunks
        if len(words) + len(p_words) <= chunk_size:
            words.extend(p_words)
        else:
            chunk = " ".join(words).strip()
            if chunk:
                chunks.append(chunk)
                if len(chunks) >= MAX_CHUNKS:
                    print(f"Warning: Limited to {MAX_CHUNKS} chunks to prevent memory issues")
                    return chunks
            words = (words[-overlap:] if overlap > 0 else []) + p_words

    if words:
        chunk = " ".join(words).strip()
        if chunk:
            chunks.append(chunk)

    if len(chunks) > MAX_CHUNKS:
        print(f"Warning: Limited to {MAX_CHUNKS} chunks to prevent memory issues")
        chunks = chunks[:MAX_CHUNKS]

    return chunks


def clean_text(text: str) -> str:
    # Preserve paragraph breaks, normalize spaces/tabs, collapse excessive blank lines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_store(docs_dir: str, store_dir: str, embed_model: str = DEFAULT_EMBED_MODEL) -> None:
    docs_path = Path(docs_dir)
    store_path = Path(store_dir)
    store_path.mkdir(parents=True, exist_ok=True)

    # Allow passing either a folder or a single file
    if docs_path.is_file():
        files = [docs_path] if docs_path.suffix.lower() in {".txt", ".md", ".pdf"} else []
    else:
        files = [p for p in docs_path.glob("**/*") if p.suffix.lower() in {".txt", ".md", ".pdf"}]
    if not files:
        raise FileNotFoundError(f"No text/PDF files found in {docs_dir}")

    # Check file sizes before processing
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
    for f in files:
        file_size = f.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File {f.name} is too large ({file_size / 1024 / 1024:.1f}MB). Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB. Please split the file or use a smaller document.")

    encoder = SentenceTransformer(embed_model)
    all_chunks: List[str] = []
    meta: List[Tuple[str, int]] = []

    for f in files:
        print(f"Processing {f.name}...")
        try:
            raw = load_text_from_file(f)
            if not raw:
                print(f"Warning: {f.name} appears to be empty, skipping...")
                continue
            
            # Limit text size before processing
            if len(raw) > 500000:  # ~500K characters
                print(f"Warning: {f.name} is very large, truncating to 500K characters...")
                raw = raw[:500000]
            
            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned)
            
            if not chunks:
                print(f"Warning: No chunks created from {f.name}, skipping...")
                continue
                
            for i, ch in enumerate(chunks):
                all_chunks.append(ch)
                meta.append((str(f), i))
            
            print(f"Created {len(chunks)} chunks from {f.name}")
        except MemoryError:
            raise MemoryError(f"Out of memory while processing {f.name}. Try with a smaller file.")
        except Exception as e:
            print(f"Error processing {f.name}: {e}")
            raise

    if not all_chunks:
        raise RuntimeError("No chunks produced from documents.")

    # Process embeddings in smaller batches to prevent memory issues
    print(f"Encoding {len(all_chunks)} chunks...")
    batch_size = min(32, len(all_chunks))  # Smaller batch size for memory efficiency
    embeddings_list = []
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        batch_embeddings = encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False, 
                                         batch_size=16, normalize_embeddings=True)
        embeddings_list.append(batch_embeddings)
        print(f"Processed {min(i+batch_size, len(all_chunks))}/{len(all_chunks)} chunks...")
    
    # Concatenate all embeddings
    embeddings = np.vstack(embeddings_list)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # Clear memory
    del embeddings_list
    import gc
    gc.collect()

    faiss.write_index(index, str(store_path / "index.faiss"))
    with (store_path / "chunks.json").open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f)
    with (store_path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)
    with (store_path / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"embed_model": embed_model}, f)

    print(f"Stored {len(all_chunks)} chunks from {len(files)} files into {store_dir}")


def load_store(store_dir: str):
    store_path = Path(store_dir)
    index = faiss.read_index(str(store_path / "index.faiss"))
    chunks = json.loads((store_path / "chunks.json").read_text(encoding="utf-8"))
    meta = json.loads((store_path / "meta.json").read_text(encoding="utf-8"))
    cfg = json.loads((store_path / "config.json").read_text(encoding="utf-8"))
    encoder = SentenceTransformer(cfg.get("embed_model", DEFAULT_EMBED_MODEL))
    return index, chunks, meta, encoder


def retrieve(query: str, index, encoder, chunks: List[str], meta: List[Tuple[str, int]], top_k: int = 8,
             rerank_model: str = DEFAULT_RERANK_MODEL, candidate_multiplier: int = 6, use_reranker: bool = True):
    # Stage 1: fast dense retrieval
    candidates_k = max(top_k, top_k * candidate_multiplier)
    q_emb = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb, candidates_k)

    candidates = []
    for score, idx in zip(scores[0], idxs[0]):
        candidates.append({"score": float(score), "chunk": chunks[idx], "source": meta[idx][0], "chunk_id": meta[idx][1]})

    # Stage 2: cross-encoder reranking for better accuracy
    if use_reranker:
        try:
            reranker = _RERANKER_CACHE.get(rerank_model)
            if reranker is None:
                reranker = CrossEncoder(rerank_model)
                _RERANKER_CACHE[rerank_model] = reranker

            pairs = [[query, c["chunk"]] for c in candidates]
            rerank_scores = reranker.predict(pairs, show_progress_bar=False)
            for c, rr in zip(candidates, rerank_scores):
                c["rerank_score"] = float(rr)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception as e:
            print(f"Warning: reranker unavailable ({e}); using dense retrieval order.")

    return candidates[:top_k]


def make_generator(model_name: str, max_new_tokens: int):
    """Create/cached text generation pipeline with proper device handling and quieter logs."""
    import torch

    cache_key = model_name
    if cache_key in _GENERATOR_CACHE:
        return _GENERATOR_CACHE[cache_key]

    tok = AutoTokenizer.from_pretrained(model_name)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model with proper device handling to avoid meta tensor issues
    try:
        # For CPU, load directly without device_map
        if device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
        else:
            # For GPU, try with device_map first
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            except Exception:
                # Fallback: load and move manually
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
    except Exception as e:
        # Final fallback: simple loading
        print(f"Warning: {e}. Using simple loading method...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32
        )
        if device == "cpu":
            model = model.to(device)

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device=0 if device == "cuda" else -1  # -1 for CPU, 0+ for GPU
    )
    _GENERATOR_CACHE[cache_key] = gen
    return gen


def format_prompt(question: str, retrieved: List[dict], content_type: str = "general") -> str:
    """Format prompt based on content type."""
    context_blocks = []
    for r in retrieved:
        context_blocks.append(f"[Source: {Path(r['source']).name} | chunk {r['chunk_id']}] {r['chunk']}")
    context = "\n".join(context_blocks)
    
    template = CONTENT_TEMPLATES.get(content_type, CONTENT_TEMPLATES["general"])
    if content_type == "general":
        return template.format(question=question, context=context)
    else:
        return template.format(context=context)


def _extract_keywords(text: str, max_n: int = 8) -> List[str]:
    stop = {
        "the","and","for","with","from","that","this","your","into","using","use","are","was","were","have","has","had",
        "about","based","document","documents","content","system","project","data","more","than","into","over","under"
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    freq = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:max_n]]


def _format_hashtags(retrieved: List[dict], max_tags: int = 10) -> str:
    text = " ".join([r.get("chunk", "") for r in retrieved[:4]])
    kws = _extract_keywords(text, max_n=max_tags)
    tags = []
    for k in kws:
        k2 = re.sub(r"[^a-z0-9]", "", k.lower())
        if not k2:
            continue
        tags.append(f"#{k2}")
    # Ensure strong baseline tags
    for base in ["#AI", "#RAG", "#MachineLearning", "#ContentCreation", "#Productivity"]:
        if base.lower() not in [t.lower() for t in tags]:
            tags.append(base)
    return " ".join(tags[:max_tags])


def extractive_summary(retrieved: List[dict], max_sentences: int = 4) -> str:
    """Fast, structured summary without LLM generation, with chunk attribution."""
    sent_rows = []
    for r in retrieved:
        src = Path(r.get("source", "unknown")).name
        cid = r.get("chunk_id", "?")
        chunk = r.get("chunk", "")
        for s in re.split(r"(?<=[.!?])\s+", chunk):
            s = s.strip()
            if len(s) > 40:
                sent_rows.append({"sent": s, "src": src, "cid": cid})

    if not sent_rows:
        return "No content available to summarize."

    # Deduplicate near-identical sentences
    seen = set()
    deduped = []
    for row in sent_rows:
        key = re.sub(r"\W+", " ", row["sent"].lower()).strip()
        key = " ".join(key.split()[:16])
        if key and key not in seen:
            seen.add(key)
            deduped.append(row)

    # Prioritize informative sentences with numbers/keywords
    keywords = {"model", "rag", "accuracy", "hallucination", "precision", "faithfulness", "summary", "retrieval", "citation"}
    def sent_score(row: dict) -> float:
        sent = row["sent"]
        low = sent.lower()
        has_num = 1.0 if re.search(r"\d", sent) else 0.0
        kw = sum(1 for k in keywords if k in low)
        length = min(len(sent), 240) / 240.0
        return 1.2 * has_num + 0.8 * kw + 0.6 * length

    ranked = sorted(deduped, key=sent_score, reverse=True)
    picked = ranked[:max_sentences]

    overview = picked[0] if picked else None
    bullets = picked[1:max_sentences]

    out = ["📌 **Executive Summary**"]
    if overview:
        out.append(f"✨ **Overview:** {overview['sent']} _(Source: {overview['src']}#chunk{overview['cid']})_")
    if bullets:
        out.append("📚 **Key Points**")
        for b in bullets:
            out.append(f"- ✅ {b['sent']} _(Source: {b['src']}#chunk{b['cid']})_")

    # compact chunk list
    uniq = []
    seen_refs = set()
    for p in picked:
        ref = f"{p['src']}#chunk{p['cid']}"
        if ref not in seen_refs:
            seen_refs.add(ref)
            uniq.append(ref)
    if uniq:
        out.append("🧩 **Chunks used:** " + ", ".join(uniq))

    return "\n".join(out)


def fast_blog_intro(retrieved: List[dict]) -> str:
    s = extractive_summary(retrieved, max_sentences=3)
    core = [ln for ln in s.splitlines() if ln.startswith("- ✅")]
    hook = "🚀 Want to turn complex documents into clear, actionable content in minutes?"
    body = core[0].replace("- ✅ ", "") if core else "This approach converts raw documents into concise insights and share-ready outputs."
    return f"{hook}\n\n{body}\n\nIn this post, we break down a practical workflow you can apply immediately."


def fast_linkedin_post(retrieved: List[dict]) -> str:
    s = extractive_summary(retrieved, max_sentences=4)
    lines = [ln for ln in s.splitlines() if ln.startswith("- ✅")]
    p1 = "✨ Turning dense documents into decision-ready insights is a superpower in modern teams."
    p2 = lines[0].replace("- ✅ ", "") if lines else "A retrieval-first workflow improves quality, consistency, and trust in generated content."
    p3 = "If you care about speed + reliability, this pattern is worth adopting now."
    tags = _format_hashtags(retrieved, max_tags=12)
    return f"{p1}\n\n{p2}\n\n{p3}\n\n{tags}"


def run_query(question: str, store_dir: str = None, llm: str = DEFAULT_LLM, top_k: int = 8, max_tokens: int = 256, content_type: str = "general", 
              index=None, chunks=None, meta=None, encoder=None, fast_mode: bool = False):
    """Run RAG query with optional content type.
    
    Can use either store_dir (loads from disk) or pre-loaded data (index, chunks, meta, encoder).
    If pre-loaded data is provided, it will be used instead of loading from disk.
    """
    # Use pre-loaded data if available, otherwise load from disk
    if index is not None and chunks is not None and meta is not None and encoder is not None:
        # Use provided data directly
        pass
    elif store_dir:
        store_path = Path(store_dir)
        if not store_path.exists():
            raise FileNotFoundError(f"Store directory not found: {store_dir}")
        index, chunks, meta, encoder = load_store(store_dir)
    else:
        raise ValueError("Either store_dir or pre-loaded data (index, chunks, meta, encoder) must be provided.")
    
    # Fast mode disables reranker by default to cut latency
    retrieved = retrieve(question, index, encoder, chunks, meta, top_k=top_k, use_reranker=not fast_mode)

    if fast_mode:
        if content_type == "summary":
            answer = extractive_summary(retrieved, max_sentences=5)
            return {"answer": answer, "retrieved": retrieved, "full_output": answer}
        if content_type == "general":
            answer = extractive_summary(retrieved, max_sentences=3)
            return {"answer": answer, "retrieved": retrieved, "full_output": answer}
        if content_type == "blog_intro":
            answer = fast_blog_intro(retrieved)
            return {"answer": answer, "retrieved": retrieved, "full_output": answer}
        if content_type == "linkedin_post":
            answer = fast_linkedin_post(retrieved)
            return {"answer": answer, "retrieved": retrieved, "full_output": answer}

    gen = make_generator(llm, max_tokens)
    prompt = format_prompt(question, retrieved, content_type=content_type)
    out = gen(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=True,
    )[0]["generated_text"]
    
    # Extract only the generated part (remove the prompt)
    if prompt in out:
        # Remove the prompt from the output
        answer = out.replace(prompt, "").strip()
    else:
        # If prompt not found, try to extract after common markers
        answer = out
        for marker in ["Answer:", "Summary:", "Blog Introduction:", "LinkedIn Post:"]:
            if marker in answer:
                answer = answer.split(marker)[-1].strip()
                break
    
    # Clean up any remaining prompt artifacts
    if "Context:" in answer:
        answer = answer.split("Context:")[0].strip()
    if "Question:" in answer:
        answer = answer.split("Question:")[-1].strip()
    
    # Remove any leaked source/chunk references (e.g. [Source: file.pdf | chunk 1])
    answer = re.sub(r'\[Source:[^\]]+\]', '', answer)
    answer = re.sub(r'\(Note:\s*Ensure proper citation[^)]*\)', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\s{2,}', ' ', answer).strip()
    
    return {"answer": answer, "retrieved": retrieved, "full_output": out}


def generate_image_from_context(retrieved: List[dict], model_name: str = DEFAULT_IMAGE_MODEL, num_inference_steps: int = 20):
    """Generate image prompt from retrieved context and create image using Stable Diffusion."""
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise ImportError("Install diffusers and torch for image generation: pip install diffusers torch") from e

    # Build a cleaner prompt from top chunks
    context_text = " ".join([r.get('chunk', '')[:220] for r in retrieved[:3]])
    context_text = re.sub(r"\s+", " ", context_text).strip()
    key_words = [w for w in context_text.split() if len(w) > 3][:36]
    core = " ".join(key_words) if key_words else "document concept"

    positive_prompt = (
        f"high quality professional illustration, clean composition, informative visual, {core}"
    )
    negative_prompt = "blurry, distorted, text artifacts, watermark, low quality, noisy"

    # Determine device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)

        with torch.no_grad():
            steps = max(1, min(8, int(num_inference_steps)))
            result = pipe(
                positive_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=0.0,
                height=384,
                width=384,
            )
        image = result.images[0]
        return image, positive_prompt
    except Exception:
        # Fallback: create a readable placeholder image instead of failing hard
        image = Image.new("RGB", (1024, 576), color=(245, 248, 252))
        draw = ImageDraw.Draw(image)
        title = "Image generation fallback"
        line1 = "Stable Diffusion failed on this machine/model."
        line2 = "Using extracted visual prompt instead:"
        prompt_preview = (positive_prompt[:180] + "...") if len(positive_prompt) > 180 else positive_prompt
        draw.text((30, 30), title, fill=(20, 40, 80))
        draw.text((30, 90), line1, fill=(40, 40, 40))
        draw.text((30, 125), line2, fill=(40, 40, 40))
        draw.text((30, 170), prompt_preview, fill=(10, 70, 120))
        return image, positive_prompt


def main():
    parser = argparse.ArgumentParser(description="Local RAG demo (API-free).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest documents and build FAISS store.")
    p_ingest.add_argument("--docs", required=True, help="Folder with .txt/.md/.pdf files")
    p_ingest.add_argument("--store", required=True, help="Output folder for FAISS + chunks")
    p_ingest.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="SentenceTransformer model name")

    p_query = sub.add_parser("query", help="Query the store with RAG generation.")
    p_query.add_argument("--question", required=True, help="User question")
    p_query.add_argument("--store", required=True, help="Store folder created by ingest")
    p_query.add_argument("--llm", default=DEFAULT_LLM, help="Local HF causal LM name")
    p_query.add_argument("--top-k", type=int, default=8, help="Retrieved chunks")
    p_query.add_argument("--max-tokens", type=int, default=256, help="Generation length")
    p_query.add_argument("--content-type", choices=["general", "summary", "blog_intro", "linkedin_post"], 
                        default="general", help="Type of content to generate")

    args = parser.parse_args()

    if args.cmd == "ingest":
        build_store(args.docs, args.store, embed_model=args.embed_model)
    elif args.cmd == "query":
        result = run_query(args.question, args.store, llm=args.llm, top_k=args.top_k, 
                          max_tokens=args.max_tokens, content_type=args.content_type)
        print("\n=== Answer ===")
        print(result["answer"])
        print("\n=== Retrieved Chunks ===")
        for r in result["retrieved"]:
            print(f"{r['score']:.3f} :: {Path(r['source']).name}#chunk{r['chunk_id']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

