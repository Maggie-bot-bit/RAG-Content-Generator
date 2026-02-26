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
from pathlib import Path
from typing import List, Tuple

# Disable TensorFlow integrations in transformers to avoid tf-keras / Keras 3 issues
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# Light defaults; swap with larger local models if you have GPU/VRAM
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_IMAGE_MODEL = "runwayml/stable-diffusion-v1-5"

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
        "You are a helpful assistant. Use ONLY the provided context to answer.\n"
        "If the context is insufficient, say you don't know.\n"
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


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    # Simple whitespace chunking with memory optimization
    # Limit total text size to prevent memory issues
    MAX_TEXT_LENGTH = 500000  # ~500K characters max
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
        print(f"Warning: Text truncated to {MAX_TEXT_LENGTH} characters to prevent memory issues")
    
    words = text.split()
    # Limit number of chunks to prevent memory overflow
    MAX_CHUNKS = 1000
    chunks = []
    start = 0
    chunk_count = 0
    
    while start < len(words) and chunk_count < MAX_CHUNKS:
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
            chunk_count += 1
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(words):
            break
    
    if chunk_count >= MAX_CHUNKS:
        print(f"Warning: Limited to {MAX_CHUNKS} chunks to prevent memory issues")
    
    return chunks


def clean_text(text: str) -> str:
    # Normalize whitespace and strip control chars
    return re.sub(r"\s+", " ", text).strip()


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


def retrieve(query: str, index, encoder, chunks: List[str], meta: List[Tuple[str, int]], top_k: int = 5):
    q_emb = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        results.append({"score": float(score), "chunk": chunks[idx], "source": meta[idx][0], "chunk_id": meta[idx][1]})
    return results


def make_generator(model_name: str, max_new_tokens: int):
    """Create a text generation pipeline with proper device handling to avoid meta tensor errors."""
    import torch
    
    tok = AutoTokenizer.from_pretrained(model_name)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model with proper device handling to avoid meta tensor issues
    try:
        # For CPU, load directly without device_map
        if device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
        else:
            # For GPU, try with device_map first
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            except Exception:
                # Fallback: load and move manually
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
    except Exception as e:
        # Final fallback: simple loading
        print(f"Warning: {e}. Using simple loading method...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        if device == "cpu":
            model = model.to(device)
    
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
        device=0 if device == "cuda" else -1  # -1 for CPU, 0+ for GPU
    )


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


def run_query(question: str, store_dir: str = None, llm: str = DEFAULT_LLM, top_k: int = 5, max_tokens: int = 256, content_type: str = "general", 
              index=None, chunks=None, meta=None, encoder=None):
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
    
    retrieved = retrieve(question, index, encoder, chunks, meta, top_k=top_k)
    gen = make_generator(llm, max_tokens)
    prompt = format_prompt(question, retrieved, content_type=content_type)
    out = gen(prompt)[0]["generated_text"]
    
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
        from PIL import Image
    except ImportError as e:
        raise ImportError("Install diffusers and torch for image generation: pip install diffusers torch") from e
    
    # Extract key phrases from retrieved chunks to create image prompt
    context_text = " ".join([r['chunk'][:200] for r in retrieved[:3]])
    # Create a concise, image-friendly prompt
    words = context_text.split()[:30]  # Limit to first 30 words
    image_prompt = f"Professional illustration, digital art, concept: {' '.join(words)}"
    
    # Determine device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load model (this will download on first run)
    print(f"Loading image generation model: {model_name} on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        safety_checker=None,  # Disable safety checker for faster generation
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    
    # Generate image
    print(f"Generating image with prompt: {image_prompt[:100]}...")
    with torch.no_grad():
        image = pipe(image_prompt, num_inference_steps=num_inference_steps).images[0]
    
    return image, image_prompt


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
    p_query.add_argument("--top-k", type=int, default=5, help="Retrieved chunks")
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

