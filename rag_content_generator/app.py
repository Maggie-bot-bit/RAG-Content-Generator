"""
Streamlit frontend for AI-Based Document Understanding and Content Generation System using RAG.

Features:
- Document upload and processing
- Generate summaries, blog posts, LinkedIn posts
- Image generation from document context
- All processing done locally (API-free)
"""

import streamlit as st
from pathlib import Path
import tempfile
import os
from rag_local import build_store, run_query, load_store, retrieve, generate_image_from_context
from linkedin_integration import (
    get_config as li_get_config,
    config_ready as li_config_ready,
    build_auth_url as li_build_auth_url,
    exchange_code_for_token as li_exchange_code,
    get_access_token as li_get_access_token,
    post_text as li_post_text,
    new_state as li_new_state,
)
import json

# Page configuration
st.set_page_config(
    page_title="RAG Content Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional aesthetic UI theme (Light/Dark)
if 'theme_mode' not in st.session_state:
    st.session_state['theme_mode'] = 'Light'

def apply_theme(theme_mode: str):
    is_dark = theme_mode == 'Dark'
    if is_dark:
        bg = '#0b1020'; card = '#111a2e'; ink = '#e6edff'; muted = '#94a3b8'; line = '#24324d'
        brand = '#22d3ee'; brand2 = '#a78bfa'; app_grad = 'radial-gradient(circle at top right, #0b1228 0%, #0b1020 45%, #090f1d 100%)'
        side_grad = 'linear-gradient(180deg, #0f172a 0%, #0b1220 100%)'
        code_bg = '#0f172a'; code_ink = '#e2e8f0'
    else:
        bg = '#f4f7fb'; card = '#ffffff'; ink = '#0f172a'; muted = '#475569'; line = '#e2e8f0'
        brand = '#2563eb'; brand2 = '#7c3aed'; app_grad = 'radial-gradient(circle at top right, #eef2ff 0%, #f4f7fb 40%, #f8fafc 100%)'
        side_grad = 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)'
        code_bg = '#f8fafc'; code_ink = '#0f172a'

    st.markdown(f"""
    <style>
        :root {{ --bg:{bg}; --card:{card}; --ink:{ink}; --muted:{muted}; --brand:{brand}; --brand2:{brand2}; --line:{line}; }}
        html, body, [data-testid="stAppViewContainer"], .stApp, [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{
            background: {app_grad} !important;
            background-color: var(--bg) !important;
        }}
        [data-testid="stHeader"] {{ background: transparent !important; }}
        section[data-testid="stSidebar"] {{ background: {side_grad} !important; border-right: 1px solid var(--line); }}
        .main-header {{ font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em;
            background: linear-gradient(90deg, var(--brand), var(--brand2)); -webkit-background-clip: text;
            -webkit-text-fill-color: transparent; text-align: center; margin-bottom: .4rem; }}
        .sub-header {{ font-size: 1.06rem; color: var(--muted) !important; text-align: center; margin-bottom: 1.2rem; }}
        .stButton>button {{ width: 100%; border: 0 !important; border-radius: 12px !important;
            background: linear-gradient(90deg, var(--brand), var(--brand2)) !important; color: #fff !important;
            font-weight: 700 !important; box-shadow: 0 8px 18px rgba(37,99,235,.25); }}
        /* Lighter style for link buttons like Connect LinkedIn */
        [data-testid="stLinkButton"] a {{
            width: 100%;
            display: inline-flex;
            justify-content: center;
            align-items: center;
            border-radius: 12px;
            padding: .55rem .8rem;
            text-decoration: none !important;
            font-weight: 700;
            border: 1px solid var(--line);
            background: linear-gradient(90deg, #e0ecff, #ede9fe) !important;
            color: #1e293b !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: .35rem; }}
        .stTabs [data-baseweb="tab"] {{ border-radius: 10px; padding: .5rem .85rem; }}
        .info-box {{ padding: 1.1rem 1.2rem; border-radius: 14px; background: var(--card);
            border: 1px solid var(--line); box-shadow: 0 10px 25px rgba(15,23,42,.10); color: var(--ink) !important; margin: 0.8rem 0; }}
        .output-card {{ background: var(--card); color: var(--ink); border: 1px solid var(--line); border-radius: 12px; padding: 1rem; line-height: 1.65; }}
        .footer-card {{ background: var(--card); color: var(--ink); border: 1px solid var(--line); border-radius: 10px; }}
        p, div, span, h1, h2, h3, h4, h5, h6, label {{ color: var(--ink) !important; }}
        .stAlert {{ border-radius: 12px !important; }}
        pre {{ background-color: {code_bg}; color: {code_ink}; border: 1px solid var(--line); border-radius: 10px; }}

        /* File uploader styling */
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] > div,
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploaderDropzone"] * {{
            background: var(--card) !important;
            color: var(--ink) !important;
            border-color: var(--line) !important;
        }}
    </style>
    """, unsafe_allow_html=True)

apply_theme(st.session_state['theme_mode'])

# Initialize session state
if 'store_dir' not in st.session_state:
    st.session_state['store_dir'] = None
if 'store_loaded' not in st.session_state:
    st.session_state['store_loaded'] = False
if 'index' not in st.session_state:
    st.session_state['index'] = None
if 'chunks' not in st.session_state:
    st.session_state['chunks'] = None
if 'meta' not in st.session_state:
    st.session_state['meta'] = None
if 'encoder' not in st.session_state:
    st.session_state['encoder'] = None
if 'li_last_post' not in st.session_state:
    st.session_state['li_last_post'] = ""
if 'li_state' not in st.session_state:
    st.session_state['li_state'] = ""
if 'li_auth_msg' not in st.session_state:
    st.session_state['li_auth_msg'] = ""
if 'li_connected' not in st.session_state:
    st.session_state['li_connected'] = False

# Handle LinkedIn OAuth callback (when redirected back with ?code=...)
try:
    q = st.query_params
    if 'code' in q:
        cfg = li_get_config()
        if li_config_ready(cfg):
            code_val = q.get('code')
            if isinstance(code_val, list):
                code_val = code_val[0]
            if code_val and code_val != st.session_state.get('li_last_code'):
                li_exchange_code(cfg, code_val)
                st.session_state['li_last_code'] = code_val
                st.session_state['li_connected'] = True
                st.session_state['li_auth_msg'] = "✅ LinkedIn connected successfully."
except Exception as _e:
    st.session_state['li_auth_msg'] = f"❌ LinkedIn auth failed: {_e}"

# Header
st.markdown('<h1 class="main-header">🤖 RAG Content Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional document intelligence for summaries, blog drafts, LinkedIn posts, and visual outputs.</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box" style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;">
  <div><b>⚡ Fast Mode</b><br><span style='color:var(--muted)'>Low-latency content generation</span></div>
  <div><b>🔒 Local-first</b><br><span style='color:var(--muted)'>Privacy-friendly processing</span></div>
  <div><b>📎 Traceable</b><br><span style='color:var(--muted)'>Chunk/source-grounded outputs</span></div>
</div>
""", unsafe_allow_html=True)

# Sidebar for document management
with st.sidebar:
    st.header("📁 Document Management")

    st.subheader("🎨 Theme")
    selected_theme = st.radio("Choose theme", ["Light", "Dark"], index=0 if st.session_state.get('theme_mode', 'Light') == 'Light' else 1, horizontal=True)
    if selected_theme != st.session_state.get('theme_mode'):
        st.session_state['theme_mode'] = selected_theme
        st.rerun()

    st.subheader("⚡ Performance")
    fast_mode = st.checkbox("Enable Fast Mode (recommended)", value=True, help="Uses quick extractive summaries for much faster results on low-end machines.")
    fast_image_mode = st.checkbox("⚡ Fast Image Mode (Prompt only)", value=False, help="If enabled, skips local image rendering and returns an instant prompt.")
    default_top_k = 3 if fast_mode else 8

    st.subheader("🔗 LinkedIn")
    li_cfg = li_get_config()
    if li_config_ready(li_cfg):
        if st.session_state.get('li_auth_msg'):
            st.caption(st.session_state['li_auth_msg'])
        state = st.session_state.get('li_state') or li_new_state()
        st.session_state['li_state'] = state
        auth_url = li_build_auth_url(li_cfg, state)
        st.link_button("Connect LinkedIn", auth_url, use_container_width=True)
        if st.session_state.get('li_connected'):
            st.success("Connected")

            # Always-available sidebar post box (more reliable than tab button)
            st.caption("LinkedIn Post (editable)")
            li_draft = st.text_area(
                "",
                value=st.session_state.get('li_last_post', ''),
                key='li_sidebar_draft',
                height=120,
                placeholder='Generate a LinkedIn post, edit here, then click Post Now'
            )
            if st.button("🚀 Post Now", key="sb_btn_li_post", use_container_width=True):
                token = li_get_access_token()
                if not token:
                    st.error("Not connected. Click Connect LinkedIn again.")
                elif not (li_draft or '').strip():
                    st.error("Draft is empty. Generate/edit content first.")
                else:
                    res = li_post_text(token, li_draft.strip())
                    if res.get('ok'):
                        st.success("Posted successfully ✅")
                    else:
                        st.error(f"Post failed ({res.get('status')}): {res.get('body')}")
        else:
            st.info("Not connected")
    else:
        st.caption("Add LinkedIn credentials in .env")

    # Option 1: Upload files
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload one or more documents to process"
    )
    
    if st.button("🔄 Process Documents", type="primary"):
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("📁 Saving uploaded files...")
                progress_bar.progress(10)
                
                # Create temporary directory for processing
                with tempfile.TemporaryDirectory() as tmpdir:
                    docs_dir = Path(tmpdir) / "docs"
                    docs_dir.mkdir()
                    store_dir = Path(tmpdir) / "store"
                    
                    # Save uploaded files
                    for i, file in enumerate(uploaded_files):
                        file_path = docs_dir / file.name
                        file_path.write_bytes(file.read())
                        progress_bar.progress(20 + (i * 10 // len(uploaded_files)))
                    
                    status_text.text("🔍 Processing documents and creating embeddings...")
                    progress_bar.progress(40)
                    
                    # Build store
                    build_store(str(docs_dir), str(store_dir))
                    progress_bar.progress(70)
                    
                    status_text.text("📚 Loading knowledge base...")
                    
                    # Load store into session state
                    index, chunks, meta, encoder = load_store(str(store_dir))
                    st.session_state['index'] = index
                    st.session_state['chunks'] = chunks
                    st.session_state['meta'] = meta
                    st.session_state['encoder'] = encoder
                    st.session_state['store_loaded'] = True
                    progress_bar.progress(85)
                    
                    # Save store to permanent location
                    permanent_store = Path("rag_store")
                    
                    # Copy store files (allow overwriting existing directory)
                    import shutil
                    import stat
                    import time

                    def _on_rm_error(func, path, exc_info):
                        # Windows: clear read-only and retry
                        try:
                            os.chmod(path, stat.S_IWRITE)
                            func(path)
                        except Exception:
                            pass

                    if permanent_store.exists():
                        try:
                            shutil.rmtree(permanent_store, onerror=_on_rm_error)
                        except PermissionError:
                            # If files are still locked, move old store aside and continue
                            backup = Path(f"rag_store_old_{int(time.time())}")
                            try:
                                permanent_store.rename(backup)
                                st.warning(f"Previous rag_store was locked; moved to {backup}.")
                            except Exception:
                                raise

                    shutil.copytree(store_dir, permanent_store)
                    st.session_state['store_dir'] = str(permanent_store.resolve())
                    progress_bar.progress(100)
                    
                    status_text.empty()
                    progress_bar.empty()
                    st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)! Generated {len(chunks)} chunks.")
                    st.balloons()
                    st.rerun()  # Refresh the page to show tabs
            except MemoryError as e:
                status_text.empty()
                progress_bar.empty()
                st.error("❌ **Memory Error**: Document is too large or system ran out of memory.")
                st.warning("""
                **Solutions:**
                1. Try a smaller document (under 50 pages or 100KB)
                2. Close other applications to free up memory
                3. Restart the app and try again
                4. Split large documents into smaller files
                """)
                st.exception(e)
            except Exception as e:
                status_text.empty()
                progress_bar.empty()
                error_msg = str(e)
                st.error(f"❌ **Error processing documents:** {error_msg}")
                # Show user-friendly error message
                if "MemoryError" in error_msg or "memory" in error_msg.lower():
                    st.warning("💡 **Memory Issue Detected**: Try with a smaller document or close other applications.")
                elif "No chunks produced" in error_msg:
                    st.warning("💡 **Empty Document**: The file appears to be empty or couldn't be read. Try a different file.")
                else:
                    st.info("💡 Check the error details below. If the issue persists, try a different document format.")
                with st.expander("🔍 Technical Details"):
                    st.exception(e)
        else:
            st.warning("⚠️ Please upload at least one document")
    
    # Option 2: Load existing store
    st.subheader("Load Existing Store")
    store_path = st.text_input("Store directory path", value="rag_store", help="Path to existing RAG store")
    if st.button("📂 Load Store"):
        store_path_obj = Path(store_path)
        if store_path_obj.exists() and (store_path_obj / "index.faiss").exists():
            try:
                with st.spinner("Loading store..."):
                    # Convert to absolute path to ensure it's valid
                    abs_store_path = str(store_path_obj.resolve())
                    index, chunks, meta, encoder = load_store(abs_store_path)
                    st.session_state['index'] = index
                    st.session_state['chunks'] = chunks
                    st.session_state['meta'] = meta
                    st.session_state['encoder'] = encoder
                    st.session_state['store_dir'] = abs_store_path  # Use absolute path
                    st.session_state['store_loaded'] = True
                    st.success(f"✅ Store loaded successfully! ({len(chunks)} chunks)")
                    st.rerun()  # Refresh to show tabs
            except Exception as e:
                st.error(f"Error loading store: {str(e)}")
        else:
            st.error("❌ Store not found. Please process documents first.")
    
    # Status indicator
    st.divider()
    if st.session_state['store_loaded']:
        st.success("✅ Store Ready")
        if st.session_state['chunks']:
            st.info(f"📊 {len(st.session_state['chunks'])} chunks available")
    else:
        st.info("👈 Upload and process documents to get started")

# Main content area
if st.session_state['store_loaded']:
    # Create tabs for different content types
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Summary", 
        "📝 Blog Post", 
        "💼 LinkedIn Post", 
        "🖼️ Image Generation",
        "🔍 Custom Query"
    ])
    
    with tab1:
        st.header("📄 Document Summary")
        st.markdown("Generate a comprehensive summary of your documents.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            max_tokens_summary = st.slider("Max tokens", 256, 1024, 256, key="summary_tokens")
        with col2:
            top_k_summary = st.number_input("Top K chunks", 1, 20, default_top_k, key="summary_k")
        
        if st.button("✨ Generate Summary", key="btn_summary", type="primary"):
            with st.spinner("Generating summary from your documents..."):
                try:
                    # Check if store is loaded
                    if not st.session_state.get('store_loaded') or not st.session_state.get('index'):
                        st.error("❌ Store not loaded. Please process documents or load a store first.")
                        st.stop()
                    
                    question = "Summarize the main points and key information"
                    # Use session state data directly (more efficient)
                    result = run_query(
                        question,
                        store_dir=st.session_state.get('store_dir'),  # Optional, for fallback
                        top_k=top_k_summary,
                        max_tokens=max_tokens_summary,
                        content_type="summary",
                        index=st.session_state['index'],
                        chunks=st.session_state['chunks'],
                        meta=st.session_state['meta'],
                        encoder=st.session_state['encoder'],
                        fast_mode=fast_mode
                    )
                    
                    st.markdown("### 📄 Generated Summary")
                    st.markdown("---")
                    # Extract clean answer (remove prompt if present)
                    answer = result.get("answer", "")
                    for marker in ["Summary:", "Answer:"]:
                        if marker in answer:
                            answer = answer.split(marker)[-1].strip()
                            break
                    if fast_mode:
                        st.markdown(answer)
                    else:
                        st.markdown(f'<div class="output-card">{answer}</div>', unsafe_allow_html=True)
                    
                    # Sources used (group by file, list chunk IDs) with emojis
                    retrieved = result["retrieved"]
                    by_source = {}
                    for r in retrieved:
                        name = Path(r['source']).name
                        if name not in by_source:
                            by_source[name] = []
                        by_source[name].append(r['chunk_id'])
                    sources_parts = [f"📄 **{name}** (Chunks {', '.join(map(str, sorted(ids)))})" for name, ids in by_source.items()]
                    st.markdown("---")
                    st.markdown("📚 **Sources used for this summary:**")
                    for part in sources_parts:
                        st.markdown(f"  • {part}")
                    
                    with st.expander("📋 View Retrieved Chunks"):
                        for i, r in enumerate(retrieved, 1):
                            st.markdown(f"📌 **Chunk {i}** — Relevance: {r['score']:.3f}")
                            st.caption(f"📄 {Path(r['source']).name} • Chunk {r['chunk_id']}")
                            st.text(r['chunk'][:300] + "..." if len(r['chunk']) > 300 else r['chunk'])
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
    
    with tab2:
        st.header("📝 Blog Post Introduction")
        st.markdown("Create an engaging blog post introduction based on your documents.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            max_tokens_blog = st.slider("Max tokens", 256, 1024, 256, key="blog_tokens")
        with col2:
            top_k_blog = st.number_input("Top K chunks", 1, 20, default_top_k, key="blog_k")
        
        if st.button("✨ Generate Blog Introduction", key="btn_blog", type="primary"):
            with st.spinner("Generating blog post introduction..."):
                try:
                    if not st.session_state.get('store_loaded') or not st.session_state.get('index'):
                        st.error("❌ Store not loaded. Please process documents or load a store first.")
                        st.stop()
                    
                    question = "Write an engaging blog post introduction"
                    result = run_query(
                        question,
                        store_dir=st.session_state.get('store_dir'),
                        top_k=top_k_blog,
                        max_tokens=max_tokens_blog,
                        content_type="blog_intro",
                        index=st.session_state['index'],
                        chunks=st.session_state['chunks'],
                        meta=st.session_state['meta'],
                        encoder=st.session_state['encoder'],
                        fast_mode=fast_mode
                    )
                    
                    st.markdown("### 📝 Generated Blog Introduction")
                    st.markdown("---")
                    # Extract clean answer
                    answer = result.get("answer", "")
                    if "Blog Introduction:" in answer:
                        answer = answer.split("Blog Introduction:")[-1].strip()
                    elif "Answer:" in answer:
                        answer = answer.split("Answer:")[-1].strip()
                    st.markdown(f'<div class="output-card">{answer}</div>', unsafe_allow_html=True)
                    
                    with st.expander("📋 View Retrieved Chunks"):
                        for i, r in enumerate(result["retrieved"], 1):
                            st.markdown(f"**Chunk {i}** (Score: {r['score']:.3f})")
                            st.caption(f"Source: {Path(r['source']).name} | Chunk ID: {r['chunk_id']}")
                            st.text(r['chunk'][:300] + "..." if len(r['chunk']) > 300 else r['chunk'])
                except Exception as e:
                    st.error(f"Error generating blog post: {str(e)}")
    
    with tab3:
        st.header("💼 LinkedIn Post")
        st.markdown("Generate professional LinkedIn posts from your documents.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            max_tokens_linkedin = st.slider("Max tokens", 256, 1024, 256, key="linkedin_tokens")
        with col2:
            top_k_linkedin = st.number_input("Top K chunks", 1, 20, default_top_k, key="linkedin_k")
        
        if st.button("✨ Generate LinkedIn Post", key="btn_linkedin", type="primary"):
            with st.spinner("Generating LinkedIn post..."):
                try:
                    if not st.session_state.get('store_loaded') or not st.session_state.get('index'):
                        st.error("❌ Store not loaded. Please process documents or load a store first.")
                        st.stop()
                    
                    question = "Create a professional LinkedIn post"
                    result = run_query(
                        question,
                        store_dir=st.session_state.get('store_dir'),
                        top_k=top_k_linkedin,
                        max_tokens=max_tokens_linkedin,
                        content_type="linkedin_post",
                        index=st.session_state['index'],
                        chunks=st.session_state['chunks'],
                        meta=st.session_state['meta'],
                        encoder=st.session_state['encoder'],
                        fast_mode=fast_mode
                    )
                    
                    st.markdown("### 💼 Generated LinkedIn Post")
                    st.markdown("---")
                    # Extract clean answer
                    answer = result.get("answer", "")
                    if "LinkedIn Post:" in answer:
                        answer = answer.split("LinkedIn Post:")[-1].strip()
                    elif "Answer:" in answer:
                        answer = answer.split("Answer:")[-1].strip()
                    st.markdown(f'<div class="output-card">{answer}</div>', unsafe_allow_html=True)
                    
                    # LinkedIn direct integration
                    st.markdown("### 🔗 LinkedIn Posting")
                    cfg = li_get_config()
                    st.session_state['li_last_post'] = answer

                    if li_config_ready(cfg):
                        if st.button("🚀 Post to LinkedIn", key="btn_li_post"):
                            token = li_get_access_token()
                            if not token:
                                st.error("LinkedIn not connected yet. Use 'Connect LinkedIn' from the sidebar first.")
                            else:
                                res = li_post_text(token, answer)
                                if res.get('ok'):
                                    st.success("✅ Posted to LinkedIn successfully.")
                                else:
                                    st.error(f"❌ LinkedIn post failed ({res.get('status')}): {res.get('body')}")
                    else:
                        st.info("Add LinkedIn credentials in .env to enable direct posting.")
                        st.code(
                            "LINKEDIN_CLIENT_ID=...\n"
                            "LINKEDIN_CLIENT_SECRET=...\n"
                            "LINKEDIN_REDIRECT_URI=http://localhost:8501",
                            language="text"
                        )

                    with st.expander("📋 View Retrieved Chunks"):
                        for i, r in enumerate(result["retrieved"], 1):
                            st.markdown(f"**Chunk {i}** (Score: {r['score']:.3f})")
                            st.caption(f"Source: {Path(r['source']).name} | Chunk ID: {r['chunk_id']}")
                            st.text(r['chunk'][:300] + "..." if len(r['chunk']) > 300 else r['chunk'])
                except Exception as e:
                    st.error(f"Error generating LinkedIn post: {str(e)}")
    
    with tab4:
        st.header("🖼️ Image Generation")
        st.markdown("Generate images based on your document content using Stable Diffusion.")
        
        col1, col2 = st.columns(2)
        with col1:
            top_k_image = st.number_input("Top K chunks for image prompt", 1, 10, 2, key="image_k")
        with col2:
            num_steps = st.slider("Inference steps", 1, 8, 2, key="image_steps")
        
        st.warning("⚠️ **Note:** Image generation is heavy. If your device is slow, reduce inference steps (10-15) and keep Top K low.")
        
        if st.button("🎨 Generate Image", key="btn_image", type="primary"):
            spin_msg = "Building instant image prompt..." if fast_image_mode else "Generating image from document context... This may take a few minutes."
            with st.spinner(spin_msg):
                try:
                    # Retrieve relevant chunks
                    question = "Generate an image related to this content"
                    retrieved = retrieve(
                        question,
                        st.session_state['index'],
                        st.session_state['encoder'],
                        st.session_state['chunks'],
                        st.session_state['meta'],
                        top_k=top_k_image,
                        use_reranker=not fast_mode
                    )

                    if fast_image_mode:
                        # Prompt-only mode (instant)
                        context_preview = " ".join([r['chunk'][:180] for r in retrieved[:2]])
                        context_preview = " ".join(context_preview.split())
                        image_prompt = (
                            "high quality professional illustration, clean composition, "
                            "infographic style, " + context_preview[:260]
                        )
                        st.success("⚡ Fast Image Mode enabled: generated a ready-to-use prompt instantly.")
                        st.markdown("### Prompt Used")
                        st.code(image_prompt, language="text")
                        st.caption("Use this prompt in any image tool (DALL·E, Midjourney, Leonardo, SD WebUI) for quick results.")
                    else:
                        # Generate image
                        image, image_prompt = generate_image_from_context(
                            retrieved,
                            num_inference_steps=num_steps
                        )

                        st.markdown("### Generated Image")
                        st.image(image, caption="AI-Generated Image from Document Context", use_container_width=True)
                        # Keep prompt hidden by default for clean UX

                    with st.expander("📋 View Source Chunks"):
                        for i, r in enumerate(retrieved, 1):
                            st.markdown(f"**Chunk {i}** (Score: {r['score']:.3f})")
                            st.caption(f"Source: {Path(r['source']).name} | Chunk ID: {r['chunk_id']}")
                            st.text(r['chunk'][:200] + "..." if len(r['chunk']) > 200 else r['chunk'])
                except Exception as e:
                    st.error(f"Error generating image: {str(e)}")
                    st.info("💡 Make sure you have sufficient GPU memory or use CPU (slower). Install: pip install diffusers torch")
    
    with tab5:
        st.header("🔍 Custom Query")
        st.markdown("Ask any question based on your documents.")
        
        custom_question = st.text_area("Enter your question", height=100, 
                                      placeholder="e.g., What are the main findings? Explain the methodology...")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            max_tokens_custom = st.slider("Max tokens", 256, 1024, 256, key="custom_tokens")
        with col2:
            top_k_custom = st.number_input("Top K chunks", 1, 20, default_top_k, key="custom_k")
        with col3:
            content_type_custom = st.selectbox("Content type", 
                                              ["general", "summary", "blog_intro", "linkedin_post"],
                                              key="custom_type")
        
        if st.button("🔍 Query Documents", key="btn_custom", type="primary"):
            if custom_question:
                with st.spinner("Processing your query..."):
                    try:
                        if not st.session_state.get('store_loaded') or not st.session_state.get('index'):
                            st.error("❌ Store not loaded. Please process documents or load a store first.")
                            st.stop()
                        
                        result = run_query(
                            custom_question,
                            store_dir=st.session_state.get('store_dir'),
                            top_k=top_k_custom,
                            max_tokens=max_tokens_custom,
                            content_type=content_type_custom,
                            index=st.session_state['index'],
                            chunks=st.session_state['chunks'],
                            meta=st.session_state['meta'],
                            encoder=st.session_state['encoder'],
                            fast_mode=fast_mode
                        )
                        
                        st.markdown("### 💡 Answer")
                        st.markdown("---")
                        # Extract clean answer
                        answer = result.get("answer", "")
                        if "Answer:" in answer:
                            answer = answer.split("Answer:")[-1].strip()
                        st.markdown(f'<div class="output-card">{answer}</div>', unsafe_allow_html=True)
                        
                        with st.expander("📋 View Retrieved Chunks"):
                            for i, r in enumerate(result["retrieved"], 1):
                                st.markdown(f"**Chunk {i}** (Score: {r['score']:.3f})")
                                st.caption(f"Source: {Path(r['source']).name} | Chunk ID: {r['chunk_id']}")
                                st.text(r['chunk'][:300] + "..." if len(r['chunk']) > 300 else r['chunk'])
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question")

else:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to RAG Content Generator</h3>
        <p>This system uses <strong>Retrieval-Augmented Generation (RAG)</strong> to generate accurate, 
        document-grounded content without using paid APIs.</p>
        <h4>🚀 Getting Started:</h4>
        <ol>
            <li>Upload your documents (PDF, TXT, or MD files) in the sidebar</li>
            <li>Click "Process Documents" to build the knowledge base</li>
            <li>Use the tabs above to generate different types of content</li>
        </ol>
        <h4>✨ Features:</h4>
        <ul>
            <li>📄 Document Summaries</li>
            <li>📝 Blog Post Introductions</li>
            <li>💼 LinkedIn Posts</li>
            <li>🖼️ AI-Generated Images</li>
            <li>🔍 Custom Queries</li>
        </ul>
        <p><strong>All processing is done locally - your data stays private!</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👈 **Start by uploading documents in the sidebar**")

# Footer
st.divider()
st.markdown("""
<div class="footer-card" style="text-align: center; padding: 1rem; margin-top: 2rem;">
    <p style="margin: 0.5rem 0;">🤖 AI-Based Document Understanding and Content Generation System using RAG</p>
    <p style="margin: 0.5rem 0; opacity:.82;">API-Free • Privacy-Friendly • Open Source</p>
</div>
""", unsafe_allow_html=True)

