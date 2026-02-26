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
import json

# Page configuration
st.set_page_config(
    page_title="RAG Content Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI - Improved visibility with light sidebar
st.markdown("""
<style>
    /* Force light theme for better visibility */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Sidebar - Light background with good contrast */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #dee2e6;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #f8f9fa !important;
    }
    
    /* Sidebar text - dark for visibility */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #212529 !important;
    }
    
    /* Sidebar input fields */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        background-color: #ffffff !important;
        color: #212529 !important;
        border: 1px solid #ced4da !important;
    }
    
    /* Sidebar file uploader */
    section[data-testid="stSidebar"] .uploadedFile {
        background-color: #ffffff !important;
        color: #212529 !important;
        border: 1px solid #ced4da !important;
    }
    
    /* Sidebar buttons */
    section[data-testid="stSidebar"] button {
        background-color: #1f77b4 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    section[data-testid="stSidebar"] button:hover {
        background-color: #1565a0 !important;
    }
    
    /* Sidebar info boxes */
    section[data-testid="stSidebar"] .stSuccess,
    section[data-testid="stSidebar"] .stInfo,
    section[data-testid="stSidebar"] .stWarning,
    section[data-testid="stSidebar"] .stError {
        background-color: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        color: #212529 !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #333333 !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .info-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        color: #212529 !important;
        margin: 1rem 0;
    }
    .info-box h3, .info-box h4 {
        color: #1f77b4 !important;
    }
    .info-box p, .info-box li {
        color: #212529 !important;
    }
    /* Ensure all text is visible */
    p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #212529 !important;
    }
    /* Streamlit default text */
    .stMarkdown, .stText {
        color: #212529 !important;
    }
    /* Code blocks */
    pre {
        background-color: #f8f9fa;
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

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

# Header
st.markdown('<h1 class="main-header">🤖 AI-Based Document Understanding & Content Generation</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload documents and generate summaries, blog posts, LinkedIn content, and images using RAG</p>', unsafe_allow_html=True)

# Sidebar for document management
with st.sidebar:
    st.header("📁 Document Management")
    
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
                    if permanent_store.exists():
                        shutil.rmtree(permanent_store)
                    shutil.copytree(store_dir, permanent_store)
                    st.session_state['store_dir'] = str(permanent_store)
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
            max_tokens_summary = st.slider("Max tokens", 128, 1024, 256, key="summary_tokens")
        with col2:
            top_k_summary = st.number_input("Top K chunks", 1, 20, 5, key="summary_k")
        
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
                        encoder=st.session_state['encoder']
                    )
                    
                    st.markdown("### 📄 Generated Summary")
                    st.markdown("---")
                    # Extract clean answer (remove prompt if present)
                    answer = result.get("answer", "")
                    for marker in ["Summary:", "Answer:"]:
                        if marker in answer:
                            answer = answer.split(marker)[-1].strip()
                            break
                    st.markdown(f'<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; color: #212529; line-height: 1.6;">{answer}</div>', unsafe_allow_html=True)
                    
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
            max_tokens_blog = st.slider("Max tokens", 128, 1024, 512, key="blog_tokens")
        with col2:
            top_k_blog = st.number_input("Top K chunks", 1, 20, 5, key="blog_k")
        
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
                        encoder=st.session_state['encoder']
                    )
                    
                    st.markdown("### 📝 Generated Blog Introduction")
                    st.markdown("---")
                    # Extract clean answer
                    answer = result.get("answer", "")
                    if "Blog Introduction:" in answer:
                        answer = answer.split("Blog Introduction:")[-1].strip()
                    elif "Answer:" in answer:
                        answer = answer.split("Answer:")[-1].strip()
                    st.markdown(f'<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; color: #212529; line-height: 1.6;">{answer}</div>', unsafe_allow_html=True)
                    
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
            max_tokens_linkedin = st.slider("Max tokens", 128, 512, 256, key="linkedin_tokens")
        with col2:
            top_k_linkedin = st.number_input("Top K chunks", 1, 20, 5, key="linkedin_k")
        
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
                        encoder=st.session_state['encoder']
                    )
                    
                    st.markdown("### 💼 Generated LinkedIn Post")
                    st.markdown("---")
                    # Extract clean answer
                    answer = result.get("answer", "")
                    if "LinkedIn Post:" in answer:
                        answer = answer.split("LinkedIn Post:")[-1].strip()
                    elif "Answer:" in answer:
                        answer = answer.split("Answer:")[-1].strip()
                    st.markdown(f'<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; color: #212529; line-height: 1.6;">{answer}</div>', unsafe_allow_html=True)
                    
                    st.info("💡 **Note:** Due to platform security policies, automatic posting to LinkedIn requires official APIs. You can manually copy and post this content.")
                    
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
            top_k_image = st.number_input("Top K chunks for image prompt", 1, 10, 3, key="image_k")
        with col2:
            num_steps = st.slider("Inference steps", 10, 50, 20, key="image_steps")
        
        st.warning("⚠️ **Note:** Image generation requires significant GPU memory. First run will download the model (~4GB).")
        
        if st.button("🎨 Generate Image", key="btn_image", type="primary"):
            with st.spinner("Generating image from document context... This may take a few minutes."):
                try:
                    # Retrieve relevant chunks
                    question = "Generate an image related to this content"
                    retrieved = retrieve(
                        question,
                        st.session_state['index'],
                        st.session_state['encoder'],
                        st.session_state['chunks'],
                        st.session_state['meta'],
                        top_k=top_k_image
                    )
                    
                    # Generate image
                    image, image_prompt = generate_image_from_context(
                        retrieved,
                        num_inference_steps=num_steps
                    )
                    
                    st.markdown("### Generated Image")
                    st.image(image, caption="AI-Generated Image from Document Context", use_container_width=True)
                    
                    st.markdown("### Image Prompt")
                    st.code(image_prompt, language="text")
                    
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
            max_tokens_custom = st.slider("Max tokens", 128, 1024, 256, key="custom_tokens")
        with col2:
            top_k_custom = st.number_input("Top K chunks", 1, 20, 5, key="custom_k")
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
                            encoder=st.session_state['encoder']
                        )
                        
                        st.markdown("### 💡 Answer")
                        st.markdown("---")
                        # Extract clean answer
                        answer = result.get("answer", "")
                        if "Answer:" in answer:
                            answer = answer.split("Answer:")[-1].strip()
                        st.markdown(f'<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; color: #212529; line-height: 1.6;">{answer}</div>', unsafe_allow_html=True)
                        
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
<div style="text-align: center; color: #333333 !important; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; margin-top: 2rem;">
    <p style="color: #333333 !important; margin: 0.5rem 0;">🤖 AI-Based Document Understanding and Content Generation System using RAG</p>
    <p style="color: #666666 !important; margin: 0.5rem 0;">API-Free • Privacy-Friendly • Open Source</p>
</div>
""", unsafe_allow_html=True)

