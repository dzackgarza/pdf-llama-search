#!/usr/bin/env python3
"""
Enhanced Mathematical PDF Search Application
Inspired by sagemath_doc_search, optimized for mathematical research papers.
"""

import os
import glob
import json
import pickle
import traceback
from pathlib import Path
from typing import List, Optional, Any, Dict
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import PDF processing libraries
try:
    import PyPDF2
    import pdfplumber
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    st.error("PDF processing libraries not available. Please install PyPDF2 and pdfplumber.")
    PDF_PROCESSING_AVAILABLE = False

# Import Haystack components with error handling
haystack_imports_successful = True
haystack_components = {}

try:
    from haystack.components.generators.openai import OpenAIGenerator
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
    from haystack.components.writers import DocumentWriter
    from haystack.dataclasses import Document
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.document_stores.types import DuplicatePolicy
    from haystack.utils import Secret
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
    from haystack import Pipeline
    
    haystack_components.update({
        'OpenAIGenerator': OpenAIGenerator,
        'InMemoryBM25Retriever': InMemoryBM25Retriever,
        'InMemoryEmbeddingRetriever': InMemoryEmbeddingRetriever,
        'DocumentWriter': DocumentWriter,
        'Document': Document,
        'InMemoryDocumentStore': InMemoryDocumentStore,
        'DuplicatePolicy': DuplicatePolicy,
        'Secret': Secret,
        'SentenceTransformersDocumentEmbedder': SentenceTransformersDocumentEmbedder,
        'SentenceTransformersTextEmbedder': SentenceTransformersTextEmbedder,
        'Pipeline': Pipeline
    })
    
except ImportError as e:
    st.error(f"Haystack import error: {e}")
    haystack_imports_successful = False

# Import our utility functions
try:
    from enhanced_prompts import create_enhanced_prompt_adaptive, get_prompt_analysis
    ENHANCED_PROMPTS_AVAILABLE = True
except ImportError:
    st.warning("Enhanced prompts not available. Using basic prompts.")
    ENHANCED_PROMPTS_AVAILABLE = False

# ----------- CONFIGURATION -----------
PDF_DIR = os.getenv("PDF_DIR", "pdfs")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
# -------------------------------------

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF with page information and metadata."""
    documents = []
    
    try:
        # Try pdfplumber first (better for academic papers)
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    # Split into chunks if text is too long
                    if len(text) > CHUNK_SIZE:
                        chunks = split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
                        for i, chunk in enumerate(chunks):
                            documents.append({
                                'content': chunk,
                                'meta': {
                                    'name': os.path.basename(pdf_path),
                                    'page': page_num,
                                    'chunk': i + 1,
                                    'source_type': 'pdf',
                                    'file_path': pdf_path
                                }
                            })
                    else:
                        documents.append({
                            'content': text,
                            'meta': {
                                'name': os.path.basename(pdf_path),
                                'page': page_num,
                                'source_type': 'pdf',
                                'file_path': pdf_path
                            }
                        })
                        
    except Exception as e:
        st.warning(f"pdfplumber failed for {pdf_path}, trying PyPDF2: {e}")
        
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        if len(text) > CHUNK_SIZE:
                            chunks = split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
                            for i, chunk in enumerate(chunks):
                                documents.append({
                                    'content': chunk,
                                    'meta': {
                                        'name': os.path.basename(pdf_path),
                                        'page': page_num,
                                        'chunk': i + 1,
                                        'source_type': 'pdf',
                                        'file_path': pdf_path
                                    }
                                })
                        else:
                            documents.append({
                                'content': text,
                                'meta': {
                                    'name': os.path.basename(pdf_path),
                                    'page': page_num,
                                    'source_type': 'pdf',
                                    'file_path': pdf_path
                                }
                            })
        except Exception as e2:
            st.error(f"Failed to process {pdf_path} with both libraries: {e2}")
    
    return documents

def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to end at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            last_period = text.rfind('.', end - 100, end)
            last_exclaim = text.rfind('!', end - 100, end)
            last_question = text.rfind('?', end - 100, end)
            
            boundary = max(last_period, last_exclaim, last_question)
            if boundary > start:
                end = boundary + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

@st.cache_resource(show_spinner="Loading document store...")
def build_document_store(_pdf_dir: str) -> Optional[InMemoryDocumentStore]:
    """Build the document store from PDFs with caching."""
    if not haystack_imports_successful or not PDF_PROCESSING_AVAILABLE:
        return None
    
    # Check for cached index
    cache_file = Path("pdf_index_cache.pkl")
    pdf_files = glob.glob(os.path.join(_pdf_dir, "*.pdf"))
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if cached_data.get('pdf_files') == pdf_files:
                    st.success("Loaded cached PDF index")
                    return cached_data['document_store']
        except Exception as e:
            st.warning(f"Failed to load cache: {e}")
    
    if not pdf_files:
        st.warning(f"No PDF files found in {_pdf_dir}")
        return None
    
    # Process PDFs
    all_documents = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, pdf_path in enumerate(pdf_files):
        status_text.text(f"Processing {os.path.basename(pdf_path)}...")
        progress_bar.progress((i + 1) / len(pdf_files))
        
        docs = extract_text_from_pdf(pdf_path)
        for doc_data in docs:
            doc = haystack_components['Document'](
                content=doc_data['content'],
                meta=doc_data['meta']
            )
            all_documents.append(doc)
    
    # Create document store
    document_store = haystack_components['InMemoryDocumentStore']()
    
    # Write documents
    writer = haystack_components['DocumentWriter'](document_store=document_store)
    writer.run(documents=all_documents)
    
    # Cache the results
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'pdf_files': pdf_files,
                'document_store': document_store
            }, f)
        st.success("Cached PDF index for faster loading next time")
    except Exception as e:
        st.warning(f"Failed to cache index: {e}")
    
    status_text.text("PDF processing complete!")
    progress_bar.empty()
    
    return document_store

@st.cache_resource(show_spinner="Initializing embeddings...")
def setup_embeddings(_document_store: InMemoryDocumentStore, embedding_model: str, device: str):
    """Setup embeddings for the document store."""
    if not _document_store:
        return None, None
    
    try:
        # Create embedders
        doc_embedder = haystack_components['SentenceTransformersDocumentEmbedder'](
            model=embedding_model,
            device=device
        )
        
        text_embedder = haystack_components['SentenceTransformersTextEmbedder'](
            model=embedding_model,
            device=device
        )
        
        # Get all documents and embed them
        documents = _document_store.filter_documents()
        embedded_docs = doc_embedder.run(documents=documents)
        
        # Update document store with embeddings
        _document_store.write_documents(embedded_docs['documents'], policy=DuplicatePolicy.OVERWRITE)
        
        return doc_embedder, text_embedder
        
    except Exception as e:
        st.error(f"Failed to setup embeddings: {e}")
        return None, None

def get_llm_generator():
    """Create and return the LLM generator."""
    try:
        return haystack_components['OpenAIGenerator'](
            api_key=haystack_components['Secret'].from_token(OPENAI_API_KEY),
            api_base_url=OPENAI_API_BASE,
            model="gpt-3.5-turbo",
            generation_kwargs={
                "max_tokens": 2000,
                "temperature": 0.3
            }
        )
    except Exception as e:
        st.error(f"Failed to create LLM generator: {e}")
        return None

def create_enhanced_prompt(question: str, documents: List[Document]) -> str:
    """Create an enhanced prompt for mathematical content."""
    if ENHANCED_PROMPTS_AVAILABLE:
        return create_enhanced_prompt_adaptive(question, documents)
    else:
        # Fallback basic prompt
        prompt_parts = [
            "You are a mathematical research expert. Analyze the following documents and provide a comprehensive response.\n\n",
            "Requirements:\n",
            "- Include exact page numbers and citations in format [Paper: filename, Page X]\n",
            "- Use proper mathematical notation with LaTeX formatting\n",
            "- Provide precise definitions, theorems, and proofs as requested\n",
            "- Include section numbers when available\n\n",
            "Documents:\n"
        ]
        
        for i, doc in enumerate(documents, 1):
            filename = doc.meta.get("name", "Unknown")
            page = doc.meta.get("page", "?")
            content = doc.content[:700] + "..." if len(doc.content) > 700 else doc.content
            prompt_parts.append(f"Document {i}: {filename} (Page {page})\n{content}\n\n")
        
        prompt_parts.append(f"Question: {question}\n\nResponse:")
        return "".join(prompt_parts)

def search_documents(question: str, document_store: InMemoryDocumentStore, top_k: int = 10, 
                    use_hybrid: bool = True, text_embedder=None) -> List[Document]:
    """Search documents using BM25, embeddings, or hybrid approach."""
    
    retrieved_docs = []
    
    try:
        if use_hybrid and text_embedder:
            # Hybrid search: BM25 + Semantic
            bm25_retriever = haystack_components['InMemoryBM25Retriever'](
                document_store=document_store,
                top_k=top_k // 2
            )
            
            embedding_retriever = haystack_components['InMemoryEmbeddingRetriever'](
                document_store=document_store,
                top_k=top_k // 2
            )
            
            # Get BM25 results
            bm25_results = bm25_retriever.run(query=question)
            bm25_docs = bm25_results.get('documents', [])
            
            # Get embedding results
            query_embedding = text_embedder.run(text=question)
            embedding_results = embedding_retriever.run(
                query_embedding=query_embedding['embedding']
            )
            embedding_docs = embedding_results.get('documents', [])
            
            # Combine and deduplicate
            all_docs = bm25_docs + embedding_docs
            seen_content = set()
            for doc in all_docs:
                content_hash = hash(doc.content[:100])  # Use first 100 chars as identifier
                if content_hash not in seen_content:
                    retrieved_docs.append(doc)
                    seen_content.add(content_hash)
                    if len(retrieved_docs) >= top_k:
                        break
        
        else:
            # BM25 only
            bm25_retriever = haystack_components['InMemoryBM25Retriever'](
                document_store=document_store,
                top_k=top_k
            )
            
            bm25_results = bm25_retriever.run(query=question)
            retrieved_docs = bm25_results.get('documents', [])
    
    except Exception as e:
        st.error(f"Search failed: {e}")
        retrieved_docs = []
    
    return retrieved_docs

def format_sources(documents: List[Document]) -> str:
    """Format source documents for display."""
    if not documents:
        return "No sources found."
    
    sources = []
    for doc in documents:
        filename = doc.meta.get("name", "Unknown")
        page = doc.meta.get("page", "?")
        chunk = doc.meta.get("chunk")
        
        # Create a snippet
        snippet = doc.content[:300].replace("\n", " ")
        if len(doc.content) > 300:
            snippet += "..."
        
        source_info = f"**{filename}** (Page {page}"
        if chunk:
            source_info += f", Chunk {chunk}"
        source_info += f"): {snippet}"
        
        sources.append(source_info)
    
    return "\n\n".join(sources)

# ----------- STREAMLIT UI -----------
def main():
    st.set_page_config(
        page_title="Mathematical PDF Search",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Mathematical PDF Research Assistant")
    st.markdown("*Advanced search for mathematical research papers with precise citations*")
    
    # Check if required libraries are available
    if not haystack_imports_successful:
        st.error("Haystack components not available. Please install haystack-ai>=2.0.0")
        return
    
    if not PDF_PROCESSING_AVAILABLE:
        st.error("PDF processing libraries not available. Please install PyPDF2 and pdfplumber")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # PDF directory
        pdf_dir = st.text_input("PDF Directory:", value=PDF_DIR, help="Directory containing your PDF files")
        
        # Search settings
        st.subheader("Search Settings")
        top_k = st.slider("Number of results:", min_value=5, max_value=20, value=DEFAULT_TOP_K)
        use_hybrid = st.checkbox("Use hybrid search (BM25 + Embeddings)", value=USE_HYBRID_SEARCH)
        
        # Embedding settings
        if use_hybrid:
            st.subheader("Embedding Settings")
            embedding_model = st.selectbox(
                "Embedding Model:",
                [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/multi-qa-mpnet-base-dot-v1"
                ],
                index=0
            )
            device = st.selectbox("Device:", ["cpu", "cuda", "auto"], index=0)
        else:
            embedding_model = EMBEDDING_MODEL
            device = EMBEDDING_DEVICE
        
        # LLM settings
        st.subheader("LLM Settings")
        api_base = st.text_input("API Base URL:", value=OPENAI_API_BASE)
        
        if st.button("🔄 Rebuild Index"):
            st.cache_resource.clear()
            st.rerun()
    
    # Check if PDF directory exists
    if not os.path.exists(pdf_dir):
        st.error(f"PDF directory '{pdf_dir}' does not exist. Please create it and add PDF files.")
        return
    
    # Build document store
    document_store = build_document_store(pdf_dir)
    
    if not document_store:
        st.error("Failed to build document store. Please check your PDF directory and files.")
        return
    
    # Setup embeddings if using hybrid search
    doc_embedder = None
    text_embedder = None
    
    if use_hybrid:
        doc_embedder, text_embedder = setup_embeddings(document_store, embedding_model, device)
        if not text_embedder:
            st.warning("Failed to setup embeddings. Falling back to BM25 only.")
            use_hybrid = False
    
    # Get LLM generator
    llm = get_llm_generator()
    if not llm:
        st.error("Failed to setup LLM. Please check your configuration.")
        return
    
    # Display some stats
    total_docs = len(document_store.filter_documents())
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PDF Files", len(pdf_files))
    with col2:
        st.metric("Text Chunks", total_docs)
    with col3:
        search_type = "Hybrid (BM25 + Embeddings)" if use_hybrid else "BM25 Only"
        st.metric("Search Type", search_type)
    
    # Main search interface
    st.markdown("---")
    
    # Query examples
    with st.expander("💡 Example Queries"):
        st.markdown("""
        **Theorem Queries:**
        - "Find all theorems about convergence of numerical methods"
        - "What are the main results for stability analysis?"
        
        **Definition Queries:**
        - "Definition of compact operator"
        - "What is a Banach space? Provide exact definitions"
        
        **Proof Queries:**
        - "Show me proofs of the fundamental theorem"
        - "How is the uniqueness theorem proved?"
        
        **Concept Queries:**
        - "Explain the concept of functional analysis"
        - "Discuss variational methods in partial differential equations"
        """)
    
    # Search input
    user_query = st.text_area(
        "🔍 Ask about your mathematical research papers:",
        placeholder="e.g., 'Find all definitions of Sobolev spaces with exact page references'",
        height=100
    )
    
    if user_query:
        with st.spinner("🔍 Searching through your mathematical papers..."):
            
            # Analyze query if enhanced prompts are available
            if ENHANCED_PROMPTS_AVAILABLE:
                query_analysis = get_prompt_analysis(user_query)
                st.info(f"**Query Type:** {query_analysis['query_type'].title()} | **Focus:** {', '.join(query_analysis['focus_areas'])}")
            
            # Search documents
            retrieved_docs = search_documents(
                user_query, 
                document_store, 
                top_k=top_k, 
                use_hybrid=use_hybrid,
                text_embedder=text_embedder
            )
            
            if not retrieved_docs:
                st.warning("No relevant documents found. Try adjusting your query or search settings.")
                return
            
            # Create enhanced prompt
            prompt = create_enhanced_prompt(user_query, retrieved_docs)
            
            # Generate response
            try:
                response = llm.run(prompt=prompt)
                answer = response['replies'][0]
                
                # Display results
                st.markdown("## 📝 Response")
                st.markdown(answer)
                
                st.markdown("## 📚 Sources")
                st.markdown(format_sources(retrieved_docs))
                
                # Debug information
                if st.checkbox("🔧 Show Debug Information"):
                    st.markdown("### Query Analysis")
                    if ENHANCED_PROMPTS_AVAILABLE:
                        for key, value in query_analysis.items():
                            st.text(f"{key}: {value}")
                    
                    st.markdown("### Generated Prompt")
                    with st.expander("Show Full Prompt"):
                        st.text(prompt)
                    
                    st.markdown("### Retrieved Documents")
                    for i, doc in enumerate(retrieved_docs, 1):
                        with st.expander(f"Document {i}: {doc.meta.get('name', 'Unknown')} (Page {doc.meta.get('page', '?')})"):
                            st.text(doc.content[:1000] + "..." if len(doc.content) > 1000 else doc.content)
            
            except Exception as e:
                st.error(f"Failed to generate response: {e}")
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main()