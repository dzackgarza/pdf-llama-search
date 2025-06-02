import os
import glob
import streamlit as st

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PDFToTextConverter, PreProcessor, EmbeddingRetriever
from haystack.pipelines import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator

# ----------- CONFIGURATION -----------
PDF_DIR = "pdfs"  # Change to your PDF folder path
OPENAI_API_BASE = "http://localhost:8000/v1"  # Your local OpenAI-compatible endpoint
OPENAI_API_KEY = "sk-local"  # Use a dummy key if your local endpoint doesn't check

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Small, CPU-friendly
USE_GPU = False
# -------------------------------------

@st.cache_resource(show_spinner="Indexing PDFs...")
def build_pipeline(pdf_dir):
    # --- Load and preprocess PDFs ---
    converter = PDFToTextConverter()
    preprocessor = PreProcessor(
        split_by="passage",
        split_length=1,
        split_respect_sentence_boundary=False,
        add_page_number=True,
    )

    all_docs = []
    for pdf_path in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        docs = converter.convert(file_path=pdf_path, meta={"name": os.path.basename(pdf_path)})
        processed = preprocessor.process(docs)
        all_docs.extend(processed)

    # --- Store documents ---
    document_store = InMemoryDocumentStore()
    document_store.write_documents(all_docs)

    # --- Embeddings and Retriever ---
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=EMBEDDING_MODEL,
        use_gpu=USE_GPU,
    )
    document_store.update_embeddings(retriever)

    # --- OpenAI-compatible LLM ---
    os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    chat_generator = OpenAIChatGenerator(
        model="gpt-3.5-turbo",
        api_base=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
    )

    # --- Build a simple RAG pipeline ---
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("llm", chat_generator)
    rag_pipeline.connect("retriever", "llm.documents")

    return rag_pipeline

# ----------- STREAMLIT UI -----------
st.title("📚 Haystack PDF Q&A (Proof-of-Concept)")
st.info(f"Indexing PDFs from: `{PDF_DIR}` (change in script if needed)")

rag_pipeline = build_pipeline(PDF_DIR)

user_query = st.text_input("Ask a question about your PDFs:", placeholder="e.g. List all definitions of 'ring' with page references.")

if user_query:
    with st.spinner("Searching..."):
        results = rag_pipeline.run({
            "retriever": {"query": user_query},
            "llm": {"query": user_query}
        })
        answer = results["llm"]["replies"][0]
        st.markdown("### Answer")
        st.write(answer)
        st.markdown("### Sources")
        for doc in results["retriever"]["documents"]:
            fname = doc.meta.get("name", "Unknown file")
            page = doc.meta.get("page", "?")
            snippet = doc.content[:200].replace("\n", " ") + ("..." if len(doc.content) > 200 else "")
            st.write(f"**{fname}** (page {page}): {snippet}")

st.markdown("---")
st.markdown("Powered by [Haystack](https://haystack.deepset.ai/) | Local LLM: OpenAI-compatible endpoint")