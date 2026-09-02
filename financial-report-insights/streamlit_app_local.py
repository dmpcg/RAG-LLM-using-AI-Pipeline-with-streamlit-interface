"""
Streamlit frontend for the Local RAG system.
No API keys required - runs entirely on your machine.
Enhanced with Excel processing and financial insights dashboard.
"""

import os
from pathlib import Path

import streamlit as st

from config import settings

# Page configuration
st.set_page_config(page_title="Local RAG - Financial Report Insights", page_icon="📊", layout="wide")

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Supported file extensions
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".docx"]
MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024  # bytes


def _sanitize_and_save(file, docs_path: Path) -> bool:
    """Sanitize an uploaded filename and save it safely within docs_path.

    Returns True if saved successfully, False otherwise.
    """
    safe_name = os.path.basename(file.name).strip()
    if not safe_name:
        st.error("Invalid filename.")
        return False

    # Reject names that are only dots or contain path separators after basename
    if safe_name in (".", "..") or "/" in safe_name or "\\" in safe_name:
        st.error(f"Rejected unsafe filename: {file.name}")
        return False

    save_path = (docs_path / safe_name).resolve()
    # Verify the resolved path stays within the docs folder
    if not str(save_path).startswith(str(docs_path.resolve())):
        st.error(f"Rejected path traversal attempt: {file.name}")
        return False

    # File size check
    file.seek(0, 2)  # seek to end
    size = file.tell()
    file.seek(0)

    if size > MAX_FILE_SIZE:
        st.error(f"File too large: {safe_name} ({size / 1024 / 1024:.1f} MB). Max is {settings.max_file_size_mb} MB.")
        return False

    with open(save_path, "wb") as f:
        f.write(file.getbuffer())
    st.success(f"Uploaded: {safe_name}")
    return True


@st.cache_resource
def load_rag_system():
    """Load the RAG system (cached). Validates config first."""
    from config import validate_settings

    errors, warnings = validate_settings()
    for w in warnings:
        st.warning(f"Config warning: {w}")
    if errors:
        for e in errors:
            st.error(f"Config error: {e}")
        st.stop()

    from app_local import SimpleRAG

    return SimpleRAG(
        docs_folder="./documents",
        llm_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        embedding_model=os.getenv("EMBEDDING_MODEL", settings.embedding_model),
    )


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.header("🧭 Navigation")
        page = st.radio("Go to", ["Q&A Chat", "Financial Insights", "Document Manager"], label_visibility="collapsed")

        st.divider()

        st.header("⚙️ Settings")

        st.divider()

        # Documents info
        st.header("📁 Documents")
        docs_path = Path("./documents")
        docs_path.mkdir(exist_ok=True)

        doc_files = list(docs_path.rglob("*"))
        doc_files = [f for f in doc_files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

        # Group by type
        pdf_files = [f for f in doc_files if f.suffix.lower() == ".pdf"]
        text_files = [f for f in doc_files if f.suffix.lower() in [".txt", ".md"]]
        excel_files = [f for f in doc_files if f.suffix.lower() in [".xlsx", ".xlsm", ".xls", ".csv", ".tsv"]]
        docx_files = [f for f in doc_files if f.suffix.lower() == ".docx"]

        if doc_files:
            st.success(f"{len(doc_files)} document(s) loaded")
            with st.expander("View files"):
                if pdf_files:
                    st.markdown("**PDF Files:**")
                    for f in pdf_files:
                        st.caption(f"📄 {f.relative_to(docs_path)}")
                if text_files:
                    st.markdown("**Text Files:**")
                    for f in text_files:
                        st.caption(f"📝 {f.relative_to(docs_path)}")
                if excel_files:
                    st.markdown("**Excel/CSV Files:**")
                    for f in excel_files:
                        st.caption(f"📊 {f.relative_to(docs_path)}")
                if docx_files:
                    st.markdown("**Word Documents:**")
                    for f in docx_files:
                        st.caption(f"📝 {f.relative_to(docs_path)}")
        else:
            st.warning("No documents found")
            st.info("Add files to the 'documents' folder or use Document Manager")

        # Reload button
        if st.button("🔄 Reload Documents"):
            st.cache_resource.clear()
            st.rerun()

        st.divider()

        # Instructions
        st.header("📖 How to Use")
        st.markdown("""
        1. Upload documents via Document Manager
        2. Ask questions in Q&A Chat
        3. Explore data in Financial Insights

        **Supported formats:**
        - PDF, Word (docx), TXT, MD
        - Excel (xlsx, xlsm, xls)
        - CSV, TSV
        """)

    return page


def _render_system_status():
    """Show Ollama connection status in sidebar (cached to avoid repeated checks)."""
    try:
        from healthcheck import check_ollama_connection

        result = check_ollama_connection(os.environ.get("OLLAMA_HOST"))
        if result["status"] == "ok":
            st.sidebar.success("Ollama: Connected")
        else:
            st.sidebar.error(f"Ollama: {result['detail']}")
    except Exception:
        st.sidebar.warning("Ollama: Status unknown")


def main():
    """Main application entry point."""
    # Render sidebar and get selected page
    page = render_sidebar()
    _render_system_status()

    # Route to appropriate page
    if page == "Q&A Chat":
        render_qa_page()
    elif page == "Financial Insights":
        render_insights_page()
    elif page == "Document Manager":
        render_document_manager()


def render_qa_page():
    """Render the Q&A Chat page."""
    # Header
    st.title("📊 Financial Report Insights")
    st.caption("Local RAG System - Ask questions about your financial documents")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Ask a Question")

        # Load RAG system
        try:
            with st.spinner("Loading RAG system..."):
                rag = load_rag_system()

            # Display document count with type breakdown
            excel_count = sum(1 for d in rag.documents if d.get("type") == "excel")
            pdf_count = sum(1 for d in rag.documents if d.get("type") == "pdf")
            text_count = sum(1 for d in rag.documents if d.get("type") == "text")

            info_text = f"📚 {len(rag.documents)} chunks indexed"
            if excel_count > 0:
                info_text += f" | 📊 {excel_count} Excel"
            if pdf_count > 0:
                info_text += f" | 📄 {pdf_count} PDF"
            if text_count > 0:
                info_text += f" | 📝 {text_count} Text"

            st.info(info_text)

        except Exception as e:
            st.error(f"Failed to load RAG system: {e}")
            st.warning("Make sure Ollama is running: `ollama serve`")
            return

        # Query input
        user_query = st.text_input(
            "Enter your question:", placeholder="What are the key financial highlights? What is the revenue trend?"
        )

        # Example queries
        with st.expander("📝 Example Questions"):
            st.markdown("""
            **Financial Analysis:**
            - What are the key financial highlights?
            - What is the revenue for the period?
            - How does actual compare to budget?
            - What are the main expense categories?

            **Trend Analysis:**
            - What is the revenue growth trend?
            - How have margins changed over time?

            **Ratio Analysis:**
            - What is the current ratio?
            - Calculate the profit margin.
            """)

        # Submit button
        if st.button("🔍 Get Answer", type="primary"):
            if user_query:
                try:
                    # Retrieve documents (fast, with spinner)
                    with st.spinner("Searching documents..."):
                        relevant_docs = rag.retrieve(user_query, top_k=settings.top_k)

                    # Stream the answer (tokens appear as they arrive)
                    st.markdown("### Answer")
                    answer = st.write_stream(rag.answer_stream(user_query, retrieved_docs=relevant_docs))

                    # Show retrieved documents (reusing same results)
                    with st.expander("📄 Retrieved Context"):
                        for i, doc in enumerate(relevant_docs, 1):
                            doc_type = doc.get("type", "unknown")
                            icon = {"excel": "📊", "pdf": "📄", "text": "📝"}.get(doc_type, "📁")
                            st.markdown(f"**Source {i}:** {icon} {doc['source']}")
                            content = doc["content"]
                            st.text(content[:500] + "..." if len(content) > 500 else content)
                            st.divider()

                    # Add to history
                    st.session_state.chat_history.append({"query": user_query, "answer": answer})

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.warning("Make sure Ollama is running with the selected model")

    with col2:
        st.subheader("💬 History")

        if st.session_state.chat_history:
            for i, item in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.container():
                    st.markdown(f"**Q:** {item['query'][:50]}...")
                    st.caption(f"A: {item['answer'][:100]}...")
                    st.divider()
        else:
            st.caption("No questions asked yet")

        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()


def render_insights_page():
    """Render the Financial Insights dashboard page."""
    try:
        from insights_page import render_insights_page as render_dashboard

        render_dashboard(docs_folder="./documents")
    except ImportError as e:
        st.error(f"Financial Insights module not available: {e}")
        st.info("Make sure plotly and pandas are installed: `pip install plotly pandas openpyxl`")
    except Exception as e:
        st.error(f"Error loading Financial Insights: {e}")


def render_document_manager():
    """Render the Document Manager page."""
    st.title("📁 Document Manager")
    st.caption("Upload and manage financial documents")

    docs_path = Path("./documents")
    docs_path.mkdir(exist_ok=True)

    # File uploader
    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload financial documents",
        type=["pdf", "xlsx", "xlsm", "xls", "csv", "txt", "md", "tsv", "docx"],
        accept_multiple_files=True,
        help="Supports PDF, Word (docx), Excel (xlsx, xlsm, xls), CSV, TSV, and text files",
    )

    if uploaded_files:
        for file in uploaded_files:
            _sanitize_and_save(file, docs_path)

        if st.button("🔄 Reindex Documents", type="primary"):
            st.cache_resource.clear()
            st.success("Documents reindexed! Navigate to Q&A Chat to query them.")
            st.rerun()

    st.divider()

    # Current documents
    st.subheader("Current Documents")

    doc_files = list(docs_path.rglob("*"))
    doc_files = [f for f in doc_files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if doc_files:
        # Create a table of documents
        for f in sorted(doc_files, key=lambda x: x.stat().st_mtime, reverse=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            rel_path = f.relative_to(docs_path)

            with col1:
                if f.suffix.lower() in [".xlsx", ".xlsm", ".xls", ".csv", ".tsv"]:
                    icon = "📊"
                elif f.suffix.lower() == ".docx":
                    icon = "📝"
                else:
                    icon = "📄"
                st.markdown(f"{icon} **{rel_path}**")

            with col2:
                size_kb = f.stat().st_size / 1024
                if size_kb > 1024:
                    st.caption(f"{size_kb / 1024:.1f} MB")
                else:
                    st.caption(f"{size_kb:.1f} KB")

            with col3:
                if st.button("🗑️", key=f"delete_{rel_path}", help=f"Delete {rel_path}"):
                    f.unlink()
                    st.success(f"Deleted {rel_path}")
                    st.cache_resource.clear()
                    st.rerun()

    else:
        st.info("No documents uploaded yet. Use the uploader above to add files.")

    st.divider()

    # Sample data generator
    st.subheader("📊 Generate Sample Data")
    st.caption("Create sample financial data for testing")

    if st.button("Generate Sample Income Statement"):
        _generate_sample_income_statement(docs_path)
        st.success("Sample income statement generated!")
        st.cache_resource.clear()
        st.rerun()

    if st.button("Generate Sample Budget vs Actual"):
        _generate_sample_budget(docs_path)
        st.success("Sample budget vs actual generated!")
        st.cache_resource.clear()
        st.rerun()


def _generate_sample_income_statement(docs_path: Path):
    """Generate a sample income statement Excel file."""
    import pandas as pd

    data = {
        "Line Item": [
            "Revenue",
            "Cost of Goods Sold",
            "Gross Profit",
            "Operating Expenses",
            "SG&A",
            "R&D",
            "Depreciation",
            "Operating Income",
            "Interest Expense",
            "Net Income",
        ],
        "Q1 2024": [1000000, 400000, 600000, 200000, 100000, 50000, 30000, 220000, 20000, 200000],
        "Q2 2024": [1100000, 440000, 660000, 210000, 105000, 55000, 30000, 260000, 20000, 240000],
        "Q3 2024": [1200000, 480000, 720000, 220000, 110000, 60000, 30000, 300000, 20000, 280000],
        "Q4 2024": [1350000, 540000, 810000, 230000, 115000, 65000, 30000, 370000, 20000, 350000],
    }

    df = pd.DataFrame(data)
    df.to_excel(docs_path / "sample_income_statement.xlsx", index=False)


def _generate_sample_budget(docs_path: Path):
    """Generate a sample budget vs actual Excel file."""
    import pandas as pd

    data = {
        "Category": [
            "Revenue",
            "Marketing",
            "Salaries",
            "Technology",
            "Office",
            "Travel",
            "Professional Services",
            "Other",
        ],
        "Budget": [1200000, 100000, 500000, 80000, 50000, 30000, 40000, 20000],
        "Actual": [1150000, 120000, 480000, 95000, 45000, 35000, 38000, 22000],
    }

    df = pd.DataFrame(data)
    df["Variance"] = df["Actual"] - df["Budget"]
    df["Variance %"] = ((df["Actual"] - df["Budget"]) / df["Budget"] * 100).round(1)
    df.to_excel(docs_path / "sample_budget_vs_actual.xlsx", index=False)


if __name__ == "__main__":
    main()
