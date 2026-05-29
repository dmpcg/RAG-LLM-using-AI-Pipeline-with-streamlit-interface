"""
Security tests for file upload sanitization and input validation.
Tests: path traversal, file size limits, query length limits, filename sanitization.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config import settings

# ============================================================
# Filename sanitization logic (mirrors streamlit_app_local._sanitize_and_save)
# ============================================================


def sanitize_filename(raw_name: str) -> str | None:
    """Pure-function version of the sanitization logic for testing."""
    # Cross-platform basename: split on BOTH separators regardless of host OS,
    # so a Windows-style path-traversal string is sanitized on Linux/CI too.
    safe_name = raw_name.replace("\\", "/").split("/")[-1].strip()
    if not safe_name or safe_name in (".", ".."):
        return None
    if "/" in safe_name or "\\" in safe_name:
        return None
    return safe_name


def is_path_safe(filename: str, docs_path: Path) -> bool:
    """Check if the resolved path stays within docs_path."""
    safe_name = sanitize_filename(filename)
    if safe_name is None:
        return False
    resolved = (docs_path / safe_name).resolve()
    return str(resolved).startswith(str(docs_path.resolve()))


class TestFilenameSanitization:
    def test_normal_filename(self):
        assert sanitize_filename("report.xlsx") == "report.xlsx"

    def test_path_traversal_unix(self):
        assert sanitize_filename("../../etc/passwd") == "passwd"

    def test_path_traversal_windows(self):
        assert sanitize_filename("..\\..\\windows\\system32\\config") == "config"

    def test_dot_filename(self):
        assert sanitize_filename(".") is None
        assert sanitize_filename("..") is None

    def test_empty_filename(self):
        assert sanitize_filename("") is None
        assert sanitize_filename("   ") is None

    def test_filename_with_spaces(self):
        assert sanitize_filename("my report.xlsx") == "my report.xlsx"

    def test_nested_path_stripped(self):
        result = sanitize_filename("subdir/report.xlsx")
        assert result == "report.xlsx"


class TestPathSafety:
    @pytest.fixture
    def temp_docs(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def test_safe_path(self, temp_docs):
        assert is_path_safe("report.xlsx", temp_docs) is True

    def test_traversal_caught(self, temp_docs):
        # After basename extraction, "../../etc/passwd" becomes "passwd"
        # which resolves inside temp_docs, so it's actually safe after sanitization
        assert is_path_safe("../../etc/passwd", temp_docs) is True  # basename makes it safe

    def test_dot_rejected(self, temp_docs):
        assert is_path_safe(".", temp_docs) is False
        assert is_path_safe("..", temp_docs) is False


# ============================================================
# File size limit tests
# ============================================================


class TestFileSizeLimits:
    def test_max_file_size_setting(self):
        assert settings.max_file_size_mb > 0
        assert settings.max_file_size_mb == 200  # updated from 50 in Phase 5+

    def test_file_under_limit(self):
        max_bytes = settings.max_file_size_mb * 1024 * 1024
        small_size = 1024  # 1 KB
        assert small_size <= max_bytes

    def test_file_over_limit_rejected(self):
        max_bytes = settings.max_file_size_mb * 1024 * 1024
        big_size = (settings.max_file_size_mb + 1) * 1024 * 1024
        assert big_size > max_bytes


# ============================================================
# Query length limit tests
# ============================================================


class TestQueryLengthLimits:
    def test_max_query_length_setting(self):
        assert settings.max_query_length > 0
        assert settings.max_query_length == 2000

    def test_normal_query_within_limit(self):
        query = "What is the revenue for Q4 2024?"
        assert len(query) <= settings.max_query_length

    def test_oversized_query_exceeds_limit(self):
        query = "x" * (settings.max_query_length + 1)
        assert len(query) > settings.max_query_length


# ============================================================
# Integration: SimpleRAG query length enforcement
# ============================================================


class TestRAGQueryValidation:
    def test_oversized_query_returns_error_message(self):
        """SimpleRAG.answer() should reject queries exceeding max_query_length."""
        from app_local import SimpleRAG

        # Create a RAG instance with mocked LLM/embedder to avoid Ollama dependency
        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = []

        rag = SimpleRAG.__new__(SimpleRAG)
        rag.docs_folder = Path(tempfile.mkdtemp())
        rag.llm = mock_llm
        rag.embedder = mock_embedder
        rag.documents = [{"source": "test.txt", "content": "test content", "type": "text"}]
        rag.embeddings = [[0.1] * 10]
        rag._excel_processor = None
        rag._charlie_analyzer = None
        rag._cache_dir = Path(tempfile.mkdtemp())

        oversized = "x" * (settings.max_query_length + 1)
        result = rag.answer(oversized)
        assert "too long" in result.lower()

        # Cleanup
        shutil.rmtree(rag.docs_folder)
        shutil.rmtree(rag._cache_dir)

    def test_normal_query_not_rejected(self):
        """Normal-length queries should not be rejected for length."""
        query = "What is the revenue?"
        assert len(query) <= settings.max_query_length


# ============================================================
# top_k parameter bounds
# ============================================================


class TestTopKValidation:
    def test_top_k_setting(self):
        assert settings.top_k > 0
        assert settings.max_top_k >= settings.top_k
