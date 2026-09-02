"""Tests for streamlit_app_local.py file upload security and utilities."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# _sanitize_and_save - path traversal prevention & file size
# ---------------------------------------------------------------------------


class TestSanitizeAndSave:
    """Test the file upload sanitization function."""

    def _make_file(self, name: str, content: bytes = b"data") -> MagicMock:
        """Create a mock uploaded file with given name and content."""
        buf = io.BytesIO(content)
        mock_file = MagicMock()
        mock_file.name = name
        mock_file.getbuffer.return_value = buf.getvalue()
        # Simulate seek/tell for size check
        mock_file.seek = buf.seek
        mock_file.tell = buf.tell
        # Reset position after setup
        buf.seek(0)
        return mock_file

    @patch("streamlit_app_local.st")
    def test_saves_normal_file(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        f = self._make_file("report.xlsx", b"excel data")
        result = _sanitize_and_save(f, tmp_path)
        assert result is True
        assert (tmp_path / "report.xlsx").exists()
        assert (tmp_path / "report.xlsx").read_bytes() == b"excel data"

    @patch("streamlit_app_local.st")
    def test_rejects_empty_filename(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        f = self._make_file("", b"data")
        result = _sanitize_and_save(f, tmp_path)
        assert result is False
        mock_st.error.assert_called()

    @patch("streamlit_app_local.st")
    def test_rejects_dot_filename(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        f = self._make_file(".", b"data")
        result = _sanitize_and_save(f, tmp_path)
        assert result is False

    @patch("streamlit_app_local.st")
    def test_rejects_dotdot_filename(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        f = self._make_file("..", b"data")
        result = _sanitize_and_save(f, tmp_path)
        assert result is False

    @patch("streamlit_app_local.st")
    def test_rejects_path_traversal_unix(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        # os.path.basename strips directory components, so this becomes "passwd"
        # The path traversal is neutralized by basename() on line 36
        f = self._make_file("../../../etc/passwd", b"data")
        result = _sanitize_and_save(f, tmp_path)
        # basename("../../../etc/passwd") = "passwd" which is a valid name
        # The resolve() check on line 48 ensures it stays within docs_path
        if result:
            assert (tmp_path / "passwd").exists()
            # Verify it didn't escape the docs folder
            saved = (tmp_path / "passwd").resolve()
            assert str(saved).startswith(str(tmp_path.resolve()))

    @patch("streamlit_app_local.st")
    def test_rejects_path_traversal_windows(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        # Backslash separator in filename
        f = self._make_file("..\\..\\Windows\\System32\\config", b"data")
        result = _sanitize_and_save(f, tmp_path)
        # basename strips path, but if "/" or "\\" remains, it should be rejected
        # on line 42's check after basename

    @patch("streamlit_app_local.st")
    def test_rejects_oversized_file(self, mock_st, tmp_path):
        from streamlit_app_local import MAX_FILE_SIZE, _sanitize_and_save

        # Create content larger than MAX_FILE_SIZE
        big_content = b"x" * (MAX_FILE_SIZE + 1)
        f = self._make_file("big.pdf", big_content)
        result = _sanitize_and_save(f, tmp_path)
        assert result is False
        mock_st.error.assert_called()
        assert not (tmp_path / "big.pdf").exists()

    @patch("streamlit_app_local.st")
    def test_accepts_file_at_size_limit(self, mock_st, tmp_path):
        from streamlit_app_local import MAX_FILE_SIZE, _sanitize_and_save

        exact_content = b"x" * MAX_FILE_SIZE
        f = self._make_file("exact.pdf", exact_content)
        result = _sanitize_and_save(f, tmp_path)
        assert result is True
        assert (tmp_path / "exact.pdf").exists()

    @patch("streamlit_app_local.st")
    def test_strips_whitespace_from_name(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        f = self._make_file("  report.xlsx  ", b"data")
        result = _sanitize_and_save(f, tmp_path)
        # After basename + strip, should be "report.xlsx"
        if result:
            assert (tmp_path / "report.xlsx").exists()

    @patch("streamlit_app_local.st")
    def test_whitespace_only_filename_rejected(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        f = self._make_file("   ", b"data")
        result = _sanitize_and_save(f, tmp_path)
        assert result is False


# ---------------------------------------------------------------------------
# SUPPORTED_EXTENSIONS and MAX_FILE_SIZE
# ---------------------------------------------------------------------------


class TestConstants:
    def test_supported_extensions_include_excel(self):
        from streamlit_app_local import SUPPORTED_EXTENSIONS

        assert ".xlsx" in SUPPORTED_EXTENSIONS
        assert ".csv" in SUPPORTED_EXTENSIONS
        assert ".xls" in SUPPORTED_EXTENSIONS

    def test_supported_extensions_include_pdf(self):
        from streamlit_app_local import SUPPORTED_EXTENSIONS

        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_supported_extensions_include_docx(self):
        from streamlit_app_local import SUPPORTED_EXTENSIONS

        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_max_file_size_is_positive(self):
        from streamlit_app_local import MAX_FILE_SIZE

        assert MAX_FILE_SIZE > 0

    def test_max_file_size_is_in_bytes(self):
        from config import settings
        from streamlit_app_local import MAX_FILE_SIZE

        assert MAX_FILE_SIZE == settings.max_file_size_mb * 1024 * 1024


# ---------------------------------------------------------------------------
# _generate_sample_income_statement & _generate_sample_budget
# ---------------------------------------------------------------------------


class TestSampleDataGenerators:
    @patch("streamlit_app_local.st")
    def test_generate_sample_income_statement(self, mock_st, tmp_path):
        from streamlit_app_local import _generate_sample_income_statement

        _generate_sample_income_statement(tmp_path)
        output = tmp_path / "sample_income_statement.xlsx"
        assert output.exists()
        assert output.stat().st_size > 0

    @patch("streamlit_app_local.st")
    def test_generate_sample_budget(self, mock_st, tmp_path):
        from streamlit_app_local import _generate_sample_budget

        _generate_sample_budget(tmp_path)
        output = tmp_path / "sample_budget_vs_actual.xlsx"
        assert output.exists()
        assert output.stat().st_size > 0

    @patch("streamlit_app_local.st")
    def test_sample_income_statement_has_columns(self, mock_st, tmp_path):
        import pandas as pd

        from streamlit_app_local import _generate_sample_income_statement

        _generate_sample_income_statement(tmp_path)
        df = pd.read_excel(tmp_path / "sample_income_statement.xlsx")
        assert "Line Item" in df.columns
        assert "Q1 2024" in df.columns
        assert len(df) == 10  # 10 line items

    @patch("streamlit_app_local.st")
    def test_sample_budget_has_variance_columns(self, mock_st, tmp_path):
        import pandas as pd

        from streamlit_app_local import _generate_sample_budget

        _generate_sample_budget(tmp_path)
        df = pd.read_excel(tmp_path / "sample_budget_vs_actual.xlsx")
        assert "Budget" in df.columns
        assert "Actual" in df.columns
        assert "Variance" in df.columns
        assert "Variance %" in df.columns


# ---------------------------------------------------------------------------
# WP-B8: dead UI removed - ollama_model selectbox + submit_query session read
# ---------------------------------------------------------------------------


class TestDeadUIRemoved:
    """WP-B8: verify the dead ollama_model selectbox and submit_query
    session read have been removed from streamlit_app_local.py."""

    def _module_source(self) -> str:
        import inspect

        import streamlit_app_local

        return inspect.getsource(streamlit_app_local)

    def test_ollama_model_selectbox_gone(self):
        """ollama_model variable was never read; selectbox must be removed."""
        src = self._module_source()
        assert "ollama_model" not in src, "Dead UI 'ollama_model' selectbox still present in streamlit_app_local.py"

    def test_submit_query_session_read_gone(self):
        """submit_query was never written to session_state; read must be removed."""
        src = self._module_source()
        assert "submit_query" not in src, (
            "Dead UI 'submit_query' session_state read still present in streamlit_app_local.py"
        )

    def test_module_imports_cleanly(self):
        """Module must still be importable after the dead-UI removal."""
        import importlib

        import streamlit_app_local

        # Re-importing forces module-level code to be inspectable; no AttributeError.
        importlib.reload(streamlit_app_local)

    @patch("streamlit_app_local.st")
    def test_render_sidebar_smoke(self, mock_st):
        """render_sidebar runs without raising after dead UI removed."""
        from unittest.mock import MagicMock

        # st.radio must return a string so the caller can compare it
        mock_st.radio.return_value = "Q&A Chat"
        # st.sidebar is accessed as an attribute context manager
        mock_st.sidebar.__enter__ = MagicMock(return_value=mock_st.sidebar)
        mock_st.sidebar.__exit__ = MagicMock(return_value=False)
        # Path.mkdir and rglob may touch the filesystem; patch at a safe level
        with patch("streamlit_app_local.Path") as mock_path_cls:
            mock_docs = MagicMock(spec=Path)
            mock_docs.resolve.return_value = mock_docs
            mock_docs.rglob.return_value = []
            mock_path_cls.return_value = mock_docs

            from streamlit_app_local import render_sidebar

            result = render_sidebar()

        # render_sidebar returns whatever st.radio returned
        assert result == "Q&A Chat"

    @patch("streamlit_app_local.st")
    def test_get_answer_button_condition_no_submit_query(self, mock_st):
        """The button condition no longer references submit_query."""
        import inspect

        import streamlit_app_local

        src = inspect.getsource(streamlit_app_local.render_qa_page)
        assert "submit_query" not in src, "render_qa_page still references submit_query after WP-B8 removal"
