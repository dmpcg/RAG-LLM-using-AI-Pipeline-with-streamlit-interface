"""Tests for ingestion_pipeline module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from document_chunker import RAGChunk
from ingestion_pipeline import (
    EXCEL_EXTENSIONS,
    PDF_EXTENSIONS,
    TEXT_EXTENSIONS,
    _detect_sheet_section_type,
    _df_to_markdown,
    _find_label_column,
    chunks_to_documents,
    ingest_excel,
    ingest_file,
    ingest_text,
)


class TestDetectSheetSectionType:
    def test_sheet_name_income(self):
        df = pd.DataFrame({"A": [1, 2]})
        assert _detect_sheet_section_type(df, "Income Statement") == "income_statement"

    def test_sheet_name_balance(self):
        df = pd.DataFrame({"A": [1, 2]})
        assert _detect_sheet_section_type(df, "Balance Sheet") == "balance_sheet"

    def test_sheet_name_cash_flow(self):
        df = pd.DataFrame({"A": [1, 2]})
        assert _detect_sheet_section_type(df, "Cash Flow") == "cash_flow_statement"

    def test_sheet_name_budget(self):
        df = pd.DataFrame({"A": [1, 2]})
        assert _detect_sheet_section_type(df, "Annual Budget") == "budget"

    def test_sheet_name_280e(self):
        df = pd.DataFrame({"A": [1, 2]})
        assert _detect_sheet_section_type(df, "280E Classification") == "280e_tax"

    def test_sheet_name_dcf(self):
        df = pd.DataFrame({"A": [1, 2]})
        assert _detect_sheet_section_type(df, "DCF Model") == "dcf"

    def test_sheet_name_lbo(self):
        df = pd.DataFrame({"A": [1, 2]})
        assert _detect_sheet_section_type(df, "LBO Returns") == "lbo"

    def test_content_based_detection(self):
        df = pd.DataFrame(
            {
                "Label": ["Total Revenue", "Cost of Goods Sold", "Gross Profit", "Net Income"],
                "2024": [1000, 500, 500, 200],
            }
        )
        result = _detect_sheet_section_type(df, "Sheet1")
        assert result == "income_statement"

    def test_unknown_sheet(self):
        df = pd.DataFrame({"Col1": [1, 2, 3]})
        result = _detect_sheet_section_type(df, "Sheet1")
        assert result == "general"


class TestFindLabelColumn:
    def test_first_column_labels(self):
        df = pd.DataFrame(
            {
                "Items": ["Revenue", "COGS", "Gross Profit", "Net Income", "Tax"],
                "2024": [1000, 500, 500, 200, 50],
            }
        )
        result = _find_label_column(df)
        assert result == 0

    def test_second_column_labels(self):
        df = pd.DataFrame(
            {
                "Section": [None, None, None, None, None],
                "Items": ["Revenue", "COGS", "Gross Profit", "Net Income", "Tax"],
                "2024": [1000, 500, 500, 200, 50],
            }
        )
        result = _find_label_column(df)
        assert result == 1

    def test_empty_df(self):
        df = pd.DataFrame()
        result = _find_label_column(df)
        assert result is None

    def test_all_numeric(self):
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
            }
        )
        result = _find_label_column(df)
        assert result is None


class TestDfToMarkdown:
    def test_basic(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        md = _df_to_markdown(df)
        assert "|" in md
        assert "A" in md

    def test_empty(self):
        df = pd.DataFrame()
        md = _df_to_markdown(df)
        assert md == ""

    def test_max_rows(self):
        df = pd.DataFrame({"A": range(500)})
        md = _df_to_markdown(df, max_rows=10)
        lines = [l for l in md.split("\n") if l.strip() and "|" in l]
        # Header + separator + 10 data rows = 12 lines
        assert len(lines) <= 15


class TestIngestExcel:
    def test_csv_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Item,2024,2023\nRevenue,1000,900\nCOGS,500,400\nGross Profit,500,500\n")
            f.flush()
            path = Path(f.name)

        try:
            chunks = ingest_excel(path)
            assert len(chunks) >= 1
            assert all(isinstance(c, RAGChunk) for c in chunks)
            # Check source is set
            assert all(c.source == path.name for c in chunks)
        finally:
            os.unlink(path)

    def test_xlsx_file(self):
        import gc

        path = Path(tempfile.mktemp(suffix=".xlsx"))

        try:
            df = pd.DataFrame(
                {
                    "Item": ["Revenue", "COGS", "Gross Profit"],
                    "2024": [1000, 500, 500],
                    "2023": [900, 400, 500],
                }
            )
            df.to_excel(path, index=False, sheet_name="Income Statement")
            chunks = ingest_excel(path)
            assert len(chunks) >= 1
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                pass  # Windows file lock - temp file will be cleaned up later

    def test_unnamed_columns_fixed(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # CSV without proper headers (simulating Excel export)
            f.write(",Revenue,COGS\n,1000,500\n")
            f.flush()
            path = Path(f.name)

        try:
            chunks = ingest_excel(path)
            # Should not crash, columns should be renamed
            assert len(chunks) >= 1
            # Check no "Unnamed" in chunk text
            for c in chunks:
                assert "Unnamed:" not in c.text
        finally:
            os.unlink(path)


class TestIngestText:
    def test_text_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("This is a test document with some financial information about revenue and costs.")
            f.flush()
            path = Path(f.name)

        try:
            chunks = ingest_text(path)
            assert len(chunks) >= 1
            assert all(isinstance(c, RAGChunk) for c in chunks)
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            path = Path(f.name)

        try:
            chunks = ingest_text(path)
            assert len(chunks) == 0
        finally:
            os.unlink(path)

    def test_markdown_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Financial Report\n\nRevenue was $1M in Q4 2024.")
            f.flush()
            path = Path(f.name)

        try:
            chunks = ingest_text(path)
            assert len(chunks) >= 1
        finally:
            os.unlink(path)


class TestIngestFile:
    def test_dispatches_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Item,Value\nRevenue,1000\n")
            f.flush()
            path = Path(f.name)

        try:
            chunks = ingest_file(path)
            assert len(chunks) >= 1
        finally:
            os.unlink(path)

    def test_dispatches_txt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Hello world.")
            f.flush()
            path = Path(f.name)

        try:
            chunks = ingest_file(path)
            assert len(chunks) >= 1
        finally:
            os.unlink(path)

    def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = Path(f.name)

        try:
            chunks = ingest_file(path)
            assert len(chunks) == 0
        finally:
            os.unlink(path)


class TestChunksToDocuments:
    def test_basic_conversion(self):
        chunks = [
            RAGChunk(
                chunk_id="abc123",
                text="Revenue was $1M.",
                source="report.pdf",
                section_type="income_statement",
                section_title="Revenue",
                metadata={"file_type": "pdf"},
            )
        ]
        docs = chunks_to_documents(chunks)
        assert len(docs) == 1
        doc = docs[0]
        assert doc["source"] == "report.pdf"
        assert doc["content"] == "Revenue was $1M."
        assert doc["type"] == "pdf"
        assert doc["metadata"]["chunk_id"] == "abc123"
        assert doc["metadata"]["section_type"] == "income_statement"

    def test_nl_description_prepended(self):
        chunks = [
            RAGChunk(
                chunk_id="def456",
                text="| Rev | 1000 |",
                source="test.xlsx",
                nl_description="Contains revenue data.",
                metadata={"file_type": "excel"},
            )
        ]
        docs = chunks_to_documents(chunks)
        assert len(docs) == 1
        # NL description should be prepended to content
        assert docs[0]["content"].startswith("Contains revenue data.")
        assert "| Rev | 1000 |" in docs[0]["content"]

    def test_parent_info_preserved(self):
        chunks = [
            RAGChunk(
                chunk_id="child1",
                text="Small chunk.",
                parent_id="parent1",
                parent_text="Full parent text here.",
                source="test.pdf",
                metadata={"file_type": "pdf"},
            )
        ]
        docs = chunks_to_documents(chunks)
        assert docs[0]["metadata"]["parent_id"] == "parent1"
        assert docs[0]["metadata"]["parent_text"] == "Full parent text here."

    def test_empty_chunks(self):
        docs = chunks_to_documents([])
        assert len(docs) == 0


class TestExtensions:
    def test_excel_extensions(self):
        assert ".xlsx" in EXCEL_EXTENSIONS
        assert ".xlsm" in EXCEL_EXTENSIONS
        assert ".xls" in EXCEL_EXTENSIONS
        assert ".csv" in EXCEL_EXTENSIONS
        assert ".tsv" in EXCEL_EXTENSIONS

    def test_pdf_extensions(self):
        assert ".pdf" in PDF_EXTENSIONS

    def test_text_extensions(self):
        assert ".txt" in TEXT_EXTENSIONS
        assert ".md" in TEXT_EXTENSIONS
        assert ".docx" in TEXT_EXTENSIONS


# ---------------------------------------------------------------------------
# WP-PDF P1-C3 tests -- ingestion_pipeline
# ---------------------------------------------------------------------------


class TestIngestPdfExcInfo:
    """ingest_pdf error path must call logger.error with exc_info=True."""

    def test_ingest_pdf_logs_exc_info_on_failure(self, tmp_path):
        """When parse_pdf raises, ingest_pdf logs with exc_info=True."""
        import ingestion_pipeline as _ip_mod
        from ingestion_pipeline import ingest_pdf

        dummy = tmp_path / "bad.pdf"
        dummy.write_bytes(b"not a pdf")

        with (
            patch.object(_ip_mod.logger, "error") as mock_log,
            patch("pdf_parser.parse_pdf", side_effect=RuntimeError("parse fail")),
        ):
            chunks = ingest_pdf(dummy)

        assert chunks == []
        # logger.error must have been called at least once with exc_info=True
        assert mock_log.called, "logger.error was not called"
        calls = mock_log.call_args_list
        assert any(
            call.kwargs.get("exc_info") is True or (len(call.args) > 0 and call.kwargs.get("exc_info", False))
            for call in calls
        ), "logger.error was not called with exc_info=True"

    def test_ingest_pdf_returns_empty_list_on_parse_failure(self, tmp_path):
        """ingest_pdf returns [] and does not re-raise when parse_pdf raises."""
        from ingestion_pipeline import ingest_pdf

        dummy = tmp_path / "fail.pdf"
        dummy.write_bytes(b"junk")

        with patch("pdf_parser.parse_pdf", side_effect=ValueError("bad pdf")):
            try:
                result = ingest_pdf(dummy)
            except Exception as exc:
                pytest.fail(f"ingest_pdf raised unexpectedly: {exc!r}")

        assert result == []
