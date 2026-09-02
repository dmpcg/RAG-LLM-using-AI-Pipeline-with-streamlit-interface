"""Tests for ingestion_pipeline module."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

import pandas as pd

from ingestion_pipeline import (
    ingest_excel,
    ingest_text,
    ingest_file,
    chunks_to_documents,
    _detect_sheet_section_type,
    _find_label_column,
    _df_to_markdown,
    _read_csv_robust,
    _detect_excel_header_row,
    EXCEL_EXTENSIONS,
    PDF_EXTENSIONS,
    TEXT_EXTENSIONS,
)
from document_chunker import RAGChunk


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
        df = pd.DataFrame({
            "Label": ["Total Revenue", "Cost of Goods Sold", "Gross Profit", "Net Income"],
            "2024": [1000, 500, 500, 200],
        })
        result = _detect_sheet_section_type(df, "Sheet1")
        assert result == "income_statement"

    def test_unknown_sheet(self):
        df = pd.DataFrame({"Col1": [1, 2, 3]})
        result = _detect_sheet_section_type(df, "Sheet1")
        assert result == "general"


class TestFindLabelColumn:
    def test_first_column_labels(self):
        df = pd.DataFrame({
            "Items": ["Revenue", "COGS", "Gross Profit", "Net Income", "Tax"],
            "2024": [1000, 500, 500, 200, 50],
        })
        result = _find_label_column(df)
        assert result == 0

    def test_second_column_labels(self):
        df = pd.DataFrame({
            "Section": [None, None, None, None, None],
            "Items": ["Revenue", "COGS", "Gross Profit", "Net Income", "Tax"],
            "2024": [1000, 500, 500, 200, 50],
        })
        result = _find_label_column(df)
        assert result == 1

    def test_empty_df(self):
        df = pd.DataFrame()
        result = _find_label_column(df)
        assert result is None

    def test_all_numeric(self):
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
        })
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
            df = pd.DataFrame({
                "Item": ["Revenue", "COGS", "Gross Profit"],
                "2024": [1000, 500, 500],
                "2023": [900, 400, 500],
            })
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


def _write_csv(text: str) -> Path:
    """Write CSV text to a temp file and return its path."""
    path = Path(tempfile.mkdtemp()) / "sample.csv"
    path.write_text(text, encoding="utf-8")
    return path


class TestReadCsvRobust:
    """Preamble/footer-aware CSV reader.

    The reader replaced pd.read_csv to strip banner and footer rows, so these
    tests pin the behaviour pandas gave us for free: no data row may be lost.
    """

    def test_short_row_is_kept_not_dropped(self):
        # A row missing a trailing field is a transaction with a blank, not a
        # footer. pandas kept it as NaN; dropping it loses money silently.
        path = _write_csv(
            "Date,Description,Amount\n"
            "2026-01-01,Coffee,4.50\n"
            "2026-01-02,Rent\n"
            "2026-01-03,Payroll,1200.00\n"
        )
        df = _read_csv_robust(path, ",")
        assert len(df) == 3
        assert df["Description"].str.contains("Rent").any()

    def test_banner_and_footer_are_dropped(self):
        path = _write_csv(
            "ACME Corp Q1 Report\n"
            "Date,Description,Amount\n"
            "2026-01-01,Coffee,4.50\n"
            "2026-01-02,Rent\n"
            "Grand Total\n"
        )
        df = _read_csv_robust(path, ",")
        assert list(df.columns) == ["Date", "Description", "Amount"]
        assert len(df) == 2
        assert not df.astype(str).apply(lambda c: c.str.contains("Grand Total")).any().any()

    def test_width_tie_does_not_collapse_table(self):
        # Two single-cell rows and two 3-field rows: a plain modal count ties
        # and can pick width 1, collapsing the sheet to one column.
        path = _write_csv(
            "Title Banner\n"
            "A,B,C\n"
            "1,2,3\n"
            "Footer Note\n"
        )
        df = _read_csv_robust(path, ",")
        assert list(df.columns) == ["A", "B", "C"]
        assert len(df) == 1

    def test_wellformed_csv_is_unchanged(self):
        path = _write_csv("A,B,C\n1,2,3\n4,5,6\n")
        df = _read_csv_robust(path, ",")
        assert list(df.columns) == ["A", "B", "C"]
        assert len(df) == 2

    def test_row_count_is_bounded(self):
        from config import settings

        rows = "\n".join(f"{i},item{i},{i * 1.5}" for i in range(settings.max_workbook_rows + 250))
        path = _write_csv("Id,Name,Value\n" + rows + "\n")
        df = _read_csv_robust(path, ",")
        assert len(df) <= settings.max_workbook_rows

    def test_quoted_delimiter_is_honoured(self):
        path = _write_csv('Date,Description,Amount\n2026-01-01,"Rent, office",1200.00\n')
        df = _read_csv_robust(path, ",")
        assert len(df) == 1
        assert df.iloc[0]["Description"] == "Rent, office"


class TestFindLabelColumnDtypeIndependence:
    """_find_label_column must judge parsed values, not storage dtype.

    _read_csv_robust returns an all-str frame, so an amounts column arrives as
    strings and must not win over the real line-item column.
    """

    def test_string_stored_numbers_do_not_win(self):
        df = pd.DataFrame(
            [["%.2f" % (i * 10.5), f"Line item {i}"] for i in range(1, 11)],
            columns=["Amount", "Note"],
        )
        assert _find_label_column(df) == 1

    def test_native_dtype_frame_unchanged(self):
        df = pd.DataFrame(
            [[i * 10.5, f"Line item {i}"] for i in range(1, 11)],
            columns=["Amount", "Note"],
        )
        assert _find_label_column(df) == 1


class TestDetectExcelHeaderRow:
    def test_dense_first_row_returns_zero(self):
        raw = pd.DataFrame([["Line", "2026", "2025", "Var"], ["Revenue", 1, 2, 3]])
        assert _detect_excel_header_row(raw) == 0

    def test_banner_rows_are_skipped(self):
        raw = pd.DataFrame(
            [
                ["ACME P&L", None, None, None],
                ["Source: QuickBooks", None, None, None],
                ["Line", "2026", "2025", "Var"],
                ["Revenue", 1, 2, 3],
            ]
        )
        assert _detect_excel_header_row(raw) == 2

    def test_narrow_statement_left_at_zero(self):
        # Label/value statements have no wide header row; skipping would eat data.
        raw = pd.DataFrame([["Revenue", 100.0], ["COGS", 40.0], ["Net", 60.0]])
        assert _detect_excel_header_row(raw) == 0
