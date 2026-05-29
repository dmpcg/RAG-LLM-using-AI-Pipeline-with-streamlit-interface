"""Tests for pdf_parser module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from pdf_parser import (
    SECTION_PATTERNS,
    ParsedDocument,
    _detect_section_type,
    _detect_tables_in_section,
    _extract_metadata,
    _split_into_sections,
)


class TestDetectSectionType:
    def test_income_statement(self):
        assert _detect_section_type("Consolidated Statements of Income\nRevenue...") == "income_statement"
        assert _detect_section_type("INCOME STATEMENT\nLine items...") == "income_statement"
        assert _detect_section_type("Profit and Loss Statement\n...") == "income_statement"

    def test_balance_sheet(self):
        assert _detect_section_type("Consolidated Balance Sheets\n...") == "balance_sheet"
        assert _detect_section_type("Statement of Financial Position\n...") == "balance_sheet"

    def test_cash_flow(self):
        assert _detect_section_type("Consolidated Statements of Cash Flows\n...") == "cash_flow_statement"
        assert _detect_section_type("Cash Flow Statement\n...") == "cash_flow_statement"

    def test_mda(self):
        assert _detect_section_type("Management's Discussion and Analysis\n...") == "mda"
        assert _detect_section_type("MD&A\nThe following...") == "mda"

    def test_risk_factors(self):
        assert _detect_section_type("Risk Factors\nThe company faces...") == "risk_factors"

    def test_dcf(self):
        assert _detect_section_type("Discounted Cash Flow Analysis\n...") == "dcf"
        assert _detect_section_type("DCF Valuation\n...") == "dcf"

    def test_lbo(self):
        assert _detect_section_type("Leveraged Buyout Analysis\n...") == "lbo"
        assert _detect_section_type("LBO Model\n...") == "lbo"

    def test_cannabis_280e(self):
        assert _detect_section_type("280E Tax Classification\n...") == "280e_tax"

    def test_unknown_section(self):
        assert _detect_section_type("Some random text without financial headers") is None

    def test_notes(self):
        assert _detect_section_type("Notes to Financial Statements\n...") == "notes_to_financial_statements"

    def test_auditor_report(self):
        assert _detect_section_type("Report of Independent Registered Public Accounting Firm\n...") == "auditor_report"

    def test_comparable_companies(self):
        assert _detect_section_type("Comparable Companies Analysis\n...") == "comparable_companies"
        assert _detect_section_type("Trading Comps\n...") == "comparable_companies"

    def test_debt_schedule(self):
        assert _detect_section_type("Debt Schedule Summary\n...") == "debt_schedule"
        assert _detect_section_type("Capital Structure Summary\n...") == "debt_schedule"

    def test_covenant(self):
        assert _detect_section_type("Covenant Compliance Summary\n...") == "covenant_analysis"

    def test_rent_roll(self):
        assert _detect_section_type("Rent Roll\nUnit details...") == "rent_roll"

    def test_construction_budget(self):
        assert _detect_section_type("Construction Budget\nCode...") == "construction_budget"

    def test_kpi(self):
        assert _detect_section_type("KPI Dashboard\nMetrics...") == "kpi_dashboard"
        assert _detect_section_type("Key Performance Indicators\n...") == "kpi_dashboard"

    def test_sensitivity(self):
        assert _detect_section_type("Sensitivity Analysis\n...") == "sensitivity_analysis"


class TestExtractMetadata:
    def test_company_name_detection(self):
        text = "\nApple Inc.\nConsolidated Balance Sheets\n"
        meta = _extract_metadata(text, "test.pdf")
        assert meta.get("company") is not None

    def test_period_detection(self):
        text = "For the fiscal year ended December 31, 2024\n"
        meta = _extract_metadata(text, "test.pdf")
        assert "period" in meta
        assert "2024" in meta["period"]

    def test_filing_type_10k(self):
        text = "ANNUAL REPORT PURSUANT TO 10-K\n"
        meta = _extract_metadata(text, "test.pdf")
        assert meta.get("filing_type") == "10-K"

    def test_filing_type_10q(self):
        text = "QUARTERLY REPORT 10-Q\n"
        meta = _extract_metadata(text, "test.pdf")
        assert meta.get("filing_type") == "10-Q"

    def test_empty_text(self):
        meta = _extract_metadata("", "test.pdf")
        assert isinstance(meta, dict)


class TestSplitIntoSections:
    def test_markdown_headings(self):
        text = "# Introduction\nSome text.\n\n## Financial Overview\nMore text.\n\n### Revenue\nDetails."
        sections = _split_into_sections(text)
        assert len(sections) >= 2

    def test_no_headings(self):
        text = "Just a plain paragraph of text without any headings."
        sections = _split_into_sections(text)
        assert len(sections) == 1
        assert sections[0]["title"] == "Document"

    def test_preamble_captured(self):
        # Preamble needs >100 chars of real text before first heading
        preamble = "This is preamble text. " * 10  # ~230 chars
        text = f"{preamble}\n\n# First Section\nContent here with enough words."
        sections = _split_into_sections(text)
        assert any(s["title"] == "Preamble" for s in sections)


class TestDetectTables:
    def test_markdown_table(self):
        content = "Some text\n| Col1 | Col2 |\n|---|---|\n| val1 | val2 |\n| val3 | val4 |\nMore text"
        tables = _detect_tables_in_section(content)
        assert len(tables) == 1
        assert "val1" in tables[0]

    def test_no_table(self):
        content = "Just text without any table formatting."
        tables = _detect_tables_in_section(content)
        assert len(tables) == 0

    def test_multiple_tables(self):
        content = "| A | B |\n|---|---|\n| 1 | 2 |\n\nSome text between tables\n\n| C | D |\n|---|---|\n| 3 | 4 |\n"
        tables = _detect_tables_in_section(content)
        assert len(tables) == 2


class TestSectionPatterns:
    def test_all_categories_have_patterns(self):
        for section_type, patterns in SECTION_PATTERNS.items():
            assert len(patterns) > 0, f"Section {section_type} has no patterns"
            for p in patterns:
                assert hasattr(p, "search"), f"Pattern in {section_type} is not a regex"

    def test_pattern_count(self):
        total = sum(len(p) for p in SECTION_PATTERNS.values())
        assert total >= 50, f"Expected 50+ patterns, got {total}"


# ---------------------------------------------------------------------------
# WP-PDF P1-C3 tests
# ---------------------------------------------------------------------------


class TestParsePdfCorruptInput:
    """parse_pdf must return an empty ParsedDocument for corrupt/unreadable PDFs."""

    def _make_empty_doc(self, path_str: str) -> "ParsedDocument":
        """Helper: what an empty ParsedDocument looks like."""
        return ParsedDocument(
            source_path=path_str,
            title=Path(path_str).stem,
            company=None,
            period=None,
            total_pages=0,
            sections=[],
            raw_markdown="",
        )

    def test_corrupt_pdf_bytes_fitz_file_data_error(self, tmp_path):
        """Garbage bytes trigger fitz.FileDataError -> empty ParsedDocument, no raise."""
        import fitz

        from pdf_parser import parse_pdf

        corrupt = tmp_path / "corrupt.pdf"
        corrupt.write_bytes(b"not a real pdf garbage bytes 1234")

        # Ensure fitz.open raises fitz.FileDataError for this input so the
        # test is meaningful; if fitz silently recovers we skip.
        try:
            doc = fitz.open(str(corrupt))
            if len(doc) == 0:
                # fitz opened it but produced 0 pages -- still a valid corrupt case
                doc.close()
            else:
                doc.close()
                pytest.skip("fitz did not raise FileDataError for this garbage input")
        except (fitz.FileDataError, RuntimeError):
            pass  # expected -- will be caught inside parse_pdf
        except Exception:
            pytest.skip("fitz raised an unexpected exception type -- not our target")

        # Now confirm parse_pdf does NOT propagate
        result = parse_pdf(corrupt)
        assert isinstance(result, ParsedDocument)
        assert result.sections == []
        assert result.raw_markdown == ""
        assert result.total_pages == 0
        assert result.company is None

    def test_corrupt_pdf_bytes_no_raise(self, tmp_path):
        """parse_pdf with garbage bytes must not raise any exception."""
        from pdf_parser import parse_pdf

        corrupt = tmp_path / "garbage.pdf"
        corrupt.write_bytes(b"\x00\x01\x02\x03" * 50)

        # This must not raise -- any fitz.FileDataError or RuntimeError is swallowed
        try:
            result = parse_pdf(corrupt)
        except Exception as exc:
            pytest.fail(f"parse_pdf raised unexpectedly: {exc!r}")

        assert isinstance(result, ParsedDocument)

    def test_runtime_error_swallowed(self, tmp_path):
        """RuntimeError from fitz/pymupdf4llm is caught and returns empty ParsedDocument."""
        from pdf_parser import parse_pdf

        dummy = tmp_path / "runtime_err.pdf"
        dummy.write_bytes(b"%PDF-1.4 fake")

        with patch("fitz.open", side_effect=RuntimeError("simulated fitz crash")):
            result = parse_pdf(dummy)

        assert isinstance(result, ParsedDocument)
        assert result.sections == []
        assert result.total_pages == 0

    def test_file_data_error_swallowed(self, tmp_path):
        """fitz.FileDataError is caught and returns empty ParsedDocument."""
        import fitz

        from pdf_parser import parse_pdf

        dummy = tmp_path / "file_data_err.pdf"
        dummy.write_bytes(b"%PDF-1.4 fake")

        with patch("fitz.open", side_effect=fitz.FileDataError("bad data")):
            result = parse_pdf(dummy)

        assert isinstance(result, ParsedDocument)
        assert result.sections == []
        assert result.total_pages == 0

    def test_empty_parsed_document_source_path(self, tmp_path):
        """Empty ParsedDocument returned on error has source_path set to the input file."""
        import fitz

        from pdf_parser import parse_pdf

        dummy = tmp_path / "check_path.pdf"
        dummy.write_bytes(b"%PDF-1.4 fake")

        with patch("fitz.open", side_effect=fitz.FileDataError("bad")):
            result = parse_pdf(dummy)

        assert result.source_path == str(dummy)
