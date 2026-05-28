"""Tests for PDF export module."""

import fitz  # PyMuPDF
import pytest

from financial_analyzer import (
    FinancialData,
    FinancialReport,
    CompositeHealthScore,
)


class TestFinancialPDFExporter:
    """Verify FinancialPDFExporter produces valid PDF bytes."""

    @pytest.fixture
    def exporter(self):
        from export_pdf import FinancialPDFExporter

        return FinancialPDFExporter()

    @pytest.fixture
    def sample_data(self):
        return FinancialData(
            total_assets=1_000_000,
            current_assets=400_000,
            cash=100_000,
            inventory=50_000,
            accounts_receivable=80_000,
            total_liabilities=500_000,
            current_liabilities=200_000,
            total_debt=300_000,
            total_equity=500_000,
            revenue=2_000_000,
            cogs=1_200_000,
            gross_profit=800_000,
            operating_income=400_000,
            net_income=300_000,
            ebit=420_000,
            ebitda=500_000,
            interest_expense=20_000,
        )

    @pytest.fixture
    def sample_results(self):
        return {
            "current_ratio": 2.0,
            "quick_ratio": 1.75,
            "gross_margin": 0.40,
            "net_margin": 0.15,
        }

    # ------------------------------------------------------------------
    # export_full_report
    # ------------------------------------------------------------------

    def test_export_full_report_returns_bytes(
        self, exporter, sample_data, sample_results
    ):
        result = exporter.export_full_report(sample_data, sample_results)
        assert isinstance(result, bytes)
        assert len(result) > 500  # A real PDF with content should be at least 500 bytes
        assert result[:5] == b"%PDF-"

    def test_export_full_report_with_report(
        self, exporter, sample_data, sample_results
    ):
        report = FinancialReport(
            executive_summary="Strong quarter with solid revenue growth.",
            sections={"Overview": "Test"},
        )
        result = exporter.export_full_report(
            sample_data, sample_results, report=report
        )
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_export_full_report_without_report(
        self, exporter, sample_data, sample_results
    ):
        result = exporter.export_full_report(
            sample_data, sample_results, report=None
        )
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_export_empty_results(self, exporter, sample_data):
        result = exporter.export_full_report(sample_data, {})
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_export_full_report_with_health_score(
        self, exporter, sample_data
    ):
        results = {
            "current_ratio": 2.0,
            "composite_health": CompositeHealthScore(
                score=78,
                grade="B",
                component_scores={"liquidity": 80, "profitability": 75},
                interpretation="Solid health.",
            ),
        }
        result = exporter.export_full_report(sample_data, results)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_export_full_report_with_z_score(self, exporter, sample_data):
        from financial_analyzer import AltmanZScore

        results = {
            "altman_z_score": AltmanZScore(
                z_score=3.1,
                zone="safe",
                components={"x1": 0.1},
                interpretation="Safe zone.",
            ),
        }
        result = exporter.export_full_report(sample_data, results)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_export_full_report_with_f_score(self, exporter, sample_data):
        from financial_analyzer import PiotroskiFScore

        results = {
            "piotroski_f_score": PiotroskiFScore(
                score=7,
                criteria={"roa_positive": True},
                interpretation="Strong.",
            ),
        }
        result = exporter.export_full_report(sample_data, results)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    # ------------------------------------------------------------------
    # export_executive_summary
    # ------------------------------------------------------------------

    def test_export_executive_summary(self, exporter):
        report = FinancialReport(
            executive_summary="Strong Q4.",
            sections={},
        )
        health = CompositeHealthScore(
            score=78,
            grade="B",
            component_scores={"liquidity": 80},
        )
        result = exporter.export_executive_summary(report, health)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_export_executive_summary_no_health(self, exporter):
        report = FinancialReport(
            executive_summary="Test summary text.",
            sections={},
        )
        result = exporter.export_executive_summary(report)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_export_executive_summary_with_interpretation(self, exporter):
        report = FinancialReport(
            executive_summary="Detailed review.",
            sections={},
        )
        health = CompositeHealthScore(
            score=55,
            grade="C",
            component_scores={"liquidity": 40, "profitability": 70},
            interpretation="Below average liquidity needs attention.",
        )
        result = exporter.export_executive_summary(report, health)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_export_executive_summary_empty_summary(self, exporter):
        report = FinancialReport(executive_summary="", sections={})
        result = exporter.export_executive_summary(report)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    # ------------------------------------------------------------------
    # Many ratios to trigger pagination
    # ------------------------------------------------------------------

    def test_export_full_report_many_ratios(
        self, exporter, sample_data
    ):
        """Ensure multi-page rendering works with many metrics."""
        results = {f"metric_{i}_ratio": float(i) * 0.01 for i in range(50)}
        result = exporter.export_full_report(sample_data, results)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"



# ---------------------------------------------------------------------------
# WS-3 P0-8: current_ratio renders as multiplier (1.50x), not percent (150.00%)
# ---------------------------------------------------------------------------

class TestRatioFormatting_P0_8:
    """Regression: PDF must format ratios as 'X.XXx' not 'XXX.XX%'."""

    def _exporter(self):
        from export_pdf import FinancialPDFExporter
        return FinancialPDFExporter()

    def test_current_ratio_renders_as_multiplier(self):
        out = self._exporter()._format_value("current_ratio", 1.5)
        assert out == "1.50x"
        assert "%" not in out

    def test_quick_ratio_renders_as_multiplier(self):
        out = self._exporter()._format_value("quick_ratio", 0.85)
        assert out == "0.85x"
        assert "%" not in out

    def test_debt_to_equity_renders_as_multiplier(self):
        out = self._exporter()._format_value("debt_to_equity", 2.3)
        assert out == "2.30x"

    def test_debt_ratio_renders_as_multiplier(self):
        out = self._exporter()._format_value("debt_ratio", 0.4)
        assert out == "0.40x"
        assert "%" not in out

    def test_gross_margin_still_renders_as_percent(self):
        out = self._exporter()._format_value("gross_margin", 0.45)
        assert out == "45.00%"

    def test_revenue_still_renders_as_dollar(self):
        out = self._exporter()._format_value("revenue", 1_500_000)
        assert "$" in out

    def test_none_value_renders_na(self):
        assert self._exporter()._format_value("current_ratio", None) == "N/A"


# ---------------------------------------------------------------------------
# WS-3 P1-E3: PDF unicode safety -- _sanitize_text + rendered-PDF extraction
# ---------------------------------------------------------------------------

class TestUnicodeSanitization_P1_E3:
    """Regression: non-latin-1 chars must be ASCII-substituted, not crash/corrupt."""

    def _exporter(self):
        from export_pdf import FinancialPDFExporter
        return FinancialPDFExporter()

    # (1) Exact substitution-table assertions ------------------------------

    def test_sanitize_em_dash(self):
        from export_pdf import _sanitize_text
        assert _sanitize_text("Acme—Corp") == "Acme-Corp"

    def test_sanitize_en_dash(self):
        from export_pdf import _sanitize_text
        assert _sanitize_text("2024–2025") == "2024-2025"

    def test_sanitize_micro_sign(self):
        from export_pdf import _sanitize_text
        assert _sanitize_text("5µm") == "5um"

    def test_sanitize_smart_quotes(self):
        from export_pdf import _sanitize_text
        assert _sanitize_text("‘a’ “b”") == "'a' \"b\""

    def test_sanitize_comparison_operators(self):
        from export_pdf import _sanitize_text
        assert _sanitize_text("x ≥ 1 and y ≤ 2") == "x >= 1 and y <= 2"

    def test_sanitize_ellipsis(self):
        from export_pdf import _sanitize_text
        assert _sanitize_text("wait…") == "wait..."

    def test_sanitize_latin1_backstop(self):
        from export_pdf import _sanitize_text
        # An arbitrary non-latin-1 char (CJK) must be replaced, not survive.
        out = _sanitize_text("price中")
        assert out.encode("latin-1")  # must not raise -> all bytes <= 0xFF
        assert "中" not in out

    def test_sanitize_is_idempotent(self):
        from export_pdf import _sanitize_text
        once = _sanitize_text("a—bµc")
        assert _sanitize_text(once) == once == "a-buc"

    # (2) Rendered-PDF extraction assertions -------------------------------

    @staticmethod
    def _extract_text(pdf_bytes: bytes) -> str:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)

    def test_rendered_pdf_substitutes_and_stays_latin1(self):
        exporter = self._exporter()
        data = FinancialData(
            total_assets=1_000_000,
            total_liabilities=500_000,
            total_equity=500_000,
            revenue=2_000_000,
            net_income=300_000,
        )
        # em-dash + micro sign + smart quotes, kept within first 50 chars.
        summary = "Acme—µ “Q4” review of margins and runway."
        report = FinancialReport(executive_summary=summary, sections={})

        pdf_bytes = exporter.export_full_report(data, {"current_ratio": 2.0}, report=report)
        assert pdf_bytes[:5] == b"%PDF-"

        text = self._extract_text(pdf_bytes)
        # ASCII-substituted form must be present.
        assert "Acme-u" in text
        assert "\"Q4\"" in text
        # No original non-latin-1 byte may survive anywhere in the document.
        for ch in ("—", "µ", "“", "”"):
            assert ch not in text
        # Entire extracted layer must be latin-1 encodable.
        text.encode("latin-1")

    def test_rendered_pdf_table_cell_sanitized(self):
        exporter = self._exporter()
        data = FinancialData(
            total_assets=1_000_000,
            total_liabilities=500_000,
            total_equity=500_000,
            revenue=2_000_000,
            net_income=300_000,
        )
        # Non-numeric value flows through _add_table cell loop (before 50-char cut).
        results = {"altman_z_score": {"interpretation": "Safe—zone ≥ 2.99"}}
        pdf_bytes = exporter.export_full_report(data, results)
        text = self._extract_text(pdf_bytes)
        assert "Safe-zone >= 2.99" in text
        assert "—" not in text and "≥" not in text
        text.encode("latin-1")
