"""Tests for PDF export module."""

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
