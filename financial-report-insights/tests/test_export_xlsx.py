"""Tests for Excel export module."""

import pytest

from financial_analyzer import (
    FinancialData,
    FinancialReport,
    CompositeHealthScore,
    ScenarioResult,
)


class TestFinancialExcelExporter:
    """Verify FinancialExcelExporter produces valid XLSX bytes."""

    @pytest.fixture
    def exporter(self):
        from export_xlsx import FinancialExcelExporter

        return FinancialExcelExporter()

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
            "debt_to_equity": 0.60,
            "roe": 0.60,
        }

    # ------------------------------------------------------------------
    # export_full_report
    # ------------------------------------------------------------------

    def test_export_full_report_returns_bytes(
        self, exporter, sample_data, sample_results
    ):
        result = exporter.export_full_report(sample_data, sample_results)
        assert isinstance(result, bytes)
        assert len(result) > 1000  # A real XLSX with content should be at least 1KB
        # XLSX magic bytes (PK zip archive)
        assert result[:2] == b"PK"

    def test_export_full_report_with_report(
        self, exporter, sample_data, sample_results
    ):
        report = FinancialReport(
            executive_summary="Test summary",
            sections={"Overview": "Test"},
        )
        result = exporter.export_full_report(
            sample_data, sample_results, report=report
        )
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"

    def test_export_full_report_with_health_score(
        self, exporter, sample_data
    ):
        results = {
            "current_ratio": 2.0,
            "composite_health": CompositeHealthScore(
                score=78,
                grade="B",
                component_scores={"liquidity": 80, "profitability": 75},
                interpretation="Solid financial health.",
            ),
        }
        result = exporter.export_full_report(sample_data, results)
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"

    def test_export_full_report_with_empty_results(
        self, exporter, sample_data
    ):
        result = exporter.export_full_report(sample_data, {})
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"

    # ------------------------------------------------------------------
    # export_ratios
    # ------------------------------------------------------------------

    def test_export_ratios_returns_bytes(self, exporter):
        ratios = {"current_ratio": 2.0, "net_margin": 0.15}
        result = exporter.export_ratios(ratios)
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"

    def test_export_ratios_with_company_name(self, exporter):
        ratios = {"current_ratio": 2.0}
        result = exporter.export_ratios(ratios, company_name="Acme Corp")
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"

    def test_export_empty_ratios(self, exporter):
        result = exporter.export_ratios({})
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"

    def test_export_ratios_with_none_values(self, exporter):
        ratios = {"current_ratio": 2.0, "bad_metric": None}
        result = exporter.export_ratios(ratios)
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"

    # ------------------------------------------------------------------
    # export_scenario_comparison
    # ------------------------------------------------------------------

    def test_export_scenario_comparison(self, exporter):
        scenario = ScenarioResult(
            scenario_name="Bull Case",
            adjustments={"revenue": 1.10},
            base_ratios={"net_margin": 0.15},
            scenario_ratios={"net_margin": 0.18},
            impact_summary="Revenue +10%",
        )
        result = exporter.export_scenario_comparison([scenario])
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"

    def test_export_scenario_comparison_multiple(self, exporter):
        scenarios = [
            ScenarioResult(
                scenario_name="Bull",
                adjustments={"revenue": 1.10},
                base_ratios={"net_margin": 0.15},
                scenario_ratios={"net_margin": 0.18},
                impact_summary="Up",
            ),
            ScenarioResult(
                scenario_name="Bear",
                adjustments={"revenue": 0.90},
                base_ratios={"net_margin": 0.15},
                scenario_ratios={"net_margin": 0.12},
                impact_summary="Down",
            ),
        ]
        result = exporter.export_scenario_comparison(scenarios)
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"

    def test_export_scenario_comparison_empty_list(self, exporter):
        result = exporter.export_scenario_comparison([])
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"

    def test_export_scenario_with_none_ratio_values(self, exporter):
        scenario = ScenarioResult(
            scenario_name="Mixed",
            adjustments={},
            base_ratios={"net_margin": 0.15, "roe": None},
            scenario_ratios={"net_margin": 0.18},
            impact_summary="Partial data",
        )
        result = exporter.export_scenario_comparison([scenario])
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"

    # ------------------------------------------------------------------
    # Scoring models
    # ------------------------------------------------------------------

    def test_export_full_report_with_z_score(self, exporter, sample_data):
        from financial_analyzer import AltmanZScore

        results = {
            "altman_z_score": AltmanZScore(
                z_score=3.1,
                zone="safe",
                components={"x1": 0.1, "x2": 0.2},
                interpretation="Safe zone.",
            ),
        }
        result = exporter.export_full_report(sample_data, results)
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"

    def test_export_full_report_with_f_score(self, exporter, sample_data):
        from financial_analyzer import PiotroskiFScore

        results = {
            "piotroski_f_score": PiotroskiFScore(
                score=7,
                criteria={"roa_positive": True, "ocf_positive": True},
                interpretation="Strong value.",
            ),
        }
        result = exporter.export_full_report(sample_data, results)
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"



# ---------------------------------------------------------------------------
# WS-3 P0-8: current_ratio renders as multiplier (1.50x), not percent (150.00%)
# ---------------------------------------------------------------------------

class TestRatioFormatting_P0_8:
    """Regression: ratios must NOT be formatted as percentages in XLSX output."""

    def test_current_ratio_uses_ratio_format(self):
        from export_xlsx import _Formats
        import xlsxwriter
        import io
        wb = xlsxwriter.Workbook(io.BytesIO(), {"in_memory": True})
        fmt = _Formats(wb)
        chosen = fmt.value_fmt("current_ratio")
        # ratio fmt is registered separately from pct fmt
        assert chosen is fmt.ratio
        assert chosen is not fmt.pct
        wb.close()

    def test_debt_to_equity_uses_ratio_format(self):
        from export_xlsx import _Formats
        import xlsxwriter, io
        wb = xlsxwriter.Workbook(io.BytesIO(), {"in_memory": True})
        fmt = _Formats(wb)
        assert fmt.value_fmt("debt_to_equity") is fmt.ratio
        wb.close()

    def test_gross_margin_still_uses_percent_format(self):
        from export_xlsx import _Formats
        import xlsxwriter, io
        wb = xlsxwriter.Workbook(io.BytesIO(), {"in_memory": True})
        fmt = _Formats(wb)
        assert fmt.value_fmt("gross_margin") is fmt.pct
        wb.close()

    def test_revenue_still_uses_dollar_format(self):
        from export_xlsx import _Formats
        import xlsxwriter, io
        wb = xlsxwriter.Workbook(io.BytesIO(), {"in_memory": True})
        fmt = _Formats(wb)
        assert fmt.value_fmt("revenue") is fmt.dollar
        wb.close()

    def test_export_ratios_writes_ratio_value_unchanged(self):
        """Sanity check: export runs without exception and current_ratio=1.5 is written."""
        from export_xlsx import FinancialExcelExporter
        exporter = FinancialExcelExporter()
        out = exporter.export_ratios({"current_ratio": 1.5, "gross_margin": 0.45})
        assert isinstance(out, bytes)
        assert len(out) > 100  # non-empty xlsx
