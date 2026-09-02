"""Tests for Phase 4: Intelligence Integration & Polish.

Covers:
- Financial analysis context generation (_get_financial_analysis_context)
- Enhanced financial query detection (_is_financial_query)
- Financial prompt building with computed analysis (_build_financial_prompt)
- Cache invalidation on reload_documents
- Report download generation (_render_report_download)
- Industry benchmark comparison logic (_render_industry_benchmarks)
- INDUSTRY_BENCHMARKS constant structure
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    CompositeHealthScore,
    FinancialData,
    FinancialReport,
)
from insights_page import FinancialInsightsPage

# ===== Fixtures =====


@pytest.fixture
def analyzer():
    return CharlieAnalyzer(tax_rate=0.25)


@pytest.fixture
def healthy_data():
    """A financially healthy company for testing."""
    return FinancialData(
        revenue=10_000_000,
        cogs=6_000_000,
        gross_profit=4_000_000,
        operating_income=2_000_000,
        ebit=2_000_000,
        ebt=1_800_000,
        interest_expense=200_000,
        net_income=1_350_000,
        total_assets=20_000_000,
        current_assets=8_000_000,
        cash=3_000_000,
        inventory=2_000_000,
        accounts_receivable=2_500_000,
        total_liabilities=10_000_000,
        current_liabilities=4_000_000,
        accounts_payable=1_500_000,
        total_debt=6_000_000,
        total_equity=10_000_000,
        retained_earnings=5_000_000,
        operating_cash_flow=2_500_000,
        capex=500_000,
    )


@pytest.fixture
def sample_df():
    """DataFrame with financial data for insights page tests."""
    return pd.DataFrame(
        {
            "Revenue": [10_000_000],
            "Cost of Goods Sold": [6_000_000],
            "Gross Profit": [4_000_000],
            "Operating Income": [2_000_000],
            "Net Income": [1_350_000],
            "Total Assets": [20_000_000],
            "Current Assets": [8_000_000],
            "Current Liabilities": [4_000_000],
            "Total Liabilities": [10_000_000],
            "Total Equity": [10_000_000],
            "Total Debt": [6_000_000],
            "Interest Expense": [200_000],
            "Accounts Receivable": [2_500_000],
            "Inventory": [2_000_000],
            "Accounts Payable": [1_500_000],
            "Cash": [3_000_000],
        }
    )


def _make_mock_rag(charlie_analyzer=None, excel_processor=None, cache=None):
    """Create a mock SimpleRAG instance with controlled properties."""
    # Import the class
    # Use __new__ to bypass __init__
    import threading

    from app_local import SimpleRAG

    rag = SimpleRAG.__new__(SimpleRAG)
    rag._financial_analysis_cache = cache
    rag._financial_analysis_lock = threading.Lock()
    rag._charlie_analyzer = charlie_analyzer
    rag._excel_processor = excel_processor

    # Patch lazy properties to return the private attributes directly
    type(rag).charlie_analyzer = property(lambda self: self._charlie_analyzer)
    type(rag).excel_processor = property(lambda self: self._excel_processor)

    return rag


# ===== _is_financial_query Tests =====


class TestIsFinancialQuery:
    """Test enhanced financial query detection with Phase 4 keywords."""

    def _check_query(self, query: str) -> bool:
        rag = _make_mock_rag()
        return rag._is_financial_query(query)

    def test_classic_keywords(self):
        """Original keywords still work."""
        assert self._check_query("What is the profit margin?")
        assert self._check_query("Show me the revenue breakdown")
        assert self._check_query("What are the liquidity ratios?")
        assert self._check_query("Balance sheet analysis")

    def test_phase4_scoring_keywords(self):
        """New scoring model keywords are detected."""
        assert self._check_query("What is the Z-Score?")
        assert self._check_query("Show the zscore")
        assert self._check_query("What is the F-Score for this company?")
        assert self._check_query("Calculate the fscore")
        assert self._check_query("Run Piotroski analysis")
        assert self._check_query("Altman bankruptcy prediction")

    def test_phase4_health_keywords(self):
        """Health score and composite keywords are detected."""
        assert self._check_query("What is the health score?")
        assert self._check_query("Show composite rating")
        assert self._check_query("What grade does this company get?")
        assert self._check_query("DuPont decomposition analysis")

    def test_phase4_distress_keywords(self):
        """Distress and bankruptcy keywords are detected."""
        assert self._check_query("Is this company in distress?")
        assert self._check_query("Bankruptcy risk assessment")
        assert self._check_query("Working capital adequacy")

    def test_non_financial_queries(self):
        """Non-financial queries return False."""
        assert not self._check_query("What is the weather today?")
        assert not self._check_query("Tell me about machine learning")
        assert not self._check_query("Hello, how are you?")

    def test_case_insensitive(self):
        """Query detection is case-insensitive."""
        assert self._check_query("WHAT IS THE Z-SCORE?")
        assert self._check_query("Altman Z-Score")
        assert self._check_query("piotroski f-score")

    def test_empty_query(self):
        """Empty query returns False."""
        assert not self._check_query("")


# ===== _get_financial_analysis_context Tests =====


class TestGetFinancialAnalysisContext:
    """Test financial analysis context generation and caching."""

    def test_returns_cached_value(self):
        """When cache is populated, returns cached value without recomputing."""
        rag = _make_mock_rag(cache="cached analysis result")
        result = rag._get_financial_analysis_context()
        assert result == "cached analysis result"

    def test_no_analyzer_returns_empty(self):
        """When analyzer is None, returns empty string and caches it."""
        rag = _make_mock_rag(charlie_analyzer=None, excel_processor=MagicMock())
        result = rag._get_financial_analysis_context()
        assert result == ""
        assert rag._financial_analysis_cache == ""

    def test_no_excel_processor_returns_empty(self):
        """When excel_processor is None, returns empty string."""
        rag = _make_mock_rag(charlie_analyzer=MagicMock(), excel_processor=None)
        result = rag._get_financial_analysis_context()
        assert result == ""

    def test_no_excel_files_returns_empty(self):
        """When no Excel files found, returns empty string."""
        mock_excel = MagicMock()
        mock_excel.scan_for_excel_files.return_value = []
        rag = _make_mock_rag(charlie_analyzer=MagicMock(), excel_processor=mock_excel)
        result = rag._get_financial_analysis_context()
        assert result == ""

    def test_successful_analysis_contains_sections(self):
        """Successful analysis contains expected report sections."""
        # Create mock Excel processor
        mock_excel = MagicMock()
        mock_file = Path("test.xlsx")
        mock_excel.scan_for_excel_files.return_value = [mock_file]

        # Create mock workbook with merged DataFrame
        mock_df = pd.DataFrame(
            {
                "Revenue": [10_000_000],
                "Net Income": [1_000_000],
                "Total Assets": [20_000_000],
                "Total Equity": [10_000_000],
                "Current Assets": [5_000_000],
                "Current Liabilities": [3_000_000],
            }
        )
        mock_combined = MagicMock()
        mock_combined.merged_df = mock_df
        mock_workbook = MagicMock()
        mock_excel.load_workbook.return_value = mock_workbook
        mock_excel.combine_sheets_intelligently.return_value = mock_combined

        # Use real analyzer
        real_analyzer = CharlieAnalyzer()
        rag = _make_mock_rag(charlie_analyzer=real_analyzer, excel_processor=mock_excel)

        result = rag._get_financial_analysis_context()

        assert "Computed Analysis:" in result
        assert "test.xlsx" in result
        # Should include report sections
        assert "Key Ratios:" in result or "Scoring Models:" in result

    def test_caches_result_after_first_call(self):
        """Result is cached so second call doesn't recompute."""
        mock_excel = MagicMock()
        mock_excel.scan_for_excel_files.return_value = []
        rag = _make_mock_rag(charlie_analyzer=MagicMock(), excel_processor=mock_excel)

        # First call
        rag._get_financial_analysis_context()
        assert rag._financial_analysis_cache == ""

        # Second call - excel processor shouldn't be called again
        mock_excel.scan_for_excel_files.reset_mock()
        rag._get_financial_analysis_context()
        mock_excel.scan_for_excel_files.assert_not_called()

    def test_exception_returns_empty(self):
        """Exceptions are caught and empty string returned."""
        mock_excel = MagicMock()
        mock_excel.scan_for_excel_files.side_effect = RuntimeError("DB error")
        rag = _make_mock_rag(charlie_analyzer=MagicMock(), excel_processor=mock_excel)
        result = rag._get_financial_analysis_context()
        assert result == ""
        assert rag._financial_analysis_cache == ""

    def test_skips_empty_financial_data(self):
        """Files with no recognizable financial data are skipped."""
        mock_excel = MagicMock()
        mock_file = Path("empty.xlsx")
        mock_excel.scan_for_excel_files.return_value = [mock_file]

        # DataFrame with no financial columns
        mock_df = pd.DataFrame({"Name": ["Alice"], "Age": [30]})
        mock_combined = MagicMock()
        mock_combined.merged_df = mock_df
        mock_excel.load_workbook.return_value = MagicMock()
        mock_excel.combine_sheets_intelligently.return_value = mock_combined

        real_analyzer = CharlieAnalyzer()
        rag = _make_mock_rag(charlie_analyzer=real_analyzer, excel_processor=mock_excel)

        result = rag._get_financial_analysis_context()
        assert result == ""

    def test_limits_to_three_files(self):
        """Only first 3 files are processed."""
        mock_excel = MagicMock()
        files = [Path(f"file{i}.xlsx") for i in range(5)]
        mock_excel.scan_for_excel_files.return_value = files

        # Make all files fail gracefully
        mock_excel.load_workbook.side_effect = Exception("fail")

        rag = _make_mock_rag(charlie_analyzer=CharlieAnalyzer(), excel_processor=mock_excel)
        rag._get_financial_analysis_context()

        # Should have attempted exactly 3 files
        assert mock_excel.load_workbook.call_count == 3


# ===== _build_financial_prompt Tests =====


class TestBuildFinancialPrompt:
    """Test enhanced prompt building with computed analysis."""

    def test_includes_query(self):
        """Prompt contains the user query."""
        rag = _make_mock_rag(cache="")
        prompt = rag._build_financial_prompt("What is the Z-Score?", "some context", [])
        assert "What is the Z-Score?" in prompt

    def test_includes_context(self):
        """Prompt contains retrieved context."""
        rag = _make_mock_rag(cache="")
        prompt = rag._build_financial_prompt("query", "RETRIEVED CONTEXT DATA", [])
        assert "RETRIEVED CONTEXT DATA" in prompt

    def test_includes_computed_analysis(self):
        """When computed analysis is available, it's included in prompt."""
        cached = "Z-Score: 3.45\nGrade: B\nHealth: 72/100"
        rag = _make_mock_rag(cache=cached)
        prompt = rag._build_financial_prompt("analysis query", "context", [])
        assert "COMPUTED FINANCIAL ANALYSIS" in prompt
        assert "Z-Score: 3.45" in prompt
        assert "Grade: B" in prompt

    def test_no_computed_section_when_empty(self):
        """When no computed analysis, section is excluded."""
        rag = _make_mock_rag(cache="")
        prompt = rag._build_financial_prompt("query", "context", [])
        assert "COMPUTED FINANCIAL ANALYSIS" not in prompt

    def test_charlie_munger_framework(self):
        """Prompt includes the Charlie Munger analysis framework."""
        rag = _make_mock_rag(cache="")
        prompt = rag._build_financial_prompt("query", "context", [])
        assert "Charlie Munger" in prompt
        assert "margin of safety" in prompt
        assert "cash flows" in prompt

    def test_cite_exact_values_instruction(self):
        """Prompt instructs LLM to cite exact values from computed analysis."""
        rag = _make_mock_rag(cache="some analysis")
        prompt = rag._build_financial_prompt("query", "context", [])
        assert "cite those EXACT values" in prompt


# ===== reload_documents Cache Invalidation Tests =====


class TestReloadDocumentsCacheInvalidation:
    """Test that reload_documents clears the financial analysis cache."""

    def test_cache_cleared_on_reload(self):
        """reload_documents resets _financial_analysis_cache to None."""
        import threading

        from app_local import SimpleRAG

        rag = SimpleRAG.__new__(SimpleRAG)
        rag.documents = ["doc1"]
        rag.embeddings = [[0.1, 0.2]]
        rag._doc_matrix = "something"
        rag._doc_norms = "something"
        rag._bm25_index = "something"
        rag._financial_analysis_cache = "stale cached analysis"
        rag._lock = threading.Lock()

        # Mock _load_documents to do nothing
        with patch.object(SimpleRAG, "_load_documents"):
            # Patch the properties to avoid __init__ dependency issues
            type(rag).charlie_analyzer = property(lambda s: None)
            type(rag).excel_processor = property(lambda s: None)

            rag.reload_documents()

        assert rag._financial_analysis_cache is None
        assert rag.documents == []
        assert rag.embeddings == []


# ===== INDUSTRY_BENCHMARKS Constant Tests =====


class TestIndustryBenchmarks:
    """Test the INDUSTRY_BENCHMARKS class constant structure."""

    def test_all_required_keys(self):
        """Each benchmark has label, benchmark, good, and unit."""
        for key, bench in FinancialInsightsPage.INDUSTRY_BENCHMARKS.items():
            assert "label" in bench, f"{key} missing 'label'"
            assert "benchmark" in bench, f"{key} missing 'benchmark'"
            assert "good" in bench, f"{key} missing 'good'"
            assert "unit" in bench, f"{key} missing 'unit'"

    def test_benchmark_values_are_numeric(self):
        """Benchmark and good values are numbers."""
        for key, bench in FinancialInsightsPage.INDUSTRY_BENCHMARKS.items():
            assert isinstance(bench["benchmark"], (int, float)), f"{key} benchmark not numeric"
            assert isinstance(bench["good"], (int, float)), f"{key} good not numeric"

    def test_unit_values_valid(self):
        """Units are either 'x' (multiple) or '%' (percentage)."""
        for key, bench in FinancialInsightsPage.INDUSTRY_BENCHMARKS.items():
            assert bench["unit"] in ("x", "%"), f"{key} has invalid unit: {bench['unit']}"

    def test_expected_ratios_present(self):
        """Key financial ratios are included."""
        benchmarks = FinancialInsightsPage.INDUSTRY_BENCHMARKS
        expected = [
            "current_ratio",
            "quick_ratio",
            "net_margin",
            "roe",
            "roa",
            "debt_to_equity",
            "interest_coverage",
            "asset_turnover",
            "gross_margin",
            "operating_margin",
        ]
        for key in expected:
            assert key in benchmarks, f"Missing benchmark: {key}"

    def test_debt_to_equity_lower_is_better(self):
        """debt_to_equity should have lower_is_better=True."""
        bench = FinancialInsightsPage.INDUSTRY_BENCHMARKS["debt_to_equity"]
        assert bench.get("lower_is_better") is True

    def test_good_thresholds_better_than_benchmark(self):
        """'good' values represent better-than-average performance."""
        for key, bench in FinancialInsightsPage.INDUSTRY_BENCHMARKS.items():
            lower_is_better = bench.get("lower_is_better", False)
            if lower_is_better:
                assert bench["good"] <= bench["benchmark"], (
                    f"{key}: good ({bench['good']}) should be <= benchmark ({bench['benchmark']})"
                )
            else:
                assert bench["good"] >= bench["benchmark"], (
                    f"{key}: good ({bench['good']}) should be >= benchmark ({bench['benchmark']})"
                )


# ===== Industry Benchmark Logic Tests =====


class TestBenchmarkComparison:
    """Test benchmark comparison logic (extracted from _render_industry_benchmarks)."""

    def _classify_ratio(self, value, benchmark_key):
        """Apply the same classification logic from _render_industry_benchmarks."""
        bench = FinancialInsightsPage.INDUSTRY_BENCHMARKS[benchmark_key]
        lower_is_better = bench.get("lower_is_better", False)

        if lower_is_better:
            if value <= bench["good"]:
                return "Above Average"
            elif value <= bench["benchmark"]:
                return "Average"
            else:
                return "Below Average"
        else:
            if value >= bench["good"]:
                return "Above Average"
            elif value >= bench["benchmark"]:
                return "Average"
            else:
                return "Below Average"

    def test_high_current_ratio_above_average(self):
        """Current ratio >= 2.0 is Above Average."""
        assert self._classify_ratio(2.5, "current_ratio") == "Above Average"

    def test_average_current_ratio(self):
        """Current ratio >= 1.5 but < 2.0 is Average."""
        assert self._classify_ratio(1.7, "current_ratio") == "Average"

    def test_low_current_ratio_below_average(self):
        """Current ratio < 1.5 is Below Average."""
        assert self._classify_ratio(0.8, "current_ratio") == "Below Average"

    def test_low_debt_to_equity_above_average(self):
        """Low debt_to_equity is Above Average (lower_is_better)."""
        assert self._classify_ratio(0.3, "debt_to_equity") == "Above Average"

    def test_high_debt_to_equity_below_average(self):
        """High debt_to_equity is Below Average (lower_is_better)."""
        assert self._classify_ratio(2.0, "debt_to_equity") == "Below Average"

    def test_average_debt_to_equity(self):
        """Moderate debt_to_equity between good and benchmark is Average."""
        assert self._classify_ratio(0.8, "debt_to_equity") == "Average"

    def test_high_roe_above_average(self):
        """ROE >= 0.20 is Above Average."""
        assert self._classify_ratio(0.25, "roe") == "Above Average"

    def test_low_roe_below_average(self):
        """ROE < 0.12 is Below Average."""
        assert self._classify_ratio(0.05, "roe") == "Below Average"

    def test_percentage_formatting(self):
        """Percentage ratios are formatted correctly."""
        bench = FinancialInsightsPage.INDUSTRY_BENCHMARKS["net_margin"]
        assert bench["unit"] == "%"
        # 0.08 = 8.0%
        assert f"{bench['benchmark']:.1%}" == "8.0%"

    def test_multiple_formatting(self):
        """Multiple ratios are formatted correctly."""
        bench = FinancialInsightsPage.INDUSTRY_BENCHMARKS["current_ratio"]
        assert bench["unit"] == "x"
        assert f"{bench['benchmark']:.2f}x" == "1.50x"


# ===== Report Generation Tests =====


class TestReportGeneration:
    """Test report generation for download (logic used by _render_report_download)."""

    def test_report_has_all_sections(self, analyzer, healthy_data):
        """Generated report has all expected sections."""
        report = analyzer.generate_report(healthy_data)
        assert isinstance(report, FinancialReport)
        assert report.executive_summary != ""
        assert "ratio_analysis" in report.sections
        assert "scoring_models" in report.sections
        assert "risk_assessment" in report.sections
        assert "recommendations" in report.sections

    def test_report_text_assembly(self, analyzer, healthy_data):
        """Report text can be assembled into download format."""
        report = analyzer.generate_report(healthy_data)

        # Replicate the assembly logic from _render_report_download
        report_lines = [
            "=" * 60,
            "FINANCIAL ANALYSIS REPORT",
            f"Generated: {report.generated_at}",
            "=" * 60,
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
            report.executive_summary,
            "",
        ]

        for section_key, section_title in [
            ("ratio_analysis", "RATIO ANALYSIS"),
            ("scoring_models", "SCORING MODELS"),
            ("risk_assessment", "RISK ASSESSMENT"),
            ("recommendations", "RECOMMENDATIONS"),
        ]:
            if section_key in report.sections:
                report_lines.extend(
                    [
                        section_title,
                        "-" * 40,
                        report.sections[section_key],
                        "",
                    ]
                )

        report_lines.append("=" * 60)
        report_text = "\n".join(report_lines)

        assert "FINANCIAL ANALYSIS REPORT" in report_text
        assert "EXECUTIVE SUMMARY" in report_text
        assert "RATIO ANALYSIS" in report_text
        assert "SCORING MODELS" in report_text
        assert report.generated_at in report_text
        assert len(report_text) > 100  # Non-trivial content

    def test_report_without_prior_period(self, analyzer, healthy_data):
        """Report without prior period should not have period_comparison."""
        report = analyzer.generate_report(healthy_data)
        assert "period_comparison" not in report.sections

    def test_report_empty_data(self, analyzer):
        """Report can be generated even with empty FinancialData."""
        report = analyzer.generate_report(FinancialData())
        assert isinstance(report, FinancialReport)
        assert report.executive_summary != ""


# ===== Integration Test: Full Analysis Pipeline =====


class TestAnalysisPipeline:
    """End-to-end integration tests for the Phase 4 analysis pipeline."""

    def test_analyze_returns_composite_health(self, analyzer, healthy_data):
        """analyze() includes composite_health for integration with prompts."""
        results = analyzer.analyze(healthy_data)
        assert "composite_health" in results
        health = results["composite_health"]
        assert isinstance(health, CompositeHealthScore)
        assert 0 <= health.score <= 100
        assert health.grade in ("A", "B", "C", "D", "F")

    def test_dataframe_analysis_pipeline(self, analyzer):
        """Full pipeline: DataFrame -> FinancialData -> analyze -> report."""
        df = pd.DataFrame(
            {
                "Revenue": [10_000_000],
                "Net Income": [1_000_000],
                "Total Assets": [20_000_000],
                "Total Equity": [10_000_000],
                "Current Assets": [5_000_000],
                "Current Liabilities": [3_000_000],
            }
        )

        # Step 1: Convert to FinancialData
        data = analyzer._dataframe_to_financial_data(df)
        assert data.revenue == 10_000_000

        # Step 2: Analyze
        results = analyzer.analyze(data)
        assert "composite_health" in results
        assert "insights" in results

        # Step 3: Generate report
        report = analyzer.generate_report(data)
        assert isinstance(report, FinancialReport)
        assert report.executive_summary != ""

    def test_benchmark_data_matches_analysis(self, analyzer, healthy_data):
        """Analysis results contain keys that match benchmark definitions."""
        results = analyzer.analyze(healthy_data)

        # Collect all ratio keys from analysis
        available_keys = set()
        for category in ("liquidity_ratios", "profitability_ratios", "leverage_ratios", "efficiency_ratios"):
            for key, value in results.get(category, {}).items():
                if value is not None:
                    available_keys.add(key)

        # At least some benchmark keys should match analysis keys
        benchmark_keys = set(FinancialInsightsPage.INDUSTRY_BENCHMARKS.keys())
        overlap = available_keys & benchmark_keys
        assert len(overlap) >= 3, f"Expected at least 3 matching benchmarks, got {len(overlap)}: {overlap}"
