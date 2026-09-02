"""Phase 10 Tests: Custom KPI Builder.

Tests for evaluate_custom_kpis() and related dataclasses.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    CustomKPIDefinition,
    CustomKPIReport,
    CustomKPIResult,
    FinancialData,
)


@pytest.fixture
def analyzer():
    return CharlieAnalyzer()


@pytest.fixture
def sample_data():
    return FinancialData(
        revenue=1_000_000,
        cogs=600_000,
        gross_profit=400_000,
        operating_expenses=200_000,
        operating_income=200_000,
        net_income=150_000,
        ebit=200_000,
        ebitda=250_000,
        total_assets=2_000_000,
        total_liabilities=800_000,
        total_equity=1_200_000,
        current_assets=500_000,
        current_liabilities=200_000,
        inventory=100_000,
        accounts_receivable=150_000,
        accounts_payable=80_000,
        total_debt=400_000,
        retained_earnings=600_000,
        depreciation=50_000,
        interest_expense=30_000,
        operating_cash_flow=220_000,
        capex=80_000,
    )


# ===== DATACLASS TESTS =====


class TestCustomKPIDefinitionDataclass:
    def test_defaults(self):
        d = CustomKPIDefinition()
        assert d.name == ""
        assert d.formula == ""
        assert d.target_min is None
        assert d.target_max is None

    def test_fields_assignable(self):
        d = CustomKPIDefinition(name="ROA", formula="net_income / total_assets", target_min=0.05)
        assert d.name == "ROA"
        assert d.target_min == 0.05


class TestCustomKPIResultDataclass:
    def test_defaults(self):
        r = CustomKPIResult()
        assert r.value is None
        assert r.meets_target is None
        assert r.error is None

    def test_fields_assignable(self):
        r = CustomKPIResult(name="test", value=0.15, meets_target=True)
        assert r.value == 0.15


class TestCustomKPIReportDataclass:
    def test_defaults(self):
        r = CustomKPIReport()
        assert r.results == []
        assert r.summary == ""


# ===== EVALUATE CUSTOM KPIS =====


class TestEvaluateCustomKPIs:
    def test_returns_report(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="AT", formula="revenue / total_assets")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert isinstance(result, CustomKPIReport)

    def test_simple_ratio(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="AT", formula="revenue / total_assets")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert len(result.results) == 1
        assert result.results[0].value is not None
        assert abs(result.results[0].value - 0.5) < 0.01  # 1M / 2M

    def test_subtraction(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="FCF", formula="ebitda - capex")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert abs(result.results[0].value - 170_000) < 1  # 250k - 80k

    def test_complex_formula(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="GM%", formula="(revenue - cogs) / revenue")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert abs(result.results[0].value - 0.4) < 0.01  # 400k / 1M

    def test_multiplication(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="TA2x", formula="total_assets * 2")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert abs(result.results[0].value - 4_000_000) < 1

    def test_meets_target_min(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="AT", formula="revenue / total_assets", target_min=0.3)]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].meets_target is True  # 0.5 >= 0.3

    def test_misses_target_min(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="AT", formula="revenue / total_assets", target_min=0.8)]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].meets_target is False  # 0.5 < 0.8

    def test_meets_target_max(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="DE", formula="total_debt / total_equity", target_max=1.0)]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].meets_target is True  # 0.33 <= 1.0

    def test_meets_target_range(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="AT", formula="revenue / total_assets", target_min=0.3, target_max=0.8)]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].meets_target is True  # 0.5 in [0.3, 0.8]

    def test_misses_target_range(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="AT", formula="revenue / total_assets", target_min=0.6, target_max=0.8)]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].meets_target is False  # 0.5 < 0.6

    def test_no_target_means_none(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="AT", formula="revenue / total_assets")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].meets_target is None

    def test_division_by_zero(self, analyzer):
        data = FinancialData(revenue=100_000, total_assets=0)
        kpis = [CustomKPIDefinition(name="bad", formula="revenue / total_assets")]
        result = analyzer.evaluate_custom_kpis(data, kpis)
        assert result.results[0].error is not None
        assert "zero" in result.results[0].error.lower() or "invalid" in result.results[0].error.lower()

    def test_unknown_variable(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="bad", formula="revenue / nonexistent")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].error is not None

    def test_invalid_token_rejected(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="bad", formula="__import__('os')")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].error is not None

    def test_empty_formula(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="empty", formula="")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].error == "Empty formula"

    def test_multiple_kpis(self, analyzer, sample_data):
        kpis = [
            CustomKPIDefinition(name="AT", formula="revenue / total_assets"),
            CustomKPIDefinition(name="GM", formula="gross_profit / revenue"),
            CustomKPIDefinition(name="DE", formula="total_debt / total_equity"),
        ]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert len(result.results) == 3
        assert all(r.value is not None for r in result.results)

    def test_summary_with_targets(self, analyzer, sample_data):
        kpis = [
            CustomKPIDefinition(name="AT", formula="revenue / total_assets", target_min=0.3),
            CustomKPIDefinition(name="GM", formula="gross_profit / revenue", target_min=0.5),
        ]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert "meeting targets" in result.summary.lower() or "/" in result.summary

    def test_summary_with_errors(self, analyzer, sample_data):
        kpis = [
            CustomKPIDefinition(name="bad", formula="xyz"),
        ]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert "error" in result.summary.lower()

    def test_empty_data_handles_missing_fields(self, analyzer):
        data = FinancialData()
        kpis = [CustomKPIDefinition(name="AT", formula="revenue / total_assets")]
        result = analyzer.evaluate_custom_kpis(data, kpis)
        # Both None -> error
        assert result.results[0].error is not None

    def test_parentheses_work(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="test", formula="(revenue + net_income) / total_assets")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        expected = (1_000_000 + 150_000) / 2_000_000
        assert abs(result.results[0].value - expected) < 0.01

    def test_numeric_literal_in_formula(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="pct", formula="net_income / revenue * 100")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert abs(result.results[0].value - 15.0) < 0.1


# ===== SECURITY TESTS =====


class TestCustomKPISecurity:
    def test_no_builtins_access(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="hack", formula="print('hello')")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].error is not None

    def test_no_import(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="hack", formula="__import__('os').system('ls')")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].error is not None

    def test_no_dunder_access(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="hack", formula="revenue.__class__")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].error is not None

    def test_no_lambda(self, analyzer, sample_data):
        """AST evaluator rejects lambda expressions."""
        kpis = [CustomKPIDefinition(name="hack", formula="(lambda: revenue)()")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].error is not None

    def test_no_list_comprehension(self, analyzer, sample_data):
        """AST evaluator rejects list comprehensions."""
        kpis = [CustomKPIDefinition(name="hack", formula="[x for x in [1,2,3]]")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].error is not None

    def test_no_attribute_access(self, analyzer, sample_data):
        """AST evaluator rejects attribute access (no dot notation)."""
        kpis = [CustomKPIDefinition(name="hack", formula="revenue.real")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].error is not None

    def test_no_subscript(self, analyzer, sample_data):
        """AST evaluator rejects subscript access."""
        kpis = [CustomKPIDefinition(name="hack", formula="revenue[0]")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].error is not None

    def test_unary_minus_allowed(self, analyzer, sample_data):
        """Unary minus should work in formulas."""
        kpis = [CustomKPIDefinition(name="neg", formula="-revenue")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].value == -1_000_000


# ===== EDGE CASES =====


class TestPhase10EdgeCases:
    def test_zero_kpis(self, analyzer, sample_data):
        result = analyzer.evaluate_custom_kpis(sample_data, [])
        assert isinstance(result, CustomKPIReport)
        assert len(result.results) == 0

    def test_negative_result(self, analyzer):
        data = FinancialData(revenue=100_000, cogs=120_000)
        kpis = [CustomKPIDefinition(name="GM", formula="revenue - cogs")]
        result = analyzer.evaluate_custom_kpis(data, kpis)
        assert result.results[0].value == -20_000

    def test_very_large_numbers(self, analyzer):
        data = FinancialData(revenue=1e12, total_assets=5e12)
        kpis = [CustomKPIDefinition(name="AT", formula="revenue / total_assets")]
        result = analyzer.evaluate_custom_kpis(data, kpis)
        assert abs(result.results[0].value - 0.2) < 0.01

    def test_decimal_in_formula(self, analyzer, sample_data):
        kpis = [CustomKPIDefinition(name="adj", formula="revenue * 0.95")]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert abs(result.results[0].value - 950_000) < 1

    def test_mixed_valid_and_error(self, analyzer, sample_data):
        kpis = [
            CustomKPIDefinition(name="good", formula="revenue / total_assets"),
            CustomKPIDefinition(name="bad", formula="xyz"),
        ]
        result = analyzer.evaluate_custom_kpis(sample_data, kpis)
        assert result.results[0].value is not None
        assert result.results[1].error is not None
