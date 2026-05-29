"""Tests for ratio_framework.py generic parameterized ratio analysis engine."""

from dataclasses import dataclass

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_financial_data():
    """Create a FinancialData instance with representative values."""
    from financial_analyzer import FinancialData

    return FinancialData(
        revenue=1_000_000,
        cogs=600_000,
        gross_profit=400_000,
        operating_income=200_000,
        ebit=200_000,
        ebitda=250_000,
        net_income=150_000,
        total_assets=2_000_000,
        current_assets=800_000,
        cash=300_000,
        inventory=100_000,
        accounts_receivable=150_000,
        total_equity=1_000_000,
        total_debt=500_000,
        current_liabilities=400_000,
        interest_expense=30_000,
        operating_cash_flow=180_000,
    )


@pytest.fixture
def sparse_financial_data():
    """FinancialData with minimal fields set."""
    from financial_analyzer import FinancialData

    return FinancialData(revenue=500_000, net_income=50_000)


@pytest.fixture
def simple_definition():
    """A simple RatioDefinition for testing."""
    from ratio_framework import Adjustment, Operator, RatioDefinition

    return RatioDefinition(
        name="Test Ratio",
        description="A test ratio",
        numerator_field="net_income",
        denominator_field="total_assets",
        higher_is_better=True,
        scoring_thresholds=[
            (0.15, 10.0),
            (0.10, 8.0),
            (0.05, 6.0),
            (0.02, 4.0),
        ],
        adjustments=[
            Adjustment("operating_income", Operator.GT, 0, 0.5, "Positive OI"),
        ],
        unit="%",
    )


# ---------------------------------------------------------------------------
# Operator enum
# ---------------------------------------------------------------------------


class TestOperator:
    def test_all_operators_defined(self):
        from ratio_framework import Operator

        assert Operator.GT.value == ">"
        assert Operator.GTE.value == ">="
        assert Operator.LT.value == "<"
        assert Operator.LTE.value == "<="
        assert Operator.EQ.value == "=="


# ---------------------------------------------------------------------------
# _apply_operator
# ---------------------------------------------------------------------------


class TestApplyOperator:
    def test_gt(self):
        from ratio_framework import Operator, _apply_operator

        assert _apply_operator(10, Operator.GT, 5) is True
        assert _apply_operator(5, Operator.GT, 5) is False
        assert _apply_operator(3, Operator.GT, 5) is False

    def test_gte(self):
        from ratio_framework import Operator, _apply_operator

        assert _apply_operator(5, Operator.GTE, 5) is True
        assert _apply_operator(6, Operator.GTE, 5) is True
        assert _apply_operator(4, Operator.GTE, 5) is False

    def test_lt(self):
        from ratio_framework import Operator, _apply_operator

        assert _apply_operator(3, Operator.LT, 5) is True
        assert _apply_operator(5, Operator.LT, 5) is False

    def test_lte(self):
        from ratio_framework import Operator, _apply_operator

        assert _apply_operator(5, Operator.LTE, 5) is True
        assert _apply_operator(3, Operator.LTE, 5) is True
        assert _apply_operator(6, Operator.LTE, 5) is False

    def test_eq(self):
        from ratio_framework import Operator, _apply_operator

        assert _apply_operator(5.0, Operator.EQ, 5.0) is True
        assert _apply_operator(5.0, Operator.EQ, 5.1) is False
        # Float precision tolerance
        assert _apply_operator(5.0000000001, Operator.EQ, 5.0) is True


# ---------------------------------------------------------------------------
# _get_field_value
# ---------------------------------------------------------------------------


class TestGetFieldValue:
    def test_existing_field(self, sample_financial_data):
        from ratio_framework import _get_field_value

        assert _get_field_value(sample_financial_data, "revenue") == 1_000_000

    def test_missing_field(self, sample_financial_data):
        from ratio_framework import _get_field_value

        assert _get_field_value(sample_financial_data, "nonexistent_field") is None

    def test_none_value(self, sparse_financial_data):
        from ratio_framework import _get_field_value

        assert _get_field_value(sparse_financial_data, "total_assets") is None


# ---------------------------------------------------------------------------
# _compute_base_score
# ---------------------------------------------------------------------------


class TestComputeBaseScore:
    def test_higher_is_better_excellent(self):
        from ratio_framework import _compute_base_score

        thresholds = [(0.15, 10.0), (0.10, 8.0), (0.05, 6.0), (0.02, 4.0)]
        assert _compute_base_score(0.20, thresholds, higher_is_better=True) == 10.0

    def test_higher_is_better_good(self):
        from ratio_framework import _compute_base_score

        thresholds = [(0.15, 10.0), (0.10, 8.0), (0.05, 6.0), (0.02, 4.0)]
        assert _compute_base_score(0.12, thresholds, higher_is_better=True) == 8.0

    def test_higher_is_better_adequate(self):
        from ratio_framework import _compute_base_score

        thresholds = [(0.15, 10.0), (0.10, 8.0), (0.05, 6.0), (0.02, 4.0)]
        assert _compute_base_score(0.07, thresholds, higher_is_better=True) == 6.0

    def test_higher_is_better_below_all(self):
        from ratio_framework import _compute_base_score

        thresholds = [(0.15, 10.0), (0.10, 8.0), (0.05, 6.0), (0.02, 4.0)]
        # Below lowest threshold: score = last_score - 1.0 = 4.0 - 1.0 = 3.0
        assert _compute_base_score(0.01, thresholds, higher_is_better=True) == 3.0

    def test_lower_is_better_excellent(self):
        from ratio_framework import _compute_base_score

        thresholds = [(0.3, 10.0), (0.5, 8.0), (1.0, 6.0), (2.0, 4.0)]
        assert _compute_base_score(0.2, thresholds, higher_is_better=False) == 10.0

    def test_lower_is_better_above_all(self):
        from ratio_framework import _compute_base_score

        thresholds = [(0.3, 10.0), (0.5, 8.0), (1.0, 6.0), (2.0, 4.0)]
        # Above all thresholds: score = 4.0 - 1.0 = 3.0
        assert _compute_base_score(3.0, thresholds, higher_is_better=False) == 3.0

    def test_empty_thresholds(self):
        from ratio_framework import _compute_base_score

        assert _compute_base_score(0.5, [], higher_is_better=True) == 5.0

    def test_exact_threshold_boundary(self):
        from ratio_framework import _compute_base_score

        thresholds = [(0.15, 10.0), (0.10, 8.0)]
        # Exactly at threshold should match
        assert _compute_base_score(0.15, thresholds, higher_is_better=True) == 10.0
        assert _compute_base_score(0.10, thresholds, higher_is_better=True) == 8.0


# ---------------------------------------------------------------------------
# _apply_adjustments
# ---------------------------------------------------------------------------


class TestApplyAdjustments:
    def test_positive_adjustment(self, sample_financial_data):
        from ratio_framework import Adjustment, Operator, _apply_adjustments

        adjustments = [
            Adjustment("operating_income", Operator.GT, 0, 0.5, "Positive OI"),
        ]
        score = _apply_adjustments(7.0, adjustments, sample_financial_data, {})
        assert score == 7.5

    def test_negative_adjustment(self, sample_financial_data):
        from ratio_framework import Adjustment, Operator, _apply_adjustments

        adjustments = [
            Adjustment("total_debt", Operator.GT, 100, -1.0, "High debt"),
        ]
        score = _apply_adjustments(7.0, adjustments, sample_financial_data, {})
        assert score == 6.0

    def test_clamped_to_10(self, sample_financial_data):
        from ratio_framework import Adjustment, Operator, _apply_adjustments

        adjustments = [
            Adjustment("operating_income", Operator.GT, 0, 5.0, "Big boost"),
        ]
        score = _apply_adjustments(9.0, adjustments, sample_financial_data, {})
        assert score == 10.0

    def test_clamped_to_0(self, sample_financial_data):
        from ratio_framework import Adjustment, Operator, _apply_adjustments

        adjustments = [
            Adjustment("operating_income", Operator.GT, 0, -10.0, "Big penalty"),
        ]
        score = _apply_adjustments(5.0, adjustments, sample_financial_data, {})
        assert score == 0.0

    def test_no_adjustments(self, sample_financial_data):
        from ratio_framework import _apply_adjustments

        score = _apply_adjustments(7.0, [], sample_financial_data, {})
        assert score == 7.0

    def test_condition_not_met(self, sample_financial_data):
        from ratio_framework import Adjustment, Operator, _apply_adjustments

        adjustments = [
            Adjustment("operating_income", Operator.LT, 0, -2.0, "Negative OI"),
        ]
        # OI is 200k (> 0), so LT 0 is False, no adjustment
        score = _apply_adjustments(7.0, adjustments, sample_financial_data, {})
        assert score == 7.0

    def test_uses_computed_ratios_first(self, sample_financial_data):
        from ratio_framework import Adjustment, Operator, _apply_adjustments

        adjustments = [
            Adjustment("custom_ratio", Operator.GT, 0.5, 1.0, "Custom boost"),
        ]
        # custom_ratio isn't a field on FinancialData, but is in computed_ratios
        computed = {"custom_ratio": 0.8}
        score = _apply_adjustments(7.0, adjustments, sample_financial_data, computed)
        assert score == 8.0

    def test_none_field_skips_adjustment(self, sparse_financial_data):
        from ratio_framework import Adjustment, Operator, _apply_adjustments

        adjustments = [
            Adjustment("total_assets", Operator.GT, 0, 1.0, "Has assets"),
        ]
        # total_assets is None on sparse data, should skip
        score = _apply_adjustments(7.0, adjustments, sparse_financial_data, {})
        assert score == 7.0


# ---------------------------------------------------------------------------
# RatioDefinition.get_grade
# ---------------------------------------------------------------------------


class TestGetGrade:
    def test_excellent(self, simple_definition):
        assert simple_definition.get_grade(9.0) == "Excellent"

    def test_good(self, simple_definition):
        assert simple_definition.get_grade(7.0) == "Good"

    def test_adequate(self, simple_definition):
        assert simple_definition.get_grade(5.0) == "Adequate"

    def test_weak(self, simple_definition):
        assert simple_definition.get_grade(2.0) == "Weak"

    def test_perfect_score(self, simple_definition):
        assert simple_definition.get_grade(10.0) == "Excellent"

    def test_zero_score(self, simple_definition):
        assert simple_definition.get_grade(0.0) == "Weak"

    def test_boundary_8(self, simple_definition):
        # 8.0 is start of Excellent range [8.0, 10.0)
        assert simple_definition.get_grade(8.0) == "Excellent"


# ---------------------------------------------------------------------------
# _build_summary
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_with_value(self):
        from ratio_framework import _build_summary

        result = _build_summary("ROA", 0.075, 6.0, "Adequate", "%", "Profit efficiency")
        assert "ROA" in result
        assert "0.07%" in result  # 0.075 formatted with :.2f = "0.07"
        assert "6.0/10" in result
        assert "Adequate" in result

    def test_with_none_value(self):
        from ratio_framework import _build_summary

        result = _build_summary("ROA", None, 0.0, "Insufficient Data", "%", "")
        assert "Insufficient data" in result

    def test_no_unit(self):
        from ratio_framework import _build_summary

        result = _build_summary("Test", 1.50, 8.0, "Good", "", "Test ratio")
        assert "1.50" in result
        assert "%" not in result.split("|")[0]  # No unit in value portion


# ---------------------------------------------------------------------------
# compute_ratio (full pipeline)
# ---------------------------------------------------------------------------


class TestComputeRatio:
    def test_full_pipeline(self, sample_financial_data, simple_definition):
        from ratio_framework import compute_ratio

        result = compute_ratio(sample_financial_data, simple_definition)
        assert result.name == "Test Ratio"
        # net_income/total_assets = 150000/2000000 = 0.075
        assert result.value == pytest.approx(0.075, abs=1e-6)
        # 0.075 >= 0.05 → base score 6.0, +0.5 for positive OI = 6.5
        assert result.score == pytest.approx(6.5, abs=0.1)
        assert result.grade == "Good"
        assert "Test Ratio" in result.summary

    def test_missing_numerator(self, sparse_financial_data, simple_definition):
        from ratio_framework import compute_ratio

        # sparse_financial_data has no total_assets, so ratio is None
        result = compute_ratio(sparse_financial_data, simple_definition)
        assert result.value is None
        assert result.score == 0.0
        assert result.grade == "Insufficient Data"

    def test_zero_denominator(self, sample_financial_data):
        from ratio_framework import RatioDefinition, compute_ratio
        from financial_analyzer import FinancialData

        data = FinancialData(net_income=100_000, total_equity=0)
        defn = RatioDefinition(
            name="ROE",
            description="Test",
            numerator_field="net_income",
            denominator_field="total_equity",
        )
        result = compute_ratio(data, defn)
        assert result.value is None
        assert result.grade == "Insufficient Data"

    def test_secondary_ratios_tracked(self, sample_financial_data, simple_definition):
        from ratio_framework import compute_ratio

        result = compute_ratio(sample_financial_data, simple_definition)
        # The adjustment references "operating_income", so it should appear
        assert "operating_income" in result.secondary_ratios

    def test_no_thresholds_gives_default_score(self, sample_financial_data):
        from ratio_framework import RatioDefinition, compute_ratio

        defn = RatioDefinition(
            name="Plain Ratio",
            description="No thresholds",
            numerator_field="net_income",
            denominator_field="revenue",
            scoring_thresholds=[],  # Empty
        )
        result = compute_ratio(sample_financial_data, defn)
        assert result.score == 5.0  # Default mid-range

    def test_computed_ratios_passed(self, sample_financial_data):
        from ratio_framework import (
            Adjustment, Operator, RatioDefinition, compute_ratio,
        )

        defn = RatioDefinition(
            name="With Pre-computed",
            description="Test",
            numerator_field="net_income",
            denominator_field="total_assets",
            scoring_thresholds=[(0.05, 8.0), (0.02, 5.0)],
            adjustments=[
                Adjustment("prev_ratio", Operator.GT, 0.5, 1.0, "Previous high"),
            ],
        )
        pre = {"prev_ratio": 0.8}
        result = compute_ratio(sample_financial_data, defn, pre)
        # Base 8.0 + 1.0 adjustment = 9.0
        assert result.score == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# RATIO_CATALOG
# ---------------------------------------------------------------------------


class TestRatioCatalog:
    def test_has_expected_ratios(self):
        from ratio_framework import RATIO_CATALOG

        expected = {
            "roa", "roe", "ebit_to_total_assets", "gross_margin", "operating_margin", "net_margin",
            "current_ratio", "cash_ratio",
            "debt_to_equity", "debt_to_ebitda", "interest_coverage",
            "asset_turnover", "inventory_turnover", "receivables_turnover",
            "fcf_yield", "ocf_to_ni", "cash_conversion_cycle",
        }
        assert set(RATIO_CATALOG.keys()) == expected

    def test_all_definitions_valid(self):
        from ratio_framework import RATIO_CATALOG, RatioDefinition

        for key, defn in RATIO_CATALOG.items():
            assert isinstance(defn, RatioDefinition), f"{key} is not a RatioDefinition"
            assert defn.name, f"{key} has empty name"
            assert defn.numerator_field, f"{key} has empty numerator_field"
            assert defn.denominator_field, f"{key} has empty denominator_field"
            assert len(defn.scoring_thresholds) > 0, f"{key} has no thresholds"

    def test_leverage_ratios_lower_is_better(self):
        from ratio_framework import RATIO_CATALOG

        assert RATIO_CATALOG["debt_to_equity"].higher_is_better is False
        assert RATIO_CATALOG["debt_to_ebitda"].higher_is_better is False
        assert RATIO_CATALOG["cash_conversion_cycle"].higher_is_better is False

    def test_profitability_ratios_higher_is_better(self):
        from ratio_framework import RATIO_CATALOG

        for key in ["roa", "roe", "ebit_to_total_assets", "gross_margin", "operating_margin", "net_margin"]:
            assert RATIO_CATALOG[key].higher_is_better is True, f"{key} should be higher_is_better"


# ---------------------------------------------------------------------------
# run_all_ratios
# ---------------------------------------------------------------------------


class TestRunAllRatios:
    def test_returns_all_ratios(self, sample_financial_data):
        from ratio_framework import RATIO_CATALOG, run_all_ratios

        results = run_all_ratios(sample_financial_data)
        assert len(results) == len(RATIO_CATALOG)
        assert set(results.keys()) == set(RATIO_CATALOG.keys())

    def test_all_results_are_ratio_result(self, sample_financial_data):
        from ratio_framework import RatioResult, run_all_ratios

        results = run_all_ratios(sample_financial_data)
        for key, result in results.items():
            assert isinstance(result, RatioResult), f"{key} is not RatioResult"

    def test_strong_company_scores(self, sample_financial_data):
        from ratio_framework import run_all_ratios

        results = run_all_ratios(sample_financial_data)
        # ROA: 150k/2M = 0.075, should be Adequate-Good range
        roa = results["roa"]
        assert roa.value == pytest.approx(0.075, abs=1e-4)
        assert roa.score >= 5.0

    def test_sparse_data_handles_none(self, sparse_financial_data):
        from ratio_framework import run_all_ratios

        results = run_all_ratios(sparse_financial_data)
        # Most ratios should be "Insufficient Data" with sparse data
        insufficient_count = sum(
            1 for r in results.values() if r.grade == "Insufficient Data"
        )
        assert insufficient_count >= 10  # Most fields are None

    def test_cash_ratio_uses_cash_field(self, sample_financial_data):
        """Regression: cash_ratio must use FinancialData.cash, not cash_and_equivalents."""
        from ratio_framework import run_all_ratios

        results = run_all_ratios(sample_financial_data)
        cash_ratio = results["cash_ratio"]
        # cash=300k / current_liabilities=400k = 0.75
        assert cash_ratio.value is not None, "cash_ratio should compute with 'cash' field"
        assert cash_ratio.value == pytest.approx(0.75, abs=0.01)


# ---------------------------------------------------------------------------
# get_ratio_by_category
# ---------------------------------------------------------------------------


class TestGetRatioByCategory:
    def test_returns_all_categories(self):
        from ratio_framework import get_ratio_by_category

        categories = get_ratio_by_category()
        assert set(categories.keys()) == {
            "Profitability", "Liquidity", "Leverage", "Efficiency", "Cash Flow"
        }

    def test_profitability_ratios(self):
        from ratio_framework import get_ratio_by_category

        categories = get_ratio_by_category()
        assert "roa" in categories["Profitability"]
        assert "roe" in categories["Profitability"]
        assert len(categories["Profitability"]) == 6

    def test_all_catalog_ratios_categorized(self):
        from ratio_framework import RATIO_CATALOG, get_ratio_by_category

        categories = get_ratio_by_category()
        categorized = set()
        for ratio_keys in categories.values():
            categorized.update(ratio_keys)
        assert categorized == set(RATIO_CATALOG.keys())


# ---------------------------------------------------------------------------
# compute_category
# ---------------------------------------------------------------------------


class TestComputeCategory:
    def test_profitability(self, sample_financial_data):
        from ratio_framework import compute_category

        results = compute_category(sample_financial_data, "Profitability")
        assert "roa" in results
        assert "roe" in results
        assert len(results) == 6

    def test_unknown_category(self, sample_financial_data):
        from ratio_framework import compute_category

        results = compute_category(sample_financial_data, "NonExistent")
        assert results == {}

    def test_liquidity_values(self, sample_financial_data):
        from ratio_framework import compute_category

        results = compute_category(sample_financial_data, "Liquidity")
        # current_ratio = 800k/400k = 2.0
        assert results["current_ratio"].value == pytest.approx(2.0, abs=0.01)

    def test_leverage_values(self, sample_financial_data):
        from ratio_framework import compute_category

        results = compute_category(sample_financial_data, "Leverage")
        # debt_to_equity = 500k/1M = 0.5
        assert results["debt_to_equity"].value == pytest.approx(0.5, abs=0.01)
        # interest_coverage = 200k/30k ≈ 6.67
        assert results["interest_coverage"].value == pytest.approx(6.667, abs=0.01)


# ---------------------------------------------------------------------------
# WP-7(f): ratio threshold provenance docs (docs-only, no scoring change)
# ---------------------------------------------------------------------------


class TestThresholdProvenance:
    """RatioDefinition exposes a threshold_source provenance field, populated
    for catalog entries; threshold VALUES are invariant (snapshot)."""

    def test_field_exists_with_empty_default(self):
        from ratio_framework import RatioDefinition

        definition = RatioDefinition(
            name="X",
            description="x",
            numerator_field="net_income",
            denominator_field="total_assets",
        )
        # New optional field defaults to empty string (no behavior change).
        assert definition.threshold_source == ""

    def test_catalog_entries_have_nonempty_threshold_source(self):
        from ratio_framework import RATIO_CATALOG

        # Every catalog entry that defines scoring_thresholds must document
        # their provenance/rationale.
        documented = 0
        for key, definition in RATIO_CATALOG.items():
            if definition.scoring_thresholds:
                assert isinstance(definition.threshold_source, str)
                assert definition.threshold_source.strip(), (
                    f"{key} has scoring_thresholds but empty threshold_source"
                )
                documented += 1
        assert documented == len(RATIO_CATALOG)

    def test_threshold_values_unchanged_snapshot(self):
        """Snapshot of every catalog entry's scoring_thresholds tuples; the
        docs-only change must not alter any threshold or score value."""
        from ratio_framework import RATIO_CATALOG

        expected = {
            "roa": [(0.15, 10.0), (0.10, 8.0), (0.05, 6.0), (0.02, 4.0)],
            "roe": [(0.20, 10.0), (0.15, 8.0), (0.10, 6.0), (0.05, 4.0)],
            "ebit_to_total_assets": [(0.15, 10.0), (0.10, 8.0), (0.07, 6.0), (0.05, 4.0)],
            "gross_margin": [(0.50, 10.0), (0.35, 8.0), (0.25, 6.0), (0.15, 4.0)],
            "operating_margin": [(0.20, 10.0), (0.15, 8.0), (0.10, 6.0), (0.05, 4.0)],
            "net_margin": [(0.15, 10.0), (0.10, 8.0), (0.05, 6.0), (0.02, 4.0)],
            "current_ratio": [(2.0, 10.0), (1.5, 8.0), (1.0, 6.0), (0.75, 4.0)],
            "cash_ratio": [(0.75, 10.0), (0.50, 8.0), (0.30, 6.0), (0.15, 4.0)],
            "debt_to_equity": [(0.3, 10.0), (0.5, 8.0), (1.0, 6.0), (2.0, 4.0)],
            "debt_to_ebitda": [(2.0, 10.0), (3.0, 8.0), (4.0, 6.0), (5.0, 4.0)],
            "interest_coverage": [(8.0, 10.0), (5.0, 8.0), (2.5, 6.0), (1.5, 4.0)],
            "asset_turnover": [(2.0, 10.0), (1.5, 8.0), (1.0, 6.0), (0.5, 4.0)],
            "inventory_turnover": [(12.0, 10.0), (8.0, 8.0), (6.0, 6.0), (4.0, 4.0)],
            "receivables_turnover": [(12.0, 10.0), (10.0, 8.0), (8.0, 6.0), (6.0, 4.0)],
            "fcf_yield": [(0.10, 10.0), (0.07, 8.0), (0.05, 6.0), (0.03, 4.0)],
            "ocf_to_ni": [(1.2, 10.0), (1.0, 8.0), (0.8, 6.0), (0.6, 4.0)],
            "cash_conversion_cycle": [(0.05, 10.0), (0.10, 8.0), (0.15, 6.0), (0.20, 4.0)],
        }
        actual = {
            key: list(definition.scoring_thresholds)
            for key, definition in RATIO_CATALOG.items()
        }
        assert actual == expected

    def test_computed_scores_unchanged_snapshot(self, sample_financial_data):
        """Computed scores for a fixed FinancialData are invariant: thresholds
        and adjustments are unchanged by the docs-only edit."""
        from ratio_framework import run_all_ratios

        results = run_all_ratios(sample_financial_data)
        scores = {key: result.score for key, result in results.items()}

        # Locked baseline of computed scores for the fixed fixture. Captured
        # pre-edit; the docs-only threshold_source change must not alter any
        # of these (thresholds and adjustments are untouched).
        expected_scores = {
            "roa": 6.5,
            "roe": 7.5,
            "ebit_to_total_assets": 8.0,
            "gross_margin": 8.0,
            "operating_margin": 10.0,
            "net_margin": 10.0,
            "current_ratio": 10.0,
            "cash_ratio": 10.0,
            "debt_to_equity": 8.5,
            "debt_to_ebitda": 10.0,
            "interest_coverage": 8.0,
            "asset_turnover": 4.0,
            "inventory_turnover": 6.0,
            "receivables_turnover": 4.0,
            "fcf_yield": 8.0,
            "ocf_to_ni": 10.0,
            "cash_conversion_cycle": 8.0,
        }
        assert scores == expected_scores
