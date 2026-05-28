"""Tests for shared export utility helpers in export_utils.py."""

import pytest

from export_utils import (
    _PERCENT_KEYWORDS,
    _DOLLAR_KEYWORDS,
    _RATIO_KEYWORDS,
    _is_percent_key,
    _is_ratio_key,
    _is_dollar_key,
    _CATEGORY_MAP,
    _categorize,
    score_to_grade,
)


# ---------------------------------------------------------------------------
# score_to_grade (canonical 100-scale grading)
# ---------------------------------------------------------------------------

class TestScoreToGrade:
    def test_grade_a_at_80(self):
        assert score_to_grade(80) == "A"

    def test_grade_a_at_100(self):
        assert score_to_grade(100) == "A"

    def test_grade_b_at_65(self):
        assert score_to_grade(65) == "B"

    def test_grade_b_at_79(self):
        assert score_to_grade(79) == "B"

    def test_grade_c_at_50(self):
        assert score_to_grade(50) == "C"

    def test_grade_d_at_35(self):
        assert score_to_grade(35) == "D"

    def test_grade_f_at_34(self):
        assert score_to_grade(34) == "F"

    def test_grade_f_at_0(self):
        assert score_to_grade(0) == "F"

    def test_grade_f_negative(self):
        assert score_to_grade(-5) == "F"

    # WS-3 WP-8 (lock-only): None must raise TypeError (fail-fast contract).
    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            score_to_grade(None)


# ---------------------------------------------------------------------------
# _is_percent_key
# ---------------------------------------------------------------------------

class TestIsPercentKey:
    def test_gross_margin_is_percent(self):
        assert _is_percent_key("gross_margin") is True

    def test_roe_is_percent(self):
        assert _is_percent_key("roe") is True

    def test_coverage_ratio_is_percent(self):
        assert _is_percent_key("coverage_ratio") is True

    def test_operating_margin_is_percent(self):
        assert _is_percent_key("operating_margin") is True

    def test_asset_turnover_is_percent(self):
        assert _is_percent_key("asset_turnover") is True

    def test_interest_rate_is_percent(self):
        assert _is_percent_key("interest_rate") is True

    def test_revenue_is_not_percent(self):
        assert _is_percent_key("revenue") is False

    def test_total_assets_is_not_percent(self):
        assert _is_percent_key("total_assets") is False

    def test_net_income_is_not_percent(self):
        assert _is_percent_key("net_income") is False

    def test_case_insensitive(self):
        assert _is_percent_key("Gross_Margin") is True
        assert _is_percent_key("GROSS_MARGIN") is True


# ---------------------------------------------------------------------------
# _is_dollar_key
# ---------------------------------------------------------------------------

class TestIsDollarKey:
    def test_revenue_is_dollar(self):
        assert _is_dollar_key("revenue") is True

    def test_total_assets_is_dollar(self):
        assert _is_dollar_key("total_assets") is True

    def test_net_income_is_dollar(self):
        assert _is_dollar_key("net_income") is True

    def test_total_debt_is_dollar(self):
        assert _is_dollar_key("total_debt") is True

    def test_cash_and_equivalents_is_dollar(self):
        assert _is_dollar_key("cash_and_equivalents") is True

    def test_ebitda_is_dollar(self):
        assert _is_dollar_key("ebitda") is True

    def test_accounts_payable_is_dollar(self):
        assert _is_dollar_key("accounts_payable") is True

    def test_accounts_receivable_is_dollar(self):
        assert _is_dollar_key("accounts_receivable") is True

    def test_gross_margin_is_not_dollar(self):
        assert _is_dollar_key("gross_margin") is False

    def test_current_ratio_is_not_dollar(self):
        assert _is_dollar_key("current_ratio") is False

    def test_case_insensitive(self):
        assert _is_dollar_key("Total_Assets") is True
        assert _is_dollar_key("REVENUE") is True


# ---------------------------------------------------------------------------
# _categorize
# ---------------------------------------------------------------------------

class TestCategorize:
    # Liquidity
    def test_current_ratio_liquidity(self):
        assert _categorize("current_ratio") == "Liquidity"

    def test_quick_ratio_liquidity(self):
        assert _categorize("quick_ratio") == "Liquidity"

    def test_cash_ratio_liquidity(self):
        assert _categorize("cash_ratio") == "Liquidity"

    def test_working_capital_liquidity(self):
        assert _categorize("working_capital") == "Liquidity"

    # Profitability
    def test_gross_margin_profitability(self):
        assert _categorize("gross_margin") == "Profitability"

    def test_operating_margin_profitability(self):
        assert _categorize("operating_margin") == "Profitability"

    def test_net_margin_profitability(self):
        assert _categorize("net_margin") == "Profitability"

    def test_roe_profitability(self):
        assert _categorize("roe") == "Profitability"

    def test_roa_profitability(self):
        assert _categorize("roa") == "Profitability"

    def test_roic_profitability(self):
        assert _categorize("roic") == "Profitability"

    # Leverage
    def test_debt_to_equity_leverage(self):
        assert _categorize("debt_to_equity") == "Leverage"

    def test_debt_to_assets_leverage(self):
        assert _categorize("debt_to_assets") == "Leverage"

    def test_debt_ratio_leverage(self):
        assert _categorize("debt_ratio") == "Leverage"

    def test_equity_multiplier_leverage(self):
        assert _categorize("equity_multiplier") == "Leverage"

    def test_interest_coverage_leverage(self):
        assert _categorize("interest_coverage") == "Leverage"

    # Efficiency
    def test_asset_turnover_efficiency(self):
        assert _categorize("asset_turnover") == "Efficiency"

    def test_inventory_turnover_efficiency(self):
        assert _categorize("inventory_turnover") == "Efficiency"

    def test_receivables_turnover_efficiency(self):
        assert _categorize("receivables_turnover") == "Efficiency"

    def test_payables_turnover_efficiency(self):
        assert _categorize("payables_turnover") == "Efficiency"

    # Unknown key falls back to Other
    def test_unknown_key_returns_other(self):
        assert _categorize("unknown_metric") == "Other"

    def test_empty_string_returns_other(self):
        assert _categorize("") == "Other"

    def test_arbitrary_string_returns_other(self):
        assert _categorize("price_to_earnings") == "Other"


# ---------------------------------------------------------------------------
# _CATEGORY_MAP completeness - all 19 keys are present
# ---------------------------------------------------------------------------

class TestCategoryMapCompleteness:
    EXPECTED_KEYS = {
        "current_ratio", "quick_ratio", "cash_ratio", "working_capital",
        "gross_margin", "operating_margin", "net_margin", "roe", "roa", "roic",
        "debt_to_equity", "debt_to_assets", "debt_ratio", "equity_multiplier",
        "interest_coverage",
        "asset_turnover", "inventory_turnover", "receivables_turnover",
        "payables_turnover",
    }

    def test_category_map_has_19_entries(self):
        assert len(_CATEGORY_MAP) == 19

    def test_all_expected_keys_present(self):
        assert set(_CATEGORY_MAP.keys()) == self.EXPECTED_KEYS

    def test_all_categories_are_valid(self):
        valid_categories = {"Liquidity", "Profitability", "Leverage", "Efficiency"}
        assert set(_CATEGORY_MAP.values()).issubset(valid_categories)



# ---------------------------------------------------------------------------
# WS-3 P0-8: ratio keys must NOT be classified as percent
# ---------------------------------------------------------------------------

class TestIsRatioKey:
    """Regression tests for P0-8: 'ratio' keys must render as multipliers (1.50x), not percent (150%)."""

    def test_current_ratio_is_ratio_not_percent(self):
        assert _is_ratio_key("current_ratio") is True
        assert _is_percent_key("current_ratio") is False

    def test_quick_ratio_is_ratio_not_percent(self):
        assert _is_ratio_key("quick_ratio") is True
        assert _is_percent_key("quick_ratio") is False

    def test_cash_ratio_is_ratio_not_percent(self):
        assert _is_ratio_key("cash_ratio") is True
        assert _is_percent_key("cash_ratio") is False

    def test_debt_ratio_is_ratio_not_percent(self):
        assert _is_ratio_key("debt_ratio") is True
        assert _is_percent_key("debt_ratio") is False

    def test_debt_to_equity_is_ratio(self):
        assert _is_ratio_key("debt_to_equity") is True

    def test_debt_to_assets_is_ratio(self):
        assert _is_ratio_key("debt_to_assets") is True

    def test_equity_multiplier_is_ratio(self):
        assert _is_ratio_key("equity_multiplier") is True

    def test_coverage_ratio_still_percent(self):
        # coverage_ratio matches both -- PDF/XLSX route ratio FIRST so it renders as 'x',
        # but _is_percent_key is True via 'coverage' keyword which is acceptable.
        assert _is_ratio_key("coverage_ratio") is True

    def test_revenue_is_not_ratio(self):
        assert _is_ratio_key("revenue") is False

    def test_gross_margin_is_not_ratio(self):
        assert _is_ratio_key("gross_margin") is False

    def test_case_insensitive(self):
        assert _is_ratio_key("Current_Ratio") is True
        assert _is_ratio_key("DEBT_TO_EQUITY") is True

    def test_ratio_keywords_excludes_legacy(self):
        # ensure 'ratio' was removed from percent keywords
        assert "ratio" not in _PERCENT_KEYWORDS
        assert "ratio" in _RATIO_KEYWORDS
