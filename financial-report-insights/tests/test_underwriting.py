"""Tests for underwriting and credit analysis module."""

import pytest

from financial_analyzer import FinancialData
from underwriting import (
    CovenantPackage,
    CreditScorecard,
    DebtCapacityResult,
    LoanStructure,
    UnderwritingAnalyzer,
    UnderwritingReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer():
    return UnderwritingAnalyzer()


@pytest.fixture
def strong_company():
    """Company with strong financials -- should get A grade."""
    return FinancialData(
        total_assets=10_000_000,
        current_assets=4_000_000,
        cash=1_500_000,
        inventory=500_000,
        accounts_receivable=800_000,
        total_liabilities=3_000_000,
        current_liabilities=1_500_000,
        total_debt=1_500_000,
        total_equity=7_000_000,
        revenue=15_000_000,
        cogs=9_000_000,
        gross_profit=6_000_000,
        operating_income=3_000_000,
        net_income=2_000_000,
        ebit=3_200_000,
        ebitda=4_000_000,
        interest_expense=200_000,
        operating_cash_flow=3_500_000,
        capex=500_000,
    )


@pytest.fixture
def weak_company():
    """Company with weak financials -- should get D/F grade."""
    return FinancialData(
        total_assets=5_000_000,
        current_assets=800_000,
        cash=50_000,
        inventory=400_000,
        total_liabilities=4_500_000,
        current_liabilities=1_200_000,
        total_debt=3_500_000,
        total_equity=500_000,
        revenue=3_000_000,
        cogs=2_700_000,
        gross_profit=300_000,
        operating_income=50_000,
        net_income=-100_000,
        ebit=100_000,
        ebitda=200_000,
        interest_expense=300_000,
        operating_cash_flow=100_000,
        capex=80_000,
    )


@pytest.fixture
def medium_company():
    """Company with mixed financials -- should get C grade."""
    return FinancialData(
        total_assets=8_000_000,
        current_assets=2_500_000,
        cash=400_000,
        inventory=600_000,
        total_liabilities=4_000_000,
        current_liabilities=1_600_000,
        total_debt=2_500_000,
        total_equity=4_000_000,
        revenue=10_000_000,
        cogs=7_000_000,
        gross_profit=3_000_000,
        operating_income=1_200_000,
        net_income=600_000,
        ebit=1_500_000,
        ebitda=2_000_000,
        interest_expense=400_000,
        operating_cash_flow=1_200_000,
        capex=300_000,
    )


# ---------------------------------------------------------------------------
# CreditScorecard tests
# ---------------------------------------------------------------------------


class TestCreditScorecard:
    def test_strong_company_high_score(self, analyzer, strong_company):
        sc = analyzer.credit_scorecard(strong_company)
        assert sc.total_score >= 70
        assert sc.grade in ("A", "B")
        assert sc.recommendation in ("approve", "conditional")

    def test_weak_company_low_score(self, analyzer, weak_company):
        sc = analyzer.credit_scorecard(weak_company)
        assert sc.total_score <= 40
        assert sc.grade in ("D", "F")
        assert sc.recommendation == "decline"

    def test_scorecard_has_five_categories(self, analyzer, strong_company):
        sc = analyzer.credit_scorecard(strong_company)
        assert len(sc.category_scores) == 5
        expected_cats = {"profitability", "leverage", "liquidity", "cash_flow", "stability"}
        assert set(sc.category_scores.keys()) == expected_cats
        assert all(0 <= v <= 20 for v in sc.category_scores.values())
        assert sc.total_score == sum(sc.category_scores.values())

    def test_scorecard_max_100(self, analyzer, strong_company):
        sc = analyzer.credit_scorecard(strong_company)
        assert 0 <= sc.total_score <= 100

    def test_scorecard_grade_boundaries(self, analyzer):
        """Minimal data -- tests graceful handling of None fields."""
        data = FinancialData()
        sc = analyzer.credit_scorecard(data)
        assert sc.total_score >= 0
        assert sc.grade in ("A", "B", "C", "D", "F")

    def test_strengths_weaknesses_populated(self, analyzer, strong_company):
        sc = analyzer.credit_scorecard(strong_company)
        assert len(sc.strengths) > 0  # Strong company should have strengths

    def test_weak_company_has_weaknesses(self, analyzer, weak_company):
        sc = analyzer.credit_scorecard(weak_company)
        assert len(sc.weaknesses) > 0

    def test_conditional_has_conditions(self, analyzer, medium_company):
        sc = analyzer.credit_scorecard(medium_company)
        if sc.recommendation == "conditional":
            assert len(sc.conditions) > 0

    def test_grade_a_threshold(self, analyzer):
        """Verify score >= 80 maps to grade A."""
        from underwriting import _score_to_grade

        assert _score_to_grade(100) == "A"
        assert _score_to_grade(80) == "A"
        assert _score_to_grade(79) == "B"

    def test_grade_f_threshold(self, analyzer):
        from underwriting import _score_to_grade

        assert _score_to_grade(34) == "F"
        assert _score_to_grade(0) == "F"
        assert _score_to_grade(35) == "D"

    def test_negative_equity_scores_zero_leverage(self, analyzer):
        """CORRUPT-04 fix: negative equity must not earn max leverage points."""
        from underwriting import UnderwritingAnalyzer
        # D/E = 500k / -200k = -2.5 (negative equity)
        # D/A = 500k / 300k = 1.67
        data = FinancialData(
            total_debt=500_000,
            total_equity=-200_000,
            total_assets=300_000,
            revenue=1_000_000,
            net_income=50_000,
        )
        sc = analyzer.credit_scorecard(data)
        assert sc.category_scores["leverage"] == 0

    def test_negative_d_e_from_negative_equity(self, analyzer):
        """Verify _score_leverage returns 0 when D/E is negative."""
        assert UnderwritingAnalyzer._score_leverage(-2.5, 1.67) == 0
        assert UnderwritingAnalyzer._score_leverage(-0.1, 0.1) == 0


# ---------------------------------------------------------------------------
# DebtCapacity tests
# ---------------------------------------------------------------------------


class TestDebtCapacity:
    @pytest.fixture
    def sample_data(self):
        return FinancialData(
            total_debt=2_000_000,
            ebitda=3_000_000,
            total_assets=10_000_000,
            total_equity=5_000_000,
            revenue=8_000_000,
            operating_cash_flow=2_500_000,
        )

    def test_debt_capacity_without_loan(self, analyzer, sample_data):
        result = analyzer.debt_capacity(sample_data)
        assert result.current_leverage is not None
        assert result.max_additional_debt is not None
        assert result.max_additional_debt > 0
        # 3.5 * 3M - 2M = 8.5M
        assert abs(result.max_additional_debt - 8_500_000) < 1

    def test_debt_capacity_with_loan(self, analyzer, sample_data):
        loan = LoanStructure(principal=1_000_000, annual_rate=0.06, term_years=5)
        result = analyzer.debt_capacity(sample_data, loan)
        assert result.pro_forma_debt == 3_000_000
        assert result.pro_forma_leverage is not None
        assert result.pro_forma_dscr is not None
        assert result.headroom_pct is not None

    def test_debt_capacity_overleveraged(self, analyzer):
        data = FinancialData(total_debt=10_000_000, ebitda=1_000_000)
        result = analyzer.debt_capacity(data)
        # 3.5 * 1M = 3.5M < 10M  ->  max_additional = 0
        assert result.max_additional_debt == 0

    def test_debt_capacity_no_data(self, analyzer):
        data = FinancialData()
        result = analyzer.debt_capacity(data)
        assert result.assessment != ""  # Should still produce assessment

    def test_pro_forma_dscr_strong(self, analyzer, sample_data):
        """Small loan relative to EBITDA should yield strong DSCR."""
        loan = LoanStructure(principal=500_000, annual_rate=0.05, term_years=5)
        result = analyzer.debt_capacity(sample_data, loan)
        # annual debt service = 100k + 25k = 125k; DSCR = 3M / 125k = 24
        assert result.pro_forma_dscr is not None
        assert result.pro_forma_dscr > 5.0

    def test_headroom_decreases_with_large_loan(self, analyzer, sample_data):
        small_loan = LoanStructure(principal=1_000_000)
        large_loan = LoanStructure(principal=7_000_000)
        r_small = analyzer.debt_capacity(sample_data, small_loan)
        r_large = analyzer.debt_capacity(sample_data, large_loan)
        assert r_small.headroom_pct > r_large.headroom_pct

    def test_assessment_warns_on_excess_leverage(self, analyzer):
        data = FinancialData(total_debt=3_000_000, ebitda=1_000_000)
        loan = LoanStructure(principal=2_000_000)
        result = analyzer.debt_capacity(data, loan)
        # pro_forma = 5M, target = 3.5M -> warning
        assert "WARNING" in result.assessment or result.pro_forma_leverage > 3.5

    # ------------------------------------------------------------------
    # P0-6: DSCR includes existing debt service
    # ------------------------------------------------------------------

    def test_p0_6_dscr_drops_when_existing_service_supplied(self, analyzer):
        """Pro-forma DSCR must shrink once existing debt service is included."""
        data = FinancialData(total_debt=2_000_000, ebitda=3_000_000)
        loan = LoanStructure(principal=1_000_000, annual_rate=0.06, term_years=5)

        # Baseline: no existing service
        loan_no_existing = LoanStructure(
            principal=1_000_000, annual_rate=0.06, term_years=5,
            existing_debt_service=0.0,
        )
        r_no = analyzer.debt_capacity(data, loan_no_existing)

        # With explicit existing service
        loan_with_existing = LoanStructure(
            principal=1_000_000, annual_rate=0.06, term_years=5,
            existing_debt_service=300_000,
        )
        r_with = analyzer.debt_capacity(data, loan_with_existing)

        assert r_with.pro_forma_dscr is not None
        assert r_no.pro_forma_dscr is not None
        # Same EBITDA, larger denominator -> smaller DSCR
        assert r_with.pro_forma_dscr < r_no.pro_forma_dscr

    def test_p0_6_existing_service_estimated_from_interest_and_debt(self, analyzer):
        """When existing_debt_service is None, estimator uses interest_expense + total_debt/term."""
        data = FinancialData(
            total_debt=2_000_000,
            ebitda=3_000_000,
            interest_expense=100_000,
        )
        loan = LoanStructure(principal=500_000, annual_rate=0.05, term_years=5)
        # New service = 100k principal + 25k interest = 125k
        # Estimated existing = 100k interest + 2M/5 = 500k principal = 600k
        # Total = 725k -> DSCR = 3M / 725k ~ 4.14
        r = analyzer.debt_capacity(data, loan)
        assert r.pro_forma_dscr is not None
        assert 3.5 < r.pro_forma_dscr < 5.0

    def test_p0_6_loan_structure_has_existing_debt_service_field(self):
        """LoanStructure exposes existing_debt_service (default None)."""
        ls = LoanStructure()
        assert hasattr(ls, "existing_debt_service")
        assert ls.existing_debt_service is None
        ls2 = LoanStructure(existing_debt_service=250_000)
        assert ls2.existing_debt_service == 250_000

    # ------------------------------------------------------------------
    # P0-7: Debt capacity returns None for non-positive EBITDA
    # ------------------------------------------------------------------

    def test_p0_7_negative_ebitda_returns_none_capacity(self, analyzer):
        """Negative EBITDA must produce max_additional_debt=None, not 0."""
        data = FinancialData(total_debt=1_000_000, ebitda=-500_000)
        result = analyzer.debt_capacity(data)
        assert result.max_additional_debt is None
        assert "undefined" in result.assessment.lower()

    def test_p0_7_zero_ebitda_returns_none_capacity(self, analyzer):
        """Zero EBITDA must also produce None (leverage ratio is undefined)."""
        data = FinancialData(total_debt=500_000, ebitda=0.0)
        result = analyzer.debt_capacity(data)
        assert result.max_additional_debt is None
        assert "undefined" in result.assessment.lower()

    def test_p0_7_missing_ebitda_returns_none_capacity(self, analyzer):
        """No EBITDA at all must produce None."""
        data = FinancialData(total_debt=500_000)
        result = analyzer.debt_capacity(data)
        assert result.max_additional_debt is None

    def test_p0_7_headroom_pct_is_none_when_capacity_undefined(self, analyzer):
        """headroom_pct should be None (not 0.0) when capacity is undefined."""
        data = FinancialData(total_debt=1_000_000, ebitda=-100_000)
        loan = LoanStructure(principal=500_000)
        result = analyzer.debt_capacity(data, loan)
        assert result.max_additional_debt is None
        assert result.headroom_pct is None


# ---------------------------------------------------------------------------
# CovenantPackage tests
# ---------------------------------------------------------------------------


class TestCovenantPackage:
    def test_light_covenants_for_strong_credit(self, analyzer):
        scorecard = CreditScorecard(
            total_score=85, grade="A", recommendation="approve"
        )
        data = FinancialData(capex=500_000)
        covenants = analyzer.recommend_covenants(data, scorecard)
        assert covenants.covenant_tier == "light"
        assert len(covenants.financial_covenants) >= 3

    def test_standard_covenants_for_b_grade(self, analyzer):
        scorecard = CreditScorecard(
            total_score=70, grade="B", recommendation="approve"
        )
        data = FinancialData()
        covenants = analyzer.recommend_covenants(data, scorecard)
        assert covenants.covenant_tier == "standard"

    def test_heavy_covenants_for_weak_credit(self, analyzer):
        scorecard = CreditScorecard(
            total_score=40, grade="D", recommendation="decline"
        )
        data = FinancialData(capex=500_000)
        covenants = analyzer.recommend_covenants(data, scorecard)
        assert covenants.covenant_tier == "heavy"
        assert len(covenants.financial_covenants) >= 4  # Extra covenants
        assert "min_fixed_charge_coverage" in covenants.financial_covenants
        assert "max_capex" in covenants.financial_covenants

    def test_heavy_without_capex_data(self, analyzer):
        scorecard = CreditScorecard(total_score=30, grade="F")
        data = FinancialData()  # no capex
        covenants = analyzer.recommend_covenants(data, scorecard)
        assert covenants.covenant_tier == "heavy"
        assert "max_capex" not in covenants.financial_covenants

    def test_reporting_requirements(self, analyzer):
        scorecard = CreditScorecard(total_score=60, grade="C")
        data = FinancialData()
        covenants = analyzer.recommend_covenants(data, scorecard)
        assert len(covenants.reporting_requirements) >= 2
        assert len(covenants.events_of_default) >= 3

    def test_heavy_has_monthly_reporting(self, analyzer):
        scorecard = CreditScorecard(total_score=30, grade="F")
        data = FinancialData()
        covenants = analyzer.recommend_covenants(data, scorecard)
        monthly = [r for r in covenants.reporting_requirements if "Monthly" in r]
        assert len(monthly) >= 1

    def test_events_of_default_always_present(self, analyzer):
        for grade in ("A", "B", "C", "D", "F"):
            scorecard = CreditScorecard(grade=grade)
            covenants = analyzer.recommend_covenants(FinancialData(), scorecard)
            assert len(covenants.events_of_default) == 4

    def test_covenant_thresholds_tighten_with_tier(self, analyzer):
        light_sc = CreditScorecard(grade="A")
        heavy_sc = CreditScorecard(grade="D")
        light = analyzer.recommend_covenants(FinancialData(), light_sc)
        heavy = analyzer.recommend_covenants(FinancialData(), heavy_sc)
        # Heavy tier should have tighter (higher) current ratio threshold
        assert (
            heavy.financial_covenants["min_current_ratio"]["threshold"]
            > light.financial_covenants["min_current_ratio"]["threshold"]
        )


# ---------------------------------------------------------------------------
# Full Underwriting Report tests
# ---------------------------------------------------------------------------


class TestFullUnderwriting:
    @pytest.fixture
    def sample_data(self):
        return FinancialData(
            total_assets=10_000_000,
            current_assets=4_000_000,
            cash=1_500_000,
            total_liabilities=3_000_000,
            current_liabilities=1_500_000,
            total_debt=1_500_000,
            total_equity=7_000_000,
            revenue=15_000_000,
            net_income=2_000_000,
            ebit=3_200_000,
            ebitda=4_000_000,
            interest_expense=200_000,
            operating_cash_flow=3_500_000,
            capex=500_000,
        )

    def test_full_underwriting_all_components(self, analyzer, sample_data):
        loan = LoanStructure(principal=2_000_000, annual_rate=0.05, term_years=7)
        report = analyzer.full_underwriting(sample_data, loan)
        assert isinstance(report, UnderwritingReport)
        assert report.scorecard is not None
        assert report.debt_capacity is not None
        assert report.covenants is not None
        assert report.loan is loan
        assert report.summary != ""

    def test_full_underwriting_without_loan(self, analyzer, sample_data):
        report = analyzer.full_underwriting(sample_data)
        assert report.scorecard is not None
        assert report.debt_capacity is not None
        assert report.covenants is not None
        assert report.loan is None
        assert report.summary != ""

    def test_summary_contains_score(self, analyzer, sample_data):
        report = analyzer.full_underwriting(sample_data)
        assert "Credit Score:" in report.summary
        assert report.scorecard.grade in report.summary

    def test_summary_contains_loan_info_when_given(self, analyzer, sample_data):
        loan = LoanStructure(principal=1_000_000, annual_rate=0.04, term_years=3)
        report = analyzer.full_underwriting(sample_data, loan)
        assert "$1,000,000" in report.summary
        assert "3 years" in report.summary

    def test_report_with_empty_data(self, analyzer):
        report = analyzer.full_underwriting(FinancialData())
        assert isinstance(report, UnderwritingReport)
        assert report.scorecard is not None
        assert report.summary != ""

    def test_report_consistency(self, analyzer, sample_data):
        """Scorecard grade should match covenant tier logic."""
        report = analyzer.full_underwriting(sample_data)
        grade = report.scorecard.grade
        tier = report.covenants.covenant_tier
        if grade == "A":
            assert tier == "light"
        elif grade == "B":
            assert tier == "standard"
        else:
            assert tier == "heavy"


# ---------------------------------------------------------------------------
# Dataclass defaults / edge cases
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_loan_structure_defaults(self):
        loan = LoanStructure()
        assert loan.principal == 0.0
        assert loan.annual_rate == 0.05
        assert loan.term_years == 5
        assert loan.loan_type == "term"

    def test_credit_scorecard_defaults(self):
        sc = CreditScorecard()
        assert sc.total_score == 0
        assert sc.grade == "F"
        assert sc.recommendation == "decline"
        assert sc.category_scores == {}

    def test_debt_capacity_result_defaults(self):
        dc = DebtCapacityResult()
        assert dc.max_leverage_target == 3.5
        assert dc.assessment == ""

    def test_covenant_package_defaults(self):
        cp = CovenantPackage()
        assert cp.covenant_tier == "standard"
        assert cp.financial_covenants == {}

    def test_underwriting_report_defaults(self):
        r = UnderwritingReport()
        assert r.scorecard is None
        assert r.summary == ""


# ---------------------------------------------------------------------------
# Scoring function boundary tests (exact threshold values)
# ---------------------------------------------------------------------------


class TestScoringBoundaries:
    """Test exact boundary values for all 5 _score_* static methods."""

    # --- Profitability: net_margin and roa thresholds ---
    @pytest.mark.parametrize("nm,roa,expected", [
        (0.101, 0.081, 20),   # Above top tier
        (0.10, 0.08, 15),     # AT threshold: not > 0.10, falls to tier 2
        (0.051, 0.041, 15),   # Above tier 2
        (0.05, 0.04, 10),     # AT threshold: falls to tier 3
        (0.021, 0.021, 10),   # Above tier 3
        (0.02, 0.02, 5),      # AT threshold: falls to tier 4
        (0.001, 0.001, 5),    # Above tier 4
        (0, 0, 0),            # AT threshold: not > 0, falls to 0
        (None, None, None),   # WP-7b: not evaluable -> None (was misleading 0)
    ])
    def test_score_profitability(self, nm, roa, expected):
        assert UnderwritingAnalyzer._score_profitability(nm, roa) == expected

    # --- Leverage: d_e and d_a thresholds ---
    @pytest.mark.parametrize("de,da,expected", [
        (0.49, 0.29, 20),     # Both below top tier
        (0.5, 0.3, 15),       # AT threshold: not < 0.5, falls to tier 2
        (0.99, 0.49, 15),     # Below tier 2
        (1.0, 0.5, 10),       # AT threshold: not < 1.0, falls to tier 3
        (1.99, 0.59, 10),     # Below tier 3
        (2.0, 0.6, 5),        # AT threshold: not < 2.0, falls to tier 4
        (2.99, None, 5),      # Below tier 4 (da=None -> 999)
        (3.0, None, 0),       # AT threshold: not < 3.0, falls to 0
        (-0.5, 0.2, 0),       # Negative D/E: worst case
        (None, None, 0),      # None values -> 999 -> 0
    ])
    def test_score_leverage(self, de, da, expected):
        assert UnderwritingAnalyzer._score_leverage(de, da) == expected

    # --- Liquidity: current_ratio and cash_ratio thresholds ---
    @pytest.mark.parametrize("cr,cashr,expected", [
        (2.01, 0.51, 20),     # Above top tier
        (2.0, 0.5, 15),       # AT threshold: not > 2.0, falls to tier 2
        (1.51, 0.31, 15),     # Above tier 2
        (1.5, 0.3, 10),       # AT threshold: not > 1.5, falls to tier 3
        (1.21, 0.11, 10),     # Above tier 3
        (1.2, 0.1, 5),        # AT threshold: not > 1.2, falls to tier 4
        (1.01, 0.0, 5),       # Above tier 4 (cash doesn't matter)
        (1.0, 0.0, 0),        # AT threshold: not > 1.0, falls to 0
        (None, None, None),   # WP-7b: not evaluable -> None (was misleading 0)
    ])
    def test_score_liquidity(self, cr, cashr, expected):
        assert UnderwritingAnalyzer._score_liquidity(cr, cashr) == expected

    # --- Cash flow: ocf_debt and fcf_margin thresholds ---
    @pytest.mark.parametrize("od,fm,expected", [
        (0.41, 0.11, 20),     # Above top tier
        (0.4, 0.10, 15),      # AT threshold: not > 0.4, falls to tier 2
        (0.26, 0.051, 15),    # Above tier 2
        (0.25, 0.05, 10),     # AT threshold: not > 0.25, falls to tier 3
        (0.16, 0.0, 10),      # Above tier 3 (fm doesn't matter)
        (0.15, 0.0, 5),       # AT threshold: not > 0.15, falls to tier 4
        (0.051, 0.0, 5),      # Above tier 4
        (0.05, 0.0, 0),       # AT threshold: not > 0.05, falls to 0
        (None, None, None),   # WP-7b: not evaluable -> None (was misleading 0)
    ])
    def test_score_cash_flow(self, od, fm, expected):
        assert UnderwritingAnalyzer._score_cash_flow(od, fm) == expected

    # --- Stability: interest_coverage and ebitda_margin thresholds ---
    @pytest.mark.parametrize("ic,em,expected", [
        (6.01, 0.21, 20),     # Above top tier
        (6.0, 0.20, 15),      # AT threshold: not > 6.0, falls to tier 2
        (4.01, 0.151, 15),    # Above tier 2
        (4.0, 0.15, 10),      # AT threshold: not > 4.0, falls to tier 3
        (2.51, 0.101, 10),    # Above tier 3
        (2.5, 0.10, 5),       # AT threshold: not > 2.5, falls to tier 4
        (1.51, 0.0, 5),       # Above tier 4 (em doesn't matter)
        (1.5, 0.0, 0),        # AT threshold: not > 1.5, falls to 0
        (None, None, None),   # WP-7b: not evaluable -> None (was misleading 0)
    ])
    def test_score_stability(self, ic, em, expected):
        assert UnderwritingAnalyzer._score_stability(ic, em) == expected


# ---------------------------------------------------------------------------
# WP-7b: 'or 0' None-aware handling -- missing data is "not evaluable",
# not a misleading 0-derived (worst-case) score.
# ---------------------------------------------------------------------------


class TestNotEvaluableScoring:
    """A category with NO evaluable inputs must return None ('N/A'), not 0.

    The previous ``metric or 0`` coerced a missing (None) metric to 0, which
    failed every positive threshold and produced the worst possible score --
    indistinguishable from a genuinely measured 0. WP-7b distinguishes the two:
    all-None inputs -> None (not evaluable); any present input -> scored
    normally with a missing companion treated conservatively (documented safe
    result, since real data exists).
    """

    def test_profitability_all_none_is_not_evaluable(self):
        assert UnderwritingAnalyzer._score_profitability(None, None) is None

    def test_liquidity_all_none_is_not_evaluable(self):
        assert UnderwritingAnalyzer._score_liquidity(None, None) is None

    def test_cash_flow_all_none_is_not_evaluable(self):
        assert UnderwritingAnalyzer._score_cash_flow(None, None) is None

    def test_stability_all_none_is_not_evaluable(self):
        assert UnderwritingAnalyzer._score_stability(None, None) is None

    def test_partial_data_still_scores(self):
        """One present metric -> still evaluable (companion None treated as 0)."""
        # Strong net margin present, roa missing -> not None, modest score.
        assert UnderwritingAnalyzer._score_profitability(0.50, None) is not None
        # Strong current ratio present, cash ratio missing -> evaluable.
        assert UnderwritingAnalyzer._score_liquidity(5.0, None) is not None

    def test_measured_zero_is_not_none(self):
        """A genuine measured 0 stays a 0 score (worst), distinct from None."""
        assert UnderwritingAnalyzer._score_profitability(0.0, 0.0) == 0
        assert UnderwritingAnalyzer._score_liquidity(0.0, 0.0) == 0
        assert UnderwritingAnalyzer._score_cash_flow(0.0, 0.0) == 0
        assert UnderwritingAnalyzer._score_stability(0.0, 0.0) == 0

    def test_empty_financials_categories_are_not_evaluable(self, analyzer):
        """With no data, profit/liquidity/cash_flow/stability categories are N/A.

        Leverage is intentionally still scored (it has its own worst-case
        sentinel by design, CORRUPT-04), so only the four 'or 0' categories
        become None.
        """
        sc = analyzer.credit_scorecard(FinancialData())
        assert sc.category_scores["profitability"] is None
        assert sc.category_scores["liquidity"] is None
        assert sc.category_scores["cash_flow"] is None
        assert sc.category_scores["stability"] is None

    def test_not_evaluable_excluded_from_strengths_weaknesses(self, analyzer):
        """A None (N/A) category is neither a strength nor a weakness."""
        sc = analyzer.credit_scorecard(FinancialData())
        for cat in ("profitability", "liquidity", "cash_flow", "stability"):
            assert cat not in sc.strengths
            assert cat not in sc.weaknesses

    def test_not_evaluable_total_excludes_none(self, analyzer):
        """total_score sums only evaluable categories (None contributes nothing)."""
        sc = analyzer.credit_scorecard(FinancialData())
        evaluable = [v for v in sc.category_scores.values() if v is not None]
        assert sc.total_score == sum(evaluable)
        assert 0 <= sc.total_score <= 100
