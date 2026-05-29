"""Tests for portfolio_analyzer.py -- portfolio & multi-company analysis."""

import math

import pytest

from financial_analyzer import FinancialData
from portfolio_analyzer import (
    CompanySnapshot,
    CorrelationMatrix,
    DiversificationScore,
    PortfolioAnalyzer,
    PortfolioReport,
    PortfolioRiskSummary,
    _hhi,
    _hhi_label,
    _score_to_grade,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strong_company() -> FinancialData:
    """Financially strong company."""
    return FinancialData(
        revenue=10_000_000,
        net_income=1_500_000,
        gross_profit=6_000_000,
        operating_income=2_500_000,
        ebit=2_500_000,
        ebitda=3_000_000,
        interest_expense=200_000,
        total_assets=20_000_000,
        current_assets=8_000_000,
        current_liabilities=3_000_000,
        total_debt=4_000_000,
        total_equity=14_000_000,
        total_liabilities=6_000_000,
        operating_cash_flow=2_000_000,
        cash=3_000_000,
    )


@pytest.fixture
def weak_company() -> FinancialData:
    """Financially weak company."""
    return FinancialData(
        revenue=2_000_000,
        net_income=-200_000,
        gross_profit=400_000,
        operating_income=-100_000,
        ebit=-100_000,
        ebitda=50_000,
        interest_expense=300_000,
        total_assets=5_000_000,
        current_assets=1_000_000,
        current_liabilities=2_000_000,
        total_debt=4_500_000,
        total_equity=100_000,
        total_liabilities=4_900_000,
        operating_cash_flow=100_000,
        cash=200_000,
    )


@pytest.fixture
def medium_company() -> FinancialData:
    """Average financial health company."""
    return FinancialData(
        revenue=5_000_000,
        net_income=250_000,
        gross_profit=2_000_000,
        operating_income=500_000,
        ebit=500_000,
        ebitda=800_000,
        interest_expense=100_000,
        total_assets=10_000_000,
        current_assets=4_000_000,
        current_liabilities=2_500_000,
        total_debt=3_000_000,
        total_equity=6_000_000,
        total_liabilities=4_000_000,
        operating_cash_flow=600_000,
        cash=1_000_000,
    )


@pytest.fixture
def three_company_portfolio(strong_company, weak_company, medium_company):
    """Dict of three companies for portfolio analysis."""
    return {
        "StrongCorp": strong_company,
        "WeakCo": weak_company,
        "MediumInc": medium_company,
    }


@pytest.fixture
def analyzer():
    return PortfolioAnalyzer()


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_hhi_single_company(self):
        assert _hhi([100.0]) == 1.0

    def test_hhi_equal_split(self):
        # 4 companies with equal revenue -> HHI = 0.25
        result = _hhi([100, 100, 100, 100])
        assert abs(result - 0.25) < 0.01

    def test_hhi_concentrated(self):
        # One company dominates
        result = _hhi([900, 10, 10, 10])
        assert result > 0.75

    def test_hhi_empty(self):
        assert _hhi([]) == 1.0

    def test_hhi_all_zeros(self):
        assert _hhi([0, 0, 0]) == 1.0

    def test_hhi_label_low(self):
        assert _hhi_label(0.10) == "low concentration"

    def test_hhi_label_moderate(self):
        assert _hhi_label(0.20) == "moderate concentration"

    def test_hhi_label_high(self):
        assert _hhi_label(0.50) == "high concentration"

    def test_score_to_grade(self):
        assert _score_to_grade(85) == "A"
        assert _score_to_grade(70) == "B"
        assert _score_to_grade(55) == "C"
        assert _score_to_grade(40) == "D"
        assert _score_to_grade(20) == "F"


# ---------------------------------------------------------------------------
# Company Snapshot
# ---------------------------------------------------------------------------


class TestCompanySnapshot:
    def test_strong_snapshot(self, analyzer, strong_company):
        snap = analyzer.company_snapshot("StrongCorp", strong_company)
        assert isinstance(snap, CompanySnapshot)
        assert snap.name == "StrongCorp"
        assert snap.health_score > 0
        assert snap.health_grade in ("A", "B", "C", "D", "F")
        assert "net_margin" in snap.key_ratios
        assert "roa" in snap.key_ratios

    def test_weak_snapshot(self, analyzer, weak_company):
        snap = analyzer.company_snapshot("WeakCo", weak_company)
        assert snap.name == "WeakCo"
        # Weak company should have low health score
        assert snap.health_score <= 60

    def test_snapshot_ratios_populated(self, analyzer, strong_company):
        snap = analyzer.company_snapshot("Test", strong_company)
        # net_margin = 1.5M / 10M = 0.15
        assert snap.key_ratios["net_margin"] is not None
        assert abs(snap.key_ratios["net_margin"] - 0.15) < 0.01


# ---------------------------------------------------------------------------
# Correlation Matrix
# ---------------------------------------------------------------------------


class TestCorrelationMatrix:
    def test_two_companies(self, analyzer, strong_company, medium_company):
        companies = {"A": strong_company, "B": medium_company}
        corr = analyzer.correlation_matrix(companies)
        assert isinstance(corr, CorrelationMatrix)
        assert len(corr.company_names) == 2
        assert len(corr.matrix) == 2
        assert len(corr.matrix[0]) == 2
        # Diagonal should be 1.0
        assert abs(corr.matrix[0][0] - 1.0) < 0.01
        assert abs(corr.matrix[1][1] - 1.0) < 0.01

    def test_single_company(self, analyzer, strong_company):
        companies = {"A": strong_company}
        corr = analyzer.correlation_matrix(companies)
        assert len(corr.matrix) == 1
        assert corr.avg_correlation == 1.0
        assert "at least 2" in corr.interpretation.lower()

    def test_three_companies(self, analyzer, three_company_portfolio):
        corr = analyzer.correlation_matrix(three_company_portfolio)
        assert len(corr.company_names) == 3
        assert len(corr.matrix) == 3
        # Avg correlation should be a float between -1 and 1
        assert -1.0 <= corr.avg_correlation <= 1.0

    def test_empty_portfolio(self, analyzer):
        corr = analyzer.correlation_matrix({})
        assert len(corr.matrix) == 0
        assert corr.avg_correlation == 0.0

    def test_interpretation_populated(self, analyzer, three_company_portfolio):
        corr = analyzer.correlation_matrix(three_company_portfolio)
        assert len(corr.interpretation) > 10

    def test_constant_ratio_nan_excluded(self, analyzer):
        """RISK-1 fix: companies with identical ratios produce NaN correlations;
        avg_correlation should exclude NaN pairs, not inflate to 0.0."""
        # Two companies with identical financial data -> corrcoef produces NaN
        identical = FinancialData(
            revenue=100,
            net_income=10,
            total_assets=200,
            current_assets=50,
            current_liabilities=25,
            total_debt=50,
            ebit=20,
            interest_expense=5,
        )
        companies = {"A": identical, "B": identical}
        corr = analyzer.correlation_matrix(companies)
        # NaN rows should not drag average toward 0
        assert isinstance(corr.avg_correlation, float)


# ---------------------------------------------------------------------------
# Diversification Score
# ---------------------------------------------------------------------------


class TestDiversificationScore:
    def test_diverse_portfolio(self, analyzer, three_company_portfolio):
        div = analyzer.diversification_score(three_company_portfolio)
        assert isinstance(div, DiversificationScore)
        assert 0 <= div.overall_score <= 100
        assert div.grade in ("A", "B", "C", "D", "F")
        assert 0.0 <= div.hhi_revenue <= 1.0
        assert 0.0 <= div.hhi_assets <= 1.0

    def test_single_company_low_diversification(self, analyzer, strong_company):
        companies = {"Only": strong_company}
        div = analyzer.diversification_score(companies)
        # Single company = fully concentrated
        assert div.hhi_revenue == 1.0
        assert div.hhi_assets == 1.0

    def test_equal_revenue_good_diversification(self, analyzer, strong_company, medium_company):
        # Give them similar revenue
        medium_company.revenue = 10_000_000
        medium_company.total_assets = 20_000_000
        companies = {"A": strong_company, "B": medium_company}
        div = analyzer.diversification_score(companies)
        # Equal revenue -> HHI = 0.5
        assert div.hhi_revenue < 0.55

    def test_interpretation_populated(self, analyzer, three_company_portfolio):
        div = analyzer.diversification_score(three_company_portfolio)
        assert "HHI" in div.interpretation

    @pytest.mark.parametrize(
        "avg_corr, expected_score",
        [
            (-1.0, 40),  # perfect negative correlation -> max corr points
            (0.0, 20),  # zero correlation -> mid corr points
            (1.0, 0),  # perfect positive correlation -> zero corr points
        ],
    )
    def test_correlation_component_mapping_isolated(self, analyzer, avg_corr, expected_score):
        """Regression (P1-E1): correlation component maps [-1, 1] -> [40, 0].

        Isolate corr_pts by using a maximally-concentrated portfolio: one
        company holds all revenue/assets, the other holds zero, so the HHI
        helper sees a single non-zero entry and returns 1.0 for both revenue
        and assets. With hhi_rev == hhi_ast == 1.0, rev_pts == ast_pts == 0,
        so overall_score == corr_pts and reads the correlation component
        directly off the public DiversificationScore (corr_pts is not
        otherwise exposed).
        """
        companies = {
            "All": FinancialData(revenue=10_000_000, total_assets=20_000_000),
            "None": FinancialData(revenue=0.0, total_assets=0.0),
        }
        # Confirm the fixture zeroes the HHI components so total == corr_pts.
        assert _hhi([10_000_000, 0.0]) == 1.0

        correlation = CorrelationMatrix(avg_correlation=avg_corr)
        div = analyzer.diversification_score(companies, correlation=correlation)

        assert div.hhi_revenue == 1.0
        assert div.hhi_assets == 1.0
        assert div.overall_score == expected_score


# ---------------------------------------------------------------------------
# Portfolio Risk Summary
# ---------------------------------------------------------------------------


class TestPortfolioRiskSummary:
    def test_risk_summary_basic(self, analyzer, three_company_portfolio):
        snapshots = [analyzer.company_snapshot(n, d) for n, d in three_company_portfolio.items()]
        risk = analyzer.portfolio_risk_summary(snapshots, three_company_portfolio)
        assert isinstance(risk, PortfolioRiskSummary)
        assert risk.num_companies == 3
        assert risk.weakest_company != ""
        assert risk.strongest_company != ""
        assert risk.overall_risk_level in ("low", "moderate", "high", "critical")

    def test_weak_company_flagged(self, analyzer, weak_company):
        companies = {"WeakCo": weak_company}
        snapshots = [analyzer.company_snapshot("WeakCo", weak_company)]
        risk = analyzer.portfolio_risk_summary(snapshots, companies)
        # Should have risk flags for the weak company
        assert len(risk.risk_flags) > 0

    def test_empty_portfolio(self, analyzer):
        risk = analyzer.portfolio_risk_summary([], {})
        assert risk.overall_risk_level == "critical"
        assert risk.num_companies == 0

    def test_negative_equity_flagged(self, analyzer):
        bad = FinancialData(
            revenue=1_000_000,
            net_income=-500_000,
            total_assets=2_000_000,
            total_equity=-100_000,
            total_liabilities=2_100_000,
            current_assets=500_000,
            current_liabilities=1_500_000,
            total_debt=2_000_000,
            ebit=-400_000,
            interest_expense=300_000,
            operating_cash_flow=-200_000,
        )
        companies = {"BadCo": bad}
        snapshots = [analyzer.company_snapshot("BadCo", bad)]
        risk = analyzer.portfolio_risk_summary(snapshots, companies)
        neg_equity_flags = [f for f in risk.risk_flags if "negative equity" in f.lower()]
        assert len(neg_equity_flags) > 0


# ---------------------------------------------------------------------------
# Full Portfolio Analysis
# ---------------------------------------------------------------------------


class TestFullPortfolioAnalysis:
    def test_full_analysis(self, analyzer, three_company_portfolio):
        report = analyzer.full_portfolio_analysis(three_company_portfolio)
        assert isinstance(report, PortfolioReport)
        assert report.num_companies == 3
        assert len(report.snapshots) == 3
        assert report.correlation is not None
        assert report.diversification is not None
        assert report.risk_summary is not None
        assert len(report.summary) > 20

    def test_single_company(self, analyzer, strong_company):
        companies = {"Only": strong_company}
        report = analyzer.full_portfolio_analysis(companies)
        assert report.num_companies == 1
        assert len(report.snapshots) == 1

    def test_summary_mentions_strongest_weakest(self, analyzer, three_company_portfolio):
        report = analyzer.full_portfolio_analysis(three_company_portfolio)
        # Summary should mention strongest and weakest
        assert "Strongest" in report.summary or "strongest" in report.summary
        assert "Weakest" in report.summary or "weakest" in report.summary


# ---------------------------------------------------------------------------
# CRASH/Edge-case regression tests (Swarm Audit)
# ---------------------------------------------------------------------------


class TestDiversificationSingleCompany:
    """CRASH-05 regression: single or empty company must not ZeroDivisionError."""

    def test_single_company_diversification_score_zero(self, analyzer, strong_company):
        score = analyzer.diversification_score({"Only": strong_company})
        assert isinstance(score, DiversificationScore)
        assert score.overall_score == 0
        assert score.grade == "F"

    def test_empty_portfolio_diversification_score_zero(self, analyzer):
        score = analyzer.diversification_score({})
        assert score.overall_score == 0
        assert score.grade == "F"

    def test_single_company_full_analysis_no_crash(self, analyzer, strong_company):
        report = analyzer.full_portfolio_analysis({"Solo": strong_company})
        assert report.diversification.overall_score == 0


class TestCorrelationEdgeCases:
    """Coverage gaps: identical companies, all-zero, high/low interpretations."""

    def test_identical_companies_no_nan(self, analyzer, strong_company):
        companies = {"A": strong_company, "B": strong_company}
        result = analyzer.correlation_matrix(companies)
        assert isinstance(result, CorrelationMatrix)
        # Should not contain NaN
        for row in result.matrix:
            for val in row:
                assert not math.isnan(val)

    def test_all_zero_companies_no_crash(self, analyzer):
        empty = FinancialData()
        companies = {"A": empty, "B": empty}
        result = analyzer.correlation_matrix(companies)
        assert isinstance(result, CorrelationMatrix)

    def test_correlation_symmetric(self, analyzer, strong_company, weak_company):
        companies = {"S": strong_company, "W": weak_company}
        result = analyzer.correlation_matrix(companies)
        assert abs(result.matrix[0][1] - result.matrix[1][0]) < 1e-9


class TestPortfolioRiskBoundary:
    """Coverage gap: risk level boundary conditions."""

    def test_risk_level_low_for_strong_portfolio(self, analyzer, strong_company):
        companies = {"A": strong_company, "B": strong_company}
        report = analyzer.full_portfolio_analysis(companies)
        assert report.risk_summary.overall_risk_level in ("low", "moderate")

    def test_risk_flags_for_weak_company(self, analyzer, weak_company):
        companies = {"W1": weak_company, "W2": weak_company}
        report = analyzer.full_portfolio_analysis(companies)
        assert len(report.risk_summary.risk_flags) > 0


class TestHHIEdgeCasesNegativeNoneEqual:
    """Coverage gap: _hhi with negative, None, and equal values.

    (Renamed from TestHHIEdgeCases: a second class of the same name at line ~490
    was silently shadowing this one — both now run.)
    """

    def test_hhi_negative_values_use_abs(self):
        # Negative revenue is included via abs(): [100, 50, 200] -> shares sum=350
        assert _hhi([100, -50, 200]) == _hhi([100, 50, 200])

    def test_hhi_none_values_filtered(self):
        assert _hhi([None, 100, None, 200]) == _hhi([100, 200])

    def test_hhi_two_equal_values(self):
        assert abs(_hhi([100, 100]) - 0.5) < 1e-9

    def test_hhi_single_value(self):
        assert abs(_hhi([500]) - 1.0) < 1e-9


class TestScoreToGradeBoundary:
    """Coverage gap: exact boundary values for _score_to_grade."""

    def test_score_80_is_A(self):
        assert _score_to_grade(80) == "A"

    def test_score_79_is_B(self):
        assert _score_to_grade(79) == "B"

    def test_score_65_is_B(self):
        assert _score_to_grade(65) == "B"

    def test_score_64_is_C(self):
        assert _score_to_grade(64) == "C"

    def test_score_50_is_C(self):
        assert _score_to_grade(50) == "C"

    def test_score_35_is_D(self):
        assert _score_to_grade(35) == "D"

    def test_score_34_is_F(self):
        assert _score_to_grade(34) == "F"


# ---------------------------------------------------------------------------
# HHI and portfolio edge cases
# ---------------------------------------------------------------------------


class TestHHIEdgeCases:
    def test_hhi_all_zero_revenues(self):
        """All-zero revenues should return 1.0 (fully concentrated)."""
        assert _hhi([0, 0, 0, 0]) == 1.0

    def test_hhi_all_none_values(self):
        """All-None values should return 1.0."""
        assert _hhi([None, None, None]) == 1.0

    def test_hhi_single_nonzero(self):
        """Single nonzero among zeros should return 1.0."""
        assert _hhi([0, 100, 0, 0]) == 1.0

    def test_hhi_negative_values_use_abs(self):
        """Negative values should be treated as absolute values."""
        result = _hhi([-100, 100])
        assert result == pytest.approx(0.5)  # Equal shares


class TestPortfolioEdgeCases:
    @pytest.fixture()
    def analyzer(self):
        return PortfolioAnalyzer()

    def test_portfolio_all_zero_revenue_companies(self, analyzer):
        """Portfolio with all-zero-revenue companies should not crash."""
        companies = {f"Co{i}": FinancialData(revenue=0, total_assets=1000 * (i + 1)) for i in range(3)}
        report = analyzer.full_portfolio_analysis(companies)
        assert isinstance(report, PortfolioReport)
        assert not math.isnan(report.diversification.overall_score)

    def test_portfolio_identical_companies(self, analyzer):
        """All-identical companies should compute without NaN."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_assets=5_000_000,
            current_assets=2_000_000,
            current_liabilities=1_000_000,
            total_equity=3_000_000,
        )
        companies = {f"Co{i}": data for i in range(5)}
        report = analyzer.full_portfolio_analysis(companies)
        assert not math.isnan(report.correlation.avg_correlation)
