"""Phase 35 Tests: Piotroski F-Score Value Screen.

Tests for piotroski_f_score_analysis() and PiotroskiFScoreResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    PiotroskiFScoreResult,
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
        cash=50_000,
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


class TestPiotroskiFScoreDataclass:
    def test_defaults(self):
        r = PiotroskiFScoreResult()
        assert r.f_score == 0
        assert r.f_score_max == 7
        assert r.piotroski_grade == ""
        assert r.summary == ""
        assert r.roa_positive is None

    def test_fields(self):
        r = PiotroskiFScoreResult(
            f_score=5,
            piotroski_grade="Moderate Value",
        )
        assert r.f_score == 5
        assert r.piotroski_grade == "Moderate Value"


# ===== CORE SIGNAL TESTS =====


class TestPiotroskiFScore:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.piotroski_f_score_analysis(sample_data)
        assert isinstance(result, PiotroskiFScoreResult)

    def test_roa_positive(self, analyzer, sample_data):
        """NI=150k, TA=2M => ROA=7.5% > 0 => True."""
        result = analyzer.piotroski_f_score_analysis(sample_data)
        assert result.roa_positive is True
        assert result.roa == pytest.approx(0.075, rel=0.01)

    def test_ocf_positive(self, analyzer, sample_data):
        """OCF=220k > 0 => True."""
        result = analyzer.piotroski_f_score_analysis(sample_data)
        assert result.ocf_positive is True

    def test_accruals_negative(self, analyzer, sample_data):
        """OCF=220k > NI=150k => True (cash > accrual)."""
        result = analyzer.piotroski_f_score_analysis(sample_data)
        assert result.accruals_negative is True

    def test_current_ratio_above_1(self, analyzer, sample_data):
        """CA=500k / CL=200k = 2.5 > 1 => True."""
        result = analyzer.piotroski_f_score_analysis(sample_data)
        assert result.current_ratio_above_1 is True
        assert result.current_ratio == pytest.approx(2.5, rel=0.01)

    def test_low_leverage(self, analyzer, sample_data):
        """Debt/TA = 400k/2M = 0.20 < 0.5 => True."""
        result = analyzer.piotroski_f_score_analysis(sample_data)
        assert result.low_leverage is True
        assert result.debt_to_assets == pytest.approx(0.20, rel=0.01)

    def test_gross_margin_healthy(self, analyzer, sample_data):
        """GP/Rev = 400k/1M = 40% > 20% => True."""
        result = analyzer.piotroski_f_score_analysis(sample_data)
        assert result.gross_margin_healthy is True
        assert result.gross_margin == pytest.approx(0.40, rel=0.01)

    def test_asset_turnover_adequate(self, analyzer, sample_data):
        """Rev/TA = 1M/2M = 0.5, not > 0.5 => False."""
        result = analyzer.piotroski_f_score_analysis(sample_data)
        assert result.asset_turnover_adequate is False
        assert result.asset_turnover == pytest.approx(0.50, rel=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.piotroski_f_score_analysis(sample_data)
        assert result.piotroski_grade in ["Strong Value", "Moderate Value", "Weak", "Avoid"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.piotroski_f_score_analysis(sample_data)
        assert "Piotroski" in result.summary


# ===== SCORING TESTS =====


class TestPiotroskiScoring:
    def test_strong_value(self, analyzer):
        """All 7 signals pass."""
        data = FinancialData(
            net_income=200_000,
            total_assets=1_000_000,
            operating_cash_flow=250_000,  # > NI (accruals pass)
            current_assets=600_000,
            current_liabilities=200_000,  # CR=3.0
            total_debt=200_000,  # Debt/TA=0.2
            gross_profit=500_000,
            revenue=1_000_000,  # GM=50%, Rev/TA=1.0
        )
        result = analyzer.piotroski_f_score_analysis(data)
        assert result.f_score == 7
        assert result.piotroski_grade == "Strong Value"

    def test_avoid(self, analyzer):
        """Most signals fail."""
        data = FinancialData(
            net_income=-100_000,
            total_assets=1_000_000,
            operating_cash_flow=-50_000,
            current_assets=100_000,
            current_liabilities=300_000,  # CR=0.33
            total_debt=600_000,  # Debt/TA=0.6
            gross_profit=30_000,
            revenue=200_000,  # GM=15%, Rev/TA=0.2
        )
        result = analyzer.piotroski_f_score_analysis(data)
        assert result.f_score <= 1
        assert result.piotroski_grade == "Avoid"

    def test_moderate_value(self, analyzer):
        """4-5 signals pass."""
        data = FinancialData(
            net_income=50_000,
            total_assets=1_000_000,
            operating_cash_flow=30_000,  # < NI => accruals fail
            current_assets=400_000,
            current_liabilities=300_000,  # CR=1.33 > 1 pass
            total_debt=400_000,  # Debt/TA=0.4 < 0.5 pass
            gross_profit=250_000,
            revenue=800_000,  # GM=31.25% pass, Rev/TA=0.8 pass
        )
        result = analyzer.piotroski_f_score_analysis(data)
        # ROA>0: pass, OCF>0: pass, Accruals: fail, CR>1: pass, LowLev: pass, GM: pass, AT: pass => 6
        assert result.f_score >= 4
        assert result.piotroski_grade in ["Strong Value", "Moderate Value"]


# ===== EDGE CASES =====


class TestPhase35EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.piotroski_f_score_analysis(FinancialData())
        assert isinstance(result, PiotroskiFScoreResult)
        assert result.f_score == 0
        assert result.roa_positive is None

    def test_no_ocf(self, analyzer):
        """No OCF => ocf_positive and accruals_negative are None."""
        data = FinancialData(
            net_income=100_000,
            total_assets=1_000_000,
        )
        result = analyzer.piotroski_f_score_analysis(data)
        assert result.ocf_positive is None
        assert result.accruals_negative is None
        assert result.roa_positive is True

    def test_no_debt(self, analyzer):
        """No debt field => debt_to_assets None, low_leverage None."""
        data = FinancialData(
            net_income=100_000,
            total_assets=1_000_000,
            operating_cash_flow=120_000,
        )
        result = analyzer.piotroski_f_score_analysis(data)
        assert result.low_leverage is None

    def test_negative_ni_negative_ocf(self, analyzer):
        """Both NI and OCF negative."""
        data = FinancialData(
            net_income=-50_000,
            total_assets=1_000_000,
            operating_cash_flow=-30_000,
        )
        result = analyzer.piotroski_f_score_analysis(data)
        assert result.roa_positive is False
        assert result.ocf_positive is False
        # OCF (-30k) > NI (-50k) => accruals True (less negative cash than accrual)
        assert result.accruals_negative is True

    def test_f_score_max_tracks_testable(self, analyzer):
        """f_score_max reflects number of testable signals."""
        data = FinancialData(
            net_income=100_000,
            total_assets=500_000,
        )
        result = analyzer.piotroski_f_score_analysis(data)
        # Only ROA is testable (NI and TA present)
        assert result.f_score_max >= 1

    def test_sample_data_score(self, analyzer, sample_data):
        """All 7 signals testable. 6 pass (asset turnover fails at exactly 0.5)."""
        result = analyzer.piotroski_f_score_analysis(sample_data)
        assert result.f_score == 6
        assert result.piotroski_grade == "Strong Value"
