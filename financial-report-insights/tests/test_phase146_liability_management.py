"""Phase 146 Tests: Liability Management Analysis.

Tests for liability_management_analysis() and LiabilityManagementResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    LiabilityManagementResult,
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
        dividends_paid=40_000,
    )


# ===== DATACLASS TESTS =====


class TestLiabilityManagementDataclass:
    def test_defaults(self):
        r = LiabilityManagementResult()
        assert r.liability_to_assets is None
        assert r.liability_to_equity is None
        assert r.current_liability_ratio is None
        assert r.liability_coverage is None
        assert r.liability_to_revenue is None
        assert r.net_liability is None
        assert r.lm_score == 0.0
        assert r.lm_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = LiabilityManagementResult(liability_to_assets=0.40, lm_grade="Good")
        assert r.liability_to_assets == 0.40
        assert r.lm_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestLiabilityManagementAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.liability_management_analysis(sample_data)
        assert isinstance(result, LiabilityManagementResult)

    def test_liability_to_assets(self, analyzer, sample_data):
        """TL/TA = 800k / 2M = 0.40."""
        result = analyzer.liability_management_analysis(sample_data)
        assert result.liability_to_assets == pytest.approx(0.40, abs=0.01)

    def test_liability_to_equity(self, analyzer, sample_data):
        """TL/TE = 800k / 1.2M = 0.667."""
        result = analyzer.liability_management_analysis(sample_data)
        assert result.liability_to_equity == pytest.approx(0.667, abs=0.01)

    def test_current_liability_ratio(self, analyzer, sample_data):
        """CL/TL = 200k / 800k = 0.25."""
        result = analyzer.liability_management_analysis(sample_data)
        assert result.current_liability_ratio == pytest.approx(0.25, abs=0.01)

    def test_liability_coverage(self, analyzer, sample_data):
        """EBITDA/TL = 250k / 800k = 0.3125."""
        result = analyzer.liability_management_analysis(sample_data)
        assert result.liability_coverage == pytest.approx(0.3125, abs=0.01)

    def test_liability_to_revenue(self, analyzer, sample_data):
        """TL/Rev = 800k / 1M = 0.80."""
        result = analyzer.liability_management_analysis(sample_data)
        assert result.liability_to_revenue == pytest.approx(0.80, abs=0.01)

    def test_net_liability(self, analyzer, sample_data):
        """(TL - Cash) / TA = (800k - 50k) / 2M = 0.375."""
        result = analyzer.liability_management_analysis(sample_data)
        assert result.net_liability == pytest.approx(0.375, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.liability_management_analysis(sample_data)
        assert result.lm_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.liability_management_analysis(sample_data)
        assert "Liability Management" in result.summary


# ===== SCORING TESTS =====


class TestLiabilityManagementScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """L/A=0.40 => base 7.0. LC=0.3125 >=0.30 => +0.5. CLR=0.25 <=0.40 => +0.5. Score=8.0."""
        result = analyzer.liability_management_analysis(sample_data)
        assert result.lm_score == pytest.approx(8.0, abs=0.5)
        assert result.lm_grade == "Excellent"

    def test_excellent_management(self, analyzer):
        """L/A <= 0.30 => base 10."""
        data = FinancialData(
            total_liabilities=400_000,
            total_assets=3_000_000,
            total_equity=2_600_000,
            current_liabilities=100_000,
            ebitda=500_000,
            revenue=2_000_000,
            cash=200_000,
        )
        # L/A=0.133 => 10. LC=1.25 >=0.30 => +0.5. CLR=0.25 <=0.40 => +0.5. Capped 10.
        result = analyzer.liability_management_analysis(data)
        assert result.lm_score >= 10.0
        assert result.lm_grade == "Excellent"

    def test_weak_management(self, analyzer):
        """L/A > 0.70 => base 1.0."""
        data = FinancialData(
            total_liabilities=2_500_000,
            total_assets=3_000_000,
            total_equity=500_000,
            current_liabilities=2_000_000,
            ebitda=200_000,
            revenue=2_000_000,
            cash=50_000,
        )
        # L/A=0.833 => 1.0. LC=0.08 <0.10 => -0.5. CLR=0.80 >0.70 => -0.5. Score=0.0.
        result = analyzer.liability_management_analysis(data)
        assert result.lm_score <= 1.0
        assert result.lm_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase146EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.liability_management_analysis(FinancialData())
        assert isinstance(result, LiabilityManagementResult)
        assert result.lm_score == 0.0

    def test_no_liabilities(self, analyzer):
        """TL=None => L/A=None, score 0."""
        data = FinancialData(
            total_assets=2_000_000,
            total_equity=1_200_000,
            ebitda=250_000,
            revenue=1_000_000,
        )
        result = analyzer.liability_management_analysis(data)
        assert result.liability_to_assets is None
        assert result.lm_score == 0.0

    def test_no_assets(self, analyzer):
        """TA=None => L/A=None => score 0."""
        data = FinancialData(
            total_liabilities=800_000,
            total_equity=1_200_000,
            ebitda=250_000,
        )
        result = analyzer.liability_management_analysis(data)
        assert result.liability_to_assets is None

    def test_no_cash(self, analyzer):
        """Cash=None => net_liability uses 0 for cash."""
        data = FinancialData(
            total_liabilities=800_000,
            total_assets=2_000_000,
            total_equity=1_200_000,
            current_liabilities=200_000,
            ebitda=250_000,
            revenue=1_000_000,
        )
        result = analyzer.liability_management_analysis(data)
        assert result.net_liability == pytest.approx(0.40, abs=0.01)  # (800k-0)/2M
