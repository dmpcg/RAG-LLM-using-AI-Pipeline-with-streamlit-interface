"""Phase 54 Tests: Return on Equity (ROE) Analysis.

Tests for roe_analysis() and ROEAnalysisResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ROEAnalysisResult,
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


class TestROEAnalysisDataclass:
    def test_defaults(self):
        r = ROEAnalysisResult()
        assert r.roe_pct is None
        assert r.net_margin_pct is None
        assert r.asset_turnover is None
        assert r.equity_multiplier is None
        assert r.roa_pct is None
        assert r.retention_ratio is None
        assert r.sustainable_growth_rate is None
        assert r.roe_score == 0.0
        assert r.roe_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ROEAnalysisResult(
            roe_pct=12.5,
            roe_grade="Good",
        )
        assert r.roe_pct == 12.5
        assert r.roe_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestROEAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.roe_analysis(sample_data)
        assert isinstance(result, ROEAnalysisResult)

    def test_roe(self, analyzer, sample_data):
        """NI/Equity = 150k/1.2M = 12.5%."""
        result = analyzer.roe_analysis(sample_data)
        assert result.roe_pct == pytest.approx(12.5, abs=0.1)

    def test_net_margin(self, analyzer, sample_data):
        """NI/Revenue = 150k/1M = 15%."""
        result = analyzer.roe_analysis(sample_data)
        assert result.net_margin_pct == pytest.approx(15.0, abs=0.1)

    def test_asset_turnover(self, analyzer, sample_data):
        """Revenue/TA = 1M/2M = 0.50."""
        result = analyzer.roe_analysis(sample_data)
        assert result.asset_turnover == pytest.approx(0.50, abs=0.01)

    def test_equity_multiplier(self, analyzer, sample_data):
        """TA/TE = 2M/1.2M = 1.667."""
        result = analyzer.roe_analysis(sample_data)
        assert result.equity_multiplier == pytest.approx(1.667, abs=0.01)

    def test_roa(self, analyzer, sample_data):
        """NI/TA = 150k/2M = 7.5%."""
        result = analyzer.roe_analysis(sample_data)
        assert result.roa_pct == pytest.approx(7.5, abs=0.1)

    def test_dupont_identity(self, analyzer, sample_data):
        """ROE = Net Margin * Asset Turnover * Equity Multiplier."""
        result = analyzer.roe_analysis(sample_data)
        dupont_roe = result.net_margin_pct * result.asset_turnover * result.equity_multiplier
        assert dupont_roe == pytest.approx(result.roe_pct, abs=0.5)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.roe_analysis(sample_data)
        assert result.roe_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.roe_analysis(sample_data)
        assert "Return on Equity" in result.summary


# ===== SCORING TESTS =====


class TestROEScoring:
    def test_high_roe(self, analyzer):
        """ROE 30% => base 10."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=360_000,
            total_assets=2_000_000,
            total_equity=1_200_000,
        )
        result = analyzer.roe_analysis(data)
        assert result.roe_score >= 8.0
        assert result.roe_grade == "Excellent"

    def test_moderate_roe(self, analyzer, sample_data):
        """ROE 12.5% => base 7.0."""
        result = analyzer.roe_analysis(sample_data)
        assert result.roe_score >= 6.0

    def test_low_roe(self, analyzer):
        """ROE 5% => base 3.5."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=60_000,
            total_assets=2_000_000,
            total_equity=1_200_000,
        )
        result = analyzer.roe_analysis(data)
        assert result.roe_score < 6.0

    def test_negative_roe(self, analyzer):
        """ROE -20% => base 0.5, Weak."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=-240_000,
            total_assets=2_000_000,
            total_equity=1_200_000,
        )
        result = analyzer.roe_analysis(data)
        assert result.roe_score < 4.0
        assert result.roe_grade == "Weak"

    def test_low_equity_multiplier_bonus(self, analyzer):
        """Equity multiplier < 2.0 => +0.5 bonus."""
        data = FinancialData(
            revenue=500_000,
            net_income=100_000,
            total_assets=600_000,
            total_equity=500_000,
        )
        result = analyzer.roe_analysis(data)
        # ROE = 100k/500k = 20% => base 8.5
        # EM = 600k/500k = 1.2 < 2.0 => +0.5
        assert result.roe_score >= 8.5


# ===== EDGE CASES =====


class TestPhase54EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.roe_analysis(FinancialData())
        assert isinstance(result, ROEAnalysisResult)
        assert result.roe_pct is None

    def test_no_equity(self, analyzer):
        data = FinancialData(revenue=1_000_000, net_income=100_000, total_assets=2_000_000)
        result = analyzer.roe_analysis(data)
        assert result.roe_pct is None

    def test_zero_equity(self, analyzer):
        data = FinancialData(revenue=1_000_000, net_income=100_000, total_assets=2_000_000, total_equity=0)
        result = analyzer.roe_analysis(data)
        assert result.roe_pct is None

    def test_no_net_income(self, analyzer):
        """Revenue & equity but no NI => roe_pct is None."""
        data = FinancialData(revenue=1_000_000, total_assets=2_000_000, total_equity=1_000_000)
        result = analyzer.roe_analysis(data)
        assert result.roe_pct is None
