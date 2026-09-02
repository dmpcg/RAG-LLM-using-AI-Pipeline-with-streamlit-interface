"""Phase 133 Tests: Defensive Posture Analysis.

Tests for defensive_posture_analysis() and DefensivePostureResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    DefensivePostureResult,
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


class TestDefensivePostureDataclass:
    def test_defaults(self):
        r = DefensivePostureResult()
        assert r.defensive_interval is None
        assert r.cash_ratio is None
        assert r.quick_ratio is None
        assert r.cash_flow_coverage is None
        assert r.equity_buffer is None
        assert r.debt_shield is None
        assert r.dp_score == 0.0
        assert r.dp_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = DefensivePostureResult(cash_ratio=0.25, dp_grade="Good")
        assert r.cash_ratio == 0.25
        assert r.dp_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestDefensivePostureAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.defensive_posture_analysis(sample_data)
        assert isinstance(result, DefensivePostureResult)

    def test_defensive_interval(self, analyzer, sample_data):
        """DI = 500k / (200k/365) = 500k / 547.95 = 912.5 days."""
        result = analyzer.defensive_posture_analysis(sample_data)
        assert result.defensive_interval == pytest.approx(912.5, abs=1.0)

    def test_cash_ratio(self, analyzer, sample_data):
        """CR = 50k / 200k = 0.25."""
        result = analyzer.defensive_posture_analysis(sample_data)
        assert result.cash_ratio == pytest.approx(0.25, abs=0.01)

    def test_quick_ratio(self, analyzer, sample_data):
        """QR = (500k - 100k) / 200k = 400k / 200k = 2.0."""
        result = analyzer.defensive_posture_analysis(sample_data)
        assert result.quick_ratio == pytest.approx(2.0, abs=0.01)

    def test_cash_flow_coverage(self, analyzer, sample_data):
        """CFC = 220k / 2M = 0.11."""
        result = analyzer.defensive_posture_analysis(sample_data)
        assert result.cash_flow_coverage == pytest.approx(0.11, abs=0.01)

    def test_equity_buffer(self, analyzer, sample_data):
        """EB = 1.2M / 2M = 0.60."""
        result = analyzer.defensive_posture_analysis(sample_data)
        assert result.equity_buffer == pytest.approx(0.60, abs=0.01)

    def test_debt_shield(self, analyzer, sample_data):
        """DS = 250k / 400k = 0.625."""
        result = analyzer.defensive_posture_analysis(sample_data)
        assert result.debt_shield == pytest.approx(0.625, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.defensive_posture_analysis(sample_data)
        assert result.dp_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.defensive_posture_analysis(sample_data)
        assert "Defensive Posture" in result.summary


# ===== SCORING TESTS =====


class TestDefensivePostureScoring:
    def test_strong_posture(self, analyzer, sample_data):
        """DI=912.5 => base 10. CR=0.25 (not >=0.50, not <0.10) => 0. DS=0.625 (not >=2.0, not <0.5) => 0. Score=10."""
        result = analyzer.defensive_posture_analysis(sample_data)
        assert result.dp_score == pytest.approx(10.0, abs=0.5)
        assert result.dp_grade == "Excellent"

    def test_excellent_posture(self, analyzer):
        """DI >= 365 with strong cash."""
        data = FinancialData(
            current_assets=1_000_000,
            operating_expenses=200_000,
            cash=500_000,
            current_liabilities=200_000,
            inventory=50_000,
            total_assets=2_000_000,
            total_equity=1_500_000,
            ebitda=500_000,
            total_debt=200_000,
            operating_cash_flow=400_000,
        )
        # DI=1M/(200k/365)=1825. CR=500k/200k=2.5 >=0.50 => +0.5. DS=500k/200k=2.5 >=2.0 => +0.5. Capped 10.
        result = analyzer.defensive_posture_analysis(data)
        assert result.dp_score >= 10.0
        assert result.dp_grade == "Excellent"

    def test_weak_posture(self, analyzer):
        """DI < 30 => base 1.0."""
        data = FinancialData(
            current_assets=10_000,
            operating_expenses=500_000,
            cash=5_000,
            current_liabilities=300_000,
            inventory=2_000,
            total_assets=500_000,
            total_equity=100_000,
            ebitda=30_000,
            total_debt=300_000,
            operating_cash_flow=10_000,
        )
        # DI=10k/(500k/365)=7.3 => 1.0. CR=5k/300k=0.017 <0.10 => -0.5. DS=30k/300k=0.10 <0.5 => -0.5. Score=0.0.
        result = analyzer.defensive_posture_analysis(data)
        assert result.dp_score <= 1.0
        assert result.dp_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase133EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.defensive_posture_analysis(FinancialData())
        assert isinstance(result, DefensivePostureResult)
        assert result.dp_score == 0.0

    def test_no_opex(self, analyzer):
        """OpEx=None => DI=None => score 0."""
        data = FinancialData(
            current_assets=500_000,
            cash=50_000,
            current_liabilities=200_000,
            total_assets=1_000_000,
            total_equity=600_000,
        )
        result = analyzer.defensive_posture_analysis(data)
        assert result.defensive_interval is None
        assert result.dp_score == 0.0

    def test_no_inventory(self, analyzer):
        """Inv=None => uses 0 for quick ratio."""
        data = FinancialData(
            current_assets=500_000,
            operating_expenses=200_000,
            cash=50_000,
            current_liabilities=200_000,
            total_assets=1_000_000,
            total_equity=600_000,
            ebitda=200_000,
            total_debt=300_000,
        )
        result = analyzer.defensive_posture_analysis(data)
        # QR = (500k - 0) / 200k = 2.5
        assert result.quick_ratio == pytest.approx(2.5, abs=0.01)

    def test_no_cash(self, analyzer):
        """Cash=None => cash_ratio=None."""
        data = FinancialData(
            current_assets=500_000,
            operating_expenses=200_000,
            current_liabilities=200_000,
            inventory=100_000,
            total_assets=1_000_000,
            total_equity=600_000,
            ebitda=200_000,
            total_debt=300_000,
        )
        result = analyzer.defensive_posture_analysis(data)
        assert result.cash_ratio is None
        assert result.defensive_interval is not None
