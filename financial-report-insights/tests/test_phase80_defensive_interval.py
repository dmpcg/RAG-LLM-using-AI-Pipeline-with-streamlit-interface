"""Phase 80 Tests: Defensive Interval Analysis.

Tests for defensive_interval_analysis() and DefensiveIntervalResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    DefensiveIntervalResult,
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


class TestDefensiveIntervalDataclass:
    def test_defaults(self):
        r = DefensiveIntervalResult()
        assert r.defensive_interval_days is None
        assert r.cash_interval_days is None
        assert r.liquid_assets_ratio is None
        assert r.days_cash_on_hand is None
        assert r.liquid_reserve_adequacy is None
        assert r.operating_expense_coverage is None
        assert r.di_score == 0.0
        assert r.di_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = DefensiveIntervalResult(defensive_interval_days=90.0, di_grade="Good")
        assert r.defensive_interval_days == 90.0
        assert r.di_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestDefensiveIntervalAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.defensive_interval_analysis(sample_data)
        assert isinstance(result, DefensiveIntervalResult)

    def test_defensive_interval_days(self, analyzer, sample_data):
        """Liquid=(50k+150k)=200k, DailyOpEx=(600k+200k)/365=2191.78, DID=200k/2191.78=91.2."""
        result = analyzer.defensive_interval_analysis(sample_data)
        assert result.defensive_interval_days == pytest.approx(91.2, abs=1.0)

    def test_cash_interval_days(self, analyzer, sample_data):
        """Cash=50k, DailyOpEx=2191.78, CID=50k/2191.78=22.8."""
        result = analyzer.defensive_interval_analysis(sample_data)
        assert result.cash_interval_days == pytest.approx(22.8, abs=1.0)

    def test_liquid_assets_ratio(self, analyzer, sample_data):
        """Liquid/TA = 200k/2M = 0.10."""
        result = analyzer.defensive_interval_analysis(sample_data)
        assert result.liquid_assets_ratio == pytest.approx(0.10, abs=0.01)

    def test_days_cash_on_hand(self, analyzer, sample_data):
        """Same as cash_interval_days: 22.8."""
        result = analyzer.defensive_interval_analysis(sample_data)
        assert result.days_cash_on_hand == pytest.approx(22.8, abs=1.0)

    def test_liquid_reserve_adequacy(self, analyzer, sample_data):
        """Liquid/CL = 200k/200k = 1.0."""
        result = analyzer.defensive_interval_analysis(sample_data)
        assert result.liquid_reserve_adequacy == pytest.approx(1.0, abs=0.01)

    def test_operating_expense_coverage(self, analyzer, sample_data):
        """Liquid/AnnualOpEx = 200k/800k = 0.25."""
        result = analyzer.defensive_interval_analysis(sample_data)
        assert result.operating_expense_coverage == pytest.approx(0.25, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.defensive_interval_analysis(sample_data)
        assert result.di_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.defensive_interval_analysis(sample_data)
        assert "Defensive Interval" in result.summary


# ===== SCORING TESTS =====


class TestDefensiveIntervalScoring:
    def test_very_high_interval(self, analyzer):
        """DID >= 180 => base 10."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=200_000,
            operating_expenses=100_000,
            total_assets=5_000_000,
            current_liabilities=100_000,
            cash=500_000,
            accounts_receivable=300_000,
        )
        result = analyzer.defensive_interval_analysis(data)
        # Liquid=800k, DailyOpEx=300k/365=821.9, DID=800k/821.9=973 days
        # base 10 + LRA=800k/100k=8.0 (+0.5) + LAR=800k/5M=0.16 (no adj) => 10 capped
        assert result.di_score >= 10.0
        assert result.di_grade == "Excellent"

    def test_moderate_interval(self, analyzer, sample_data):
        """DID ~ 91 => base 7.0."""
        result = analyzer.defensive_interval_analysis(sample_data)
        # DID=91.2 (base 7.0) + LRA=1.0 (no adj) + LAR=0.10 (no adj) => 7.0
        assert result.di_score >= 7.0

    def test_low_interval(self, analyzer):
        """DID ~ 20 => base 2.5."""
        data = FinancialData(
            revenue=2_000_000,
            cogs=1_200_000,
            operating_expenses=500_000,
            total_assets=3_000_000,
            current_liabilities=800_000,
            cash=30_000,
            accounts_receivable=50_000,
        )
        result = analyzer.defensive_interval_analysis(data)
        # Liquid=80k, DailyOpEx=1.7M/365=4658, DID=80k/4658=17.2 days
        # base 2.5 + LRA=80k/800k=0.1 (<0.5 => -0.5) + LAR=0.027 (<0.05 => -0.5) => 1.5
        assert result.di_score < 3.0
        assert result.di_grade == "Weak"

    def test_lra_bonus(self, analyzer):
        """LRA >= 2.0 => +0.5."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=400_000,
            operating_expenses=200_000,
            total_assets=2_000_000,
            current_liabilities=100_000,
            cash=200_000,
            accounts_receivable=100_000,
        )
        result = analyzer.defensive_interval_analysis(data)
        # Liquid=300k, DailyOpEx=600k/365=1643.8, DID=182.5 days
        # base 10 + LRA=300k/100k=3.0 (+0.5) + LAR=300k/2M=0.15 (no adj) => 10 capped
        assert result.di_score >= 10.0

    def test_lar_bonus(self, analyzer):
        """LAR >= 0.30 => +0.5."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=300_000,
            operating_expenses=100_000,
            total_assets=500_000,
            current_liabilities=100_000,
            cash=100_000,
            accounts_receivable=100_000,
        )
        result = analyzer.defensive_interval_analysis(data)
        # Liquid=200k, LAR=200k/500k=0.40 => +0.5
        assert result.di_score >= 7.0


# ===== EDGE CASES =====


class TestPhase80EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.defensive_interval_analysis(FinancialData())
        assert isinstance(result, DefensiveIntervalResult)
        assert result.defensive_interval_days is None

    def test_no_opex(self, analyzer):
        """No COGS or OpEx => empty result."""
        data = FinancialData(
            revenue=1_000_000,
            cash=100_000,
            accounts_receivable=50_000,
        )
        result = analyzer.defensive_interval_analysis(data)
        assert result.di_score == 0.0

    def test_no_cash_no_ar(self, analyzer):
        """Liquid=0 => DID=0."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=500_000,
            operating_expenses=200_000,
            total_assets=1_000_000,
            current_liabilities=200_000,
        )
        result = analyzer.defensive_interval_analysis(data)
        assert result.defensive_interval_days == pytest.approx(0.0, abs=0.1)

    def test_no_current_liabilities(self, analyzer):
        """CL=0 => LRA is None."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=400_000,
            operating_expenses=200_000,
            total_assets=1_000_000,
            cash=100_000,
            accounts_receivable=50_000,
        )
        result = analyzer.defensive_interval_analysis(data)
        assert result.liquid_reserve_adequacy is None
