"""Phase 34 Tests: Altman Z-Score Bankruptcy Prediction.

Tests for altman_z_score_analysis() and AltmanZScoreResult dataclass.
"""

import pytest

from financial_analyzer import (
    AltmanZScoreResult,
    CharlieAnalyzer,
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
    )


# ===== DATACLASS TESTS =====


class TestAltmanZScoreDataclass:
    def test_defaults(self):
        r = AltmanZScoreResult()
        assert r.z_score is None
        assert r.z_zone == ""
        assert r.altman_score == 0.0
        assert r.altman_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = AltmanZScoreResult(
            z_score=3.5,
            z_zone="Safe",
            altman_grade="Strong",
        )
        assert r.z_score == 3.5
        assert r.z_zone == "Safe"
        assert r.altman_grade == "Strong"


# ===== CORE COMPUTATION TESTS =====


class TestAltmanZScore:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.altman_z_score_analysis(sample_data)
        assert isinstance(result, AltmanZScoreResult)

    def test_wc_to_assets(self, analyzer, sample_data):
        """WC/TA = (500k - 200k) / 2M = 0.15."""
        result = analyzer.altman_z_score_analysis(sample_data)
        assert result.working_capital_to_assets == pytest.approx(0.15, rel=0.01)

    def test_re_to_assets(self, analyzer, sample_data):
        """RE/TA = 600k / 2M = 0.30."""
        result = analyzer.altman_z_score_analysis(sample_data)
        assert result.retained_earnings_to_assets == pytest.approx(0.30, rel=0.01)

    def test_ebit_to_assets(self, analyzer, sample_data):
        """EBIT/TA = 200k / 2M = 0.10."""
        result = analyzer.altman_z_score_analysis(sample_data)
        assert result.ebit_to_assets == pytest.approx(0.10, rel=0.01)

    def test_equity_to_liabilities(self, analyzer, sample_data):
        """Equity/TL = 1.2M / 800k = 1.50."""
        result = analyzer.altman_z_score_analysis(sample_data)
        assert result.equity_to_liabilities == pytest.approx(1.50, rel=0.01)

    def test_revenue_to_assets(self, analyzer, sample_data):
        """Rev/TA = 1M / 2M = 0.50."""
        result = analyzer.altman_z_score_analysis(sample_data)
        assert result.revenue_to_assets == pytest.approx(0.50, rel=0.01)

    def test_weighted_components(self, analyzer, sample_data):
        """X1=1.2*0.15=0.18, X2=1.4*0.30=0.42, X3=3.3*0.10=0.33, X4=0.6*1.50=0.90, X5=1.0*0.50=0.50."""
        result = analyzer.altman_z_score_analysis(sample_data)
        assert result.x1_weighted == pytest.approx(0.18, rel=0.01)
        assert result.x2_weighted == pytest.approx(0.42, rel=0.01)
        assert result.x3_weighted == pytest.approx(0.33, rel=0.01)
        assert result.x4_weighted == pytest.approx(0.90, rel=0.01)
        assert result.x5_weighted == pytest.approx(0.50, rel=0.01)

    def test_z_score_calculation(self, analyzer, sample_data):
        """Z = 0.18 + 0.42 + 0.33 + 0.90 + 0.50 = 2.33."""
        result = analyzer.altman_z_score_analysis(sample_data)
        assert result.z_score == pytest.approx(2.33, rel=0.01)

    def test_z_zone_gray(self, analyzer, sample_data):
        """Z=2.33 is in Gray zone (1.81-2.99)."""
        result = analyzer.altman_z_score_analysis(sample_data)
        assert result.z_zone == "Gray"

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.altman_z_score_analysis(sample_data)
        assert result.altman_grade in ["Strong", "Adequate", "Watch", "Critical"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.altman_z_score_analysis(sample_data)
        assert "Altman Z-Score" in result.summary


# ===== SCORING TESTS =====


class TestAltmanZScoreScoring:
    def test_strong_safe_zone(self, analyzer):
        """High Z-Score => Safe zone, Strong grade."""
        data = FinancialData(
            current_assets=800_000,
            current_liabilities=200_000,
            total_assets=1_000_000,
            retained_earnings=500_000,
            ebit=300_000,
            total_equity=700_000,
            total_liabilities=300_000,
            revenue=1_500_000,
        )
        result = analyzer.altman_z_score_analysis(data)
        # WC/TA=0.6, RE/TA=0.5, EBIT/TA=0.3, Eq/TL=2.33, Rev/TA=1.5
        # Z = 1.2*0.6 + 1.4*0.5 + 3.3*0.3 + 0.6*2.33 + 1.0*1.5 = 0.72+0.7+0.99+1.4+1.5=5.31
        assert result.z_zone == "Safe"
        assert result.z_score > 3.0
        assert result.altman_grade == "Strong"

    def test_distress_zone(self, analyzer):
        """Low Z-Score => Distress zone."""
        data = FinancialData(
            current_assets=100_000,
            current_liabilities=400_000,
            total_assets=1_000_000,
            retained_earnings=-200_000,
            ebit=-50_000,
            total_equity=200_000,
            total_liabilities=800_000,
            revenue=300_000,
        )
        result = analyzer.altman_z_score_analysis(data)
        # WC/TA=-0.3, RE/TA=-0.2, EBIT/TA=-0.05, Eq/TL=0.25, Rev/TA=0.3
        # Z = 1.2*(-0.3)+1.4*(-0.2)+3.3*(-0.05)+0.6*0.25+1.0*0.3 = -0.36-0.28-0.165+0.15+0.30 = -0.335
        assert result.z_zone == "Distress"
        assert result.z_score < 1.81

    def test_score_capped_at_10(self, analyzer):
        data = FinancialData(
            current_assets=5_000_000,
            current_liabilities=100_000,
            total_assets=5_000_000,
            retained_earnings=3_000_000,
            ebit=2_000_000,
            total_equity=4_000_000,
            total_liabilities=1_000_000,
            revenue=10_000_000,
        )
        result = analyzer.altman_z_score_analysis(data)
        assert result.altman_score <= 10.0

    def test_score_floored_at_0(self, analyzer):
        data = FinancialData(
            current_assets=10_000,
            current_liabilities=900_000,
            total_assets=1_000_000,
            retained_earnings=-500_000,
            ebit=-300_000,
            total_equity=50_000,
            total_liabilities=950_000,
            revenue=100_000,
        )
        result = analyzer.altman_z_score_analysis(data)
        assert result.altman_score >= 0.0


# ===== EDGE CASES =====


class TestPhase34EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.altman_z_score_analysis(FinancialData())
        assert isinstance(result, AltmanZScoreResult)
        assert result.z_score is None
        assert result.z_zone == ""

    def test_missing_one_component(self, analyzer):
        """Missing EBIT => z_score is None."""
        data = FinancialData(
            current_assets=500_000,
            current_liabilities=200_000,
            total_assets=2_000_000,
            retained_earnings=600_000,
            total_equity=1_200_000,
            total_liabilities=800_000,
            revenue=1_000_000,
        )
        result = analyzer.altman_z_score_analysis(data)
        assert result.z_score is None  # Missing EBIT component
        assert result.working_capital_to_assets is not None  # Individual ratios still work

    def test_zero_total_assets(self, analyzer):
        """Zero TA => most ratios None."""
        data = FinancialData(
            total_assets=0,
            current_assets=100_000,
            current_liabilities=50_000,
        )
        result = analyzer.altman_z_score_analysis(data)
        assert result.working_capital_to_assets is None
        assert result.ebit_to_assets is None

    def test_zero_total_liabilities(self, analyzer):
        """Zero TL => equity_to_liabilities None."""
        data = FinancialData(
            total_assets=1_000_000,
            total_liabilities=0,
            total_equity=1_000_000,
            current_assets=500_000,
            current_liabilities=0,
            retained_earnings=400_000,
            ebit=200_000,
            revenue=800_000,
        )
        result = analyzer.altman_z_score_analysis(data)
        assert result.equity_to_liabilities is None
        assert result.z_score is None  # Can't compute without X4

    def test_negative_working_capital(self, analyzer):
        """CL > CA => negative WC/TA."""
        data = FinancialData(
            current_assets=100_000,
            current_liabilities=300_000,
            total_assets=1_000_000,
            retained_earnings=200_000,
            ebit=100_000,
            total_equity=500_000,
            total_liabilities=500_000,
            revenue=600_000,
        )
        result = analyzer.altman_z_score_analysis(data)
        assert result.working_capital_to_assets < 0

    def test_sample_data_score(self, analyzer, sample_data):
        """Z=2.33 (Gray zone, >=1.81 and <3.0: +0.0 from z, but from base 5.0)
        Z>=1.81: no adjustment from z level.
        WC/TA=0.15 (not >0.20 and not <0: no adj).
        EBIT/TA=0.10 (not >0.15 and not <0: no adj).
        => score = 5.0 => Watch... but let me recalculate:
        Z=2.33: >=1.81 and <2.5: +0.0
        WC/TA=0.15: >0 but <=0.20: no adj
        EBIT/TA=0.10: >0 but <=0.15: no adj
        => 5.0 => Watch (4.0-5.99)
        Wait: grade thresholds: >=8.0 Strong, >=6.0 Adequate, >=4.0 Watch, <4.0 Critical
        So 5.0 => Watch."""
        result = analyzer.altman_z_score_analysis(sample_data)
        assert result.altman_score == pytest.approx(5.0, abs=0.1)
        assert result.altman_grade == "Watch"
