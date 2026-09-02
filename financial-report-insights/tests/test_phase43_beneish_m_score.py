"""Phase 43 Tests: Beneish M-Score (Earnings Manipulation Detection).

Tests for beneish_m_score_analysis() and BeneishMScoreResult dataclass.
"""

import pytest

from financial_analyzer import (
    BeneishMScoreResult,
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


class TestBeneishMScoreDataclass:
    def test_defaults(self):
        r = BeneishMScoreResult()
        assert r.m_score is None
        assert r.dsri is None
        assert r.gmi is None
        assert r.aqi is None
        assert r.sgi is None
        assert r.depi is None
        assert r.sgai is None
        assert r.lvgi is None
        assert r.tata is None
        assert r.manipulation_score == 0.0
        assert r.manipulation_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = BeneishMScoreResult(
            m_score=-2.50,
            manipulation_grade="Unlikely",
        )
        assert r.m_score == -2.50
        assert r.manipulation_grade == "Unlikely"


# ===== CORE COMPUTATION TESTS =====


class TestBeneishMScoreAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.beneish_m_score_analysis(sample_data)
        assert isinstance(result, BeneishMScoreResult)

    def test_dsri(self, analyzer, sample_data):
        """DSRI: AR/Revenue = 150k/1M = 0.15; dsri = 0.15/0.15 = 1.0."""
        result = analyzer.beneish_m_score_analysis(sample_data)
        assert result.dsri == pytest.approx(1.0, abs=0.01)

    def test_gmi(self, analyzer, sample_data):
        """GMI: GM = 400k/1M = 0.40; gmi = (1 - 0.40)/0.60 = 1.0."""
        result = analyzer.beneish_m_score_analysis(sample_data)
        assert result.gmi == pytest.approx(1.0, abs=0.01)

    def test_aqi(self, analyzer, sample_data):
        """AQI: CA/TA = 500k/2M = 0.25; aqi = 1.0 + (1-0.25)*0.5 = 1.375."""
        result = analyzer.beneish_m_score_analysis(sample_data)
        assert result.aqi == pytest.approx(1.375, abs=0.01)

    def test_sgi(self, analyzer, sample_data):
        """SGI: default 1.0 (no prior period)."""
        result = analyzer.beneish_m_score_analysis(sample_data)
        assert result.sgi == pytest.approx(1.0, abs=0.01)

    def test_depi(self, analyzer, sample_data):
        """DEPI: dep_rate = 50k/2M = 0.025; depi = 0.05/0.025 = 2.0."""
        result = analyzer.beneish_m_score_analysis(sample_data)
        assert result.depi == pytest.approx(2.0, abs=0.01)

    def test_sgai(self, analyzer, sample_data):
        """SGAI: opex/revenue = 200k/1M = 0.20; sgai = 0.20/0.20 = 1.0."""
        result = analyzer.beneish_m_score_analysis(sample_data)
        assert result.sgai == pytest.approx(1.0, abs=0.01)

    def test_lvgi(self, analyzer, sample_data):
        """LVGI: TL/TA = 800k/2M = 0.40; lvgi = 0.40/0.40 = 1.0."""
        result = analyzer.beneish_m_score_analysis(sample_data)
        assert result.lvgi == pytest.approx(1.0, abs=0.01)

    def test_tata(self, analyzer, sample_data):
        """TATA: (NI - OCF) / TA = (150k - 220k) / 2M = -0.035."""
        result = analyzer.beneish_m_score_analysis(sample_data)
        assert result.tata == pytest.approx(-0.035, abs=0.001)

    def test_m_score(self, analyzer, sample_data):
        """M = -4.84 + 0.920(1.0) + 0.528(1.0) + 0.404(1.375) + 0.892(1.0)
        + 0.115(2.0) - 0.172(1.0) + 4.679(-0.035) - 0.327(1.0)
        = -4.84 + 0.920 + 0.528 + 0.5555 + 0.892 + 0.230 - 0.172
          - 0.16377 - 0.327 = ~-2.377."""
        result = analyzer.beneish_m_score_analysis(sample_data)
        assert result.m_score == pytest.approx(-2.377, abs=0.05)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.beneish_m_score_analysis(sample_data)
        assert result.manipulation_grade in ["Unlikely", "Possible", "Likely", "Highly Likely"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.beneish_m_score_analysis(sample_data)
        assert "M-Score" in result.summary or "Manipulation" in result.summary


# ===== SCORING TESTS =====


class TestBeneishMScoreScoring:
    def test_unlikely(self, analyzer):
        """Very negative TATA drives M far below -2.22 => Unlikely."""
        data = FinancialData(
            revenue=1_000_000,
            gross_profit=500_000,
            operating_expenses=100_000,
            net_income=50_000,
            operating_cash_flow=300_000,  # TATA = (50k-300k)/1M = -0.25
            total_assets=1_000_000,
            total_liabilities=300_000,
            depreciation=30_000,
            accounts_receivable=100_000,
            current_assets=400_000,
        )
        result = analyzer.beneish_m_score_analysis(data)
        # M-Score driven very negative by TATA=-0.25 (*4.679 = -1.17)
        assert result.manipulation_score >= 8.0
        assert result.manipulation_grade == "Unlikely"

    def test_highly_likely(self, analyzer):
        """Large positive TATA + high indices => Highly Likely."""
        data = FinancialData(
            revenue=500_000,
            gross_profit=50_000,  # GM = 10%; gmi = (1-0.10)/0.60 = 1.50
            operating_expenses=200_000,  # sgai = 0.40/0.20 = 2.0
            net_income=400_000,
            operating_cash_flow=10_000,  # TATA = (400k-10k)/1M = 0.39
            total_assets=1_000_000,
            total_liabilities=700_000,  # lvgi = 0.70/0.40 = 1.75
            accounts_receivable=200_000,  # dsri = (200k/500k)/0.15 = 2.667
            current_assets=300_000,
            depreciation=5_000,  # dep_rate = 0.005; depi = 0.05/0.005 = 10
        )
        result = analyzer.beneish_m_score_analysis(data)
        # M-Score pushed highly positive by TATA=0.39 and large indices
        assert result.manipulation_score < 4.0
        assert result.manipulation_grade == "Highly Likely"


# ===== EDGE CASES =====


class TestPhase43EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.beneish_m_score_analysis(FinancialData())
        assert isinstance(result, BeneishMScoreResult)
        assert result.m_score is None

    def test_no_revenue(self, analyzer):
        """No revenue => insufficient data."""
        data = FinancialData(
            total_assets=1_000_000,
            net_income=100_000,
        )
        result = analyzer.beneish_m_score_analysis(data)
        assert result.m_score is None

    def test_no_total_assets(self, analyzer):
        """No total assets => insufficient data."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
        )
        result = analyzer.beneish_m_score_analysis(data)
        assert result.m_score is None

    def test_minimal_data(self, analyzer):
        """Just revenue + TA => M-Score computed with defaults."""
        data = FinancialData(
            revenue=1_000_000,
            total_assets=2_000_000,
        )
        result = analyzer.beneish_m_score_analysis(data)
        assert result.m_score is not None
        # All indices default to 1.0, TATA = 0.0
        # M = -4.84 + 0.920 + 0.528 + 1.0 + 0.892 + 0.115 - 0.172 + 0 - 0.327
        # AQI = 1.0 (no CA), so 0.404*1.0 = 0.404
        # M = -4.84 + 0.920 + 0.528 + 0.404 + 0.892 + 0.115 - 0.172 + 0 - 0.327
        # M = -2.48
        assert result.m_score == pytest.approx(-2.48, abs=0.05)

    def test_sample_data_score(self, analyzer, sample_data):
        """M ~= -2.38, which is < -2.22 => score 9.0.
        TATA=-0.035 (not <= -0.05): no adj => score 9.0 => Unlikely."""
        result = analyzer.beneish_m_score_analysis(sample_data)
        assert result.manipulation_score == pytest.approx(9.0, abs=0.3)
        assert result.manipulation_grade == "Unlikely"
