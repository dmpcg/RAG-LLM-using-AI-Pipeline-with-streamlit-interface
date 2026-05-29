"""Phase 8 Tests: Working Capital Analysis & Narrative Intelligence.

Tests for CCC/DSO/DIO/DPO metrics and auto-generated SWOT narratives.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    NarrativeReport,
    WorkingCapitalResult,
)


@pytest.fixture
def analyzer():
    return CharlieAnalyzer()


@pytest.fixture
def sample_data():
    """Complete FinancialData for testing."""
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
        inventory=100_000,
        accounts_receivable=150_000,
        accounts_payable=80_000,
        total_debt=400_000,
        retained_earnings=600_000,
        depreciation=50_000,
        interest_expense=30_000,
        operating_cash_flow=220_000,
        investing_cash_flow=-80_000,
        financing_cash_flow=-50_000,
        capex=80_000,
    )


@pytest.fixture
def weak_data():
    """Financially weak company for narrative testing."""
    return FinancialData(
        revenue=100_000,
        cogs=90_000,
        gross_profit=10_000,
        operating_expenses=15_000,
        net_income=-5_000,
        ebit=-5_000,
        ebitda=0,
        total_assets=200_000,
        total_liabilities=180_000,
        total_equity=20_000,
        current_assets=30_000,
        current_liabilities=60_000,
        total_debt=150_000,
        interest_expense=20_000,
    )


# ===== WORKING CAPITAL RESULT DATACLASS =====


class TestWorkingCapitalResultDataclass:
    def test_defaults(self):
        r = WorkingCapitalResult()
        assert r.dso is None
        assert r.ccc is None
        assert r.insights == []

    def test_fields_assignable(self):
        r = WorkingCapitalResult(dso=30.0, ccc=45.0)
        assert r.dso == 30.0
        assert r.ccc == 45.0


# ===== NARRATIVE REPORT DATACLASS =====


class TestNarrativeReportDataclass:
    def test_defaults(self):
        r = NarrativeReport()
        assert r.headline == ""
        assert r.strengths == []
        assert r.recommendation == ""

    def test_fields_assignable(self):
        r = NarrativeReport(headline="Test", recommendation="Do it")
        assert r.headline == "Test"


# ===== WORKING CAPITAL ANALYSIS =====


class TestWorkingCapitalAnalysis:
    def test_returns_working_capital_result(self, analyzer, sample_data):
        result = analyzer.working_capital_analysis(sample_data)
        assert isinstance(result, WorkingCapitalResult)

    def test_dso_calculation(self, analyzer, sample_data):
        result = analyzer.working_capital_analysis(sample_data)
        # DSO = 150k / 1M * 365 = 54.75
        assert result.dso is not None
        assert abs(result.dso - 54.8) < 0.5

    def test_dio_calculation(self, analyzer, sample_data):
        result = analyzer.working_capital_analysis(sample_data)
        # DIO = 100k / 600k * 365 = 60.83
        assert result.dio is not None
        assert abs(result.dio - 60.8) < 0.5

    def test_dpo_calculation(self, analyzer, sample_data):
        result = analyzer.working_capital_analysis(sample_data)
        # DPO = 80k / 600k * 365 = 48.67
        assert result.dpo is not None
        assert abs(result.dpo - 48.7) < 0.5

    def test_ccc_calculation(self, analyzer, sample_data):
        result = analyzer.working_capital_analysis(sample_data)
        # CCC = DSO + DIO - DPO = 54.8 + 60.8 - 48.7 = 66.9
        assert result.ccc is not None
        expected = result.dso + result.dio - result.dpo
        assert abs(result.ccc - expected) < 0.1

    def test_net_working_capital(self, analyzer, sample_data):
        result = analyzer.working_capital_analysis(sample_data)
        # NWC = 500k - 200k = 300k
        assert result.net_working_capital == 300_000

    def test_working_capital_ratio(self, analyzer, sample_data):
        result = analyzer.working_capital_analysis(sample_data)
        # WC ratio = 500k / 200k = 2.5
        assert result.working_capital_ratio == 2.5

    def test_insights_generated(self, analyzer, sample_data):
        result = analyzer.working_capital_analysis(sample_data)
        assert len(result.insights) > 0

    def test_no_ar_no_dso(self, analyzer):
        data = FinancialData(revenue=100_000, cogs=50_000)
        result = analyzer.working_capital_analysis(data)
        assert result.dso is None

    def test_no_inventory_no_dio(self, analyzer):
        data = FinancialData(revenue=100_000, cogs=50_000, accounts_receivable=10_000)
        result = analyzer.working_capital_analysis(data)
        assert result.dio is None

    def test_no_cogs_no_dio_dpo(self, analyzer):
        data = FinancialData(revenue=100_000, inventory=10_000, accounts_payable=5_000)
        result = analyzer.working_capital_analysis(data)
        assert result.dio is None
        assert result.dpo is None

    def test_ccc_none_if_incomplete(self, analyzer):
        data = FinancialData(revenue=100_000, cogs=50_000, accounts_receivable=10_000)
        result = analyzer.working_capital_analysis(data)
        # Missing inventory and AP, so DIO and DPO are None => CCC is None
        assert result.ccc is None

    def test_empty_data(self, analyzer):
        result = analyzer.working_capital_analysis(FinancialData())
        assert result.dso is None
        assert result.dio is None
        assert result.dpo is None
        assert result.ccc is None

    def test_uses_avg_receivables_fallback(self, analyzer):
        """Uses avg_receivables when accounts_receivable is None."""
        data = FinancialData(revenue=100_000, avg_receivables=20_000)
        result = analyzer.working_capital_analysis(data)
        # DSO = 20k / 100k * 365 = 73.0
        assert result.dso is not None
        assert abs(result.dso - 73.0) < 0.5

    def test_high_dso_insight(self, analyzer):
        """DSO > 60 should trigger warning insight."""
        data = FinancialData(revenue=100_000, accounts_receivable=20_000)
        result = analyzer.working_capital_analysis(data)
        # DSO = 73 days
        assert any("high" in i.lower() or "tightening" in i.lower() for i in result.insights)


# ===== NARRATIVE INTELLIGENCE =====


class TestNarrativeIntelligence:
    def test_returns_narrative_report(self, analyzer, sample_data):
        result = analyzer.generate_narrative(sample_data)
        assert isinstance(result, NarrativeReport)

    def test_headline_not_empty(self, analyzer, sample_data):
        result = analyzer.generate_narrative(sample_data)
        assert len(result.headline) > 0

    def test_strong_company_has_strengths(self, analyzer, sample_data):
        result = analyzer.generate_narrative(sample_data)
        assert len(result.strengths) > 0

    def test_strong_company_headline_positive(self, analyzer, sample_data):
        result = analyzer.generate_narrative(sample_data)
        assert "strong" in result.headline.lower() or "moderate" in result.headline.lower()

    def test_weak_company_has_risks(self, analyzer, weak_data):
        result = analyzer.generate_narrative(weak_data)
        assert len(result.risks) > 0 or len(result.weaknesses) > 0

    def test_weak_company_headline_negative(self, analyzer, weak_data):
        result = analyzer.generate_narrative(weak_data)
        assert "weak" in result.headline.lower() or "moderate" in result.headline.lower()

    def test_recommendation_not_empty(self, analyzer, sample_data):
        result = analyzer.generate_narrative(sample_data)
        assert len(result.recommendation) > 0

    def test_profitability_strength_detected(self, analyzer, sample_data):
        """Net margin = 15%, should be flagged as strength."""
        result = analyzer.generate_narrative(sample_data)
        has_margin = any("margin" in s.lower() or "profit" in s.lower() for s in result.strengths)
        assert has_margin

    def test_leverage_strength_detected(self, analyzer, sample_data):
        """D/E = 0.67, should be flagged as conservative leverage."""
        result = analyzer.generate_narrative(sample_data)
        has_leverage = any("leverage" in s.lower() or "d/e" in s.lower() for s in result.strengths)
        assert has_leverage

    def test_interest_coverage_strength(self, analyzer, sample_data):
        """EBIT/interest = 200k/30k = 6.67, should be flagged."""
        result = analyzer.generate_narrative(sample_data)
        has_ic = any("interest coverage" in s.lower() for s in result.strengths)
        assert has_ic

    def test_empty_data_no_crash(self, analyzer):
        result = analyzer.generate_narrative(FinancialData())
        assert isinstance(result, NarrativeReport)
        assert len(result.headline) > 0

    def test_z_score_in_narrative(self, analyzer, sample_data):
        """Z-Score should appear somewhere in SWOT (strengths or risks)."""
        result = analyzer.generate_narrative(sample_data)
        all_items = result.strengths + result.risks
        has_z = any("z-score" in s.lower() or "z score" in s.lower() or "altman" in s.lower() for s in all_items)
        assert has_z

    def test_breakeven_in_narrative(self, analyzer, sample_data):
        """Profitable company should mention margin of safety."""
        result = analyzer.generate_narrative(sample_data)
        all_items = result.strengths + result.opportunities
        has_breakeven = any("breakeven" in s.lower() or "margin of safety" in s.lower() for s in all_items)
        assert has_breakeven

    def test_weak_company_liquidity_warning(self, analyzer, weak_data):
        """Current ratio < 1 should trigger weakness."""
        result = analyzer.generate_narrative(weak_data)
        has_liquidity = any("current ratio" in w.lower() or "liquidity" in w.lower() for w in result.weaknesses)
        assert has_liquidity

    def test_unprofitable_company_weakness(self, analyzer, weak_data):
        """Negative net income should trigger weakness."""
        result = analyzer.generate_narrative(weak_data)
        has_profit_weakness = any("margin" in w.lower() or "unprofitable" in w.lower() for w in result.weaknesses)
        assert has_profit_weakness


# ===== EDGE CASES =====


class TestPhase8EdgeCases:
    def test_wc_negative_ccc(self, analyzer):
        """Company with very high DPO can have negative CCC."""
        data = FinancialData(
            revenue=100_000,
            cogs=50_000,
            accounts_receivable=5_000,  # DSO = 18.25
            inventory=5_000,  # DIO = 36.5
            accounts_payable=30_000,  # DPO = 219
        )
        result = analyzer.working_capital_analysis(data)
        assert result.ccc is not None
        assert result.ccc < 0

    def test_narrative_swot_counts(self, analyzer, sample_data):
        """Total SWOT items should be reasonable."""
        result = analyzer.generate_narrative(sample_data)
        total = len(result.strengths) + len(result.weaknesses) + len(result.opportunities) + len(result.risks)
        assert total >= 3  # Should have at least a few items
        assert total <= 20  # Not excessively verbose

    def test_wc_zero_revenue(self, analyzer):
        data = FinancialData(revenue=0, accounts_receivable=10_000)
        result = analyzer.working_capital_analysis(data)
        assert result.dso is None

    def test_narrative_high_leverage_risk(self, analyzer):
        """Very high leverage should trigger risk."""
        data = FinancialData(
            revenue=100_000,
            net_income=5_000,
            ebit=10_000,
            total_assets=200_000,
            total_liabilities=180_000,
            total_equity=20_000,
            total_debt=150_000,
            current_assets=30_000,
            current_liabilities=20_000,
        )
        result = analyzer.generate_narrative(data)
        has_leverage_risk = any("debt" in r.lower() or "leverage" in r.lower() for r in result.risks)
        assert has_leverage_risk
