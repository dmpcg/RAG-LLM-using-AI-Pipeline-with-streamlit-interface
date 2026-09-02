"""Phase 12 Tests: Financial Rating & Variance Waterfall.

Tests for financial_rating(), variance_waterfall() and related dataclasses.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    FinancialRating,
    RatingCategory,
    VarianceWaterfall,
    WaterfallItem,
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


@pytest.fixture
def prior_data():
    return FinancialData(
        revenue=900_000,
        cogs=560_000,
        gross_profit=340_000,
        operating_expenses=190_000,
        operating_income=150_000,
        net_income=110_000,
        ebit=150_000,
        interest_expense=25_000,
    )


# ===== DATACLASS TESTS =====


class TestRatingCategoryDataclass:
    def test_defaults(self):
        c = RatingCategory()
        assert c.score == 0.0
        assert c.grade == ""

    def test_fields(self):
        c = RatingCategory(name="Liquidity", score=7.5, grade="A")
        assert c.name == "Liquidity"


class TestFinancialRatingDataclass:
    def test_defaults(self):
        r = FinancialRating()
        assert r.overall_score == 0.0
        assert r.categories == []

    def test_fields(self):
        r = FinancialRating(overall_grade="AA", overall_score=8.5)
        assert r.overall_grade == "AA"


class TestWaterfallItemDataclass:
    def test_defaults(self):
        w = WaterfallItem()
        assert w.item_type == "delta"
        assert w.value == 0.0


class TestVarianceWaterfallDataclass:
    def test_defaults(self):
        v = VarianceWaterfall()
        assert v.items == []
        assert v.total_variance == 0.0


# ===== GRADE CONVERSION TESTS =====


class TestScoreToGrade:
    def test_aaa(self):
        assert CharlieAnalyzer._score_to_grade(9.5) == "AAA"

    def test_aa(self):
        assert CharlieAnalyzer._score_to_grade(8.5) == "AA"

    def test_a(self):
        assert CharlieAnalyzer._score_to_grade(7.5) == "A"

    def test_bbb(self):
        assert CharlieAnalyzer._score_to_grade(6.5) == "BBB"

    def test_bb(self):
        assert CharlieAnalyzer._score_to_grade(5.5) == "BB"

    def test_b(self):
        assert CharlieAnalyzer._score_to_grade(4.5) == "B"

    def test_ccc(self):
        assert CharlieAnalyzer._score_to_grade(3.5) == "CCC"

    def test_cc(self):
        assert CharlieAnalyzer._score_to_grade(2.5) == "CC"

    def test_c(self):
        assert CharlieAnalyzer._score_to_grade(1.0) == "C"

    def test_zero(self):
        assert CharlieAnalyzer._score_to_grade(0.0) == "C"

    def test_boundary_nine(self):
        assert CharlieAnalyzer._score_to_grade(9.0) == "AAA"

    def test_boundary_eight(self):
        assert CharlieAnalyzer._score_to_grade(8.0) == "AA"


# ===== FINANCIAL RATING TESTS =====


class TestFinancialRatingMethod:
    def test_returns_rating(self, analyzer, sample_data):
        result = analyzer.financial_rating(sample_data)
        assert isinstance(result, FinancialRating)

    def test_five_categories(self, analyzer, sample_data):
        result = analyzer.financial_rating(sample_data)
        assert len(result.categories) == 5

    def test_category_names(self, analyzer, sample_data):
        result = analyzer.financial_rating(sample_data)
        names = [c.name for c in result.categories]
        assert "Liquidity" in names
        assert "Profitability" in names
        assert "Leverage" in names
        assert "Efficiency" in names
        assert "Cash Flow" in names

    def test_overall_score_range(self, analyzer, sample_data):
        result = analyzer.financial_rating(sample_data)
        assert 0 <= result.overall_score <= 10

    def test_overall_grade_assigned(self, analyzer, sample_data):
        result = analyzer.financial_rating(sample_data)
        assert result.overall_grade in ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C"]

    def test_category_scores_range(self, analyzer, sample_data):
        result = analyzer.financial_rating(sample_data)
        for cat in result.categories:
            assert 0 <= cat.score <= 10

    def test_strong_company_high_rating(self, analyzer):
        strong = FinancialData(
            revenue=10_000_000,
            cogs=3_000_000,
            gross_profit=7_000_000,
            net_income=2_000_000,
            ebit=3_000_000,
            total_equity=5_000_000,
            total_assets=8_000_000,
            total_debt=1_000_000,
            current_assets=4_000_000,
            current_liabilities=1_000_000,
            inventory=500_000,
            interest_expense=100_000,
            operating_cash_flow=2_500_000,
            capex=500_000,
        )
        result = analyzer.financial_rating(strong)
        assert result.overall_score >= 7.0
        assert result.overall_grade in ["AAA", "AA", "A"]

    def test_weak_company_low_rating(self, analyzer):
        weak = FinancialData(
            revenue=500_000,
            cogs=450_000,
            gross_profit=50_000,
            net_income=5_000,
            ebit=10_000,
            total_equity=100_000,
            total_assets=500_000,
            total_debt=350_000,
            current_assets=80_000,
            current_liabilities=200_000,
            inventory=60_000,
            interest_expense=40_000,
            operating_cash_flow=-10_000,
            capex=20_000,
        )
        result = analyzer.financial_rating(weak)
        assert result.overall_score < 5.0

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.financial_rating(sample_data)
        assert "Overall" in result.summary

    def test_details_present(self, analyzer, sample_data):
        result = analyzer.financial_rating(sample_data)
        for cat in result.categories:
            assert cat.details != ""

    def test_empty_data(self, analyzer):
        result = analyzer.financial_rating(FinancialData())
        assert isinstance(result, FinancialRating)
        assert result.overall_score == 0.0


# ===== VARIANCE WATERFALL TESTS =====


class TestVarianceWaterfall:
    def test_returns_waterfall(self, analyzer, sample_data, prior_data):
        result = analyzer.variance_waterfall(sample_data, prior_data)
        assert isinstance(result, VarianceWaterfall)

    def test_start_and_end(self, analyzer, sample_data, prior_data):
        result = analyzer.variance_waterfall(sample_data, prior_data)
        assert result.start_value == 110_000  # prior net income
        assert result.end_value == 150_000  # current net income

    def test_total_variance(self, analyzer, sample_data, prior_data):
        result = analyzer.variance_waterfall(sample_data, prior_data)
        assert result.total_variance == 40_000  # 150k - 110k

    def test_has_start_item(self, analyzer, sample_data, prior_data):
        result = analyzer.variance_waterfall(sample_data, prior_data)
        start_items = [i for i in result.items if i.item_type == "start"]
        assert len(start_items) == 1
        assert start_items[0].value == 110_000

    def test_has_total_item(self, analyzer, sample_data, prior_data):
        result = analyzer.variance_waterfall(sample_data, prior_data)
        total_items = [i for i in result.items if i.item_type == "total"]
        assert len(total_items) == 1
        assert total_items[0].value == 150_000

    def test_revenue_delta(self, analyzer, sample_data, prior_data):
        result = analyzer.variance_waterfall(sample_data, prior_data)
        rev_item = [i for i in result.items if "Revenue" in i.label][0]
        assert rev_item.value == 100_000  # 1M - 900k

    def test_cogs_impact(self, analyzer, sample_data, prior_data):
        result = analyzer.variance_waterfall(sample_data, prior_data)
        cogs_item = [i for i in result.items if "COGS" in i.label][0]
        # COGS went from 560k to 600k -> unfavorable -> -(600k - 560k) = -40k
        assert cogs_item.value == -40_000

    def test_opex_impact(self, analyzer, sample_data, prior_data):
        result = analyzer.variance_waterfall(sample_data, prior_data)
        opex_item = [i for i in result.items if "OpEx" in i.label][0]
        # OpEx went from 190k to 200k -> unfavorable -> -(200k - 190k) = -10k
        assert opex_item.value == -10_000

    def test_interest_impact(self, analyzer, sample_data, prior_data):
        result = analyzer.variance_waterfall(sample_data, prior_data)
        int_item = [i for i in result.items if "Interest" in i.label][0]
        # Interest went from 25k to 30k -> unfavorable -> -(30k - 25k) = -5k
        assert int_item.value == -5_000

    def test_summary_present(self, analyzer, sample_data, prior_data):
        result = analyzer.variance_waterfall(sample_data, prior_data)
        assert "changed" in result.summary.lower()
        assert "40,000" in result.summary

    def test_cumulative_makes_sense(self, analyzer, sample_data, prior_data):
        result = analyzer.variance_waterfall(sample_data, prior_data)
        # Last item's cumulative should equal end value
        assert result.items[-1].cumulative == result.end_value

    def test_identical_periods(self, analyzer, sample_data):
        result = analyzer.variance_waterfall(sample_data, sample_data)
        assert result.total_variance == 0
        # All deltas should be zero
        deltas = [i for i in result.items if i.item_type == "delta"]
        for d in deltas:
            assert d.value == 0

    def test_zero_prior(self, analyzer, sample_data):
        prior = FinancialData(net_income=0)
        result = analyzer.variance_waterfall(sample_data, prior)
        assert result.start_value == 0
        assert result.end_value == 150_000


# ===== EDGE CASES =====


class TestPhase12EdgeCases:
    def test_rating_partial_data(self, analyzer):
        data = FinancialData(revenue=1_000_000, net_income=100_000)
        result = analyzer.financial_rating(data)
        assert isinstance(result, FinancialRating)
        # Some categories will have low scores due to missing data
        assert result.overall_score >= 0

    def test_waterfall_negative_income(self, analyzer):
        current = FinancialData(net_income=-50_000, revenue=500_000, cogs=400_000)
        prior = FinancialData(net_income=100_000, revenue=800_000, cogs=500_000)
        result = analyzer.variance_waterfall(current, prior)
        assert result.total_variance == -150_000

    def test_waterfall_empty_data(self, analyzer):
        result = analyzer.variance_waterfall(FinancialData(), FinancialData())
        assert result.total_variance == 0

    def test_rating_grade_boundaries(self, analyzer):
        """Verify grade boundaries work correctly."""
        assert CharlieAnalyzer._score_to_grade(10.0) == "AAA"
        assert CharlieAnalyzer._score_to_grade(7.0) == "A"
        assert CharlieAnalyzer._score_to_grade(6.99) == "BBB"
