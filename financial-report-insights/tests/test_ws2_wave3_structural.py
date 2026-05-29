"""WS-2 Wave 3 Structural Tests.

WP-F: _scored_analysis new modes (derived_primary, band) + conversions.
WP-I: Three-composite-score unification (deprecation markers + behavior invariance).

Test-first: snapshots captured BEFORE code changes; assertions enforce byte-equal
output after refactoring.
"""

import pytest
from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    AssetLightnessResult,
    PayoutResilienceResult,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    return CharlieAnalyzer()


@pytest.fixture
def sample_data():
    """Standard fixture used across existing phase tests."""
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


# ===========================================================================
# WP-F PART 1 — New mode unit tests
# These tests exercise the two NEW opt-in modes added to _scored_analysis.
# They are written BEFORE the implementation so they define the expected contract.
# ===========================================================================

class TestScoredAnalysisDerivedPrimaryMode:
    """Unit tests for the new 'derived_primary' mode in _scored_analysis.

    In this mode the primary metric is itself derived (computed by a
    derive_primary_fn) rather than read directly from a pre-computed
    ratio field.
    """

    def test_derived_primary_computes_and_scores(self, analyzer):
        """derive_primary_fn produces the primary value used for scoring."""
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class _DPResult:
            primary_val: Optional[float] = None
            dp_score: float = 0.0
            dp_grade: str = ""
            summary: str = ""

        data = FinancialData(revenue=1_000_000, total_assets=500_000)

        result = analyzer._scored_analysis(
            data=data,
            result_class=_DPResult,
            ratio_defs=[],
            score_field="dp_score",
            grade_field="dp_grade",
            primary="primary_val",
            higher_is_better=True,
            thresholds=[
                (2.0, 10.0),
                (1.5, 8.0),
                (1.0, 6.0),
                (0.5, 4.0),
                (0.0, 2.0),
            ],
            mode="derived_primary",
            derive_primary_fn=lambda ratios, d: (
                (d.revenue or 0) / (d.total_assets or 1)
            ),
            primary_result_field="primary_val",
        )
        # revenue/total_assets = 1_000_000 / 500_000 = 2.0 -> score 10.0
        assert result.primary_val == pytest.approx(2.0, abs=1e-9)
        assert result.dp_score == pytest.approx(10.0, abs=1e-9)
        assert result.dp_grade == "Excellent"

    def test_derived_primary_none_returns_zero_score(self, analyzer):
        """When derive_primary_fn returns None, score is 0 and grade absent."""
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class _DPResult:
            primary_val: Optional[float] = None
            dp_score: float = 0.0
            dp_grade: str = ""
            summary: str = ""

        data = FinancialData()

        result = analyzer._scored_analysis(
            data=data,
            result_class=_DPResult,
            ratio_defs=[],
            score_field="dp_score",
            grade_field="dp_grade",
            primary="primary_val",
            higher_is_better=True,
            thresholds=[(1.0, 10.0)],
            mode="derived_primary",
            derive_primary_fn=lambda ratios, d: None,
            primary_result_field="primary_val",
        )
        assert result.primary_val is None
        assert result.dp_score == 0.0
        assert "Insufficient" in result.summary

    def test_derived_primary_stores_on_result(self, analyzer):
        """The derived primary value is stored on the result under primary_result_field."""
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class _DPResult:
            my_metric: Optional[float] = None
            dp_score: float = 0.0
            dp_grade: str = ""
            summary: str = ""

        data = FinancialData(net_income=100_000, total_equity=500_000)

        result = analyzer._scored_analysis(
            data=data,
            result_class=_DPResult,
            ratio_defs=[],
            score_field="dp_score",
            grade_field="dp_grade",
            primary="my_metric",
            higher_is_better=True,
            thresholds=[(0.15, 10.0), (0.08, 7.0), (0.0, 4.0)],
            mode="derived_primary",
            derive_primary_fn=lambda ratios, d: (
                (d.net_income or 0) / (d.total_equity or 1)
            ),
            primary_result_field="my_metric",
        )
        # NI/TE = 100k/500k = 0.20 -> >= 0.15 -> score 10.0
        assert result.my_metric == pytest.approx(0.20, abs=1e-9)
        assert result.dp_score == pytest.approx(10.0, abs=1e-9)


class TestScoredAnalysisBandMode:
    """Unit tests for the new 'band' mode in _scored_analysis.

    Band mode scores highest in a target band and falls off both above
    and below (mid-band-optimal), unlike the monotonic higher/lower-is-better
    modes.
    """

    def test_band_center_scores_highest(self, analyzer):
        """A value in the target band achieves the maximum band score."""
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class _BResult:
            ratio: Optional[float] = None
            b_score: float = 0.0
            b_grade: str = ""
            summary: str = ""

        # band: [0.20, 0.50] is optimal (score 10.0)
        # band_thresholds format: (low, high, score), (low, high, score) ...
        data = FinancialData(dividends_paid=35_000, net_income=100_000)

        result = analyzer._scored_analysis(
            data=data,
            result_class=_BResult,
            ratio_defs=[("ratio", "dividends_paid", "net_income")],
            score_field="b_score",
            grade_field="b_grade",
            primary="ratio",
            higher_is_better=True,  # ignored in band mode
            thresholds=[],           # ignored in band mode
            mode="band",
            band_thresholds=[
                (0.20, 0.50, 10.0),
                (0.10, 0.60, 8.5),
                (0.05, 0.70, 7.0),
                (0.00, 0.80, 5.5),
                (0.00, 0.90, 4.0),
                (0.00, 1.00, 2.5),
            ],
        )
        # div/NI = 0.35 -> in [0.20, 0.50] -> score 10.0
        assert result.ratio == pytest.approx(0.35, abs=1e-9)
        assert result.b_score == pytest.approx(10.0, abs=1e-9)
        assert result.b_grade == "Excellent"

    def test_band_above_center_scores_lower(self, analyzer):
        """A value above the target band receives a lower score."""
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class _BResult:
            ratio: Optional[float] = None
            b_score: float = 0.0
            b_grade: str = ""
            summary: str = ""

        data = FinancialData(dividends_paid=95_000, net_income=100_000)

        result = analyzer._scored_analysis(
            data=data,
            result_class=_BResult,
            ratio_defs=[("ratio", "dividends_paid", "net_income")],
            score_field="b_score",
            grade_field="b_grade",
            primary="ratio",
            higher_is_better=True,
            thresholds=[],
            mode="band",
            band_thresholds=[
                (0.20, 0.50, 10.0),
                (0.10, 0.60, 8.5),
                (0.05, 0.70, 7.0),
                (0.00, 0.80, 5.5),
                (0.00, 0.90, 4.0),
                (0.00, 1.00, 2.5),
            ],
        )
        # div/NI = 0.95 -> only in [0.00, 1.00] -> 2.5 + adjustments
        assert result.b_score < 5.0

    def test_band_below_center_scores_lower(self, analyzer):
        """A value below the target band receives a lower score than centre."""
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class _BResult:
            ratio: Optional[float] = None
            b_score: float = 0.0
            b_grade: str = ""
            summary: str = ""

        # Below 0.05 -> falls through all bands -> lowest fallback score
        data = FinancialData(dividends_paid=3_000, net_income=100_000)

        result = analyzer._scored_analysis(
            data=data,
            result_class=_BResult,
            ratio_defs=[("ratio", "dividends_paid", "net_income")],
            score_field="b_score",
            grade_field="b_grade",
            primary="ratio",
            higher_is_better=True,
            thresholds=[],
            mode="band",
            band_thresholds=[
                (0.20, 0.50, 10.0),
                (0.10, 0.60, 8.5),
                (0.05, 0.70, 7.0),
                (0.00, 0.80, 5.5),
                (0.00, 0.90, 4.0),
                (0.00, 1.00, 2.5),
            ],
        )
        # div/NI=0.03, below 0.05 on low side of all bands
        # All band lows are 0.00, so it falls in the widest band [0.00, 1.00] => 2.5
        assert result.b_score < 8.0

    def test_band_none_primary_returns_zero(self, analyzer):
        """None primary in band mode returns score 0."""
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class _BResult:
            ratio: Optional[float] = None
            b_score: float = 0.0
            b_grade: str = ""
            summary: str = ""

        data = FinancialData()

        result = analyzer._scored_analysis(
            data=data,
            result_class=_BResult,
            ratio_defs=[("ratio", "dividends_paid", "net_income")],
            score_field="b_score",
            grade_field="b_grade",
            primary="ratio",
            higher_is_better=True,
            thresholds=[],
            mode="band",
            band_thresholds=[(0.20, 0.50, 10.0), (0.00, 1.00, 2.5)],
        )
        assert result.b_score == 0.0
        assert "Insufficient" in result.summary

    def test_band_accepts_adjustments(self, analyzer):
        """Band mode still applies optional adjustments on top of band score."""
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class _BResult:
            ratio: Optional[float] = None
            b_score: float = 0.0
            b_grade: str = ""
            summary: str = ""

        data = FinancialData(
            dividends_paid=35_000, net_income=100_000, operating_cash_flow=200_000
        )

        result = analyzer._scored_analysis(
            data=data,
            result_class=_BResult,
            ratio_defs=[("ratio", "dividends_paid", "net_income")],
            score_field="b_score",
            grade_field="b_grade",
            primary="ratio",
            higher_is_better=True,
            thresholds=[],
            mode="band",
            band_thresholds=[(0.20, 0.50, 9.0), (0.00, 1.00, 2.5)],
            adjustments=[
                (lambda ratios, d: d.operating_cash_flow is not None, 0.5),
            ],
        )
        # base 9.0 + adj 0.5 = 9.5
        assert result.b_score == pytest.approx(9.5, abs=1e-9)


# ===========================================================================
# WP-F PART 2 — Snapshot invariance for converted methods
# Snapshots captured BEFORE conversion; same values asserted AFTER.
# ===========================================================================

class TestAssetLightnessSnapshotInvariance:
    """Assert byte-equal output for asset_lightness_analysis after conversion."""

    def test_sample_data_snapshot(self, analyzer, sample_data):
        """Exact snapshot: score=3.5, grade='Weak'."""
        r = analyzer.asset_lightness_analysis(sample_data)
        assert r.alt_score == pytest.approx(3.5, abs=1e-9)
        assert r.alt_grade == "Weak"
        assert r.lightness_ratio == pytest.approx(0.25, abs=1e-9)
        assert r.ca_to_ta == pytest.approx(0.25, abs=1e-9)
        assert r.revenue_to_assets == pytest.approx(0.5, abs=1e-9)
        assert r.fixed_asset_ratio == pytest.approx(0.75, abs=1e-9)
        assert r.lightness_spread == pytest.approx(-0.25, abs=1e-9)
        assert r.summary == (
            "Asset Lightness: CA/TA=0.2500, Revenue/TA=0.5000, Score=3.5/10 (Weak)."
        )

    def test_high_lightness_snapshot(self, analyzer):
        """CA/TA=0.75 -> score=10.0, grade=Excellent."""
        data = FinancialData(
            current_assets=1_500_000, total_assets=2_000_000, revenue=1_500_000
        )
        r = analyzer.asset_lightness_analysis(data)
        assert r.alt_score == pytest.approx(10.0, abs=1e-9)
        assert r.alt_grade == "Excellent"
        assert r.summary == (
            "Asset Lightness: CA/TA=0.7500, Revenue/TA=0.7500, Score=10.0/10 (Excellent)."
        )

    def test_low_lightness_snapshot(self, analyzer):
        """CA/TA=0.05 -> score=1.5, grade=Weak."""
        data = FinancialData(
            current_assets=100_000, total_assets=2_000_000, revenue=500_000
        )
        r = analyzer.asset_lightness_analysis(data)
        assert r.alt_score == pytest.approx(1.5, abs=1e-9)
        assert r.alt_grade == "Weak"
        assert r.summary == (
            "Asset Lightness: CA/TA=0.0500, Revenue/TA=0.2500, Score=1.5/10 (Weak)."
        )

    def test_empty_data_returns_zero(self, analyzer):
        r = analyzer.asset_lightness_analysis(FinancialData())
        assert r.alt_score == 0.0
        assert r.alt_grade == ""

    def test_no_total_assets_lightness_none(self, analyzer):
        data = FinancialData(current_assets=500_000)
        r = analyzer.asset_lightness_analysis(data)
        assert r.lightness_ratio is None

    def test_result_type(self, analyzer, sample_data):
        r = analyzer.asset_lightness_analysis(sample_data)
        assert isinstance(r, AssetLightnessResult)


class TestPayoutResilienceSnapshotInvariance:
    """Assert byte-equal output for payout_resilience_analysis after conversion."""

    def test_sample_data_snapshot(self, analyzer, sample_data):
        """Exact snapshot: score=10.0, grade='Excellent'."""
        r = analyzer.payout_resilience_analysis(sample_data)
        assert r.prs_score == pytest.approx(10.0, abs=1e-9)
        assert r.prs_grade == "Excellent"
        assert r.div_to_ni == pytest.approx(40_000 / 150_000, abs=1e-9)
        assert r.div_to_ocf == pytest.approx(40_000 / 220_000, abs=1e-9)
        assert r.div_to_revenue == pytest.approx(40_000 / 1_000_000, abs=1e-9)
        assert r.div_to_ebitda == pytest.approx(40_000 / 250_000, abs=1e-9)
        assert r.payout_ratio == pytest.approx(40_000 / 150_000, abs=1e-9)
        assert r.resilience_buffer == pytest.approx(1.0 - 40_000 / 150_000, abs=1e-9)
        # summary ends without a period — match exact pre-refactor format
        assert r.summary == (
            "Payout Resilience Analysis: Div/NI=0.2667, Div/OCF=0.1818, "
            "Score=10.0/10 (Excellent)"
        )

    def test_high_payout_snapshot(self, analyzer):
        """Div/NI=0.933 -> score=3.0, grade=Weak."""
        data = FinancialData(
            dividends_paid=140_000, net_income=150_000, operating_cash_flow=220_000
        )
        r = analyzer.payout_resilience_analysis(data)
        assert r.prs_score == pytest.approx(3.0, abs=1e-9)
        assert r.prs_grade == "Weak"
        assert r.summary == (
            "Payout Resilience Analysis: Div/NI=0.9333, Div/OCF=0.6364, "
            "Score=3.0/10 (Weak)"
        )

    def test_over_payout_snapshot(self, analyzer):
        """Div/NI=1.333 -> score=1.5, grade=Weak."""
        data = FinancialData(
            dividends_paid=200_000, net_income=150_000, operating_cash_flow=220_000
        )
        r = analyzer.payout_resilience_analysis(data)
        assert r.prs_score == pytest.approx(1.5, abs=1e-9)
        assert r.prs_grade == "Weak"
        assert r.summary == (
            "Payout Resilience Analysis: Div/NI=1.3333, Div/OCF=0.9091, "
            "Score=1.5/10 (Weak)"
        )

    def test_empty_data_returns_zero(self, analyzer):
        r = analyzer.payout_resilience_analysis(FinancialData())
        assert r.prs_score == 0.0
        assert r.prs_grade == ""

    def test_no_dividends_returns_zero(self, analyzer):
        data = FinancialData(net_income=150_000, operating_cash_flow=220_000)
        r = analyzer.payout_resilience_analysis(data)
        assert r.prs_score == 0.0

    def test_no_net_income_returns_zero(self, analyzer):
        data = FinancialData(dividends_paid=40_000, operating_cash_flow=220_000)
        r = analyzer.payout_resilience_analysis(data)
        assert r.prs_score == 0.0

    def test_result_type(self, analyzer, sample_data):
        r = analyzer.payout_resilience_analysis(sample_data)
        assert isinstance(r, PayoutResilienceResult)


# ===========================================================================
# WP-F PART 3 — Regression: existing _scored_analysis methods unaffected
# ===========================================================================

class TestExistingScoredAnalysisRegression:
    """Confirm a sample of pre-existing _scored_analysis methods are unchanged."""

    def test_noncurrent_asset_ratio(self, analyzer, sample_data):
        """NCA/TA = 0.75 -> nar_score=9.0, nar_grade=Excellent (baseline)."""
        r = analyzer.noncurrent_asset_ratio_analysis(sample_data)
        assert r.nca_ratio == pytest.approx(0.75, abs=1e-9)
        assert r.nar_score == pytest.approx(9.0, abs=1e-9)
        assert r.nar_grade == "Excellent"

    def test_debt_burden_index(self, analyzer, sample_data):
        """Debt/EBITDA = 400k/250k=1.6 -> dbi_score=9.5, dbi_grade=Excellent."""
        r = analyzer.debt_burden_index_analysis(sample_data)
        assert r.dbi_score == pytest.approx(9.5, abs=1e-9)
        assert r.dbi_grade == "Excellent"


# ===========================================================================
# WP-I — Three-composite-score unification (deprecation + behavior invariance)
# ===========================================================================

class TestComprehensiveHealthScoreSnapshot:
    """Canonical method output unchanged after WP-I docstring edits."""

    def test_snapshot_overall_score(self, analyzer, sample_data):
        """overall_score=80.5, grade='A', 7 dimensions."""
        r = analyzer.comprehensive_health_score(sample_data)
        assert r.overall_score == pytest.approx(80.5, abs=0.5)
        assert r.grade == "A"
        assert len(r.dimensions) == 7

    def test_snapshot_summary_contains_grade(self, analyzer, sample_data):
        r = analyzer.comprehensive_health_score(sample_data)
        assert "A" in r.summary


class TestDeprecatedCompositesBehaviorUnchanged:
    """Deprecated methods still function AND carry deprecation markers."""

    def test_financial_rating_still_works(self, analyzer, sample_data):
        """financial_rating() still produces a valid score/grade."""
        r = analyzer.financial_rating(sample_data)
        assert r.overall_score == pytest.approx(8.51, abs=0.5)
        assert r.overall_grade in ("AA", "A", "AAA", "BBB")
        assert len(r.categories) > 0

    def test_financial_health_score_analysis_still_works(self, analyzer, sample_data):
        """financial_health_score_analysis() still produces a valid composite."""
        r = analyzer.financial_health_score_analysis(sample_data)
        assert r.composite_score == pytest.approx(8.6, abs=0.5)
        assert r.fh_grade in ("Excellent", "Good")

    def test_financial_rating_has_deprecation_docstring(self, analyzer):
        """financial_rating.__doc__ must contain a deprecation marker."""
        doc = analyzer.financial_rating.__doc__ or ""
        assert "deprecat" in doc.lower(), (
            "financial_rating docstring must contain a deprecation marker"
        )

    def test_financial_rating_docstring_references_canonical(self, analyzer):
        """financial_rating docstring must reference comprehensive_health_score."""
        doc = analyzer.financial_rating.__doc__ or ""
        assert "comprehensive_health_score" in doc, (
            "financial_rating docstring must reference the canonical method"
        )

    def test_financial_health_score_analysis_has_deprecation_docstring(self, analyzer):
        """financial_health_score_analysis.__doc__ must contain a deprecation marker."""
        doc = analyzer.financial_health_score_analysis.__doc__ or ""
        assert "deprecat" in doc.lower(), (
            "financial_health_score_analysis docstring must contain a deprecation marker"
        )

    def test_financial_health_score_analysis_docstring_references_canonical(self, analyzer):
        """financial_health_score_analysis docstring must reference comprehensive_health_score."""
        doc = analyzer.financial_health_score_analysis.__doc__ or ""
        assert "comprehensive_health_score" in doc, (
            "financial_health_score_analysis must reference the canonical method"
        )

    def test_comprehensive_health_score_is_not_deprecated(self, analyzer):
        """The canonical method must NOT carry a deprecation marker."""
        doc = analyzer.comprehensive_health_score.__doc__ or ""
        assert "deprecat" not in doc.lower(), (
            "comprehensive_health_score must NOT be marked as deprecated"
        )
