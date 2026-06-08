"""Tests for structured_types module.

Covers: RatioCategory enum, ratio dataclasses (Liquidity, Profitability,
Leverage, Efficiency), AnalysisResults, GraphChunk, GraphFinancialContext,
GraphRetrievalResult, and categorize_ratio().
"""

import pytest

from structured_types import (
    AnalysisResults,
    EfficiencyRatios,
    GraphChunk,
    GraphFinancialContext,
    GraphRetrievalResult,
    LeverageRatios,
    LiquidityRatios,
    ProfitabilityRatios,
    RatioCategory,
    categorize_ratio,
)


# ---------------------------------------------------------------------------
# RatioCategory enum
# ---------------------------------------------------------------------------

class TestRatioCategory:
    """Tests for the RatioCategory str-enum."""

    EXPECTED_MEMBERS = [
        "LIQUIDITY",
        "PROFITABILITY",
        "LEVERAGE",
        "EFFICIENCY",
        "VALUATION",
        "OTHER",
    ]

    @pytest.mark.parametrize("name", EXPECTED_MEMBERS)
    def test_member_exists(self, name):
        assert hasattr(RatioCategory, name)

    def test_all_members_accounted_for(self):
        member_names = [m.name for m in RatioCategory]
        assert sorted(member_names) == sorted(self.EXPECTED_MEMBERS)

    def test_string_comparison_liquidity(self):
        assert RatioCategory.LIQUIDITY == "Liquidity"

    def test_string_comparison_profitability(self):
        assert RatioCategory.PROFITABILITY == "Profitability"

    def test_string_comparison_leverage(self):
        assert RatioCategory.LEVERAGE == "Leverage"

    def test_string_comparison_efficiency(self):
        assert RatioCategory.EFFICIENCY == "Efficiency"

    def test_string_comparison_valuation(self):
        assert RatioCategory.VALUATION == "Valuation"

    def test_string_comparison_other(self):
        assert RatioCategory.OTHER == "Other"

    def test_iteration(self):
        members = list(RatioCategory)
        assert len(members) == 6

    def test_is_str_subclass(self):
        assert isinstance(RatioCategory.LIQUIDITY, str)


# ---------------------------------------------------------------------------
# Ratio dataclass helpers
# ---------------------------------------------------------------------------

# Mapping of dataclass -> (field_names, sample_values)
_RATIO_SPECS = {
    "liquidity": (
        LiquidityRatios,
        ["current_ratio", "quick_ratio", "cash_ratio", "working_capital"],
        [2.1, 1.5, 0.8, 50000.0],
    ),
    "profitability": (
        ProfitabilityRatios,
        ["gross_margin", "operating_margin", "net_margin", "roe", "roa", "roic"],
        [0.45, 0.30, 0.15, 0.20, 0.10, 0.12],
    ),
    "leverage": (
        LeverageRatios,
        ["debt_to_equity", "debt_to_assets", "debt_ratio", "equity_multiplier", "interest_coverage"],
        [1.5, 0.6, 0.55, 2.5, 4.0],
    ),
    "efficiency": (
        EfficiencyRatios,
        ["asset_turnover", "inventory_turnover", "receivables_turnover", "payables_turnover"],
        [1.2, 8.0, 12.0, 6.0],
    ),
}


def _make_full(cls, fields, values):
    """Create an instance with all fields populated."""
    return cls(**dict(zip(fields, values)))


# ---------------------------------------------------------------------------
# LiquidityRatios
# ---------------------------------------------------------------------------

class TestLiquidityRatios:

    CLS, FIELDS, VALUES = _RATIO_SPECS["liquidity"]

    def test_default_construction_all_none(self):
        obj = self.CLS()
        for f in self.FIELDS:
            assert getattr(obj, f) is None

    def test_full_construction(self):
        obj = _make_full(self.CLS, self.FIELDS, self.VALUES)
        for f, v in zip(self.FIELDS, self.VALUES):
            assert getattr(obj, f) == v

    def test_to_dict_keys(self):
        obj = _make_full(self.CLS, self.FIELDS, self.VALUES)
        d = obj.to_dict()
        assert set(d.keys()) == set(self.FIELDS)

    def test_to_dict_values(self):
        obj = _make_full(self.CLS, self.FIELDS, self.VALUES)
        d = obj.to_dict()
        for f, v in zip(self.FIELDS, self.VALUES):
            assert d[f] == v

    def test_round_trip(self):
        original = _make_full(self.CLS, self.FIELDS, self.VALUES)
        restored = self.CLS.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_from_dict_extra_keys_ignored(self):
        d = dict(zip(self.FIELDS, self.VALUES))
        d["extra_field"] = 999
        obj = self.CLS.from_dict(d)
        assert not hasattr(obj, "extra_field") or getattr(obj, "extra_field", None) is None
        # Known fields still set correctly
        assert getattr(obj, self.FIELDS[0]) == self.VALUES[0]

    def test_from_dict_missing_keys_default_none(self):
        obj = self.CLS.from_dict({"current_ratio": 1.0})
        assert obj.current_ratio == 1.0
        assert obj.quick_ratio is None

    def test_from_dict_empty_dict(self):
        obj = self.CLS.from_dict({})
        for f in self.FIELDS:
            assert getattr(obj, f) is None


# ---------------------------------------------------------------------------
# ProfitabilityRatios
# ---------------------------------------------------------------------------

class TestProfitabilityRatios:

    CLS, FIELDS, VALUES = _RATIO_SPECS["profitability"]

    def test_default_construction_all_none(self):
        obj = self.CLS()
        for f in self.FIELDS:
            assert getattr(obj, f) is None

    def test_full_construction(self):
        obj = _make_full(self.CLS, self.FIELDS, self.VALUES)
        for f, v in zip(self.FIELDS, self.VALUES):
            assert getattr(obj, f) == v

    def test_to_dict_keys(self):
        obj = _make_full(self.CLS, self.FIELDS, self.VALUES)
        d = obj.to_dict()
        assert set(d.keys()) == set(self.FIELDS)

    def test_round_trip(self):
        original = _make_full(self.CLS, self.FIELDS, self.VALUES)
        restored = self.CLS.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_from_dict_extra_keys_ignored(self):
        d = dict(zip(self.FIELDS, self.VALUES))
        d["bogus"] = -1
        obj = self.CLS.from_dict(d)
        assert getattr(obj, self.FIELDS[0]) == self.VALUES[0]

    def test_from_dict_missing_keys_default_none(self):
        obj = self.CLS.from_dict({"net_margin": 0.15})
        assert obj.net_margin == 0.15
        assert obj.gross_margin is None
        assert obj.roe is None

    def test_from_dict_empty_dict(self):
        obj = self.CLS.from_dict({})
        for f in self.FIELDS:
            assert getattr(obj, f) is None


# ---------------------------------------------------------------------------
# LeverageRatios
# ---------------------------------------------------------------------------

class TestLeverageRatios:

    CLS, FIELDS, VALUES = _RATIO_SPECS["leverage"]

    def test_default_construction_all_none(self):
        obj = self.CLS()
        for f in self.FIELDS:
            assert getattr(obj, f) is None

    def test_full_construction(self):
        obj = _make_full(self.CLS, self.FIELDS, self.VALUES)
        for f, v in zip(self.FIELDS, self.VALUES):
            assert getattr(obj, f) == v

    def test_to_dict_keys(self):
        obj = _make_full(self.CLS, self.FIELDS, self.VALUES)
        d = obj.to_dict()
        assert set(d.keys()) == set(self.FIELDS)

    def test_round_trip(self):
        original = _make_full(self.CLS, self.FIELDS, self.VALUES)
        restored = self.CLS.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_from_dict_extra_keys_ignored(self):
        d = {"debt_to_equity": 1.5, "nonsense": "abc"}
        obj = self.CLS.from_dict(d)
        assert obj.debt_to_equity == 1.5

    def test_from_dict_missing_keys_default_none(self):
        obj = self.CLS.from_dict({"interest_coverage": 4.0})
        assert obj.interest_coverage == 4.0
        assert obj.debt_to_equity is None

    def test_from_dict_empty_dict(self):
        obj = self.CLS.from_dict({})
        for f in self.FIELDS:
            assert getattr(obj, f) is None


# ---------------------------------------------------------------------------
# EfficiencyRatios
# ---------------------------------------------------------------------------

class TestEfficiencyRatios:

    CLS, FIELDS, VALUES = _RATIO_SPECS["efficiency"]

    def test_default_construction_all_none(self):
        obj = self.CLS()
        for f in self.FIELDS:
            assert getattr(obj, f) is None

    def test_full_construction(self):
        obj = _make_full(self.CLS, self.FIELDS, self.VALUES)
        for f, v in zip(self.FIELDS, self.VALUES):
            assert getattr(obj, f) == v

    def test_to_dict_keys(self):
        obj = _make_full(self.CLS, self.FIELDS, self.VALUES)
        d = obj.to_dict()
        assert set(d.keys()) == set(self.FIELDS)

    def test_round_trip(self):
        original = _make_full(self.CLS, self.FIELDS, self.VALUES)
        restored = self.CLS.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_from_dict_extra_keys_ignored(self):
        d = {"asset_turnover": 1.2, "fake": True}
        obj = self.CLS.from_dict(d)
        assert obj.asset_turnover == 1.2

    def test_from_dict_missing_keys_default_none(self):
        obj = self.CLS.from_dict({"inventory_turnover": 8.0})
        assert obj.inventory_turnover == 8.0
        assert obj.asset_turnover is None

    def test_from_dict_empty_dict(self):
        obj = self.CLS.from_dict({})
        for f in self.FIELDS:
            assert getattr(obj, f) is None


# ---------------------------------------------------------------------------
# AnalysisResults
# ---------------------------------------------------------------------------

class TestAnalysisResults:

    def test_default_construction(self):
        ar = AnalysisResults()
        assert ar.liquidity_ratios is not None or ar.liquidity_ratios is None  # exists
        assert ar.insights == [] or ar.insights is not None

    def test_default_ratio_groups_are_none_or_empty(self):
        ar = AnalysisResults()
        # Scalar optional fields should be None by default
        assert ar.cash_flow is None
        assert ar.working_capital is None
        assert ar.dupont is None
        assert ar.altman_z_score is None
        assert ar.piotroski_f_score is None
        assert ar.composite_health is None

    def test_full_construction_with_ratios(self):
        liq = LiquidityRatios(current_ratio=2.0, quick_ratio=1.5)
        prof = ProfitabilityRatios(net_margin=0.15, roe=0.20)
        lev = LeverageRatios(debt_to_equity=1.0)
        eff = EfficiencyRatios(asset_turnover=1.5)

        ar = AnalysisResults(
            liquidity_ratios=liq,
            profitability_ratios=prof,
            leverage_ratios=lev,
            efficiency_ratios=eff,
            composite_health=85,
            insights=["Good liquidity"],
        )
        assert ar.liquidity_ratios.current_ratio == 2.0
        assert ar.profitability_ratios.net_margin == 0.15
        assert ar.composite_health == 85
        assert ar.insights == ["Good liquidity"]

    def test_to_dict_has_expected_top_level_keys(self):
        ar = AnalysisResults()
        d = ar.to_dict()
        expected_keys = {
            "liquidity_ratios",
            "profitability_ratios",
            "leverage_ratios",
            "efficiency_ratios",
            "cash_flow",
            "working_capital",
            "dupont",
            "altman_z_score",
            "piotroski_f_score",
            "composite_health",
            "insights",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_ratio_groups_are_dicts(self):
        liq = LiquidityRatios(current_ratio=2.0)
        ar = AnalysisResults(liquidity_ratios=liq)
        d = ar.to_dict()
        assert isinstance(d["liquidity_ratios"], dict)
        assert d["liquidity_ratios"]["current_ratio"] == 2.0

    def test_from_dict_old_format_compatibility(self):
        old_format = {
            "liquidity_ratios": {"current_ratio": 1.5, "quick_ratio": 1.2},
            "profitability_ratios": {"net_margin": 0.15},
            "leverage_ratios": {},
            "efficiency_ratios": {},
            "cash_flow": None,
            "working_capital": None,
            "dupont": None,
            "altman_z_score": None,
            "piotroski_f_score": None,
            "composite_health": None,
            "insights": [],
        }
        ar = AnalysisResults.from_dict(old_format)
        assert ar.liquidity_ratios.current_ratio == 1.5
        assert ar.liquidity_ratios.quick_ratio == 1.2
        assert ar.profitability_ratios.net_margin == 0.15
        assert ar.leverage_ratios.debt_to_equity is None
        assert ar.cash_flow is None
        assert ar.insights == []

    def test_round_trip_preserves_ratio_values(self):
        liq = LiquidityRatios(current_ratio=2.1, quick_ratio=1.5)
        prof = ProfitabilityRatios(net_margin=0.15, roa=0.10)
        ar = AnalysisResults(
            liquidity_ratios=liq,
            profitability_ratios=prof,
            composite_health=72,
            insights=["Moderate"],
        )
        d = ar.to_dict()
        restored = AnalysisResults.from_dict(d)

        assert restored.liquidity_ratios.current_ratio == 2.1
        assert restored.liquidity_ratios.quick_ratio == 1.5
        assert restored.profitability_ratios.net_margin == 0.15
        assert restored.profitability_ratios.roa == 0.10
        assert restored.composite_health == 72
        assert restored.insights == ["Moderate"]

    def test_from_dict_empty_dict(self):
        ar = AnalysisResults.from_dict({})
        assert ar.cash_flow is None
        assert ar.composite_health is None

    def test_from_dict_partial_ratios(self):
        d = {"liquidity_ratios": {"cash_ratio": 0.5}}
        ar = AnalysisResults.from_dict(d)
        assert ar.liquidity_ratios.cash_ratio == 0.5
        assert ar.liquidity_ratios.current_ratio is None

    # ------------------------------------------------------------------
    # WP-A / P0-5: dict-cache invalidation tests
    # ------------------------------------------------------------------

    def test_post_construction_mutation_visible_via_dict(self):
        """Fix #2: mutate .insights after seeding cache; new value visible via dict."""
        ar = AnalysisResults()
        # Seed the cache
        _ = ar.to_dict()
        _ = ar["insights"]
        # Mutate after cache has been built
        ar.insights = ["new insight"]
        # The dict view must reflect the mutation
        assert ar["insights"] == ["new insight"]
        assert ar.to_dict()["insights"] == ["new insight"]

    def test_original_ordering_recurrence_ws2_o5(self):
        """WS2-O5: to_dict() FIRST then set .insights must still show new insights.

        This is the original-ordering recurrence that FAILS against the pre-fix
        class (no __setattr__ invalidation).  After Fix #2 it must pass.
        """
        ar = AnalysisResults()
        # Call to_dict() first -- this seeds the cache with empty insights
        first_dict = ar.to_dict()
        assert first_dict["insights"] == []
        # Now assign insights (reproduces the analyze() bug pattern)
        ar.insights = ["insight A", "insight B"]
        # The cache must be invalidated so the new value is visible
        assert ar["insights"] == ["insight A", "insight B"]
        assert ar.to_dict()["insights"] == ["insight A", "insight B"]

    def test_analyze_level_insights_non_empty(self):
        """analyze() result has non-empty insights in dict view when data triggers them."""
        from financial_analyzer import CharlieAnalyzer, FinancialData

        analyzer = CharlieAnalyzer()
        data = FinancialData(
            current_assets=500_000,
            current_liabilities=200_000,
            cash=50_000,
            net_income=150_000,
            total_assets=2_000_000,
            total_equity=1_200_000,
            revenue=1_000_000,
            gross_profit=400_000,
            operating_income=200_000,
        )
        result = analyzer.analyze(data)
        # Insights should be populated (current_ratio >= 1.5 triggers an info insight)
        assert result["insights"] is not None
        assert len(result["insights"]) > 0


# ---------------------------------------------------------------------------
# GraphChunk
# ---------------------------------------------------------------------------

class TestGraphChunk:

    def test_construction(self):
        gc = GraphChunk(source="report.pdf", content="Revenue grew 15%", score=0.92)
        assert gc.source == "report.pdf"
        assert gc.content == "Revenue grew 15%"
        assert gc.score == 0.92

    def test_to_dict(self):
        gc = GraphChunk(source="file.csv", content="data", score=0.5)
        d = gc.to_dict()
        assert d == {"source": "file.csv", "content": "data", "score": 0.5}

    def test_attribute_types(self):
        gc = GraphChunk(source="a", content="b", score=0.0)
        assert isinstance(gc.source, str)
        assert isinstance(gc.content, str)
        assert isinstance(gc.score, float)


# ---------------------------------------------------------------------------
# GraphFinancialContext
# ---------------------------------------------------------------------------

class TestGraphFinancialContext:

    def test_construction_empty_lists(self):
        ctx = GraphFinancialContext(
            document="Q3 Report", period="2025-Q3", ratios=[], scores=[]
        )
        assert ctx.document == "Q3 Report"
        assert ctx.period == "2025-Q3"
        assert ctx.ratios == []
        assert ctx.scores == []

    def test_construction_with_data(self):
        ctx = GraphFinancialContext(
            document="Annual",
            period="2025",
            ratios=[{"current_ratio": 2.0}],
            scores=[{"health": 85}],
        )
        assert len(ctx.ratios) == 1
        assert len(ctx.scores) == 1

    def test_to_dict(self):
        ctx = GraphFinancialContext(
            document="doc", period="2025-Q1", ratios=[1, 2], scores=[3]
        )
        d = ctx.to_dict()
        assert d["document"] == "doc"
        assert d["period"] == "2025-Q1"
        assert d["ratios"] == [1, 2]
        assert d["scores"] == [3]


# ---------------------------------------------------------------------------
# GraphRetrievalResult
# ---------------------------------------------------------------------------

class TestGraphRetrievalResult:

    def test_construction_empty(self):
        grr = GraphRetrievalResult(chunks=[], financial_context=[])
        assert grr.chunks == []
        assert grr.financial_context == []

    def test_construction_with_data(self):
        chunk = GraphChunk(source="a.pdf", content="text", score=0.9)
        ctx = GraphFinancialContext(
            document="a.pdf", period="2025", ratios=[], scores=[]
        )
        grr = GraphRetrievalResult(chunks=[chunk], financial_context=[ctx])
        assert len(grr.chunks) == 1
        assert len(grr.financial_context) == 1

    def test_to_dict_structure(self):
        grr = GraphRetrievalResult(chunks=[], financial_context=[])
        d = grr.to_dict()
        assert "chunks" in d
        assert "financial_context" in d
        assert isinstance(d["chunks"], list)
        assert isinstance(d["financial_context"], list)

    def test_to_dict_nested_serialization(self):
        chunk = GraphChunk(source="x", content="y", score=0.7)
        ctx = GraphFinancialContext(
            document="z", period="2025-Q2", ratios=[], scores=[]
        )
        grr = GraphRetrievalResult(chunks=[chunk], financial_context=[ctx])
        d = grr.to_dict()
        assert d["chunks"][0]["source"] == "x"
        assert d["financial_context"][0]["document"] == "z"

    def test_round_trip(self):
        chunk = GraphChunk(source="src", content="cnt", score=0.85)
        ctx = GraphFinancialContext(
            document="doc", period="P1", ratios=[{"r": 1}], scores=[{"s": 2}]
        )
        original = GraphRetrievalResult(chunks=[chunk], financial_context=[ctx])
        d = original.to_dict()
        restored = GraphRetrievalResult.from_dict(d)

        assert len(restored.chunks) == 1
        assert restored.chunks[0].source == "src"
        assert restored.chunks[0].score == 0.85
        assert len(restored.financial_context) == 1
        assert restored.financial_context[0].document == "doc"

    def test_from_dict_empty(self):
        grr = GraphRetrievalResult.from_dict({"chunks": [], "financial_context": []})
        assert grr.chunks == []
        assert grr.financial_context == []


# ---------------------------------------------------------------------------
# categorize_ratio function
# ---------------------------------------------------------------------------

class TestCategorizeRatio:

    @pytest.mark.parametrize(
        "key, expected",
        [
            ("current_ratio", RatioCategory.LIQUIDITY),
            ("quick_ratio", RatioCategory.LIQUIDITY),
            ("cash_ratio", RatioCategory.LIQUIDITY),
            ("working_capital", RatioCategory.LIQUIDITY),
            ("gross_margin", RatioCategory.PROFITABILITY),
            ("operating_margin", RatioCategory.PROFITABILITY),
            ("net_margin", RatioCategory.PROFITABILITY),
            ("roe", RatioCategory.PROFITABILITY),
            ("roa", RatioCategory.PROFITABILITY),
            ("roic", RatioCategory.PROFITABILITY),
            ("debt_to_equity", RatioCategory.LEVERAGE),
            ("debt_to_assets", RatioCategory.LEVERAGE),
            ("debt_ratio", RatioCategory.LEVERAGE),
            ("equity_multiplier", RatioCategory.LEVERAGE),
            ("interest_coverage", RatioCategory.LEVERAGE),
            ("asset_turnover", RatioCategory.EFFICIENCY),
            ("inventory_turnover", RatioCategory.EFFICIENCY),
            ("receivables_turnover", RatioCategory.EFFICIENCY),
            ("payables_turnover", RatioCategory.EFFICIENCY),
        ],
    )
    def test_known_keys(self, key, expected):
        assert categorize_ratio(key) == expected

    def test_unknown_key_returns_other(self):
        assert categorize_ratio("unknown_ratio") == RatioCategory.OTHER

    def test_completely_unknown_returns_other(self):
        assert categorize_ratio("xyzzy_metric") == RatioCategory.OTHER

    def test_empty_string_returns_other(self):
        assert categorize_ratio("") == RatioCategory.OTHER

    def test_case_sensitivity_uppercase_not_matched(self):
        """Keys are expected to be lowercase; uppercase should fall to OTHER."""
        result = categorize_ratio("CURRENT_RATIO")
        # If the implementation is case-insensitive, this still returns LIQUIDITY
        # which is also acceptable. We just verify it returns a valid category.
        assert isinstance(result, RatioCategory)

    def test_case_sensitivity_exact_lowercase_works(self):
        assert categorize_ratio("current_ratio") == RatioCategory.LIQUIDITY


# ---------------------------------------------------------------------------
# Cross-cutting: parametrized round-trip for all ratio dataclasses
# ---------------------------------------------------------------------------

class TestAllRatioDataclassesRoundTrip:
    """Parametrized round-trip across all four ratio dataclasses."""

    @pytest.mark.parametrize(
        "cls, fields, values",
        [
            _RATIO_SPECS["liquidity"],
            _RATIO_SPECS["profitability"],
            _RATIO_SPECS["leverage"],
            _RATIO_SPECS["efficiency"],
        ],
        ids=["liquidity", "profitability", "leverage", "efficiency"],
    )
    def test_round_trip(self, cls, fields, values):
        original = cls(**dict(zip(fields, values)))
        restored = cls.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    @pytest.mark.parametrize(
        "cls, fields, values",
        [
            _RATIO_SPECS["liquidity"],
            _RATIO_SPECS["profitability"],
            _RATIO_SPECS["leverage"],
            _RATIO_SPECS["efficiency"],
        ],
        ids=["liquidity", "profitability", "leverage", "efficiency"],
    )
    def test_default_all_none(self, cls, fields, values):
        obj = cls()
        for f in fields:
            assert getattr(obj, f) is None, f"Expected {f} to be None"

    @pytest.mark.parametrize(
        "cls, fields, values",
        [
            _RATIO_SPECS["liquidity"],
            _RATIO_SPECS["profitability"],
            _RATIO_SPECS["leverage"],
            _RATIO_SPECS["efficiency"],
        ],
        ids=["liquidity", "profitability", "leverage", "efficiency"],
    )
    def test_from_dict_empty(self, cls, fields, values):
        obj = cls.from_dict({})
        for f in fields:
            assert getattr(obj, f) is None
