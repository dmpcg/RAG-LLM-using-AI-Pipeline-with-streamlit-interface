"""Phase 11 Tests: Peer Comparison & Ratio Decomposition.

Tests for peer_comparison(), ratio_decomposition() and related dataclasses.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    PeerCompanyData,
    PeerComparisonReport,
    PeerMetricComparison,
    RatioDecompositionNode,
    RatioDecompositionTree,
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
        ebt=180_000,
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
def peer_data():
    """Second company data for comparison."""
    return FinancialData(
        revenue=800_000,
        cogs=520_000,
        gross_profit=280_000,
        operating_expenses=180_000,
        operating_income=100_000,
        net_income=70_000,
        ebit=100_000,
        ebitda=130_000,
        total_assets=1_500_000,
        total_liabilities=700_000,
        total_equity=800_000,
        current_assets=350_000,
        current_liabilities=250_000,
        inventory=80_000,
        accounts_receivable=120_000,
        total_debt=350_000,
        interest_expense=40_000,
    )


# ===== DATACLASS TESTS =====


class TestPeerCompanyDataDataclass:
    def test_defaults(self):
        d = PeerCompanyData()
        assert d.name == ""
        assert d.data is None

    def test_fields_assignable(self):
        fd = FinancialData(revenue=100)
        d = PeerCompanyData(name="TestCo", data=fd)
        assert d.name == "TestCo"
        assert d.data.revenue == 100


class TestPeerMetricComparisonDataclass:
    def test_defaults(self):
        m = PeerMetricComparison()
        assert m.values == {}
        assert m.best_performer == ""

    def test_fields(self):
        m = PeerMetricComparison(metric_name="roe", values={"A": 0.1})
        assert m.metric_name == "roe"


class TestPeerComparisonReportDataclass:
    def test_defaults(self):
        r = PeerComparisonReport()
        assert r.peer_names == []
        assert r.comparisons == []
        assert r.rankings == {}


class TestRatioDecompositionNodeDataclass:
    def test_defaults(self):
        n = RatioDecompositionNode()
        assert n.children == []
        assert n.value is None

    def test_nested(self):
        child = RatioDecompositionNode(name="child", value=0.5)
        parent = RatioDecompositionNode(name="parent", children=[child])
        assert len(parent.children) == 1
        assert parent.children[0].name == "child"


class TestRatioDecompositionTreeDataclass:
    def test_defaults(self):
        t = RatioDecompositionTree()
        assert t.root is None
        assert t.summary == ""


# ===== PEER COMPARISON TESTS =====


class TestPeerComparison:
    def test_returns_report(self, analyzer, sample_data, peer_data):
        peers = [
            PeerCompanyData(name="Alpha", data=sample_data),
            PeerCompanyData(name="Beta", data=peer_data),
        ]
        result = analyzer.peer_comparison(peers)
        assert isinstance(result, PeerComparisonReport)

    def test_two_peers(self, analyzer, sample_data, peer_data):
        peers = [
            PeerCompanyData(name="Alpha", data=sample_data),
            PeerCompanyData(name="Beta", data=peer_data),
        ]
        result = analyzer.peer_comparison(peers)
        assert len(result.peer_names) == 2
        assert "Alpha" in result.peer_names
        assert "Beta" in result.peer_names

    def test_default_metrics_count(self, analyzer, sample_data, peer_data):
        peers = [
            PeerCompanyData(name="A", data=sample_data),
            PeerCompanyData(name="B", data=peer_data),
        ]
        result = analyzer.peer_comparison(peers)
        assert len(result.comparisons) == 9  # default metrics list

    def test_custom_metrics(self, analyzer, sample_data, peer_data):
        peers = [
            PeerCompanyData(name="A", data=sample_data),
            PeerCompanyData(name="B", data=peer_data),
        ]
        result = analyzer.peer_comparison(peers, metrics=["roe", "roa"])
        assert len(result.comparisons) == 2

    def test_best_performer_higher_is_better(self, analyzer, sample_data, peer_data):
        # sample_data has higher ROE: 150k/1.2M = 0.125 vs 70k/800k = 0.0875
        peers = [
            PeerCompanyData(name="Alpha", data=sample_data),
            PeerCompanyData(name="Beta", data=peer_data),
        ]
        result = analyzer.peer_comparison(peers, metrics=["roe"])
        assert result.comparisons[0].best_performer == "Alpha"

    def test_debt_to_equity_lower_is_better(self, analyzer, sample_data, peer_data):
        # sample_data: 400k/1.2M = 0.333, peer: 350k/800k = 0.4375
        # Lower is better -> Alpha is best
        peers = [
            PeerCompanyData(name="Alpha", data=sample_data),
            PeerCompanyData(name="Beta", data=peer_data),
        ]
        result = analyzer.peer_comparison(peers, metrics=["debt_to_equity"])
        assert result.comparisons[0].best_performer == "Alpha"

    def test_rankings(self, analyzer, sample_data, peer_data):
        peers = [
            PeerCompanyData(name="Alpha", data=sample_data),
            PeerCompanyData(name="Beta", data=peer_data),
        ]
        result = analyzer.peer_comparison(peers)
        assert 1 in result.rankings.values()
        assert 2 in result.rankings.values()

    def test_average_computed(self, analyzer, sample_data, peer_data):
        peers = [
            PeerCompanyData(name="A", data=sample_data),
            PeerCompanyData(name="B", data=peer_data),
        ]
        result = analyzer.peer_comparison(peers, metrics=["gross_margin"])
        comp = result.comparisons[0]
        assert comp.average is not None
        # (0.4 + 0.35) / 2 = 0.375
        assert abs(comp.average - 0.375) < 0.01

    def test_summary_present(self, analyzer, sample_data, peer_data):
        peers = [
            PeerCompanyData(name="A", data=sample_data),
            PeerCompanyData(name="B", data=peer_data),
        ]
        result = analyzer.peer_comparison(peers)
        assert "2 peers" in result.summary
        assert "metrics" in result.summary

    def test_empty_peers(self, analyzer):
        result = analyzer.peer_comparison([])
        assert "No peers" in result.summary

    def test_three_peers(self, analyzer, sample_data, peer_data):
        third = FinancialData(
            revenue=1_200_000,
            net_income=200_000,
            total_equity=900_000,
            total_assets=2_500_000,
        )
        peers = [
            PeerCompanyData(name="A", data=sample_data),
            PeerCompanyData(name="B", data=peer_data),
            PeerCompanyData(name="C", data=third),
        ]
        result = analyzer.peer_comparison(peers, metrics=["roe"])
        assert len(result.peer_names) == 3
        # C has highest ROE: 200k/900k = 0.222
        assert result.comparisons[0].best_performer == "C"

    def test_none_data_handled(self, analyzer, sample_data):
        peers = [
            PeerCompanyData(name="A", data=sample_data),
            PeerCompanyData(name="B", data=None),
        ]
        result = analyzer.peer_comparison(peers, metrics=["roe"])
        assert result.comparisons[0].values["B"] is None

    def test_auto_naming(self, analyzer, sample_data, peer_data):
        """Peers without names get auto-named."""
        peers = [
            PeerCompanyData(data=sample_data),
            PeerCompanyData(data=peer_data),
        ]
        result = analyzer.peer_comparison(peers)
        assert "Peer 1" in result.peer_names
        assert "Peer 2" in result.peer_names


# ===== COMPUTE PEER METRIC TESTS =====


class TestComputePeerMetric:
    def test_current_ratio(self, analyzer, sample_data):
        val = analyzer._compute_peer_metric(sample_data, "current_ratio")
        assert abs(val - 2.5) < 0.01  # 500k / 200k

    def test_gross_margin(self, analyzer, sample_data):
        val = analyzer._compute_peer_metric(sample_data, "gross_margin")
        assert abs(val - 0.4) < 0.01

    def test_roe(self, analyzer, sample_data):
        val = analyzer._compute_peer_metric(sample_data, "roe")
        assert abs(val - 0.125) < 0.01  # 150k / 1.2M

    def test_unknown_metric(self, analyzer, sample_data):
        val = analyzer._compute_peer_metric(sample_data, "nonexistent")
        assert val is None

    def test_none_data(self, analyzer):
        val = analyzer._compute_peer_metric(None, "roe")
        assert val is None

    def test_ebitda_margin(self, analyzer, sample_data):
        val = analyzer._compute_peer_metric(sample_data, "ebitda_margin")
        assert abs(val - 0.25) < 0.01  # 250k / 1M


# ===== RATIO DECOMPOSITION TESTS =====


class TestRatioDecomposition:
    def test_returns_tree(self, analyzer, sample_data):
        result = analyzer.ratio_decomposition(sample_data)
        assert isinstance(result, RatioDecompositionTree)
        assert result.root is not None

    def test_root_is_roe(self, analyzer, sample_data):
        result = analyzer.ratio_decomposition(sample_data)
        assert result.root.name == "ROE"
        # ROE = 150k / 1.2M = 0.125
        assert abs(result.root.value - 0.125) < 0.01

    def test_three_dupont_factors(self, analyzer, sample_data):
        result = analyzer.ratio_decomposition(sample_data)
        children = result.root.children
        assert len(children) == 3
        names = [c.name for c in children]
        assert "Net Profit Margin" in names
        assert "Asset Turnover" in names
        assert "Equity Multiplier" in names

    def test_net_margin_value(self, analyzer, sample_data):
        result = analyzer.ratio_decomposition(sample_data)
        nm = [c for c in result.root.children if c.name == "Net Profit Margin"][0]
        assert abs(nm.value - 0.15) < 0.01  # 150k / 1M

    def test_asset_turnover_value(self, analyzer, sample_data):
        result = analyzer.ratio_decomposition(sample_data)
        at = [c for c in result.root.children if c.name == "Asset Turnover"][0]
        assert abs(at.value - 0.5) < 0.01  # 1M / 2M

    def test_equity_multiplier_value(self, analyzer, sample_data):
        result = analyzer.ratio_decomposition(sample_data)
        em = [c for c in result.root.children if c.name == "Equity Multiplier"][0]
        # 2M / 1.2M = 1.667
        assert abs(em.value - 1.667) < 0.01

    def test_dupont_identity(self, analyzer, sample_data):
        """ROE = Net Margin * Asset Turnover * Equity Multiplier."""
        result = analyzer.ratio_decomposition(sample_data)
        children = {c.name: c.value for c in result.root.children}
        product = children["Net Profit Margin"] * children["Asset Turnover"] * children["Equity Multiplier"]
        assert abs(product - result.root.value) < 0.001

    def test_net_margin_has_sub_drivers(self, analyzer, sample_data):
        result = analyzer.ratio_decomposition(sample_data)
        nm = [c for c in result.root.children if c.name == "Net Profit Margin"][0]
        assert len(nm.children) >= 3  # gross margin, opex ratio, interest burden
        child_names = [c.name for c in nm.children]
        assert "Gross Margin" in child_names
        assert "OpEx Ratio" in child_names

    def test_gross_margin_sub_driver(self, analyzer, sample_data):
        result = analyzer.ratio_decomposition(sample_data)
        nm = [c for c in result.root.children if c.name == "Net Profit Margin"][0]
        gm = [c for c in nm.children if c.name == "Gross Margin"][0]
        assert abs(gm.value - 0.4) < 0.01  # 400k / 1M

    def test_asset_turnover_has_sub_drivers(self, analyzer, sample_data):
        result = analyzer.ratio_decomposition(sample_data)
        at = [c for c in result.root.children if c.name == "Asset Turnover"][0]
        assert len(at.children) >= 2
        child_names = [c.name for c in at.children]
        assert "Receivables Turnover" in child_names

    def test_equity_multiplier_has_sub_drivers(self, analyzer, sample_data):
        result = analyzer.ratio_decomposition(sample_data)
        em = [c for c in result.root.children if c.name == "Equity Multiplier"][0]
        assert len(em.children) >= 2
        child_names = [c.name for c in em.children]
        assert "Debt Ratio" in child_names
        assert "Debt-to-Equity" in child_names

    def test_tax_retention_included_when_ebt(self, analyzer, sample_data):
        result = analyzer.ratio_decomposition(sample_data)
        nm = [c for c in result.root.children if c.name == "Net Profit Margin"][0]
        child_names = [c.name for c in nm.children]
        assert "Tax Retention" in child_names
        tr = [c for c in nm.children if c.name == "Tax Retention"][0]
        # 150k / 180k = 0.833
        assert abs(tr.value - 0.833) < 0.01

    def test_summary_contains_roe(self, analyzer, sample_data):
        result = analyzer.ratio_decomposition(sample_data)
        assert "ROE" in result.summary

    def test_empty_data(self, analyzer):
        result = analyzer.ratio_decomposition(FinancialData())
        assert result.root is not None
        assert result.root.value is None
        assert "Insufficient" in result.summary


# ===== EDGE CASES =====


class TestPhase11EdgeCases:
    def test_single_peer(self, analyzer, sample_data):
        peers = [PeerCompanyData(name="Solo", data=sample_data)]
        result = analyzer.peer_comparison(peers)
        assert len(result.peer_names) == 1

    def test_decomposition_missing_ebt(self, analyzer):
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_equity=500_000,
            total_assets=1_000_000,
        )
        result = analyzer.ratio_decomposition(data)
        nm = [c for c in result.root.children if c.name == "Net Profit Margin"][0]
        # No ebt -> no Tax Retention child
        child_names = [c.name for c in nm.children]
        assert "Tax Retention" not in child_names

    def test_decomposition_zero_equity(self, analyzer):
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_equity=0,
            total_assets=1_000_000,
        )
        result = analyzer.ratio_decomposition(data)
        assert result.root.value is None  # division by zero -> None

    def test_peer_all_none_data(self, analyzer):
        peers = [
            PeerCompanyData(name="A", data=None),
            PeerCompanyData(name="B", data=None),
        ]
        result = analyzer.peer_comparison(peers, metrics=["roe"])
        assert result.comparisons[0].average is None
