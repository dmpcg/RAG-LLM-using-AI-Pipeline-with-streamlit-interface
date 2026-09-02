"""Tests for graph_schema.py Cypher templates — Phases 2, 4, and 7.

Covers templates that had zero direct test coverage:
  Phase 2: MERGE_FINANCIAL_STATEMENT, MERGE_LINE_ITEMS_BATCH, MERGE_DERIVED_FROM_BATCH
  Phase 4: MERGE_TEMPORAL_EDGES, CROSS_PERIOD_RATIO_TREND
  Phase 7: MERGE_PORTFOLIO, MERGE_PORTFOLIO_MEMBERSHIP_BATCH, MERGE_PORTFOLIO_RISK,
           MERGE_COMPLIANCE_REPORT_BATCH, COMPLIANCE_BY_COMPANY
"""


# ---------------------------------------------------------------------------
# Phase 2: Structured financial data population
# ---------------------------------------------------------------------------


class TestMergeFinancialStatement:
    def test_has_required_params(self):
        from graph_schema import MERGE_FINANCIAL_STATEMENT

        assert "$stmt_id" in MERGE_FINANCIAL_STATEMENT
        assert "$stmt_type" in MERGE_FINANCIAL_STATEMENT
        assert "$period_id" in MERGE_FINANCIAL_STATEMENT

    def test_node_and_relationship(self):
        from graph_schema import MERGE_FINANCIAL_STATEMENT

        assert ":FinancialStatement" in MERGE_FINANCIAL_STATEMENT
        assert "HAS_STATEMENT" in MERGE_FINANCIAL_STATEMENT
        assert ":FiscalPeriod" in MERGE_FINANCIAL_STATEMENT

    def test_uses_merge_not_create(self):
        from graph_schema import MERGE_FINANCIAL_STATEMENT

        assert "MERGE" in MERGE_FINANCIAL_STATEMENT


class TestMergeLineItemsBatch:
    def test_uses_unwind(self):
        from graph_schema import MERGE_LINE_ITEMS_BATCH

        assert "UNWIND" in MERGE_LINE_ITEMS_BATCH
        assert "$batch" in MERGE_LINE_ITEMS_BATCH

    def test_has_required_fields(self):
        from graph_schema import MERGE_LINE_ITEMS_BATCH

        assert "item_id" in MERGE_LINE_ITEMS_BATCH
        assert "name" in MERGE_LINE_ITEMS_BATCH
        assert "value" in MERGE_LINE_ITEMS_BATCH
        assert "unit" in MERGE_LINE_ITEMS_BATCH

    def test_node_and_relationship(self):
        from graph_schema import MERGE_LINE_ITEMS_BATCH

        assert ":LineItem" in MERGE_LINE_ITEMS_BATCH
        assert ":FinancialStatement" in MERGE_LINE_ITEMS_BATCH
        assert "CONTAINS" in MERGE_LINE_ITEMS_BATCH

    def test_returns_count(self):
        from graph_schema import MERGE_LINE_ITEMS_BATCH

        assert "count(" in MERGE_LINE_ITEMS_BATCH


class TestMergeDerivedFromBatch:
    def test_uses_unwind(self):
        from graph_schema import MERGE_DERIVED_FROM_BATCH

        assert "UNWIND" in MERGE_DERIVED_FROM_BATCH
        assert "$batch" in MERGE_DERIVED_FROM_BATCH

    def test_links_ratio_to_line_item(self):
        from graph_schema import MERGE_DERIVED_FROM_BATCH

        assert ":FinancialRatio" in MERGE_DERIVED_FROM_BATCH
        assert ":LineItem" in MERGE_DERIVED_FROM_BATCH
        assert "DERIVED_FROM" in MERGE_DERIVED_FROM_BATCH

    def test_has_role_property(self):
        from graph_schema import MERGE_DERIVED_FROM_BATCH

        assert "role" in MERGE_DERIVED_FROM_BATCH

    def test_returns_count(self):
        from graph_schema import MERGE_DERIVED_FROM_BATCH

        assert "count(" in MERGE_DERIVED_FROM_BATCH


# ---------------------------------------------------------------------------
# Phase 4: Temporal edges and cross-period queries
# ---------------------------------------------------------------------------


class TestMergeTemporalEdges:
    def test_uses_unwind(self):
        from graph_schema import MERGE_TEMPORAL_EDGES

        assert "UNWIND" in MERGE_TEMPORAL_EDGES
        assert "$pairs" in MERGE_TEMPORAL_EDGES

    def test_creates_bidirectional_edges(self):
        from graph_schema import MERGE_TEMPORAL_EDGES

        assert "PRECEDES" in MERGE_TEMPORAL_EDGES
        assert "FOLLOWS" in MERGE_TEMPORAL_EDGES

    def test_targets_fiscal_periods(self):
        from graph_schema import MERGE_TEMPORAL_EDGES

        assert ":FiscalPeriod" in MERGE_TEMPORAL_EDGES

    def test_has_earlier_later_params(self):
        from graph_schema import MERGE_TEMPORAL_EDGES

        assert "earlier_id" in MERGE_TEMPORAL_EDGES
        assert "later_id" in MERGE_TEMPORAL_EDGES

    def test_returns_count(self):
        from graph_schema import MERGE_TEMPORAL_EDGES

        assert "count(" in MERGE_TEMPORAL_EDGES


class TestCrossPeriodRatioTrend:
    def test_has_period_labels_param(self):
        from graph_schema import CROSS_PERIOD_RATIO_TREND

        assert "$period_labels" in CROSS_PERIOD_RATIO_TREND

    def test_queries_ratios_across_periods(self):
        from graph_schema import CROSS_PERIOD_RATIO_TREND

        assert ":FiscalPeriod" in CROSS_PERIOD_RATIO_TREND
        assert "HAS_RATIO" in CROSS_PERIOD_RATIO_TREND
        assert ":FinancialRatio" in CROSS_PERIOD_RATIO_TREND

    def test_returns_expected_columns(self):
        from graph_schema import CROSS_PERIOD_RATIO_TREND

        assert "period" in CROSS_PERIOD_RATIO_TREND
        assert "ratio_name" in CROSS_PERIOD_RATIO_TREND
        assert "value" in CROSS_PERIOD_RATIO_TREND
        assert "category" in CROSS_PERIOD_RATIO_TREND

    def test_orders_by_ratio_name(self):
        from graph_schema import CROSS_PERIOD_RATIO_TREND

        assert "ORDER BY" in CROSS_PERIOD_RATIO_TREND


# ---------------------------------------------------------------------------
# Phase 7: Portfolio graph nodes
# ---------------------------------------------------------------------------


class TestMergePortfolio:
    def test_has_required_params(self):
        from graph_schema import MERGE_PORTFOLIO

        assert "$portfolio_id" in MERGE_PORTFOLIO
        assert "$name" in MERGE_PORTFOLIO

    def test_node_type(self):
        from graph_schema import MERGE_PORTFOLIO

        assert ":Portfolio" in MERGE_PORTFOLIO

    def test_sets_timestamp(self):
        from graph_schema import MERGE_PORTFOLIO

        assert "datetime()" in MERGE_PORTFOLIO


class TestMergePortfolioMembershipBatch:
    def test_uses_unwind(self):
        from graph_schema import MERGE_PORTFOLIO_MEMBERSHIP_BATCH

        assert "UNWIND" in MERGE_PORTFOLIO_MEMBERSHIP_BATCH
        assert "$company_names" in MERGE_PORTFOLIO_MEMBERSHIP_BATCH

    def test_links_portfolio_to_company(self):
        from graph_schema import MERGE_PORTFOLIO_MEMBERSHIP_BATCH

        assert ":Portfolio" in MERGE_PORTFOLIO_MEMBERSHIP_BATCH
        assert ":Company" in MERGE_PORTFOLIO_MEMBERSHIP_BATCH
        assert "CONTAINS_COMPANY" in MERGE_PORTFOLIO_MEMBERSHIP_BATCH

    def test_has_portfolio_id_param(self):
        from graph_schema import MERGE_PORTFOLIO_MEMBERSHIP_BATCH

        assert "$portfolio_id" in MERGE_PORTFOLIO_MEMBERSHIP_BATCH

    def test_returns_count(self):
        from graph_schema import MERGE_PORTFOLIO_MEMBERSHIP_BATCH

        assert "count(" in MERGE_PORTFOLIO_MEMBERSHIP_BATCH


class TestMergePortfolioRisk:
    def test_has_required_params(self):
        from graph_schema import MERGE_PORTFOLIO_RISK

        assert "$risk_id" in MERGE_PORTFOLIO_RISK
        assert "$avg_health" in MERGE_PORTFOLIO_RISK
        assert "$min_health" in MERGE_PORTFOLIO_RISK
        assert "$max_health" in MERGE_PORTFOLIO_RISK
        assert "$risk_level" in MERGE_PORTFOLIO_RISK
        assert "$distress_count" in MERGE_PORTFOLIO_RISK
        assert "$diversification_score" in MERGE_PORTFOLIO_RISK
        assert "$risk_flags" in MERGE_PORTFOLIO_RISK

    def test_node_and_relationship(self):
        from graph_schema import MERGE_PORTFOLIO_RISK

        assert ":PortfolioRisk" in MERGE_PORTFOLIO_RISK
        assert ":Portfolio" in MERGE_PORTFOLIO_RISK
        assert "HAS_PORTFOLIO_RISK" in MERGE_PORTFOLIO_RISK

    def test_links_to_portfolio(self):
        from graph_schema import MERGE_PORTFOLIO_RISK

        assert "$portfolio_id" in MERGE_PORTFOLIO_RISK


# ---------------------------------------------------------------------------
# Phase 7: Compliance graph nodes
# ---------------------------------------------------------------------------


class TestMergeComplianceReportBatch:
    def test_uses_unwind(self):
        from graph_schema import MERGE_COMPLIANCE_REPORT_BATCH

        assert "UNWIND" in MERGE_COMPLIANCE_REPORT_BATCH
        assert "$batch" in MERGE_COMPLIANCE_REPORT_BATCH

    def test_has_compliance_fields(self):
        from graph_schema import MERGE_COMPLIANCE_REPORT_BATCH

        assert "sox_risk" in MERGE_COMPLIANCE_REPORT_BATCH
        assert "sox_score" in MERGE_COMPLIANCE_REPORT_BATCH
        assert "sec_score" in MERGE_COMPLIANCE_REPORT_BATCH
        assert "regulatory_pct" in MERGE_COMPLIANCE_REPORT_BATCH
        assert "audit_risk" in MERGE_COMPLIANCE_REPORT_BATCH
        assert "audit_score" in MERGE_COMPLIANCE_REPORT_BATCH
        assert "going_concern" in MERGE_COMPLIANCE_REPORT_BATCH

    def test_node_and_relationship(self):
        from graph_schema import MERGE_COMPLIANCE_REPORT_BATCH

        assert ":ComplianceReport" in MERGE_COMPLIANCE_REPORT_BATCH
        assert ":Company" in MERGE_COMPLIANCE_REPORT_BATCH
        assert "HAS_COMPLIANCE_REPORT" in MERGE_COMPLIANCE_REPORT_BATCH

    def test_returns_count(self):
        from graph_schema import MERGE_COMPLIANCE_REPORT_BATCH

        assert "count(" in MERGE_COMPLIANCE_REPORT_BATCH


class TestComplianceByCompany:
    def test_has_company_name_param(self):
        from graph_schema import COMPLIANCE_BY_COMPANY

        assert "$company_name" in COMPLIANCE_BY_COMPANY

    def test_returns_expected_columns(self):
        from graph_schema import COMPLIANCE_BY_COMPANY

        assert "compliance_id" in COMPLIANCE_BY_COMPANY
        assert "sox_risk" in COMPLIANCE_BY_COMPANY
        assert "sox_score" in COMPLIANCE_BY_COMPANY
        assert "sec_score" in COMPLIANCE_BY_COMPANY
        assert "regulatory_pct" in COMPLIANCE_BY_COMPANY
        assert "audit_risk" in COMPLIANCE_BY_COMPANY
        assert "audit_score" in COMPLIANCE_BY_COMPANY
        assert "going_concern" in COMPLIANCE_BY_COMPANY

    def test_relationship_traversal(self):
        from graph_schema import COMPLIANCE_BY_COMPANY

        assert ":Company" in COMPLIANCE_BY_COMPANY
        assert "HAS_COMPLIANCE_REPORT" in COMPLIANCE_BY_COMPANY
        assert ":ComplianceReport" in COMPLIANCE_BY_COMPANY

    def test_orders_results(self):
        from graph_schema import COMPLIANCE_BY_COMPANY

        assert "ORDER BY" in COMPLIANCE_BY_COMPANY


# ---------------------------------------------------------------------------
# Constraint completeness
# ---------------------------------------------------------------------------


class TestConstraintCompleteness:
    def test_has_all_13_constraints(self):
        from graph_schema import CONSTRAINTS

        assert len(CONSTRAINTS) == 13

    def test_portfolio_constraints_present(self):
        from graph_schema import CONSTRAINTS

        text = " ".join(CONSTRAINTS)
        assert "Portfolio" in text
        assert "PortfolioRisk" in text
        assert "ComplianceReport" in text

    def test_all_node_types_have_constraints(self):
        from graph_schema import CONSTRAINTS

        text = " ".join(CONSTRAINTS)
        expected_types = [
            "Document",
            "Chunk",
            "FiscalPeriod",
            "FinancialStatement",
            "LineItem",
            "FinancialRatio",
            "ScoringResult",
            "CreditAssessment",
            "CovenantPackage",
            "Company",
            "Portfolio",
            "PortfolioRisk",
            "ComplianceReport",
        ]
        for node_type in expected_types:
            assert node_type in text, f"Missing constraint for {node_type}"

    def test_all_templates_are_strings(self):
        """Every exported Cypher template should be a non-empty string."""
        import graph_schema

        template_names = [
            "MERGE_DOCUMENT",
            "MERGE_CHUNK",
            "MERGE_CHUNKS_BATCH",
            "MERGE_FISCAL_PERIOD",
            "MERGE_LINE_ITEM",
            "MERGE_RATIO",
            "MERGE_RATIOS_BATCH",
            "MERGE_SCORE",
            "MERGE_SCORES_BATCH",
            "MERGE_FINANCIAL_STATEMENT",
            "MERGE_LINE_ITEMS_BATCH",
            "MERGE_DERIVED_FROM_BATCH",
            "MERGE_TEMPORAL_EDGES",
            "CROSS_PERIOD_RATIO_TREND",
            "MERGE_PORTFOLIO",
            "MERGE_PORTFOLIO_MEMBERSHIP_BATCH",
            "MERGE_PORTFOLIO_RISK",
            "MERGE_COMPLIANCE_REPORT_BATCH",
            "COMPLIANCE_BY_COMPANY",
            "MERGE_COMPANY",
            "MERGE_CREDIT_ASSESSMENT",
            "MERGE_CREDIT_ASSESSMENTS_BATCH",
            "MERGE_COVENANT_PACKAGE",
            "MERGE_COVENANT_PACKAGES_BATCH",
            "CREDIT_ASSESSMENT_BY_COMPANY",
            "VECTOR_SEARCH",
            "GRAPH_CONTEXT_FOR_CHUNK",
            "GRAPH_CONTEXT_FOR_CHUNKS_BATCH",
            "RATIOS_BY_PERIOD",
            "SCORES_BY_PERIOD",
            "RATIOS_BY_PERIOD_LABEL",
            "SCORES_BY_PERIOD_LABEL",
        ]
        for name in template_names:
            tmpl = getattr(graph_schema, name)
            assert isinstance(tmpl, str), f"{name} is not a string"
            assert len(tmpl.strip()) > 10, f"{name} is too short"
