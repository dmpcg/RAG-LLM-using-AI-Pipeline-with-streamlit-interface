"""Tests for CreditAssessment and CovenantPackage Neo4j graph nodes.

Covers:
  - graph_schema Cypher template presence and parameterisation
  - Neo4jStore.store_credit_assessment (UNWIND batch, Company node, relationship)
  - Neo4jStore.store_covenant_package  (UNWIND batch, relationship to assessment)
  - Graceful failure on Neo4j errors
  - End-to-end: store assessment -> store covenants -> package_id returned
"""

import hashlib
import json
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


@pytest.fixture()
def store(mock_driver):
    from graph_store import Neo4jStore
    driver, _ = mock_driver
    return Neo4jStore(driver)


@pytest.fixture()
def sample_scorecard():
    from underwriting import CreditScorecard
    return CreditScorecard(
        total_score=72,
        grade="B",
        category_scores={
            "profitability": 15,
            "leverage": 15,
            "liquidity": 15,
            "cash_flow": 12,
            "stability": 15,
        },
        recommendation="approve",
        conditions=[],
        strengths=["profitability", "leverage", "liquidity", "stability"],
        weaknesses=[],
    )


@pytest.fixture()
def sample_debt_capacity():
    from underwriting import DebtCapacityResult
    return DebtCapacityResult(
        current_total_debt=1_000_000.0,
        current_ebitda=500_000.0,
        current_leverage=2.0,
        max_leverage_target=3.5,
        max_additional_debt=750_000.0,
        assessment="Current leverage is 2.00x (target max 3.5x). Additional capacity: $750,000.",
    )


@pytest.fixture()
def sample_covenant_package():
    from underwriting import CovenantPackage
    return CovenantPackage(
        covenant_tier="standard",
        financial_covenants={
            "min_current_ratio": {"threshold": 1.25, "frequency": "quarterly",
                                  "description": "Minimum current ratio"},
            "max_debt_to_ebitda": {"threshold": 3.5, "frequency": "quarterly",
                                   "description": "Maximum total debt to EBITDA"},
            "min_interest_coverage": {"threshold": 2.5, "frequency": "quarterly",
                                      "description": "Minimum interest coverage ratio"},
        },
        reporting_requirements=[
            "Quarterly unaudited financial statements within 45 days of quarter-end",
            "Annual audited financial statements within 90 days of fiscal year-end",
        ],
        events_of_default=[
            "Payment default (principal or interest)",
            "Financial covenant breach not cured within 30 days",
            "Material adverse change in financial condition",
            "Cross-default on other indebtedness exceeding $500,000",
        ],
    )


# ---------------------------------------------------------------------------
# graph_schema: CreditAssessment Cypher templates
# ---------------------------------------------------------------------------


class TestGraphSchemaCreditTemplates:
    """Verify all new credit-related Cypher templates exist and are valid."""

    def test_merge_company_template_exists(self):
        from graph_schema import MERGE_COMPANY
        assert isinstance(MERGE_COMPANY, str) and len(MERGE_COMPANY) > 10
        assert "MERGE" in MERGE_COMPANY
        assert ":Company" in MERGE_COMPANY

    def test_merge_credit_assessment_template_parameterised(self):
        from graph_schema import MERGE_CREDIT_ASSESSMENT
        # Must not contain string interpolation -- only $param placeholders
        assert "$assessment_id" in MERGE_CREDIT_ASSESSMENT
        assert "$total_score" in MERGE_CREDIT_ASSESSMENT
        assert "$grade" in MERGE_CREDIT_ASSESSMENT
        assert "$recommendation" in MERGE_CREDIT_ASSESSMENT
        assert "$category_scores" in MERGE_CREDIT_ASSESSMENT
        assert "$strengths" in MERGE_CREDIT_ASSESSMENT
        assert "$weaknesses" in MERGE_CREDIT_ASSESSMENT
        assert "$max_additional_debt" in MERGE_CREDIT_ASSESSMENT
        assert "$current_leverage" in MERGE_CREDIT_ASSESSMENT
        assert "$company_name" in MERGE_CREDIT_ASSESSMENT
        assert ":CreditAssessment" in MERGE_CREDIT_ASSESSMENT
        assert "HAS_CREDIT_ASSESSMENT" in MERGE_CREDIT_ASSESSMENT

    def test_merge_credit_assessments_batch_uses_unwind(self):
        from graph_schema import MERGE_CREDIT_ASSESSMENTS_BATCH
        assert "UNWIND" in MERGE_CREDIT_ASSESSMENTS_BATCH
        assert "$batch" in MERGE_CREDIT_ASSESSMENTS_BATCH
        assert ":CreditAssessment" in MERGE_CREDIT_ASSESSMENTS_BATCH
        assert "HAS_CREDIT_ASSESSMENT" in MERGE_CREDIT_ASSESSMENTS_BATCH

    def test_merge_covenant_package_template_parameterised(self):
        from graph_schema import MERGE_COVENANT_PACKAGE
        assert "$package_id" in MERGE_COVENANT_PACKAGE
        assert "$covenant_tier" in MERGE_COVENANT_PACKAGE
        assert "$financial_covenants" in MERGE_COVENANT_PACKAGE
        assert "$reporting_requirements" in MERGE_COVENANT_PACKAGE
        assert "$events_of_default" in MERGE_COVENANT_PACKAGE
        assert "$assessment_id" in MERGE_COVENANT_PACKAGE
        assert ":CovenantPackage" in MERGE_COVENANT_PACKAGE
        assert "REQUIRES_COVENANTS" in MERGE_COVENANT_PACKAGE

    def test_merge_covenant_packages_batch_uses_unwind(self):
        from graph_schema import MERGE_COVENANT_PACKAGES_BATCH
        assert "UNWIND" in MERGE_COVENANT_PACKAGES_BATCH
        assert "$batch" in MERGE_COVENANT_PACKAGES_BATCH
        assert ":CovenantPackage" in MERGE_COVENANT_PACKAGES_BATCH
        assert "REQUIRES_COVENANTS" in MERGE_COVENANT_PACKAGES_BATCH

    def test_credit_assessment_by_company_query_parameterised(self):
        from graph_schema import CREDIT_ASSESSMENT_BY_COMPANY
        assert "$company_name" in CREDIT_ASSESSMENT_BY_COMPANY
        assert ":Company" in CREDIT_ASSESSMENT_BY_COMPANY
        assert "HAS_CREDIT_ASSESSMENT" in CREDIT_ASSESSMENT_BY_COMPANY
        assert "REQUIRES_COVENANTS" in CREDIT_ASSESSMENT_BY_COMPANY

    def test_constraints_include_credit_types(self):
        from graph_schema import CONSTRAINTS
        constraint_text = " ".join(CONSTRAINTS)
        assert "CreditAssessment" in constraint_text
        assert "CovenantPackage" in constraint_text
        assert "Company" in constraint_text
        # All must be idempotent
        for c in CONSTRAINTS:
            assert "IF NOT EXISTS" in c


# ---------------------------------------------------------------------------
# Neo4jStore.store_credit_assessment
# ---------------------------------------------------------------------------


class TestStoreCreditAssessment:
    def test_returns_assessment_id_on_success(
        self, store, mock_driver, sample_scorecard, sample_debt_capacity
    ):
        _, session = mock_driver
        assessment_id = store.store_credit_assessment(
            "Acme Corp", sample_scorecard, sample_debt_capacity
        )
        assert assessment_id is not None
        assert isinstance(assessment_id, str)
        assert len(assessment_id) == 64  # sha256 hex digest

    def test_creates_company_node_then_assessment_batch(
        self, store, mock_driver, sample_scorecard, sample_debt_capacity
    ):
        _, session = mock_driver
        store.store_credit_assessment("Acme Corp", sample_scorecard, sample_debt_capacity)
        # Exactly 2 session.run calls: MERGE_COMPANY, then MERGE_CREDIT_ASSESSMENTS_BATCH
        assert session.run.call_count == 2

    def test_assessment_batch_contains_correct_fields(
        self, store, mock_driver, sample_scorecard, sample_debt_capacity
    ):
        _, session = mock_driver
        store.store_credit_assessment("Acme Corp", sample_scorecard, sample_debt_capacity)
        # Second call is the UNWIND batch -- grab its kwargs
        second_call_kwargs = session.run.call_args_list[1]
        batch_arg = second_call_kwargs[1].get("batch") or second_call_kwargs[0][1]
        assert len(batch_arg) == 1
        row = batch_arg[0]
        assert row["total_score"] == 72
        assert row["grade"] == "B"
        assert row["recommendation"] == "approve"
        assert row["company_name"] == "Acme Corp"
        assert isinstance(row["category_scores"], str)  # serialised JSON
        # Verify the JSON is parseable and correct
        parsed = json.loads(row["category_scores"])
        assert parsed["profitability"] == 15
        assert row["strengths"] == ["profitability", "leverage", "liquidity", "stability"]
        assert row["weaknesses"] == []
        assert row["max_additional_debt"] == pytest.approx(750_000.0)
        assert row["current_leverage"] == pytest.approx(2.0)

    def test_assessment_id_is_deterministic(
        self, store, mock_driver, sample_scorecard, sample_debt_capacity
    ):
        _, session = mock_driver
        id1 = store.store_credit_assessment("Acme Corp", sample_scorecard, sample_debt_capacity)
        id2 = store.store_credit_assessment("Acme Corp", sample_scorecard, sample_debt_capacity)
        assert id1 == id2

    def test_different_company_produces_different_id(
        self, store, mock_driver, sample_scorecard, sample_debt_capacity
    ):
        _, session = mock_driver
        id1 = store.store_credit_assessment("Acme Corp", sample_scorecard, sample_debt_capacity)
        # Reset session mock state
        session.run.reset_mock()
        id2 = store.store_credit_assessment("Beta LLC", sample_scorecard, sample_debt_capacity)
        assert id1 != id2

    def test_handles_none_debt_capacity_fields(self, store, mock_driver, sample_scorecard):
        from underwriting import DebtCapacityResult
        _, session = mock_driver
        sparse_dc = DebtCapacityResult()  # all fields None / defaults
        assessment_id = store.store_credit_assessment("NullCo", sample_scorecard, sparse_dc)
        assert assessment_id is not None
        batch_arg = session.run.call_args_list[1][1].get("batch") or \
                    session.run.call_args_list[1][0][1]
        row = batch_arg[0]
        assert row["max_additional_debt"] is None
        assert row["current_leverage"] is None

    def test_returns_none_on_neo4j_failure(
        self, store, mock_driver, sample_scorecard, sample_debt_capacity
    ):
        # WS-1 P0-3 (2026-05-07): transient failures must raise
        # Neo4jTransientError so callers can retry instead of treating None
        # as "nothing to write" (silent data loss).
        from graph_store import Neo4jTransientError
        _, session = mock_driver
        session.run.side_effect = ConnectionError("Neo4j unavailable")
        with pytest.raises(Neo4jTransientError):
            store.store_credit_assessment("Acme Corp", sample_scorecard, sample_debt_capacity)

    def test_uses_unwind_batch_query(
        self, store, mock_driver, sample_scorecard, sample_debt_capacity
    ):
        """The second session.run call must use the UNWIND batch template."""
        _, session = mock_driver
        from graph_schema import MERGE_CREDIT_ASSESSMENTS_BATCH
        store.store_credit_assessment("Acme Corp", sample_scorecard, sample_debt_capacity)
        second_call = session.run.call_args_list[1]
        cypher_used = second_call[0][0]
        assert cypher_used == MERGE_CREDIT_ASSESSMENTS_BATCH


# ---------------------------------------------------------------------------
# Neo4jStore.store_covenant_package
# ---------------------------------------------------------------------------


class TestStoreCovenantPackage:
    def test_returns_package_id_on_success(
        self, store, mock_driver, sample_covenant_package
    ):
        _, session = mock_driver
        assessment_id = "a" * 64  # fake 64-char hex id
        package_id = store.store_covenant_package(assessment_id, sample_covenant_package)
        assert package_id is not None
        assert isinstance(package_id, str)
        assert len(package_id) == 64

    def test_creates_exactly_one_unwind_batch_call(
        self, store, mock_driver, sample_covenant_package
    ):
        _, session = mock_driver
        assessment_id = "b" * 64
        store.store_covenant_package(assessment_id, sample_covenant_package)
        # Only 1 session.run call (MERGE_COVENANT_PACKAGES_BATCH via UNWIND)
        assert session.run.call_count == 1

    def test_batch_contains_correct_fields(
        self, store, mock_driver, sample_covenant_package
    ):
        _, session = mock_driver
        assessment_id = "c" * 64
        store.store_covenant_package(assessment_id, sample_covenant_package)
        call_args = session.run.call_args_list[0]
        batch_arg = call_args[1].get("batch") or call_args[0][1]
        assert len(batch_arg) == 1
        row = batch_arg[0]
        assert row["assessment_id"] == assessment_id
        assert row["covenant_tier"] == "standard"
        assert isinstance(row["financial_covenants"], str)  # serialised JSON
        parsed = json.loads(row["financial_covenants"])
        assert "min_current_ratio" in parsed
        assert parsed["min_current_ratio"]["threshold"] == pytest.approx(1.25)
        assert isinstance(row["reporting_requirements"], list)
        assert len(row["reporting_requirements"]) == 2
        assert isinstance(row["events_of_default"], list)
        assert len(row["events_of_default"]) == 4

    def test_package_id_is_deterministic(
        self, store, mock_driver, sample_covenant_package
    ):
        _, session = mock_driver
        assessment_id = "d" * 64
        id1 = store.store_covenant_package(assessment_id, sample_covenant_package)
        session.run.reset_mock()
        id2 = store.store_covenant_package(assessment_id, sample_covenant_package)
        assert id1 == id2

    def test_different_tier_produces_different_package_id(
        self, store, mock_driver
    ):
        from underwriting import CovenantPackage
        _, session = mock_driver
        assessment_id = "e" * 64
        pkg_standard = CovenantPackage(covenant_tier="standard")
        pkg_heavy = CovenantPackage(covenant_tier="heavy")
        id1 = store.store_covenant_package(assessment_id, pkg_standard)
        session.run.reset_mock()
        id2 = store.store_covenant_package(assessment_id, pkg_heavy)
        assert id1 != id2

    def test_returns_none_on_neo4j_failure(
        self, store, mock_driver, sample_covenant_package
    ):
        # WS-1 P0-3: transient failures must raise Neo4jTransientError.
        from graph_store import Neo4jTransientError
        _, session = mock_driver
        session.run.side_effect = ConnectionError("Connection refused")
        with pytest.raises(Neo4jTransientError):
            store.store_covenant_package("f" * 64, sample_covenant_package)

    def test_uses_unwind_batch_query(
        self, store, mock_driver, sample_covenant_package
    ):
        """session.run must receive the UNWIND batch template."""
        _, session = mock_driver
        from graph_schema import MERGE_COVENANT_PACKAGES_BATCH
        assessment_id = "g" * 64
        store.store_covenant_package(assessment_id, sample_covenant_package)
        call_args = session.run.call_args_list[0]
        cypher_used = call_args[0][0]
        assert cypher_used == MERGE_COVENANT_PACKAGES_BATCH

    def test_handles_empty_covenant_package(self, store, mock_driver):
        from underwriting import CovenantPackage
        _, session = mock_driver
        empty_pkg = CovenantPackage()
        assessment_id = "h" * 64
        package_id = store.store_covenant_package(assessment_id, empty_pkg)
        assert package_id is not None
        batch_arg = session.run.call_args_list[0][1].get("batch") or \
                    session.run.call_args_list[0][0][1]
        row = batch_arg[0]
        assert row["covenant_tier"] == "standard"  # default
        assert row["reporting_requirements"] == []
        assert row["events_of_default"] == []


# ---------------------------------------------------------------------------
# Relationship creation: end-to-end assessment -> covenant chain
# ---------------------------------------------------------------------------


class TestCreditAssessmentCovenantChain:
    """Verify the full write chain: assessment stored, then covenant linked."""

    def test_full_chain_returns_both_ids(
        self,
        store,
        mock_driver,
        sample_scorecard,
        sample_debt_capacity,
        sample_covenant_package,
    ):
        _, session = mock_driver
        assessment_id = store.store_credit_assessment(
            "ChainCo", sample_scorecard, sample_debt_capacity
        )
        assert assessment_id is not None

        package_id = store.store_covenant_package(assessment_id, sample_covenant_package)
        assert package_id is not None
        assert assessment_id != package_id

    def test_full_chain_total_run_calls(
        self,
        store,
        mock_driver,
        sample_scorecard,
        sample_debt_capacity,
        sample_covenant_package,
    ):
        """Assessment: 2 calls (MERGE_COMPANY + batch). Package: 1 call (batch). Total = 3."""
        _, session = mock_driver
        assessment_id = store.store_credit_assessment(
            "ChainCo", sample_scorecard, sample_debt_capacity
        )
        store.store_covenant_package(assessment_id, sample_covenant_package)
        assert session.run.call_count == 3

    def test_covenant_batch_links_to_assessment_id(
        self,
        store,
        mock_driver,
        sample_scorecard,
        sample_debt_capacity,
        sample_covenant_package,
    ):
        """The covenant batch row must reference the assessment_id returned from step 1."""
        _, session = mock_driver
        assessment_id = store.store_credit_assessment(
            "LinkCo", sample_scorecard, sample_debt_capacity
        )
        store.store_covenant_package(assessment_id, sample_covenant_package)

        # Third call is the covenant batch
        third_call = session.run.call_args_list[2]
        batch_arg = third_call[1].get("batch") or third_call[0][1]
        assert batch_arg[0]["assessment_id"] == assessment_id

    def test_assessment_failure_does_not_prevent_package_attempt(
        self,
        store,
        mock_driver,
        sample_scorecard,
        sample_debt_capacity,
        sample_covenant_package,
    ):
        """WS-1 P0-3: when the assessment write fails transiently, the typed
        Neo4jTransientError now bubbles up so the caller can decide whether to
        retry; once the connection recovers the covenant write is independent
        and still succeeds.
        """
        from graph_store import Neo4jTransientError

        _, session = mock_driver
        session.run.side_effect = ConnectionError("DB down")
        with pytest.raises(Neo4jTransientError):
            store.store_credit_assessment(
                "FailCo", sample_scorecard, sample_debt_capacity
            )

        # Reset: let covenant store succeed
        session.run.side_effect = None
        fake_id = "0" * 64
        package_id = store.store_covenant_package(fake_id, sample_covenant_package)
        assert package_id is not None


# ---------------------------------------------------------------------------
# Safe attribute access pattern (getattr guard)
# ---------------------------------------------------------------------------


class TestSafeGraphStoreAccess:
    """Verify the getattr(self, '_graph_store', None) pattern works with Neo4jStore."""

    def test_graph_store_attribute_accessible_via_getattr(self, store):
        # Simulate an object that wraps a store but bypasses __init__
        class Wrapper:
            pass

        wrapper = Wrapper()
        # Not set: should return None safely
        assert getattr(wrapper, "_graph_store", None) is None

        # Set to the store fixture: should return the store
        wrapper._graph_store = store
        assert getattr(wrapper, "_graph_store", None) is store

    def test_store_credit_assessment_method_exists(self, store):
        assert callable(getattr(store, "store_credit_assessment", None))

    def test_store_covenant_package_method_exists(self, store):
        assert callable(getattr(store, "store_covenant_package", None))
