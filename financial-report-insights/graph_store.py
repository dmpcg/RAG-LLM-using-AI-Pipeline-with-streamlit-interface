"""
Neo4j graph+vector storage layer.

Optional backend that activates when NEO4J_URI is set.
Falls back gracefully so the system works identically without Neo4j.
"""

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Neo4jTransientError(RuntimeError):
    """Raised when a Neo4j write fails with a transient error.

    Distinct from permanent errors so callers can choose to retry the write
    rather than treating a "stored=0" return as "nothing to write" (which was
    silent data loss prior to 2026-05-07 / WS-1 P0-3).
    """

    def __init__(self, operation: str, original: BaseException):
        super().__init__(f"Neo4j transient failure in {operation}: {original}")
        self.operation = operation
        self.original = original


# Neo4j exception types for error classification
try:
    from neo4j.exceptions import (
        ServiceUnavailable,
        SessionExpired,
        TransientError,
    )

    _NEO4J_TRANSIENT = (ServiceUnavailable, SessionExpired, TransientError, ConnectionError, OSError)
except ImportError:
    _NEO4J_TRANSIENT = (ConnectionError, OSError)


def _neo4j_configured() -> bool:
    """Return True if NEO4J_URI is set in the environment."""
    return bool(os.environ.get("NEO4J_URI", "").strip())


class Neo4jStore:
    """Persistent graph+vector store backed by Neo4j.

    Usage::

        store = Neo4jStore.connect()  # returns None if NEO4J_URI not set
        if store:
            store.ensure_schema(embedding_dim=1024, model_name="mxbai-embed-large")
            store.store_chunks(chunks, embeddings, doc_id="report.pdf")
            results = store.vector_search(query_embedding, top_k=5)
            store.close()
    """

    def __init__(self, driver):
        self._driver = driver

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def connect(cls) -> Optional["Neo4jStore"]:
        """Create a Neo4jStore if NEO4J_URI is configured, else return None."""
        if not _neo4j_configured():
            return None

        try:
            import neo4j

            uri = os.environ["NEO4J_URI"]
            username = os.environ.get("NEO4J_USERNAME", "neo4j")
            password = os.environ.get("NEO4J_PASSWORD", "")
            if not password:
                logger.error("NEO4J_PASSWORD is not set. Refusing to connect without credentials.")
                return None
            driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
            try:
                driver.verify_connectivity()
            except Exception:
                # Close the driver to avoid leaking its connection pool before
                # propagating the failure to the outer fallback handler.
                driver.close()
                raise
            logger.info("Connected to Neo4j at %s", uri)
            return cls(driver)
        except Exception as exc:
            logger.warning("Neo4j connection failed (%s) – falling back to in-memory", exc)
            return None

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def ensure_schema(self, embedding_dim: int, model_name: str) -> None:
        """Create constraints and vector index if they don't exist."""
        from graph_schema import CONSTRAINTS, vector_index_statement

        with self._driver.session() as session:
            for stmt in CONSTRAINTS:
                session.run(stmt)
            session.run(vector_index_statement(embedding_dim, model_name))
        logger.info("Neo4j schema ensured (dim=%d, model=%s)", embedding_dim, model_name)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        doc_id: str,
    ) -> int:
        """Persist document chunks with embedding vectors.

        Args:
            chunks: List of chunk dicts with 'content', 'source', and optional metadata.
            embeddings: Parallel list of embedding vectors.
            doc_id: Document identifier (typically filename).

        Returns:
            Number of chunks stored.
        """
        from graph_schema import MERGE_CHUNKS_BATCH, MERGE_DOCUMENT

        stored = 0
        try:
            with self._driver.session() as session:
                # Ensure document node exists
                file_type = chunks[0].get("type", "unknown") if chunks else "unknown"
                session.run(MERGE_DOCUMENT, doc_id=doc_id, filename=doc_id, file_type=file_type)

                # Build batch payload for UNWIND
                batch = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    content = chunk.get("content") or ""
                    chunk_id = hashlib.sha256(f"{doc_id}:{i}:{content[:100]}".encode()).hexdigest()
                    batch.append(
                        {
                            "chunk_id": chunk_id,
                            "content": chunk.get("content", ""),
                            "embedding": embedding,
                            "source": chunk.get("source", doc_id),
                            "chunk_index": i,
                            "doc_id": doc_id,
                        }
                    )

                if batch:
                    session.run(MERGE_CHUNKS_BATCH, batch=batch)
                    stored = len(batch)

            logger.info("Stored %d chunks for %s in Neo4j", stored, doc_id)
        except _NEO4J_TRANSIENT as exc:
            logger.warning("Neo4j store_chunks failed (transient): %s", exc)
            raise Neo4jTransientError("store_chunks", exc) from exc
        except Exception as exc:
            logger.error("Neo4j store_chunks failed (permanent): %s", exc)
            raise

        return stored

    def store_financial_data(
        self,
        doc_id: str,
        period_label: str,
        ratios: Optional[Dict[str, Dict[str, Any]]] = None,
        scores: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Persist computed financial ratios and scores as graph nodes.

        Args:
            doc_id: Document the data comes from.
            period_label: Fiscal period label (e.g. "FY2024").
            ratios: Dict of ratio_name -> {value, category}.
            scores: Dict of model_name -> {value, grade, interpretation}.
        """
        from graph_schema import MERGE_FISCAL_PERIOD, MERGE_RATIOS_BATCH, MERGE_SCORES_BATCH

        period_id = hashlib.sha256(f"{doc_id}:{period_label}".encode()).hexdigest()

        try:
            with self._driver.session() as session:
                with session.begin_transaction() as tx:
                    tx.run(MERGE_FISCAL_PERIOD, period_id=period_id, label=period_label, doc_id=doc_id)

                    # Batch all ratios into a single UNWIND query
                    ratio_batch = []
                    for name, data in (ratios or {}).items():
                        ratio_id = hashlib.sha256(f"{period_id}:{name}".encode()).hexdigest()
                        ratio_batch.append(
                            {
                                "ratio_id": ratio_id,
                                "name": name,
                                "value": data.get("value"),
                                "category": data.get("category", ""),
                                "period_id": period_id,
                            }
                        )
                    if ratio_batch:
                        tx.run(MERGE_RATIOS_BATCH, batch=ratio_batch)

                    # Batch all scores into a single UNWIND query
                    score_batch = []
                    for model, data in (scores or {}).items():
                        score_id = hashlib.sha256(f"{period_id}:{model}".encode()).hexdigest()
                        score_batch.append(
                            {
                                "score_id": score_id,
                                "model": model,
                                "value": data.get("value"),
                                "grade": data.get("grade", ""),
                                "interpretation": data.get("interpretation", ""),
                                "period_id": period_id,
                            }
                        )
                    if score_batch:
                        tx.run(MERGE_SCORES_BATCH, batch=score_batch)
                    tx.commit()

            logger.info("Stored financial data for %s / %s", doc_id, period_label)
        except _NEO4J_TRANSIENT as exc:
            logger.warning("Neo4j store_financial_data failed (transient): %s", exc)
            raise Neo4jTransientError("store_financial_data", exc) from exc
        except Exception as exc:
            logger.error("Neo4j store_financial_data failed (permanent): %s", exc)
            raise

    # ------------------------------------------------------------------
    # Structured financial data population (Phase 2)
    # ------------------------------------------------------------------

    # Field name -> (statement_type, display_name, unit)
    _STATEMENT_FIELD_MAP = {
        "balance_sheet": [
            ("total_assets", "Total Assets", "USD"),
            ("current_assets", "Current Assets", "USD"),
            ("total_equity", "Total Equity", "USD"),
            ("total_debt", "Total Debt", "USD"),
            ("current_liabilities", "Current Liabilities", "USD"),
            ("cash_and_equivalents", "Cash and Equivalents", "USD"),
            ("inventory", "Inventory", "USD"),
            ("accounts_receivable", "Accounts Receivable", "USD"),
            ("accounts_payable", "Accounts Payable", "USD"),
        ],
        "income_statement": [
            ("revenue", "Revenue", "USD"),
            ("net_income", "Net Income", "USD"),
            ("gross_profit", "Gross Profit", "USD"),
            ("operating_income", "Operating Income", "USD"),
            ("ebit", "EBIT", "USD"),
            ("ebitda", "EBITDA", "USD"),
            ("interest_expense", "Interest Expense", "USD"),
            ("cogs", "Cost of Goods Sold", "USD"),
        ],
        "cash_flow": [
            ("operating_cash_flow", "Operating Cash Flow", "USD"),
            ("capital_expenditures", "Capital Expenditures", "USD"),
        ],
    }

    def store_line_items(
        self,
        financial_data,
        period_id: str,
    ) -> int:
        """Persist FinancialData fields as FinancialStatement + LineItem nodes.

        Args:
            financial_data: FinancialData instance.
            period_id: Existing FiscalPeriod node ID to link statements to.

        Returns:
            Number of line items stored.
        """
        from graph_schema import MERGE_FINANCIAL_STATEMENT, MERGE_LINE_ITEMS_BATCH

        total_stored = 0
        try:
            with self._driver.session() as session:
                with session.begin_transaction() as tx:
                    for stmt_type, fields in self._STATEMENT_FIELD_MAP.items():
                        stmt_id = hashlib.sha256(f"{period_id}:{stmt_type}".encode()).hexdigest()

                        batch = []
                        for field_name, display_name, unit in fields:
                            value = getattr(financial_data, field_name, None)
                            if value is None:
                                continue
                            item_id = hashlib.sha256(f"{stmt_id}:{field_name}".encode()).hexdigest()
                            batch.append(
                                {
                                    "item_id": item_id,
                                    "name": display_name,
                                    "value": float(value),
                                    "unit": unit,
                                    "stmt_id": stmt_id,
                                }
                            )

                        if batch:
                            tx.run(
                                MERGE_FINANCIAL_STATEMENT,
                                stmt_id=stmt_id,
                                stmt_type=stmt_type,
                                period_id=period_id,
                            )
                            tx.run(MERGE_LINE_ITEMS_BATCH, batch=batch)
                            total_stored += len(batch)
                    tx.commit()

            logger.info("Stored %d line items for period %s", total_stored, period_id)
        except _NEO4J_TRANSIENT as exc:
            logger.warning("Neo4j store_line_items failed (transient): %s", exc)
            raise Neo4jTransientError("store_line_items", exc) from exc
        except Exception as exc:
            logger.error("Neo4j store_line_items failed (permanent): %s", exc)
            raise

        return total_stored

    def store_derived_from_edges(
        self,
        period_id: str,
    ) -> int:
        """Create DERIVED_FROM edges from FinancialRatio -> LineItem.

        Uses RATIO_CATALOG from ratio_framework to determine which line items
        each ratio is derived from (numerator_field, denominator_field).

        Args:
            period_id: FiscalPeriod ID whose ratios and items to link.

        Returns:
            Number of edges created.
        """
        from graph_schema import MERGE_DERIVED_FROM_BATCH

        try:
            from ratio_framework import RATIO_CATALOG
        except ImportError:
            logger.debug("ratio_framework not available; skipping DERIVED_FROM edges")
            return 0

        # Build reverse lookup: field_name -> (stmt_type, stmt_id_suffix)
        field_to_stmt = {}
        for stmt_type, fields in self._STATEMENT_FIELD_MAP.items():
            stmt_id = hashlib.sha256(f"{period_id}:{stmt_type}".encode()).hexdigest()
            for field_name, _, _ in fields:
                item_id = hashlib.sha256(f"{stmt_id}:{field_name}".encode()).hexdigest()
                field_to_stmt[field_name] = item_id

        batch = []
        for ratio_key, defn in RATIO_CATALOG.items():
            ratio_id = hashlib.sha256(f"{period_id}:{defn.name}".encode()).hexdigest()
            for role, field_name in [("numerator", defn.numerator_field), ("denominator", defn.denominator_field)]:
                item_id = field_to_stmt.get(field_name)
                if item_id:
                    batch.append(
                        {
                            "ratio_id": ratio_id,
                            "item_id": item_id,
                            "role": role,
                        }
                    )

        if not batch:
            return 0

        try:
            with self._driver.session() as session:
                session.run(MERGE_DERIVED_FROM_BATCH, batch=batch)
            logger.info("Created %d DERIVED_FROM edges for period %s", len(batch), period_id)
            return len(batch)
        except _NEO4J_TRANSIENT as exc:
            logger.warning("Neo4j store_derived_from_edges failed (transient): %s", exc)
            raise Neo4jTransientError("store_derived_from_edges", exc) from exc
        except Exception as exc:
            logger.error("Neo4j store_derived_from_edges failed (permanent): %s", exc)
            raise

    def store_credit_assessment(
        self,
        company_name: str,
        scorecard,
        debt_capacity,
    ) -> Optional[str]:
        """Persist a CreditScorecard and DebtCapacityResult as graph nodes.

        Creates or merges a :Company node, then creates a :CreditAssessment node
        linked via HAS_CREDIT_ASSESSMENT.  Uses UNWIND batching for the assessment
        write so the pattern stays consistent with the rest of the store.

        Args:
            company_name: Identifier for the borrower (used as :Company.name).
            scorecard: CreditScorecard instance from underwriting.py.
            debt_capacity: DebtCapacityResult instance from underwriting.py.

        Returns:
            assessment_id string if successful, None on failure.
        """
        import json

        from graph_schema import MERGE_COMPANY, MERGE_CREDIT_ASSESSMENTS_BATCH

        assessment_id = hashlib.sha256(f"{company_name}:{scorecard.grade}:{scorecard.total_score}".encode()).hexdigest()

        # Serialise category_scores dict to a JSON string so Neo4j can store it
        # as a single property (Neo4j does not support nested maps on nodes).
        category_scores_json = json.dumps(scorecard.category_scores)

        batch = [
            {
                "assessment_id": assessment_id,
                "company_name": company_name,
                "total_score": scorecard.total_score,
                "grade": scorecard.grade,
                "recommendation": scorecard.recommendation,
                "category_scores": category_scores_json,
                "strengths": list(scorecard.strengths),
                "weaknesses": list(scorecard.weaknesses),
                "max_additional_debt": (
                    float(debt_capacity.max_additional_debt) if debt_capacity.max_additional_debt is not None else None
                ),
                "current_leverage": (
                    float(debt_capacity.current_leverage) if debt_capacity.current_leverage is not None else None
                ),
            }
        ]

        try:
            with self._driver.session() as session:
                session.run(MERGE_COMPANY, name=company_name)
                session.run(MERGE_CREDIT_ASSESSMENTS_BATCH, batch=batch)
            logger.info(
                "Stored credit assessment %s for company %s (score=%d grade=%s)",
                assessment_id[:8],
                company_name,
                scorecard.total_score,
                scorecard.grade,
            )
            return assessment_id
        except _NEO4J_TRANSIENT as exc:
            logger.warning("Neo4j store_credit_assessment failed (transient): %s", exc)
            raise Neo4jTransientError("store_credit_assessment", exc) from exc
        except Exception as exc:
            logger.error("Neo4j store_credit_assessment failed (permanent): %s", exc)
            raise

    def store_covenant_package(
        self,
        assessment_id: str,
        covenant_package,
    ) -> Optional[str]:
        """Persist a CovenantPackage node linked to an existing CreditAssessment.

        Uses UNWIND batching for the covenant write.  The :CreditAssessment node
        must already exist (call store_credit_assessment first).

        Args:
            assessment_id: ID returned by store_credit_assessment.
            covenant_package: CovenantPackage instance from underwriting.py.

        Returns:
            package_id string if successful, None on failure.
        """
        import json

        from graph_schema import MERGE_COVENANT_PACKAGES_BATCH

        package_id = hashlib.sha256(f"{assessment_id}:{covenant_package.covenant_tier}".encode()).hexdigest()

        # Serialise the nested financial_covenants dict to JSON string
        financial_covenants_json = json.dumps(covenant_package.financial_covenants)

        batch = [
            {
                "package_id": package_id,
                "assessment_id": assessment_id,
                "covenant_tier": covenant_package.covenant_tier,
                "financial_covenants": financial_covenants_json,
                "reporting_requirements": list(covenant_package.reporting_requirements),
                "events_of_default": list(covenant_package.events_of_default),
            }
        ]

        try:
            with self._driver.session() as session:
                session.run(MERGE_COVENANT_PACKAGES_BATCH, batch=batch)
            logger.info(
                "Stored covenant package %s (tier=%s) for assessment %s",
                package_id[:8],
                covenant_package.covenant_tier,
                assessment_id[:8],
            )
            return package_id
        except _NEO4J_TRANSIENT as exc:
            logger.warning("Neo4j store_covenant_package failed (transient): %s", exc)
            raise Neo4jTransientError("store_covenant_package", exc) from exc
        except Exception as exc:
            logger.error("Neo4j store_covenant_package failed (permanent): %s", exc)
            raise

    def link_fiscal_periods(self, period_labels_and_ids: List[Dict[str, str]]) -> int:
        """Create temporal PRECEDES/FOLLOWS edges between fiscal periods.

        Args:
            period_labels_and_ids: List of dicts with 'label' and 'period_id' keys,
                sorted chronologically.

        Returns:
            Number of temporal edge pairs created.
        """
        from graph_schema import MERGE_TEMPORAL_EDGES

        if len(period_labels_and_ids) < 2:
            return 0

        pairs = []
        for i in range(len(period_labels_and_ids) - 1):
            pairs.append(
                {
                    "earlier_id": period_labels_and_ids[i].get("period_id", ""),
                    "later_id": period_labels_and_ids[i + 1].get("period_id", ""),
                }
            )

        try:
            with self._driver.session() as session:
                session.run(MERGE_TEMPORAL_EDGES, pairs=pairs)
            logger.info("Linked %d temporal period pairs", len(pairs))
            return len(pairs)
        except _NEO4J_TRANSIENT as exc:
            logger.warning("Neo4j link_fiscal_periods failed (transient): %s", exc)
            raise Neo4jTransientError("link_fiscal_periods", exc) from exc
        except Exception as exc:
            logger.error("Neo4j link_fiscal_periods failed (permanent): %s", exc)
            raise

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        model_name: str = "mxbai-embed-large",
    ) -> List[Dict[str, Any]]:
        """Search chunks by embedding vector using Neo4j's HNSW index.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.
            model_name: Embedding model (determines index name).

        Returns:
            List of dicts with chunk_id, content, source, score.
        """
        from graph_schema import VECTOR_SEARCH

        safe_model = model_name.replace("-", "_").replace("/", "_")
        index_name = f"chunk_embedding_{safe_model}"

        try:
            with self._driver.session() as session:
                result = session.run(
                    VECTOR_SEARCH,
                    index_name=index_name,
                    top_k=top_k,
                    query_embedding=query_embedding,
                )
                return [dict(record) for record in result]
        except Exception as exc:
            logger.warning("Neo4j vector_search failed: %s", exc)
            return []

    def graph_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        model_name: str = "mxbai-embed-large",
    ) -> List[Dict[str, Any]]:
        """Vector search + graph traversal for enriched financial context.

        Returns vector search results augmented with connected ratios and scores.
        """
        from graph_schema import GRAPH_CONTEXT_FOR_CHUNKS_BATCH

        vector_results = self.vector_search(query_embedding, top_k, model_name)
        if not vector_results:
            return []

        enriched = []
        try:
            with self._driver.session() as session:
                chunk_ids = [vr.get("chunk_id", "") for vr in vector_results if vr.get("chunk_id")]
                result = session.run(GRAPH_CONTEXT_FOR_CHUNKS_BATCH, chunk_ids=chunk_ids)

                # Build lookup from chunk_id -> graph context
                context_map: Dict[str, Dict[str, Any]] = {}
                for record in result:
                    cid = record.get("chunk_id")
                    if cid is None:
                        continue
                    context_map[cid] = {
                        "document": record.get("document", ""),
                        "period": record.get("period", ""),
                        "ratios": record.get("ratios", []),
                        "scores": record.get("scores", []),
                    }

                for vr in vector_results:
                    entry = {**vr}
                    ctx = context_map.get(vr.get("chunk_id", ""))
                    if ctx:
                        entry.update(ctx)
                    enriched.append(entry)
        except Exception as exc:
            logger.warning("Neo4j graph_search traversal failed: %s", exc)
            return vector_results  # degrade to vector-only

        return enriched

    def ratios_by_period_label(self, period_label: str) -> List[Dict[str, Any]]:
        """Return all ratios for a fiscal period identified by label."""
        from graph_schema import RATIOS_BY_PERIOD_LABEL

        try:
            with self._driver.session() as session:
                result = session.run(RATIOS_BY_PERIOD_LABEL, period_label=period_label)
                return [dict(record) for record in result]
        except Exception as exc:
            logger.warning("Neo4j ratios_by_period_label failed: %s", exc)
            return []

    def scores_by_period_label(self, period_label: str) -> List[Dict[str, Any]]:
        """Return all scores for a fiscal period identified by label."""
        from graph_schema import SCORES_BY_PERIOD_LABEL

        try:
            with self._driver.session() as session:
                result = session.run(SCORES_BY_PERIOD_LABEL, period_label=period_label)
                return [dict(record) for record in result]
        except Exception as exc:
            logger.warning("Neo4j scores_by_period_label failed: %s", exc)
            return []

    def cross_period_ratio_trend(self, period_labels: List[str]) -> List[Dict[str, Any]]:
        """Return ratio values across multiple fiscal periods for trend analysis."""
        from graph_schema import CROSS_PERIOD_RATIO_TREND

        try:
            with self._driver.session() as session:
                result = session.run(CROSS_PERIOD_RATIO_TREND, period_labels=period_labels)
                return [dict(record) for record in result]
        except Exception as exc:
            logger.warning("Neo4j cross_period_ratio_trend failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Portfolio & compliance storage (Phase 7)
    # ------------------------------------------------------------------

    def store_portfolio_analysis(
        self,
        portfolio_name: str,
        company_names: List[str],
        risk_summary,
        diversification_score: int = 0,
    ) -> Optional[str]:
        """Persist portfolio analysis as Portfolio + PortfolioRisk graph nodes.

        Creates or merges a :Portfolio node, links :Company nodes via
        CONTAINS_COMPANY, and creates a :PortfolioRisk summary node.

        Args:
            portfolio_name: Name for the portfolio.
            company_names: List of company names in the portfolio.
            risk_summary: PortfolioRiskSummary instance from portfolio_analyzer.
            diversification_score: Overall diversification score (0-100).

        Returns:
            portfolio_id string if successful, None on failure.
        """
        from graph_schema import (
            MERGE_PORTFOLIO,
            MERGE_PORTFOLIO_MEMBERSHIP_BATCH,
            MERGE_PORTFOLIO_RISK,
        )

        portfolio_id = hashlib.sha256(f"{portfolio_name}:{','.join(sorted(company_names))}".encode()).hexdigest()

        risk_id = hashlib.sha256(f"{portfolio_id}:risk:{risk_summary.overall_risk_level}".encode()).hexdigest()

        try:
            with self._driver.session() as session:
                session.run(MERGE_PORTFOLIO, portfolio_id=portfolio_id, name=portfolio_name)
                session.run(
                    MERGE_PORTFOLIO_MEMBERSHIP_BATCH,
                    portfolio_id=portfolio_id,
                    company_names=company_names,
                )
                session.run(
                    MERGE_PORTFOLIO_RISK,
                    risk_id=risk_id,
                    portfolio_id=portfolio_id,
                    avg_health=float(risk_summary.avg_health_score),
                    min_health=risk_summary.min_health_score,
                    max_health=risk_summary.max_health_score,
                    risk_level=risk_summary.overall_risk_level,
                    distress_count=risk_summary.distress_count,
                    diversification_score=diversification_score,
                    risk_flags=list(risk_summary.risk_flags),
                )
            logger.info(
                "Stored portfolio analysis %s (%d companies, risk=%s)",
                portfolio_name,
                len(company_names),
                risk_summary.overall_risk_level,
            )
            return portfolio_id
        except _NEO4J_TRANSIENT as exc:
            logger.warning("Neo4j store_portfolio_analysis failed (transient): %s", exc)
            raise Neo4jTransientError("store_portfolio_analysis", exc) from exc
        except Exception as exc:
            logger.error("Neo4j store_portfolio_analysis failed (permanent): %s", exc)
            raise

    def store_compliance_report(
        self,
        company_name: str,
        compliance_report,
    ) -> Optional[str]:
        """Persist a ComplianceReport as a graph node linked to :Company.

        Uses UNWIND batching for the compliance write.

        Args:
            company_name: Company identifier.
            compliance_report: ComplianceReport instance from compliance_scorer.

        Returns:
            compliance_id string if successful, None on failure.
        """
        from graph_schema import MERGE_COMPANY, MERGE_COMPLIANCE_REPORT_BATCH

        sox = compliance_report.sox
        sec = compliance_report.sec
        reg = compliance_report.regulatory
        audit = compliance_report.audit_risk

        compliance_id = hashlib.sha256(f"{company_name}:compliance:{audit.score if audit else 0}".encode()).hexdigest()

        batch = [
            {
                "compliance_id": compliance_id,
                "company_name": company_name,
                "sox_risk": sox.overall_risk if sox else "unknown",
                "sox_score": sox.risk_score if sox else 0,
                "sec_score": sec.disclosure_score if sec else 0,
                "regulatory_pct": reg.compliance_pct if reg else 0.0,
                "audit_risk": audit.risk_level if audit else "unknown",
                "audit_score": audit.score if audit else 0,
                "going_concern": audit.going_concern_risk if audit else False,
            }
        ]

        try:
            with self._driver.session() as session:
                session.run(MERGE_COMPANY, name=company_name)
                session.run(MERGE_COMPLIANCE_REPORT_BATCH, batch=batch)
            logger.info(
                "Stored compliance report %s for %s (audit=%s)",
                compliance_id[:8],
                company_name,
                audit.risk_level if audit else "unknown",
            )
            return compliance_id
        except _NEO4J_TRANSIENT as exc:
            logger.warning("Neo4j store_compliance_report failed (transient): %s", exc)
            raise Neo4jTransientError("store_compliance_report", exc) from exc
        except Exception as exc:
            logger.error("Neo4j store_compliance_report failed (permanent): %s", exc)
            raise

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Neo4j driver."""
        try:
            self._driver.close()
            logger.info("Neo4j connection closed")
        except Exception:
            logger.debug("Error closing Neo4j driver", exc_info=True)
