# Audit 2026-05-07

Six-agent parallel reconnaissance of the full project. Findings synthesized into the action PRD at `financial-report-insights/docs/LOKI-PRD.md` for `/loki-mode` handoff.

| File | Domain | Agent type |
|------|--------|------------|
| `01-core-analytics.md` | financial_analyzer + ratio_framework + structured_types + agents/ + ml/ + evaluation/ | code-review-expert |
| `02-data-api.md` | api + local_llm + vector + ingestion + chunkers + parsers + graph_* + reranker + protocols | code-review-expert |
| `03-ui.md` | streamlit_app_local + insights_page + viz_utils + app_local + prompts/ | code-review-expert |
| `04-domain-exports.md` | underwriting + startup_model + portfolio_analyzer + compliance_scorer + export_* | code-review-expert |
| `05-infra-security.md` | Dockerfile + compose + CI workflows + healthcheck + observability + config + logging + security docs | security-auditor |
| `06-tests-backlog.md` | tests/ + 2026-03-16 backlog rollup + README drift + MEMORY split + findings.md | qa-expert |

Base commit: `f943cc8` (P0+P1 of audit-2026-03-16 closed). Tests: 4,966 passing.
