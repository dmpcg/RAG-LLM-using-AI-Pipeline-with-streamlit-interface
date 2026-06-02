# Financial Report Insights

A privacy-first, local RAG (Retrieval-Augmented Generation) platform for CFO-grade financial analysis. Upload Excel or PDF financial reports and get interactive dashboards, 160+ financial ratios, scoring models, Monte Carlo simulations, and natural-language Q&A — all powered by Ollama (LLM) and Docker Model Runner embeddings (mxbai-embed-large) running entirely on your machine. No API keys, no cloud dependencies.

## Key Features

- **Hybrid Search** — BM25 + semantic (cosine similarity) with Reciprocal Rank Fusion for accurate document retrieval
- **160+ Financial Ratios** — Profitability, liquidity, leverage, efficiency, cash flow, and specialty ratios with automated scoring and grading
- **Scoring Models** — Altman Z-Score (bankruptcy prediction), Piotroski F-Score (financial strength), Beneish M-Score (earnings manipulation detection)
- **Monte Carlo Simulation** — Stochastic cash flow forecasting and DCF valuation with configurable scenarios
- **Interactive Dashboards** — 130+ Streamlit tabs with Plotly visualizations: executive summaries, trend analysis, budget variance, drill-downs
- **Excel Intelligence** — Automatic detection of income statements, balance sheets, and cash flow statements from raw spreadsheet data
- **Circuit Breaker** — Resilient LLM calls with automatic failure detection and recovery
- **DuPont Decomposition** — ROE breakdown into margin, turnover, and leverage components
- **Anomaly Detection** — Z-score-based outlier identification across all financial metrics
- **Report Generation** — Downloadable financial analysis reports with graded assessments

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12 or 3.13 (numpy 2.x wheels require 3.12+) |
| UI | Streamlit |
| LLM | Ollama (llama3.2 default) |
| Embeddings | Docker Model Runner (mxbai-embed-large, 1024-dim) |
| Data | Pandas, NumPy, openpyxl, xlrd |
| Visualization | Plotly |
| Documents | PyMuPDF (PDF), python-docx (DOCX) |
| Search | rank-bm25 + cosine similarity |
| Config | Pydantic Settings |
| Caching | joblib (embeddings), OrderedDict LRU (LLM responses) |
| Container | Docker (multi-stage, non-root, tini) |

## Prerequisites

- **Python 3.12 or 3.13 (numpy 2.x wheels require 3.12+)**
- **Ollama** — [Install Ollama](https://ollama.com/download) and pull a model:
  ```bash
  ollama pull llama3.2
  ```
- **Git** — to clone the repository

Or, if using Docker:
- **Docker** and **Docker Compose v2**

## Quick Start

### Option A: Docker Compose (Recommended)

This starts Ollama and the app together. No local Python or Ollama install required.

```bash
git clone https://github.com/Dono1901/RAG-LLM-using-AI-Pipeline-with-streamlit-interface.git
cd RAG-LLM-using-AI-Pipeline-with-streamlit-interface/financial-report-insights

# Start the application (Streamlit + FastAPI + optional Neo4j):
docker compose up -d

# To enable the optional graph database (Neo4j):
docker compose --profile graph up -d
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

To stop:
```bash
docker compose down
```

### Option B: Local Setup

```bash
git clone https://github.com/Dono1901/RAG-LLM-using-AI-Pipeline-with-streamlit-interface.git
cd RAG-LLM-using-AI-Pipeline-with-streamlit-interface/financial-report-insights

# Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment config (optional — defaults work out of the box)
cp .env.example .env

# Ensure Ollama is running with a model pulled
ollama pull llama3.2

# Start the app
streamlit run streamlit_app_local.py
```

Open [http://localhost:8501](http://localhost:8501).

## Architecture Overview

```
financial-report-insights/
├── streamlit_app_local.py   # Streamlit entry point, page routing, file upload
├── app_local.py             # SimpleRAG engine: chunking, embedding, hybrid search, LLM Q&A
├── local_llm.py             # Ollama wrapper (LocalLLM, LocalEmbedder, CircuitBreaker)
├── financial_analyzer.py    # CharlieAnalyzer: 161 analysis methods, FinancialData extraction
├── ratio_framework.py       # Generic parameterized ratio engine (RatioDefinition + compute_ratio)
├── excel_processor.py       # ExcelProcessor: sheet detection, financial table classification
├── insights_page.py         # FinancialInsightsPage: 130+ dashboard tabs with Plotly charts
├── viz_utils.py             # FinancialVizUtils: reusable Plotly chart components
├── config.py                # Pydantic Settings with RAG_ env prefix
├── protocols.py             # LLMProvider / EmbeddingProvider protocol interfaces
├── logging_config.py        # Structured logging (JSON or text)
├── healthcheck.py           # Ollama connectivity and model availability checks
├── requirements.txt
├── Dockerfile               # Multi-stage build, non-root user, tini init
├── docker-compose.yml       # Ollama + rag-app + setup profile
├── .env.example
└── tests/                   # 150+ test files, ~5,300 tests
    ├── conftest.py
    └── test_*.py
```

### Data Flow

```
Excel/PDF/DOCX Upload
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│ ExcelProcessor   │────▶│  FinancialData    │
│ (sheet detect,   │     │  (extracted       │
│  table classify) │     │   metrics)        │
└─────────────────┘     └────────┬──────────┘
                                 │
        ┌────────────────────────┼─────────────────────┐
        ▼                        ▼                      ▼
┌──────────────┐    ┌────────────────────┐    ┌─────────────────┐
│ SimpleRAG     │    │ CharlieAnalyzer     │    │ RatioFramework  │
│ (chunk, embed,│    │ (161 methods:       │    │ (18 catalog     │
│  BM25+cosine, │    │  DuPont, Z-Score,   │    │  ratios with    │
│  LLM Q&A)     │    │  F-Score, M-Score,  │    │  scoring +      │
└──────┬────────┘    │  Monte Carlo, etc.) │    │  grading)       │
       │             └────────┬────────────┘    └────────┬────────┘
       │                      │                          │
       ▼                      ▼                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  FinancialInsightsPage                        │
│         Streamlit dashboard with Plotly charts                │
│    (executive summary, ratios, trends, budgets, reports)     │
└──────────────────────────────────────────────────────────────┘
```

## Module Reference

| Module | Key Classes / Functions | Description |
|--------|------------------------|-------------|
| `streamlit_app_local.py` | `load_rag_system()`, `render_sidebar()` | Main Streamlit entry point. Handles page routing, file upload with path-traversal protection, and session state management. |
| `app_local.py` | `SimpleRAG` | Core RAG engine. Chunks documents, generates embeddings, performs hybrid BM25+semantic search with RRF, and queries Ollama for answers. Supports PDF, DOCX, TXT, and Excel files. |
| `local_llm.py` | `LocalLLM`, `LocalEmbedder`, `CircuitBreaker` | Ollama integration with timeout handling, retry logic, circuit breaker pattern (CLOSED/OPEN/HALF_OPEN), LRU response cache (128 entries), and streaming support. |
| `financial_analyzer.py` | `CharlieAnalyzer`, `FinancialData`, `safe_divide()`, `quick_analyze()` | 161-method financial analysis engine. Automatic metric extraction from DataFrames. DuPont decomposition, Altman Z-Score, Piotroski F-Score, Beneish M-Score, Monte Carlo simulation, scenario analysis, anomaly detection, trend analysis, and more. |
| `ratio_framework.py` | `RatioDefinition`, `compute_ratio()`, `run_all_ratios()`, `RATIO_CATALOG` | Generic parameterized ratio engine. 18 canonical ratios (ROA, ROE, ROIC, margins, liquidity, leverage, efficiency, cash flow) with declarative scoring thresholds and adjustment rules. |
| `excel_processor.py` | `ExcelProcessor`, `WorkbookData`, `SheetData`, `FinancialTable`, `DocumentChunk` | Reads xlsx/xlsm/xls/csv/tsv files. Detects header rows, classifies financial table types (income statement, balance sheet, cash flow), identifies time periods, and produces RAG-ready chunks. |
| `insights_page.py` | `FinancialInsightsPage` | Interactive Streamlit dashboard with 130+ tabs. Executive summary cards, drill-down charts, budget variance analysis, what-if scenarios, and downloadable reports. |
| `viz_utils.py` | `FinancialVizUtils` | Reusable Plotly chart factory. Waterfall charts, gauge indicators, trend lines, comparison bars, and currency formatting with K/M/B suffixes. |
| `config.py` | `Settings`, `settings` | Pydantic `BaseSettings` with `RAG_` env prefix. All application defaults in one place with environment variable overrides. |
| `protocols.py` | `LLMProvider`, `EmbeddingProvider` | Runtime-checkable Protocol classes for dependency injection. Allows swapping LLM/embedding backends without changing consuming code. |
| `logging_config.py` | `setup_logging()`, `JSONFormatter` | Configures structured logging. JSON format for production, human-readable text for development. Controlled by `LOG_LEVEL` and `LOG_FORMAT` env vars. |
| `healthcheck.py` | `check_ollama_connection()`, `check_model_available()` | Startup validation. Verifies Ollama is reachable and the configured model is pulled before the app accepts requests. |

## Environment Variables

All variables are optional. Defaults are designed to work out of the box with a standard Ollama installation.

Copy `.env.example` to `.env` to customize:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `RAG_LLM_MODEL` | `llama3.2` | Ollama model for text generation |
| `RAG_EMBEDDING_MODEL` | `mxbai-embed-large` | Docker Model Runner embedding model (1024-dim) |
| `RAG_CHUNK_SIZE` | `500` | Characters per document chunk |
| `RAG_CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `RAG_TOP_K` | `3` | Number of search results to retrieve |
| `RAG_MAX_TOP_K` | `20` | Maximum allowed top_k value |
| `RAG_MAX_FILE_SIZE_MB` | `200` | Maximum upload file size in MB |
| `RAG_MAX_QUERY_LENGTH` | `2000` | Maximum query length in characters |
| `RAG_MAX_WORKBOOK_ROWS` | `500000` | Maximum rows to process per workbook |
| `RAG_LLM_TIMEOUT_SECONDS` | `120` | Timeout for Ollama API calls |
| `RAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `3` | Consecutive failures before circuit opens |
| `RAG_CIRCUIT_BREAKER_RECOVERY_SECONDS` | `30` | Seconds before circuit attempts recovery |
| `RAG_DEFAULT_TAX_RATE` | `0.25` | Default corporate tax rate for financial calculations |
| `RAG_EMBEDDING_CACHE_DIR` | `.cache/embeddings` | Directory for embedding cache files |
| `RAG_LLM_CACHE_DIR` | `.cache/llm_responses` | Directory for LLM response cache |
| `RAG_LLM_CACHE_SIZE_LIMIT_MB` | `500` | Maximum cache size on disk |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LOG_FORMAT` | `text` | Log format: `text` (development) or `json` (production) |

## Configuration

The application uses [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration management. All settings are defined in `config.py` with the `RAG_` environment variable prefix.

```python
from config import settings

# Access any setting:
settings.chunk_size      # 500 (default) or RAG_CHUNK_SIZE env var
settings.llm_model       # "llama3.2" or RAG_LLM_MODEL env var
settings.top_k           # 3 or RAG_TOP_K env var
```

Override any setting via environment variable:

```bash
# Shell
export RAG_CHUNK_SIZE=1000
export RAG_TOP_K=5

# Or in .env file
RAG_CHUNK_SIZE=1000
RAG_TOP_K=5
```

## Testing

The project has 150+ test files with approximately 5,300 tests covering all analysis methods, edge cases, security, and integration scenarios.

```bash
cd financial-report-insights

# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_phase2_scoring.py -v

# Run tests matching a keyword
python -m pytest tests/ -k "z_score" -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing
```

Test categories:
- `test_phase2_scoring.py` — DuPont, Z-Score, F-Score scoring models
- `test_phase3_production.py` — Health scores, period comparison, report generation
- `test_phase4_integration.py` — RAG + analysis bridge, report downloads
- `test_phase5_analytics.py` — Scenario analysis, sensitivity analysis
- `test_phase6_stochastic.py` — Monte Carlo simulation, DCF, cash flow forecasting
- `test_circuit_breaker.py` — Circuit breaker state transitions
- `test_security.py` — Path traversal, input validation, injection prevention
- `test_resilience.py` — Error handling, timeout recovery, edge cases
- `test_phase34_altman_z_score.py` through `test_phase353_*.py` — Individual ratio and metric tests

## Deployment

### Docker Compose Services

The `docker-compose.yml` defines two core services plus an optional graph database:

| Service | Image | Purpose | Port |
|---------|-------|---------|------|
| `rag-app` | Built from `Dockerfile` | Streamlit (8501) + FastAPI (8504) | 8501, 8504 |
| `neo4j` (optional, `--profile graph`) | `neo4j:5-community` | Graph knowledge base with APOC | 7474, 7687 |

**Resource limits:**
- rag-app: 3 GB reservation, 4 GB limit
- neo4j: 1.5 GB limit

**Volumes:**
- `app-cache` — embedding and LLM response cache
- `neo4j-data` — graph database persistence (graph profile only)
- `./documents` — bind mount for uploaded files

**Embedding service:** Ollama-compatible API served by Docker Model Runner (DMR) at `http://model-runner.docker.internal:80` (requires Docker Desktop with model service enabled). The application uses `ollama` Python client but queries DMR, not Ollama.

### Dockerfile Highlights

- **Multi-stage build** — build dependencies are not in the final image
- **Non-root user** (`appuser`) — runs as unprivileged user
- **tini init** — proper PID 1 signal handling for graceful shutdown
- **Read-only filesystem** — immutable root FS with tmpfs and named volumes for writes
- **Health check** — probes both Streamlit (8501) and FastAPI (8504) endpoints

### Changing the LLM Model

```bash
# Pull a different model into Ollama
docker compose exec ollama ollama pull mistral

# Set the model in .env
echo "RAG_LLM_MODEL=mistral" >> .env

# Restart the app
docker compose restart rag-app
```

## Troubleshooting

### Ollama connection refused

```
Cannot connect to Ollama: Connection refused
```

**Cause:** Ollama is not running or is on a different host/port.

**Fix:**
```bash
# Check if Ollama is running
ollama list

# If not running, start it
ollama serve

# If using Docker Compose, the ollama service starts automatically
docker compose up -d
```

### Model not found

```
Model 'llama3.2' not found
```

**Fix:**
```bash
ollama pull llama3.2

# Or in Docker:
docker compose --profile setup run --rm ollama-setup
```

### Out of memory

Large models need significant RAM. `llama3.2` (3B parameters) needs ~4 GB.

**Fix:** Use a smaller model or increase Docker memory limits:
```bash
# Use a smaller model
export RAG_LLM_MODEL=llama3.2:1b

# Or increase Docker memory in docker-compose.yml
```

### File upload too large

Default maximum file size is 200 MB.

**Fix:** Adjust in `.env`:
```bash
RAG_MAX_FILE_SIZE_MB=500
```

### Slow first query or embedding errors

Embeddings are served by Docker Model Runner (DMR). Ensure:
1. Docker Desktop is running
2. Model service is enabled in Docker Desktop (Settings > AI ML > Enable Docker AI)
3. Model is pulled: `docker model pull ai/mxbai-embed-large` (not via Ollama)
4. DMR is accessible at `http://model-runner.docker.internal:80` (or set OLLAMA_HOST)

### Circuit breaker open

```
Circuit breaker is OPEN — LLM calls rejected
```

**Cause:** Ollama failed 3+ consecutive times. The circuit breaker prevents further calls for 30 seconds.

**Fix:** Check Ollama is healthy, then wait 30 seconds for automatic recovery. Adjust thresholds in `.env`:
```bash
RAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
RAG_CIRCUIT_BREAKER_RECOVERY_SECONDS=15
```

## Supported File Formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| Excel | `.xlsx`, `.xlsm`, `.xls` | Multi-sheet support with automatic financial table detection |
| CSV/TSV | `.csv`, `.tsv` | Treated as single-sheet workbooks |
| PDF | `.pdf` | Text extraction via PyMuPDF |
| Word | `.docx` | Text extraction via python-docx |
| Text | `.txt`, `.md` | Direct text ingestion |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
