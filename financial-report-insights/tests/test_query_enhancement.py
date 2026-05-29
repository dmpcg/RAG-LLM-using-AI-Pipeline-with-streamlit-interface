"""Tests for query enhancement features: BM25 tokenization, query classification,
HyDE expansion, and LLM-based query decomposition."""

from unittest.mock import MagicMock, patch

import pytest

from app_local import SimpleRAG

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rag():
    """Create a SimpleRAG with mocked LLM and embedder (no real model needed)."""
    mock_llm = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1] * 10
    with patch("app_local.Path.mkdir"):
        r = SimpleRAG(
            docs_folder="./test_docs",
            llm=mock_llm,
            embedder=mock_embedder,
        )
    return r


# ---------------------------------------------------------------------------
# _tokenize_for_bm25
# ---------------------------------------------------------------------------


class TestTokenizeForBM25:
    """Tests for BM25 tokenizer with stop-word removal and stemming."""

    def test_removes_stop_words(self):
        tokens = SimpleRAG._tokenize_for_bm25("the company is a leader in the market")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "in" not in tokens
        # Content words survive
        assert "company" in tokens
        assert "leader" in tokens
        assert "market" in tokens

    def test_basic_stemming_ing(self):
        tokens = SimpleRAG._tokenize_for_bm25("running increasing operating")
        # "running" (7 chars, ends in 'ing') -> "runn" (wait, len>5 check: 7>5 yes -> "runn")
        # Actually: running -> runn (strip 'ing')
        assert "runn" in tokens
        assert "increas" in tokens
        assert "operat" in tokens

    def test_basic_stemming_ed(self):
        tokens = SimpleRAG._tokenize_for_bm25("increased decreased")
        # increased (9 chars) ends in 'ed', len>4 -> "increas"
        assert "increas" in tokens
        assert "decreas" in tokens

    def test_basic_stemming_ly(self):
        tokens = SimpleRAG._tokenize_for_bm25("quickly slowly")
        # quickly (7 chars) ends in 'ly', len>4 -> "quick"
        assert "quick" in tokens
        assert "slow" in tokens

    def test_basic_stemming_s(self):
        tokens = SimpleRAG._tokenize_for_bm25("profits margins assets")
        # profits (7 chars) ends in 's' not 'ss', len>3 -> "profit"
        assert "profit" in tokens
        assert "margin" in tokens
        assert "asset" in tokens

    def test_no_stemming_short_words(self):
        tokens = SimpleRAG._tokenize_for_bm25("red bed")
        # "red" len=3, ends in 'ed' but len not > 4 -> unchanged
        assert "red" in tokens
        assert "bed" in tokens

    def test_no_stemming_ss(self):
        tokens = SimpleRAG._tokenize_for_bm25("gross loss")
        # "gross" ends in 'ss', should NOT strip
        assert "gross" in tokens
        assert "loss" in tokens

    def test_empty_input(self):
        assert SimpleRAG._tokenize_for_bm25("") == []

    def test_whitespace_only(self):
        assert SimpleRAG._tokenize_for_bm25("   \t\n  ") == []

    def test_removes_single_char_tokens(self):
        tokens = SimpleRAG._tokenize_for_bm25("x y z revenue")
        assert "x" not in tokens
        assert "y" not in tokens
        assert "z" not in tokens
        assert "revenue" in tokens

    def test_numeric_tokens_preserved(self):
        tokens = SimpleRAG._tokenize_for_bm25("revenue was 1500000 in 2024")
        assert "1500000" in tokens
        assert "2024" in tokens

    def test_splits_on_punctuation(self):
        tokens = SimpleRAG._tokenize_for_bm25("debt-to-equity ratio: 1.5")
        assert "debt" in tokens
        assert "equity" in tokens
        assert "ratio" in tokens


# ---------------------------------------------------------------------------
# _classify_query
# ---------------------------------------------------------------------------


class TestClassifyQuery:
    """Tests for query type classification."""

    def test_ratio_lookup_what_is(self):
        assert SimpleRAG._classify_query("What is the current ratio?") == "ratio_lookup"

    def test_ratio_lookup_calculate(self):
        assert SimpleRAG._classify_query("Calculate the debt to equity ratio") == "ratio_lookup"

    def test_ratio_lookup_roe(self):
        assert SimpleRAG._classify_query("What is ROE for 2024?") == "ratio_lookup"

    def test_ratio_lookup_margin(self):
        assert SimpleRAG._classify_query("What was the profit margin?") == "ratio_lookup"

    def test_trend_analysis_yoy(self):
        assert SimpleRAG._classify_query("How has revenue changed year over year?") == "trend_analysis"

    def test_trend_analysis_growth(self):
        assert SimpleRAG._classify_query("Show revenue growth over time") == "trend_analysis"

    def test_trend_analysis_forecast(self):
        assert SimpleRAG._classify_query("forecast earnings for next quarter") == "trend_analysis"

    def test_trend_analysis_trajectory(self):
        assert SimpleRAG._classify_query("Show the trajectory of debt levels") == "trend_analysis"

    def test_comparison_compare(self):
        assert SimpleRAG._classify_query("Compare revenue to expenses") == "comparison"

    def test_comparison_vs(self):
        assert SimpleRAG._classify_query("Company A vs Company B performance") == "comparison"

    def test_comparison_benchmark(self):
        assert SimpleRAG._classify_query("benchmark our performance against industry") == "comparison"

    def test_explanation_why(self):
        assert SimpleRAG._classify_query("Why did revenue decline?") == "explanation"

    def test_explanation_explain(self):
        assert SimpleRAG._classify_query("Explain the drop in operating income") == "explanation"

    def test_explanation_impact(self):
        assert SimpleRAG._classify_query("impact of rising interest rates on earnings") == "explanation"

    def test_general_default(self):
        assert SimpleRAG._classify_query("Tell me about the company") == "general"

    def test_general_unrelated(self):
        assert SimpleRAG._classify_query("Hello world") == "general"


# ---------------------------------------------------------------------------
# _hyde_expand_query
# ---------------------------------------------------------------------------


class TestHydeExpandQuery:
    """Tests for HyDE (Hypothetical Document Embeddings) query expansion."""

    def test_uses_llm_hypothetical_answer(self, rag):
        rag.llm.generate.return_value = "The company's revenue was $10M with a 15% margin."
        rag.embedder.embed.return_value = [0.5] * 10

        result = rag._hyde_expand_query("What is the revenue?")

        # LLM should be called with a prompt
        rag.llm.generate.assert_called_once()
        prompt_arg = rag.llm.generate.call_args[0][0]
        assert "What is the revenue?" in prompt_arg

        # Embedder should embed the hypothetical answer, not the question
        rag.embedder.embed.assert_called_once_with("The company's revenue was $10M with a 15% margin.")
        assert result == [0.5] * 10

    def test_falls_back_on_llm_failure(self, rag):
        rag.llm.generate.side_effect = RuntimeError("LLM down")
        rag.embedder.embed.return_value = [0.3] * 10

        result = rag._hyde_expand_query("What is revenue?")

        # Should fall back to embedding the query directly
        rag.embedder.embed.assert_called_once_with("What is revenue?")
        assert result == [0.3] * 10

    def test_strips_whitespace_from_hypothetical(self, rag):
        rag.llm.generate.return_value = "  Some answer with whitespace.  \n"
        rag.embedder.embed.return_value = [0.1] * 10

        rag._hyde_expand_query("test query")

        rag.embedder.embed.assert_called_once_with("Some answer with whitespace.")


# ---------------------------------------------------------------------------
# _decompose_query (LLM-based with keyword fallback)
# ---------------------------------------------------------------------------


class TestDecomposeQuery:
    """Tests for LLM-based query decomposition with keyword fallback."""

    def test_llm_decomposition_when_enabled(self, rag):
        """When LLM decomposition is enabled and succeeds, uses LLM sub-queries."""
        rag.llm.generate.return_value = (
            "What was the total revenue for the year?\n"
            "What were the operating expenses?\n"
            "What was the net income margin?"
        )
        with patch("app_local.settings") as mock_settings:
            mock_settings.enable_query_decomposition = True
            mock_settings.max_sub_queries = 4
            result = rag._decompose_query("How profitable was the company?")

        assert result[0] == "How profitable was the company?"
        assert len(result) > 1
        assert len(result) <= 4

    def test_falls_back_to_keyword_when_llm_disabled(self, rag):
        """When LLM decomposition is disabled, uses keyword fallback."""
        with patch("app_local.settings") as mock_settings:
            mock_settings.enable_query_decomposition = False
            mock_settings.max_sub_queries = 4
            result = rag._decompose_query("What is the profit margin?")

        # Should include original + keyword expansions
        assert result[0] == "What is the profit margin?"
        assert len(result) >= 1

    def test_falls_back_to_keyword_on_llm_error(self, rag):
        """When LLM fails, falls back to keyword decomposition."""
        rag.llm.generate.side_effect = RuntimeError("LLM error")
        with patch("app_local.settings") as mock_settings:
            mock_settings.enable_query_decomposition = True
            mock_settings.max_sub_queries = 4
            result = rag._decompose_query("What is the debt to equity ratio?")

        assert result[0] == "What is the debt to equity ratio?"
        assert len(result) >= 1

    def test_always_includes_original_query(self, rag):
        with patch("app_local.settings") as mock_settings:
            mock_settings.enable_query_decomposition = False
            mock_settings.max_sub_queries = 4
            result = rag._decompose_query("general question about things")

        assert result[0] == "general question about things"


# ---------------------------------------------------------------------------
# _keyword_decompose_query
# ---------------------------------------------------------------------------


class TestKeywordDecomposeQuery:
    """Tests for the keyword-based decomposition fallback."""

    def test_expands_profitability_query(self, rag):
        result = rag._keyword_decompose_query("What is the profit margin?")
        assert result[0] == "What is the profit margin?"
        # Should add related keywords
        assert len(result) > 1

    def test_expands_leverage_query(self, rag):
        result = rag._keyword_decompose_query("analyze the debt levels")
        assert result[0] == "analyze the debt levels"
        assert len(result) > 1

    def test_caps_at_max_subs(self, rag):
        result = rag._keyword_decompose_query("analyze debt and equity", max_subs=2)
        assert len(result) <= 2

    def test_caps_at_default_max(self, rag):
        result = rag._keyword_decompose_query("analyze revenue and profit margins")
        assert len(result) <= 4

    def test_adds_details_for_unmatched_query(self, rag):
        result = rag._keyword_decompose_query("tell me about the company")
        assert len(result) == 2
        assert "details about tell me about the company" in result

    def test_adds_details_question_mark(self, rag):
        result = rag._keyword_decompose_query("what happened?")
        assert len(result) == 2
        assert "what happened details?" in result


# ---------------------------------------------------------------------------
# _filter_low_quality_chunks
# ---------------------------------------------------------------------------


class TestFilterLowQualityChunks:
    """Tests for chunk quality filtering."""

    def test_preserves_good_chunks(self):
        chunks = [
            "This is a perfectly good chunk with enough words to pass the filter easily",
            "Another excellent chunk with financial data including revenue and margins",
        ]
        result = SimpleRAG._filter_low_quality_chunks(chunks)
        assert len(result) == 2

    def test_removes_short_chunks(self):
        chunks = [
            "Too short",
            "This is a perfectly good chunk with enough words to pass the filter easily",
        ]
        result = SimpleRAG._filter_low_quality_chunks(chunks)
        assert len(result) == 1
        assert "perfectly good" in result[0]

    def test_removes_mostly_punctuation(self):
        chunks = [
            "--- ... ### *** !!! ??? ,,, ;;; ::: === +++ ---",
            "This is a perfectly good chunk with enough words to pass the filter easily",
        ]
        result = SimpleRAG._filter_low_quality_chunks(chunks)
        assert len(result) == 1

    def test_removes_near_duplicates(self):
        prefix = "This is a duplicated chunk " + "x " * 50
        chunks = [prefix, prefix + " extra"]
        result = SimpleRAG._filter_low_quality_chunks(chunks)
        assert len(result) == 1

    def test_never_returns_empty(self):
        chunks = ["short"]
        result = SimpleRAG._filter_low_quality_chunks(chunks)
        assert len(result) >= 1  # Returns original if all filtered


# ---------------------------------------------------------------------------
# Config settings
# ---------------------------------------------------------------------------


class TestQueryEnhancementConfig:
    """Tests for query enhancement config settings."""

    def test_config_defaults(self):
        from config import Settings

        s = Settings()
        assert s.enable_hyde is True
        assert s.enable_query_decomposition is True
        assert s.max_sub_queries == 4
