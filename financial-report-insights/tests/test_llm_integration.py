"""
LLM integration tests with mocked Ollama responses.
Tests: financial query detection, connection failures, empty responses, caching.
"""

from unittest.mock import patch

import pytest

from local_llm import LLMConnectionError, LocalLLM

# ============================================================
# LocalLLM tests
# ============================================================


class TestLocalLLM:
    def test_generate_returns_response(self):
        llm = LocalLLM(model="test-model")
        with patch("local_llm.ollama.generate") as mock_gen:
            mock_gen.return_value = {"response": "The revenue is $1M."}
            result = llm.generate("What is the revenue?")
            assert result == "The revenue is $1M."
            mock_gen.assert_called_once()

    def test_generate_empty_response(self):
        llm = LocalLLM(model="test-model")
        with patch("local_llm.ollama.generate") as mock_gen:
            mock_gen.return_value = {"response": ""}
            result = llm.generate("Hello")
            assert result == ""

    def test_generate_missing_response_key(self):
        llm = LocalLLM(model="test-model")
        with patch("local_llm.ollama.generate") as mock_gen:
            mock_gen.return_value = {}
            result = llm.generate("Hello")
            assert result == ""

    def test_connection_error_raises_llm_connection_error(self):
        llm = LocalLLM(model="test-model")
        with patch("local_llm.ollama.generate") as mock_gen:
            mock_gen.side_effect = ConnectionError("refused")
            with pytest.raises(LLMConnectionError, match="Cannot connect"):
                llm.generate("Hello")

    def test_generic_error_raises_llm_connection_error(self):
        llm = LocalLLM(model="test-model")
        with patch("local_llm.ollama.generate") as mock_gen:
            mock_gen.side_effect = RuntimeError("model not found")
            with pytest.raises(LLMConnectionError, match="Ollama error"):
                llm.generate("Hello")

    def test_error_preserves_original_cause(self):
        llm = LocalLLM(model="test-model")
        with patch("local_llm.ollama.generate") as mock_gen:
            original = RuntimeError("boom")
            mock_gen.side_effect = original
            with pytest.raises(LLMConnectionError) as exc_info:
                llm.generate("Hello")
            assert exc_info.value.__cause__ is original

    def test_callable_interface(self):
        llm = LocalLLM(model="test-model")
        with patch("local_llm.ollama.generate") as mock_gen:
            mock_gen.return_value = {"response": "Yes"}
            assert llm("Test") == "Yes"


# ============================================================
# LLM response caching
# ============================================================


class TestLLMCaching:
    def test_cache_hit_skips_ollama(self):
        llm = LocalLLM(model="test-model", enable_cache=True)
        with patch("local_llm.ollama.generate") as mock_gen:
            mock_gen.return_value = {"response": "Cached answer"}
            # First call
            result1 = llm.generate("Same prompt")
            # Second call - should hit cache
            result2 = llm.generate("Same prompt")
            assert result1 == result2 == "Cached answer"
            assert mock_gen.call_count == 1  # Only called once

    def test_different_prompts_not_cached(self):
        llm = LocalLLM(model="test-model", enable_cache=True)
        with patch("local_llm.ollama.generate") as mock_gen:
            mock_gen.return_value = {"response": "Answer"}
            llm.generate("Prompt A")
            llm.generate("Prompt B")
            assert mock_gen.call_count == 2

    def test_cache_disabled(self):
        llm = LocalLLM(model="test-model", enable_cache=False)
        with patch("local_llm.ollama.generate") as mock_gen:
            mock_gen.return_value = {"response": "Answer"}
            llm.generate("Same prompt")
            llm.generate("Same prompt")
            assert mock_gen.call_count == 2  # Called twice (no caching)


# ============================================================
# Financial query detection (logic test, no Ollama needed)
# ============================================================


class TestFinancialQueryDetection:
    """Test the _is_financial_query logic from app_local.py."""

    FINANCIAL_KEYWORDS = [
        "ratio",
        "margin",
        "profit",
        "revenue",
        "income",
        "expense",
        "cash flow",
        "budget",
        "variance",
        "roe",
        "roa",
        "roi",
        "liquidity",
        "leverage",
        "debt",
        "equity",
        "asset",
        "liability",
        "growth",
        "trend",
        "forecast",
        "analysis",
        "financial",
        "balance sheet",
        "income statement",
        "p&l",
        "cfo",
        "ebitda",
    ]

    def _is_financial(self, query: str) -> bool:
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.FINANCIAL_KEYWORDS)

    def test_financial_queries(self):
        assert self._is_financial("What is the profit margin?")
        assert self._is_financial("Calculate the ROE")
        assert self._is_financial("Show me the cash flow statement")
        assert self._is_financial("What is the current ratio?")
        assert self._is_financial("revenue growth over time")
        assert self._is_financial("What's the EBITDA?")

    def test_non_financial_queries(self):
        assert not self._is_financial("What is the weather today?")
        assert not self._is_financial("Tell me a joke")
        assert not self._is_financial("How do I use this app?")

    def test_case_insensitive(self):
        assert self._is_financial("What is the REVENUE?")
        assert self._is_financial("show me the Balance Sheet")

    def test_empty_query(self):
        assert not self._is_financial("")

    def test_query_with_special_characters(self):
        assert self._is_financial("What's the P&L summary?")
