"""Tests for protocols.py dependency injection interfaces."""

from typing import List

# ---------------------------------------------------------------------------
# LLMProvider protocol
# ---------------------------------------------------------------------------


class TestLLMProvider:
    def test_protocol_checkable(self):
        from protocols import LLMProvider

        class GoodLLM:
            def generate(self, prompt: str) -> str:
                return "answer"

        llm = GoodLLM()
        assert isinstance(llm, LLMProvider)
        # Verify the conforming instance actually produces output
        assert llm.generate("test") == "answer"

    def test_missing_method_fails(self):
        from protocols import LLMProvider

        class BadLLM:
            pass

        assert not isinstance(BadLLM(), LLMProvider)

    def test_wrong_signature_still_satisfies(self):
        """Python runtime_checkable only checks method existence, not signatures."""
        from protocols import LLMProvider

        class WeirdLLM:
            def generate(self):
                return ""

        # runtime_checkable only checks name existence
        assert isinstance(WeirdLLM(), LLMProvider)


# ---------------------------------------------------------------------------
# EmbeddingProvider protocol
# ---------------------------------------------------------------------------


class TestEmbeddingProvider:
    def test_protocol_checkable(self):
        from protocols import EmbeddingProvider

        class GoodEmbedder:
            def embed(self, text: str) -> List[float]:
                return [0.0]

            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return [[0.0]]

        assert isinstance(GoodEmbedder(), EmbeddingProvider)

    def test_missing_embed_fails(self):
        from protocols import EmbeddingProvider

        class NoEmbed:
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return [[0.0]]

        assert not isinstance(NoEmbed(), EmbeddingProvider)

    def test_missing_embed_batch_fails(self):
        from protocols import EmbeddingProvider

        class NoEmbedBatch:
            def embed(self, text: str) -> List[float]:
                return [0.0]

        assert not isinstance(NoEmbedBatch(), EmbeddingProvider)


# ---------------------------------------------------------------------------
# Cross-protocol
# ---------------------------------------------------------------------------


class TestCrossProtocol:
    def test_both_protocols_importable(self):
        from protocols import EmbeddingProvider, LLMProvider

        assert LLMProvider is not EmbeddingProvider

    def test_class_can_satisfy_both(self):
        from protocols import EmbeddingProvider, LLMProvider

        class DualProvider:
            def generate(self, prompt: str) -> str:
                return ""

            def embed(self, text: str) -> List[float]:
                return [0.0]

            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return [[0.0]]

        provider = DualProvider()
        assert isinstance(provider, LLMProvider)
        assert isinstance(provider, EmbeddingProvider)
