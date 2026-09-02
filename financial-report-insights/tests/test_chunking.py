"""Tests for sentence-aware chunking and chunk quality filtering (Phase 1.2)."""

from unittest.mock import patch

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers – build a minimal SimpleRAG instance without hitting real services
# ---------------------------------------------------------------------------


def _make_rag():
    """Return a SimpleRAG with all heavy dependencies stubbed out."""
    with patch("app_local.LocalEmbedder"), patch("app_local.settings") as mock_settings:
        mock_settings.docs_folder = "/tmp/fake_docs"
        mock_settings.chunk_size = 500
        mock_settings.chunk_overlap = 50
        from app_local import SimpleRAG

        rag = SimpleRAG.__new__(SimpleRAG)
    return rag


# ===========================================================================
# 1. Sentence-aware chunking
# ===========================================================================


class TestChunkTextSentenceAware:
    """Tests for SimpleRAG._chunk_text."""

    def setup_method(self):
        self.rag = _make_rag()

    # --- Basic sentence boundary preservation ---

    def test_preserves_sentence_boundaries(self):
        """Chunks should not split in the middle of a sentence."""
        text = (
            "Revenue grew 15% year over year. "
            "Operating expenses decreased by 3%. "
            "Net income reached a record high. "
            "The board approved a new dividend policy."
        )
        chunks = self.rag._chunk_text(text, chunk_size=12, overlap=0)
        # Each chunk should end at a sentence boundary (period present)
        for chunk in chunks:
            # Last char of trimmed chunk should be a period (sentence end)
            assert chunk.strip()[-1] == ".", f"Chunk does not end at sentence boundary: {chunk!r}"

    def test_multiple_sentences_grouped(self):
        """Multiple short sentences should be grouped into one chunk."""
        sentences = ["Sentence number {}.".format(i) for i in range(5)]
        text = " ".join(sentences)
        # chunk_size large enough to hold all sentences
        chunks = self.rag._chunk_text(text, chunk_size=500, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == text

    # --- Overlap behaviour ---

    def test_overlap_between_chunks(self):
        """Consecutive chunks should share overlapping content."""
        # Build text with clearly distinguishable sentences
        sentences = [f"Sentence {chr(65 + i)} has several words in it." for i in range(10)]
        text = " ".join(sentences)
        chunks = self.rag._chunk_text(text, chunk_size=20, overlap=8)

        if len(chunks) >= 2:
            # The end of chunk N should overlap with the start of chunk N+1
            first_words = set(chunks[0].split())
            second_words = set(chunks[1].split())
            overlap_words = first_words & second_words
            assert len(overlap_words) > 0, "Expected overlap between consecutive chunks"

    # --- Long single sentence fallback ---

    def test_long_sentence_falls_back_to_word_split(self):
        """A single sentence longer than chunk_size should be word-split."""
        long_sentence = " ".join(["word"] * 100)  # 100-word sentence, no period
        chunks = self.rag._chunk_text(long_sentence, chunk_size=30, overlap=5)
        assert len(chunks) >= 3, "Long sentence should produce multiple chunks"
        # Each chunk should be at most chunk_size words
        for chunk in chunks:
            assert len(chunk.split()) <= 30

    # --- Edge cases ---

    def test_empty_string_returns_empty_list(self):
        """Empty text should return an empty list."""
        result = self.rag._chunk_text("", chunk_size=500, overlap=50)
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        """Whitespace-only text should return an empty list."""
        result = self.rag._chunk_text("   \n\t  ", chunk_size=500, overlap=50)
        assert result == []

    def test_single_short_sentence(self):
        """A single short sentence returns one chunk."""
        text = "Hello world."
        chunks = self.rag._chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_no_sentence_ending_punctuation(self):
        """Text without sentence-ending punctuation treated as one block."""
        text = "this is text without any punctuation or capitals just words"
        chunks = self.rag._chunk_text(text, chunk_size=5, overlap=0)
        # Should fall back to word-splitting since it's one "sentence" > chunk_size
        assert len(chunks) >= 2

    def test_returns_original_when_no_chunks_produced(self):
        """If chunking produces nothing, return original text."""
        text = "Ok."
        chunks = self.rag._chunk_text(text, chunk_size=500, overlap=0)
        assert len(chunks) >= 1
        assert "Ok." in chunks[0]


# ===========================================================================
# 2. Chunk quality filter
# ===========================================================================


class TestFilterLowQualityChunks:
    """Tests for SimpleRAG._filter_low_quality_chunks."""

    def setup_method(self):
        self.rag = _make_rag()

    def test_removes_short_chunks(self):
        """Chunks below min_words should be removed."""
        chunks = [
            "This is a sufficiently long chunk with enough words to pass the filter easily.",
            "Too short.",
        ]
        result = self.rag._filter_low_quality_chunks(chunks, min_words=10)
        assert len(result) == 1
        assert "sufficiently long" in result[0]

    def test_removes_near_duplicates(self):
        """Chunks with identical first 100 chars should be deduplicated."""
        base = "A" * 100 + " unique_suffix_"
        chunks = [base + "1 with extra words to meet minimum", base + "2 with extra words to meet minimum"]
        result = self.rag._filter_low_quality_chunks(chunks, min_words=5)
        assert len(result) == 1

    def test_removes_low_alpha_ratio(self):
        """Chunks that are mostly punctuation/numbers should be removed."""
        chunks = [
            "This is a normal sentence with enough alphabetic characters to pass the filter.",
            "1234 5678 9012 3456 7890 1234 5678 9012 3456 7890 --- ??? !!!",
        ]
        result = self.rag._filter_low_quality_chunks(chunks, min_words=5)
        assert len(result) == 1
        assert "normal sentence" in result[0]

    def test_preserves_minimum_one_chunk(self):
        """Even if all chunks are low quality, at least one is returned."""
        chunks = ["short", "tiny"]
        result = self.rag._filter_low_quality_chunks(chunks, min_words=100)
        # Should return original chunks since filter produced nothing
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        """Empty chunk list returns empty (the fallback is 'chunks' which is [])."""
        # Actually [] is falsy so fallback returns chunks which is []
        result = self.rag._filter_low_quality_chunks([])
        assert result == []

    def test_keeps_good_chunks_unchanged(self):
        """Good chunks should pass through unmodified."""
        chunks = [
            "The quarterly revenue exceeded expectations by a significant margin this period.",
            "Operating income improved due to cost reduction initiatives across all departments.",
        ]
        result = self.rag._filter_low_quality_chunks(chunks, min_words=5)
        assert result == chunks


# ===========================================================================
# 3. Excel column headers and metadata enrichment
# ===========================================================================


class TestExcelChunkEnrichment:
    """Tests for ExcelProcessor._dataframe_to_chunks summary and metadata."""

    def _make_processor(self):
        from excel_processor import ExcelProcessor

        return ExcelProcessor()

    def test_column_headers_in_every_chunk_summary(self):
        """Every chunk summary should include the full column list."""
        proc = self._make_processor()
        df = pd.DataFrame(
            {
                "Revenue": range(30),
                "2024": range(30),
                "2023": range(30),
            }
        )
        chunks = proc._dataframe_to_chunks(
            df, sheet_name="Sheet1", file_name="test.xlsx", chunk_size=500, detected_type="income_statement"
        )
        assert len(chunks) >= 1
        for chunk in chunks:
            # Every chunk's text_content should contain the Columns: line
            assert "Columns: Revenue, 2024, 2023" in chunk.text_content

    def test_summary_includes_source_file(self):
        """Chunk summary should include the source file name."""
        proc = self._make_processor()
        df = pd.DataFrame({"A": [1, 2, 3]})
        chunks = proc._dataframe_to_chunks(
            df, sheet_name="Sheet1", file_name="financials.xlsx", chunk_size=500, detected_type=None
        )
        assert len(chunks) >= 1
        assert "Source file: financials.xlsx" in chunks[0].text_content

    def test_summary_includes_statement_type(self):
        """When detected_type is set, summary shows 'Statement type:'."""
        proc = self._make_processor()
        df = pd.DataFrame({"Revenue": [100]})
        chunks = proc._dataframe_to_chunks(
            df, sheet_name="IS", file_name="f.xlsx", chunk_size=500, detected_type="income_statement"
        )
        assert "Statement type: income_statement" in chunks[0].text_content

    def test_metadata_includes_statement_type(self):
        """Chunk metadata should include statement_type."""
        proc = self._make_processor()
        df = pd.DataFrame({"Revenue": [100]})
        chunks = proc._dataframe_to_chunks(
            df, sheet_name="IS", file_name="f.xlsx", chunk_size=500, detected_type="income_statement"
        )
        assert chunks[0].metadata["statement_type"] == "income_statement"

    def test_metadata_statement_type_defaults_to_unknown(self):
        """When no detected_type, metadata statement_type should be 'unknown'."""
        proc = self._make_processor()
        df = pd.DataFrame({"A": [1]})
        chunks = proc._dataframe_to_chunks(df, sheet_name="S1", file_name="f.xlsx", chunk_size=500, detected_type=None)
        assert chunks[0].metadata["statement_type"] == "unknown"

    def test_metadata_includes_period_columns(self):
        """Metadata should list columns that look like time periods."""
        proc = self._make_processor()
        df = pd.DataFrame(
            {
                "Line Item": ["Revenue"],
                "2024": [100],
                "2023": [90],
                "Q1 2024": [25],
                "Notes": ["n/a"],
            }
        )
        chunks = proc._dataframe_to_chunks(
            df, sheet_name="IS", file_name="f.xlsx", chunk_size=500, detected_type="income_statement"
        )
        periods = chunks[0].metadata["period_columns"]
        assert "2024" in periods
        assert "2023" in periods
        assert "Q1 2024" in periods
        assert "Notes" not in periods
        assert "Line Item" not in periods

    def test_metadata_includes_total_chunks(self):
        """Metadata should include total_chunks count."""
        proc = self._make_processor()
        df = pd.DataFrame({"A": range(50)})
        chunks = proc._dataframe_to_chunks(df, sheet_name="S1", file_name="f.xlsx", chunk_size=500, detected_type=None)
        for chunk in chunks:
            assert "total_chunks" in chunk.metadata
            assert chunk.metadata["total_chunks"] >= 1

    def test_metadata_chunk_index_sequential(self):
        """Chunk indices should be sequential starting from 0."""
        proc = self._make_processor()
        df = pd.DataFrame({"A": range(100)})
        chunks = proc._dataframe_to_chunks(df, sheet_name="S1", file_name="f.xlsx", chunk_size=100, detected_type=None)
        if len(chunks) > 1:
            indices = [c.metadata["chunk_index"] for c in chunks]
            assert indices == list(range(len(chunks)))

    def test_empty_dataframe_returns_no_chunks(self):
        """An empty DataFrame should produce no chunks."""
        proc = self._make_processor()
        df = pd.DataFrame()
        chunks = proc._dataframe_to_chunks(df, sheet_name="S1", file_name="f.xlsx", chunk_size=500, detected_type=None)
        assert chunks == []
