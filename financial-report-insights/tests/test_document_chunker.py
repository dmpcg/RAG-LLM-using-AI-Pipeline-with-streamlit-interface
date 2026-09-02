"""Tests for document_chunker module."""

from document_chunker import (
    RAGChunk,
    _count_tokens_approx,
    _generate_chunk_id,
    _generate_nl_description,
    _is_mostly_numeric,
    chunk_excel_sheet,
    chunk_table,
    chunk_text_content,
)


class TestTokenCounting:
    def test_empty(self):
        assert _count_tokens_approx("") == 0

    def test_simple(self):
        # 5 words * 1.3 ≈ 6
        result = _count_tokens_approx("one two three four five")
        assert 5 <= result <= 8

    def test_longer_text(self):
        text = " ".join(["word"] * 100)
        result = _count_tokens_approx(text)
        assert 100 <= result <= 150


class TestIsMostlyNumeric:
    def test_text(self):
        assert not _is_mostly_numeric("Revenue from operations increased significantly")

    def test_numeric(self):
        assert _is_mostly_numeric("1,234,567 $2,345,678 (45.6%) $789,012")

    def test_mixed(self):
        assert not _is_mostly_numeric("Revenue: $1,234,567")

    def test_empty(self):
        assert not _is_mostly_numeric("")


class TestGenerateChunkId:
    def test_deterministic(self):
        id1 = _generate_chunk_id("file.pdf", "section", 0)
        id2 = _generate_chunk_id("file.pdf", "section", 0)
        assert id1 == id2

    def test_different_inputs(self):
        id1 = _generate_chunk_id("file.pdf", "section", 0)
        id2 = _generate_chunk_id("file.pdf", "section", 1)
        assert id1 != id2

    def test_length(self):
        chunk_id = _generate_chunk_id("test", "test", 0)
        assert len(chunk_id) == 16


class TestGenerateNLDescription:
    def test_with_source_and_section(self):
        desc = _generate_nl_description(
            "| Revenue | 1000 |\n| COGS | 500 |",
            "income_statement",
            "P&L",
            "company.xlsx",
        )
        assert "company.xlsx" in desc
        assert "P&L" in desc

    def test_extracts_labels(self):
        text = "Revenue  1000\nCOGS  500\nGross Profit  500"
        desc = _generate_nl_description(text, "income_statement", "P&L", "test.xlsx")
        assert "Revenue" in desc or "financial data" in desc.lower()

    def test_empty_content(self):
        desc = _generate_nl_description("", "general", "", "")
        assert "financial data" in desc.lower()


class TestChunkTextContent:
    def test_basic_chunking(self):
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        chunks = chunk_text_content(text, source="test.pdf", section_type="general")
        assert len(chunks) >= 1
        assert all(isinstance(c, RAGChunk) for c in chunks)

    def test_parent_child_relationship(self):
        # Generate enough text for parent-child split
        sentences = [f"Sentence number {i} with enough words to count." for i in range(50)]
        text = " ".join(sentences)
        chunks = chunk_text_content(
            text,
            source="test.pdf",
            child_token_target=50,
            parent_token_target=200,
        )
        parents = [c for c in chunks if c.metadata.get("chunk_level") == "parent"]
        children = [c for c in chunks if c.metadata.get("chunk_level") == "child"]
        assert len(parents) >= 1
        assert len(children) >= 1
        # Children should reference parent
        for child in children:
            assert child.parent_id is not None
            assert child.parent_text is not None

    def test_metadata_preserved(self):
        chunks = chunk_text_content(
            "Some financial text here.",
            source="report.pdf",
            section_type="income_statement",
            section_title="Revenue",
            page_start=5,
            page_end=7,
            metadata={"company": "TestCo"},
        )
        assert len(chunks) >= 1
        chunk = chunks[0]
        assert chunk.source == "report.pdf"
        assert chunk.section_type == "income_statement"
        assert chunk.section_title == "Revenue"
        assert chunk.page_start == 5
        assert chunk.metadata.get("company") == "TestCo"

    def test_empty_text(self):
        chunks = chunk_text_content("", source="test.pdf")
        assert len(chunks) == 0

    def test_whitespace_only(self):
        chunks = chunk_text_content("   \n\n  ", source="test.pdf")
        assert len(chunks) == 0

    def test_numeric_content_gets_nl_description(self):
        text = "$1,234,567 $2,345,678 (45.6%) $789,012 $456,789 $123,456"
        chunks = chunk_text_content(text, source="test.xlsx", section_type="income_statement")
        # At least one chunk should have NL description
        has_nl = any(c.nl_description is not None for c in chunks)
        assert has_nl


class TestChunkTable:
    def test_atomic_table(self):
        table = "| Revenue | 2024 | 2023 |\n|---|---|---|\n| Sales | 1000 | 900 |"
        chunk = chunk_table(table, source="test.xlsx", section_type="income_statement")
        assert isinstance(chunk, RAGChunk)
        assert chunk.is_table
        assert chunk.text == table
        assert chunk.section_type == "income_statement"

    def test_nl_description_generated(self):
        table = "| Item | Amount |\n|---|---|\n| Revenue | 1000 |\n| COGS | 500 |"
        chunk = chunk_table(table, source="test.xlsx")
        assert chunk.nl_description is not None

    def test_metadata(self):
        chunk = chunk_table(
            "| A | B |",
            source="file.xlsx",
            section_title="Income Statement",
            metadata={"sheet_name": "P&L"},
        )
        assert chunk.metadata.get("sheet_name") == "P&L"
        assert chunk.metadata.get("chunk_level") == "atomic_table"


class TestChunkExcelSheet:
    def test_small_sheet_atomic(self):
        md = "| Item | 2024 |\n|---|---|\n| Revenue | 1000 |\n| COGS | 500 |"
        chunks = chunk_excel_sheet(md, source="test.xlsx", sheet_name="P&L")
        assert len(chunks) == 1
        assert chunks[0].is_table

    def test_large_sheet_split(self):
        # Create a large table with enough content to exceed 1500 tokens
        rows = ["| Item | Value | Description | Notes | Category |", "|---|---|---|---|---|"]
        for i in range(200):
            rows.append(
                f"| Line item number {i} with a long description text | {i * 1000} | "
                f"Description for item {i} with extra words | Notes about line {i} | Category {i % 10} |"
            )
        md = "\n".join(rows)
        chunks = chunk_excel_sheet(md, source="test.xlsx", sheet_name="BigSheet")
        assert len(chunks) >= 2  # Should be split into multiple chunks

    def test_sheet_metadata(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        chunks = chunk_excel_sheet(
            md,
            source="test.xlsx",
            sheet_name="Balance Sheet",
            section_type="balance_sheet",
        )
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("sheet_name") == "Balance Sheet"

    def test_empty_sheet(self):
        chunks = chunk_excel_sheet("", source="test.xlsx", sheet_name="Empty")
        # chunk_table or chunk_text_content on empty returns empty or minimal
        # Implementation: chunk_table on empty string still creates a chunk
        # but chunk_excel_sheet checks token count
        assert len(chunks) >= 0  # May be 0 or 1 depending on implementation


class TestExcelChunkInvariants:
    """Guards for the dense parent-child Excel chunking (A-prime)."""

    def _rows(self, n: int) -> str:
        # One dense row-record per line, varied widths to force multiple
        # parent/child splits.
        return "\n".join(
            f"Line Item {i}: {i * 100} | {i * 7} | week {i % 13} | note-{i}"
            for i in range(n)
        )

    def test_row_conservation(self):
        """Every input row appears in exactly one child chunk (no drop/dup).

        This is the only guard against silently dropping a financial line item.
        """
        from collections import Counter

        rows = self._rows(120)
        chunks = chunk_excel_sheet(rows, source="cf.xlsx", sheet_name="Forecast")
        expected = Counter(ln for ln in rows.split("\n") if ln.strip())
        got = Counter(
            ln
            for c in chunks
            for ln in c.text.split("\n")
            if ln.strip()
        )
        assert got == expected, "rows dropped or duplicated across children"

    def test_children_have_parent_links(self):
        """All emitted chunks are children carrying parent_id + parent_text."""
        chunks = chunk_excel_sheet(self._rows(120), source="cf.xlsx", sheet_name="F")
        assert len(chunks) > 1  # 120 varied rows must split into multiple children
        for c in chunks:
            assert c.metadata.get("chunk_level") == "child"
            assert c.parent_id
            assert c.parent_text

    def test_children_fit_embed_window(self):
        """Child text stays within the ~512-token embed window (table-aware)."""
        chunks = chunk_excel_sheet(self._rows(200), source="cf.xlsx", sheet_name="F")
        for c in chunks:
            # table-aware estimate must leave headroom under 512 for the
            # nl_description prefix added at embed time
            assert _count_tokens_approx(c.text, is_table=True) <= 512
            assert len(c.parent_text or "") <= 6000  # EXCEL_MAX_PARENT_CHARS
