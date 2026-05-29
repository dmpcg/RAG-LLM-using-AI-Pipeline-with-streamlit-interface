"""
Financial-Aware Document Chunker.

Implements parent-child chunking strategy optimized for financial documents:
- Child chunks (200-300 tokens) for retrieval precision
- Parent chunks (1000-1500 tokens) for LLM context
- Tables preserved as atomic chunks (never split)
- Section boundaries respected
- Natural language descriptions generated for numeric-heavy chunks
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RAGChunk:
    """A chunk ready for embedding and indexing.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        text: The text content to embed.
        parent_id: ID of the parent chunk (if this is a child).
        parent_text: Full parent chunk text (for LLM context expansion).
        source: Source file name/path.
        section_type: Detected financial section type.
        section_title: Section heading text.
        page_start: Starting page number (PDF) or sheet name (Excel).
        page_end: Ending page number.
        is_table: Whether this chunk is a table.
        metadata: Additional metadata dict.
        nl_description: Natural language description of numeric content.
    """

    chunk_id: str
    text: str
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None
    source: str = ""
    section_type: str = "general"
    section_title: str = ""
    page_start: int = 0
    page_end: int = 0
    is_table: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    nl_description: Optional[str] = None


def _generate_chunk_id(source: str, section: str, index) -> str:
    """Generate a deterministic chunk ID."""
    key = f"{source}:{section}:{index}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _count_tokens_approx(text: str) -> int:
    """Approximate token count (words * 1.3 for English text)."""
    return int(len(text.split()) * 1.3)


def _is_mostly_numeric(text: str) -> bool:
    """Check if text is predominantly numbers/tables (poor for embedding)."""
    if not text.strip():
        return False
    # Count numeric characters vs alpha
    nums = sum(1 for c in text if c.isdigit() or c in ".,%-$()£€¥")
    alphas = sum(1 for c in text if c.isalpha())
    return nums > alphas * 2 if alphas > 0 else nums > 10


def _generate_nl_description(
    text: str,
    section_type: str,
    section_title: str,
    source: str,
) -> str:
    """Generate a natural language description for numeric-heavy chunks.

    This is prepended to the chunk text before embedding to improve
    retrieval quality for queries about specific financial data.
    """
    # Build a descriptive prefix
    parts = []
    if source:
        parts.append(f"From {source}")
    if section_title:
        parts.append(f"section '{section_title}'")
    if section_type and section_type != "general":
        readable = section_type.replace("_", " ").title()
        parts.append(f"({readable})")

    prefix = ", ".join(parts) + "." if parts else ""

    # Try to extract line item names from the table
    lines = text.strip().split("\n")
    items = []
    for line in lines[:15]:  # Sample first 15 lines
        # Strip markdown table formatting
        clean = re.sub(r"[|]", " ", line).strip()
        clean = re.sub(r"[-=]+", "", clean).strip()
        if not clean:
            continue
        # Extract the first text column (likely label)
        parts_line = [p.strip() for p in clean.split("  ") if p.strip()]
        if parts_line:
            label = parts_line[0]
            # Only include if it looks like a label (has letters)
            if re.search(r"[a-zA-Z]{2,}", label) and len(label) < 80:
                items.append(label)

    if items:
        items_str = ", ".join(items[:8])
        return f"{prefix} Contains financial data including: {items_str}."

    return f"{prefix} Contains financial data and calculations."


def _split_text_into_sentences(text: str) -> List[str]:
    """Split text into sentences, respecting abbreviations."""
    # Simple sentence splitter that handles common abbreviations
    # Don't split on periods in numbers, abbreviations like "Inc.", "vs.", etc.
    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z])",
        text,
    )
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_content(
    text: str,
    source: str,
    section_type: str = "general",
    section_title: str = "",
    page_start: int = 0,
    page_end: int = 0,
    child_token_target: int = 250,
    parent_token_target: int = 1200,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[RAGChunk]:
    """Chunk text content using parent-child strategy.

    Args:
        text: Text to chunk.
        source: Source file identifier.
        section_type: Financial section type.
        section_title: Section heading.
        page_start: Starting page.
        page_end: Ending page.
        child_token_target: Target tokens for child (retrieval) chunks.
        parent_token_target: Target tokens for parent (context) chunks.
        metadata: Additional metadata.

    Returns:
        List of RAGChunk objects. Child chunks have parent_id and parent_text set.
    """
    if not text.strip():
        return []

    meta = metadata or {}
    chunks: List[RAGChunk] = []
    chunk_idx = 0

    # First, split into parent-sized blocks
    sentences = _split_text_into_sentences(text)
    if not sentences:
        sentences = [text]

    parent_blocks: List[List[str]] = []
    current_block: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = _count_tokens_approx(sentence)
        if current_tokens + sent_tokens > parent_token_target and current_block:
            parent_blocks.append(current_block)
            # Overlap: keep last sentence for context continuity
            current_block = [current_block[-1]] if current_block else []
            current_tokens = _count_tokens_approx(current_block[0]) if current_block else 0
        current_block.append(sentence)
        current_tokens += sent_tokens

    if current_block:
        parent_blocks.append(current_block)

    # Now create parent and child chunks
    for block_idx, block_sentences in enumerate(parent_blocks):
        parent_text = " ".join(block_sentences)
        parent_id = _generate_chunk_id(source, section_title, block_idx)

        # Create parent chunk
        parent_chunk = RAGChunk(
            chunk_id=parent_id,
            text=parent_text,
            source=source,
            section_type=section_type,
            section_title=section_title,
            page_start=page_start,
            page_end=page_end,
            metadata={**meta, "chunk_level": "parent", "block_index": block_idx},
        )

        # Add NL description for numeric-heavy parent chunks
        if _is_mostly_numeric(parent_text):
            parent_chunk.nl_description = _generate_nl_description(
                parent_text,
                section_type,
                section_title,
                source,
            )

        chunks.append(parent_chunk)

        # Split parent into child chunks
        child_sentences: List[str] = []
        child_tokens = 0

        for sentence in block_sentences:
            sent_tokens = _count_tokens_approx(sentence)
            if child_tokens + sent_tokens > child_token_target and child_sentences:
                child_text = " ".join(child_sentences)
                child_id = _generate_chunk_id(source, f"{section_title}:child", chunk_idx)

                child_chunk = RAGChunk(
                    chunk_id=child_id,
                    text=child_text,
                    parent_id=parent_id,
                    parent_text=parent_text,
                    source=source,
                    section_type=section_type,
                    section_title=section_title,
                    page_start=page_start,
                    page_end=page_end,
                    metadata={**meta, "chunk_level": "child", "chunk_index": chunk_idx},
                )

                if _is_mostly_numeric(child_text):
                    child_chunk.nl_description = _generate_nl_description(
                        child_text,
                        section_type,
                        section_title,
                        source,
                    )

                chunks.append(child_chunk)
                chunk_idx += 1
                child_sentences = []
                child_tokens = 0

            child_sentences.append(sentence)
            child_tokens += sent_tokens

        # Flush remaining child sentences
        if child_sentences:
            child_text = " ".join(child_sentences)
            child_id = _generate_chunk_id(source, f"{section_title}:child", chunk_idx)
            child_chunk = RAGChunk(
                chunk_id=child_id,
                text=child_text,
                parent_id=parent_id,
                parent_text=parent_text,
                source=source,
                section_type=section_type,
                section_title=section_title,
                page_start=page_start,
                page_end=page_end,
                metadata={**meta, "chunk_level": "child", "chunk_index": chunk_idx},
            )
            if _is_mostly_numeric(child_text):
                child_chunk.nl_description = _generate_nl_description(
                    child_text,
                    section_type,
                    section_title,
                    source,
                )
            chunks.append(child_chunk)
            chunk_idx += 1

    return chunks


def chunk_table(
    table_text: str,
    source: str,
    section_type: str = "general",
    section_title: str = "",
    page_start: int = 0,
    page_end: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> RAGChunk:
    """Create an atomic chunk from a table (tables are never split).

    Args:
        table_text: The table content (markdown format).
        source: Source file identifier.
        section_type: Financial section type.
        section_title: Section heading.
        page_start: Starting page.
        page_end: Ending page.
        metadata: Additional metadata.

    Returns:
        A single RAGChunk for the entire table.
    """
    meta = metadata or {}
    table_hash = hashlib.sha256(table_text.encode("utf-8", errors="replace")).hexdigest()[:16]
    chunk_id = _generate_chunk_id(source, f"table:{section_title}", table_hash)

    nl_desc = _generate_nl_description(table_text, section_type, section_title, source)

    return RAGChunk(
        chunk_id=chunk_id,
        text=table_text,
        source=source,
        section_type=section_type,
        section_title=section_title,
        page_start=page_start,
        page_end=page_end,
        is_table=True,
        metadata={**meta, "chunk_level": "atomic_table"},
        nl_description=nl_desc,
    )


def chunk_excel_sheet(
    df_markdown: str,
    source: str,
    sheet_name: str,
    section_type: str = "general",
    metadata: Optional[Dict[str, Any]] = None,
) -> List[RAGChunk]:
    """Chunk an Excel sheet rendered as markdown.

    Treats the sheet as a table-heavy document. Small sheets become
    atomic table chunks; large sheets get parent-child splitting
    while respecting row boundaries.

    Args:
        df_markdown: Sheet content as markdown table.
        source: Source file name.
        sheet_name: Excel sheet name.
        section_type: Detected financial section type.
        metadata: Additional metadata.

    Returns:
        List of RAGChunk objects.
    """
    meta = metadata or {}
    meta["sheet_name"] = sheet_name

    tokens = _count_tokens_approx(df_markdown)

    # Small sheets -> atomic table chunk
    if tokens <= 1500:
        return [
            chunk_table(
                df_markdown,
                source=source,
                section_type=section_type,
                section_title=sheet_name,
                metadata=meta,
            )
        ]

    # Large sheets -> split by rows while keeping header
    lines = df_markdown.split("\n")
    # Find header rows (first 2-3 lines of a markdown table)
    header_lines: List[str] = []
    data_lines: List[str] = []
    header_done = False

    for i, line in enumerate(lines):
        if not header_done:
            header_lines.append(line)
            # Header separator line (e.g., |---|---|---|)
            if re.match(r"^\s*\|[\s\-:]+\|", line):
                header_done = True
        else:
            data_lines.append(line)

    if not data_lines:
        # No clear table structure, fall back to text chunking
        return chunk_text_content(
            df_markdown,
            source=source,
            section_type=section_type,
            section_title=sheet_name,
            metadata=meta,
        )

    header_text = "\n".join(header_lines)
    chunks: List[RAGChunk] = []
    current_rows: List[str] = []
    current_tokens = _count_tokens_approx(header_text)
    chunk_idx = 0
    parent_target = 1200

    for row in data_lines:
        row_tokens = _count_tokens_approx(row)
        if current_tokens + row_tokens > parent_target and current_rows:
            block_text = header_text + "\n" + "\n".join(current_rows)
            table_chunk = chunk_table(
                block_text,
                source=source,
                section_type=section_type,
                section_title=f"{sheet_name} (part {chunk_idx + 1})",
                metadata={**meta, "part_index": chunk_idx},
            )
            chunks.append(table_chunk)
            chunk_idx += 1
            current_rows = []
            current_tokens = _count_tokens_approx(header_text)

        current_rows.append(row)
        current_tokens += row_tokens

    # Flush remaining
    if current_rows:
        block_text = header_text + "\n" + "\n".join(current_rows)
        table_chunk = chunk_table(
            block_text,
            source=source,
            section_type=section_type,
            section_title=f"{sheet_name} (part {chunk_idx + 1})" if chunk_idx > 0 else sheet_name,
            metadata={**meta, "part_index": chunk_idx},
        )
        chunks.append(table_chunk)

    return chunks
