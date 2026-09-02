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


def _count_tokens_approx(text: str, is_table: bool = False) -> int:
    """Approximate token count.

    Prose: words * 1.3 (English heuristic).
    Tables/markdown (``is_table=True``): a word-based count under-estimates
    table content ~3x because digits, punctuation and pipe separators each
    split into their own WordPiece tokens. Use a conservative char-based
    estimate (~3 chars/token) instead, so table chunks stay safely under the
    embedder's 512-token window rather than overflowing and being truncated.
    """
    if is_table:
        return len(text) // 3 + 1
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


# Excel chunk sizing (REAL tokens, table-aware count). Children are the units
# actually embedded and must fit mxbai's 512-token window once the nl_description
# prefix (~60 tokens) is prepended at embed time, so the child text target leaves
# headroom. Parents are larger context units swapped in at retrieval time.
EXCEL_CHILD_TOKEN_TARGET = 350
EXCEL_PARENT_TOKEN_TARGET = 1100
EXCEL_MAX_PARENT_CHARS = 6000  # cap parent_text injected into the LLM prompt


def chunk_excel_sheet(
    df_markdown: str,
    source: str,
    sheet_name: str,
    section_type: str = "general",
    metadata: Optional[Dict[str, Any]] = None,
    child_token_target: int = EXCEL_CHILD_TOKEN_TARGET,
    parent_token_target: int = EXCEL_PARENT_TOKEN_TARGET,
) -> List[RAGChunk]:
    """Chunk an Excel sheet rendered as markdown into parent-child chunks.

    Every sheet is split into small CHILD chunks sized to the embedder's token
    window (table-aware count) and grouped under larger PARENT context blocks.
    Only children are emitted for
    embedding/indexing; each child carries ``parent_id`` + ``parent_text`` so
    ``_expand_parent_chunks`` can swap in the richer parent context at retrieval
    time. This replaces the old behaviour where sheets became single ~1200-token
    atomic blocks that overflowed the 512-token window and were silently
    truncated to ~18% coverage.

    Args:
        df_markdown: Sheet content as a (dense) markdown table.
        source: Source file name.
        sheet_name: Excel sheet name.
        section_type: Detected financial section type.
        metadata: Additional metadata.
        child_token_target: Target real-token size of child table text.
        parent_token_target: Target real-token size of parent context blocks.

    Returns:
        List of child RAGChunk objects (chunk_level == "child").
    """
    meta = metadata or {}
    meta["sheet_name"] = sheet_name

    # _df_to_markdown emits one dense row-record per line (non-empty cells only,
    # no table header/separator). Treat each non-blank line as one row.
    rows = [ln for ln in df_markdown.split("\n") if ln.strip()]
    if not rows:
        return []

    # 1) Group rows into PARENT context blocks (row boundaries preserved).
    #    NOTE: main's "small sheets -> atomic table chunk" shortcut is
    #    deliberately not reinstated here. That path is what produced single
    #    ~1200-token blocks which overflowed the 512-token embed window and were
    #    truncated to ~18% coverage -- the defect this parent/child split fixes.
    parent_blocks: List[List[str]] = []
    cur_rows: List[str] = []
    cur_tokens = 0
    for row in rows:
        rt = _count_tokens_approx(row, is_table=True)
        if cur_rows and cur_tokens + rt > parent_token_target:
            parent_blocks.append(cur_rows)
            cur_rows = []
            cur_tokens = 0
        cur_rows.append(row)
        cur_tokens += rt
    if cur_rows:
        parent_blocks.append(cur_rows)

    chunks: List[RAGChunk] = []
    multi_part = len(parent_blocks) > 1

    # 2) Split each parent block into CHILD chunks sized to the embed window.
    #    Children carry only their own rows (no repeated header); the
    #    nl_description prefix added at embed time supplies sheet/section context.
    for p_idx, p_rows in enumerate(parent_blocks):
        title = f"{sheet_name} (part {p_idx + 1})" if multi_part else sheet_name
        parent_id = _generate_chunk_id(source, f"{sheet_name}:parent", p_idx)
        parent_text = "\n".join(p_rows)
        if len(parent_text) > EXCEL_MAX_PARENT_CHARS:
            parent_text = parent_text[:EXCEL_MAX_PARENT_CHARS]

        # Sub-group the parent's rows into child-sized blocks.
        child_groups: List[List[str]] = []
        cur_child: List[str] = []
        cur_child_tokens = 0
        for row in p_rows:
            rt = _count_tokens_approx(row, is_table=True)
            if cur_child and cur_child_tokens + rt > child_token_target:
                child_groups.append(cur_child)
                cur_child = []
                cur_child_tokens = 0
            cur_child.append(row)
            cur_child_tokens += rt
        if cur_child:
            child_groups.append(cur_child)

        for child_idx, child_rows in enumerate(child_groups):
            child_text = "\n".join(child_rows)
            child_id = _generate_chunk_id(
                source,
                f"{sheet_name}:child",
                f"{p_idx}-{child_idx}",
            )
            child = RAGChunk(
                chunk_id=child_id,
                text=child_text,
                parent_id=parent_id,
                parent_text=parent_text,
                source=source,
                section_type=section_type,
                section_title=title,
                is_table=True,
                metadata={
                    **meta,
                    "chunk_level": "child",
                    "part_index": p_idx,
                    "child_index": child_idx,
                },
            )
            child.nl_description = _generate_nl_description(
                child_text,
                section_type,
                title,
                source,
            )
            chunks.append(child)

    return chunks
