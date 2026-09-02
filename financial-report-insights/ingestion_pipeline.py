"""
Ingestion Pipeline for Financial Documents.

Orchestrates: parse → chunk → embed → index.
Supports Excel (.xlsx, .xlsm, .xls, .csv) and PDF files.

Produces RAGChunk objects with embeddings ready for vector storage.
Integrates with SimpleRAG's document/embedding stores.
"""

import csv
import logging
import re
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import settings
from document_chunker import RAGChunk, chunk_excel_sheet, chunk_table, chunk_text_content

logger = logging.getLogger(__name__)

# Supported file extensions
EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".csv", ".tsv"}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".txt", ".md", ".docx"}


def _detect_sheet_section_type(df: pd.DataFrame, sheet_name: str) -> str:
    """Detect the financial section type of an Excel sheet.

    Uses column names, row labels, and sheet name to classify.
    """
    from line_item_mapper import map_label

    # Check sheet name first
    sheet_lower = sheet_name.lower()
    # Ordered list of (hint, section_type) — longer/more specific hints first
    sheet_type_hints = [
        ("income", "income_statement"),
        ("p&l", "income_statement"),
        ("profit", "income_statement"),
        ("revenue", "income_statement"),
        ("balance", "balance_sheet"),
        ("assets", "balance_sheet"),
        ("cash flow", "cash_flow_statement"),
        ("cash_flow", "cash_flow_statement"),
        ("equity", "equity_statement"),
        ("budget", "budget"),
        ("forecast", "forecast"),
        ("fcst", "forecast"),
        ("kpi", "kpi_dashboard"),
        ("metric", "kpi_dashboard"),
        ("dashboard", "kpi_dashboard"),
        ("debt", "debt_schedule"),
        ("loan", "debt_schedule"),
        ("covenant", "covenant_analysis"),
        ("280e", "280e_tax"),
        ("production", "production_schedule"),
        ("cultivation", "production_schedule"),
        ("rent roll", "rent_roll"),
        ("construction", "construction_budget"),
        ("cap rate", "cap_rate_analysis"),
        ("dcf", "dcf"),
        ("lbo", "lbo"),
        ("valuation", "dcf"),
        ("comps", "comparable_companies"),
        ("waterfall", "debt_schedule"),
        ("capital account", "capital_account"),
        ("fund", "fund_performance"),
        ("assumptions", "assumptions"),
        ("sensitivity", "sensitivity_analysis"),
        ("scenario", "scenario_analysis"),
    ]

    for hint, section in sheet_type_hints:
        if hint in sheet_lower:
            return section

    # Sample row labels to detect content type
    label_col = _find_label_column(df)
    if label_col is not None:
        labels = df.iloc[:30, label_col].dropna().astype(str).tolist()
        category_counts: Dict[str, int] = {}
        for label in labels:
            mapped = map_label(label)
            if mapped:
                cat = mapped.category
                category_counts[cat] = category_counts.get(cat, 0) + 1

        if category_counts:
            top_category = max(category_counts, key=lambda k: category_counts[k])
            # Map category to section type
            cat_to_section = {
                "income_statement": "income_statement",
                "balance_sheet": "balance_sheet",
                "cash_flow": "cash_flow_statement",
                "valuation": "dcf",
                "debt": "debt_schedule",
                "cannabis": "280e_tax",
                "cash_forecast": "forecast",
                "kpi": "kpi_dashboard",
                "saas": "kpi_dashboard",
                "construction": "construction_budget",
                "fund": "fund_performance",
            }
            return cat_to_section.get(top_category, "general")

    return "general"


def _find_label_column(df: pd.DataFrame) -> Optional[int]:
    """Find the column most likely to contain line item labels.

    Often column B (index 1) in real models, sometimes column A.
    Looks for the column with the most string values.
    """
    if df.empty:
        return None

    best_col = None
    best_score = 0

    for col_idx in range(min(5, len(df.columns))):  # Check first 5 columns
        col = df.iloc[:, col_idx]
        # Judge by parsed value, not storage dtype: the CSV reader returns an
        # all-str frame, so "1234.50" is a str and would otherwise count as a
        # label and let an amounts column win over the real line-item column.
        numeric = pd.to_numeric(col, errors="coerce")
        is_label = col.apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 2) & numeric.isna()
        str_count = int(is_label.sum())
        # Penalize columns that are mostly numeric
        num_count = int(numeric.notna().sum())
        score = str_count - num_count * 0.5
        if score > best_score:
            best_score = score
            best_col = col_idx

    return best_col if best_score > 3 else None


def _is_meaningful_header(name: str) -> bool:
    """True if a column header carries semantic meaning worth embedding.

    Excel exports without a header band become ``Unnamed: N`` (renamed to
    ``Col_N`` upstream); attaching those to every value is pure noise, so we
    emit value-only for them and ``Header: value`` only for real labels.
    """
    s = str(name).strip()
    if not s:
        return False
    if re.match(r"^(unnamed|col)[\s_:]*\d*$", s, re.IGNORECASE):
        return False
    # Date/period headers (e.g. "2026-02-20 00:00:00", "01/05/2026") carry no
    # letters but are meaningful labels on time-series sheets: without them the
    # weekly/monthly values render as bare numbers detached from their period.
    if re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}", s):
        return True
    return bool(re.search(r"[A-Za-z]", s))


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 200) -> str:
    """Render a DataFrame densely: one line per row, non-empty cells only.

    On sparse financial sheets, ``to_markdown`` pads every cell to column width
    and emits all NaN cells, so ~80% of the output is whitespace/pipe padding
    that both dilutes embeddings and inflates the volume to embed ~5x. This form
    drops empty cells and alignment padding entirely: each row becomes its
    non-empty cells joined by `` | ``, with the column name attached only when
    it is a real label (``Header: value``), else value-only. Downstream chunking
    treats each line as one row record. Limits to max_rows to bound volume.
    """
    if df.empty:
        return ""

    # Drop all-blank rows and columns before truncating, so data past the limit
    # isn't lost to the blank padding pandas reads from sparse sheets.
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if df.empty:
        return ""
    df = df.head(max_rows)

    headers = [str(c) for c in df.columns]
    meaningful = [_is_meaningful_header(h) for h in headers]
    lines: List[str] = []
    for _, row in df.iterrows():
        cells: List[str] = []
        for header, is_named, value in zip(headers, meaningful, row):
            if pd.notna(value) and str(value).strip() != "":
                val = str(value).strip()
                cells.append(f"{header}: {val}" if is_named else val)
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _read_csv_robust(file_path: Path, sep: str) -> pd.DataFrame:
    """Read a CSV/TSV that may be wrapped in banner/footer rows around the table.

    Bank/QuickBooks exports routinely surround the real table with non-table
    rows: a leading title or balance banner (e.g. "Available Balance : $X", a
    blank line, then the genuine header), and a trailing summary footer (e.g. a
    "Debit,Credit,Date" mini-header followed by a period-totals row). A naive
    ``pd.read_csv`` infers the column count from the banner and chokes on the
    wider data ("Expected 3 fields, saw 8"); even with skiprows, the narrow
    footer rows get NaN-padded into the data columns, so a totals figure lands
    under the wrong header and pollutes retrieval with a misleading chunk.

    Strategy: the genuine table is the block of rows sharing the modal field
    width. We find the header as the first modal-width row, then keep the rows
    below it, padding those a field or two short (exports commonly omit trailing
    empty fields) and dropping only what is far off-shape: banners, footers and
    repeated headers. Stray BOMs are stripped. For a normal single-header CSV the
    whole file is modal-width from row 0, so behaviour is unchanged.
    csv.reader honours quoted commas, so field counts are accurate.
    """
    # Bound the read. The pandas path this replaced passed nrows=max_workbook_rows;
    # an unbounded list() pulls an entire multi-GB CSV into memory before the row
    # limit is applied below. The headroom covers banner/footer/repeated-header
    # rows, which are dropped and so never count toward the data limit.
    row_limit = settings.max_workbook_rows + 100

    rows: list[list[str]] = []
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            with open(file_path, newline="", encoding=enc) as fh:
                rows = list(islice(csv.reader(fh, delimiter=sep), row_limit))
            break
        except UnicodeDecodeError:
            continue

    if not rows:
        # Could not read raw rows; fall back to a tolerant pandas read.
        return pd.read_csv(
            file_path,
            sep=sep,
            engine="python",
            on_bad_lines="skip",
            encoding="utf-8-sig",
            nrows=settings.max_workbook_rows,
        )

    def _clean(cell: str) -> str:
        return str(cell).replace("﻿", "").strip()

    populated = [r for r in rows if any(_clean(c) for c in r)]
    if not populated:
        return pd.DataFrame()

    # Infer the table width from rows that could actually be table rows. Banner
    # and footer lines hold a single cell; counting them lets a file with as many
    # banners as data rows infer a 1-column table and collapse the whole sheet.
    # Ties break toward the wider shape for the same reason.
    shaped = [r for r in populated if sum(1 for c in r if _clean(c)) >= 2]
    width_counts = Counter(len(r) for r in (shaped or populated))
    modal = max(width_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    header_idx = next(
        (i for i, r in enumerate(rows) if len(r) == modal and sum(1 for c in r if _clean(c)) >= max(2, modal * 0.5)),
        0,
    )
    header = [_clean(c) for c in rows[header_idx]]
    header_tokens = {h for h in header if h}

    # A ragged row is only a banner/footer when it is *far* off the table shape.
    # Real bank and QuickBooks exports routinely omit trailing empty fields, so a
    # row one or two fields short is a transaction with blanks, not a footer --
    # pandas keeps it as NaN, and dropping it silently loses the transaction.
    min_data_width = max(2, int(modal * 0.6))

    data: list[list[str]] = []
    for r in rows[header_idx + 1 :]:
        cleaned = [_clean(c) for c in r]
        if not any(cleaned):
            continue  # fully blank
        if len(cleaned) != modal:
            # Off-shape: keep it only if it still looks like a table row.
            if len(cleaned) < min_data_width or sum(1 for c in cleaned if c) < 2:
                continue  # banner / footer / summary row off the table shape
            cleaned = (cleaned + [""] * modal)[:modal]  # pad short, trim stray extras
        if {c for c in cleaned if c} <= header_tokens and len(header_tokens) > 1:
            continue  # repeated header block
        data.append(cleaned)
        if len(data) >= settings.max_workbook_rows:
            break

    dropped = len(populated) - 1 - len(data)
    if header_idx or dropped > 0:
        logger.info(
            "CSV '%s': header at row %d (width %d), kept %d data rows, dropped %d off-shape banner/footer row(s)",
            file_path.name,
            header_idx,
            modal,
            len(data),
            max(dropped, 0),
        )
    return pd.DataFrame(data, columns=header)


def _detect_excel_header_row(raw: pd.DataFrame, max_scan: int = 15) -> int:
    """Pick the header row for a sheet that may have sparse banner rows above it.

    Financial workbooks frequently prefix a table with a title row and a
    provenance/source row (each populating a single cell), and the genuine
    header is often a row of *dates* (period columns) rather than text -- so a
    text-vs-numeric heuristic misfires. Row density is the robust signal: the
    header is the first row whose populated-cell count reaches most of the
    table's width. Returns 0 (no banner) when the first row is already dense,
    so ordinary single-header sheets are unaffected.
    """
    if raw.empty:
        return 0
    pop = raw.notna().sum(axis=1)
    max_pop = int(pop.max())
    # Narrow sheets (label/value statements like a QuickBooks P&L or balance
    # sheet) have no real column header -- the line items are 2-3 cells wide and
    # any "dense" row is actually data. Skipping banners there would consume a
    # data row as the header, so leave those as header=0 (original behaviour).
    if max_pop < 4:
        return 0
    target = max(3, int(max_pop * 0.6))
    for i in range(min(max_scan, len(raw))):
        if pop.iloc[i] >= target:
            return i
    return 0


def ingest_excel(
    file_path: Path,
    child_token_target: int = 250,
    parent_token_target: int = 1200,
) -> List[RAGChunk]:
    """Ingest an Excel file into RAGChunks.

    Processes each sheet separately, detects section types,
    and creates financial-aware chunks.

    Args:
        file_path: Path to the Excel file.
        child_token_target: Target child chunk size in tokens.
        parent_token_target: Target parent chunk size in tokens.

    Returns:
        List of RAGChunk objects ready for embedding.
    """
    file_path = Path(file_path)
    source = file_path.name
    all_chunks: List[RAGChunk] = []

    try:
        suffix = file_path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            df = _read_csv_robust(file_path, sep)
            # Fix unnamed columns from headerless Excel exports
            df.columns = [c if not str(c).startswith("Unnamed") else f"Col_{i}" for i, c in enumerate(df.columns)]
            sheets = [("Sheet1", df)]
        else:
            xls = pd.ExcelFile(file_path)
            sheets = []
            for sheet_name in xls.sheet_names:
                try:
                    # Read raw (no header), detect the real header row below any
                    # title/provenance banner, then re-key the frame from it.
                    raw = pd.read_excel(
                        xls,
                        sheet_name=sheet_name,
                        header=None,
                        nrows=settings.max_workbook_rows,
                    )
                    if raw.empty:
                        continue
                    hdr = _detect_excel_header_row(raw)
                    cols = raw.iloc[hdr].tolist()
                    df = raw.iloc[hdr + 1 :].reset_index(drop=True)
                    # Name columns from the detected header; fall back to Col_N
                    # for blank/unnamed cells.
                    df.columns = [
                        str(c).strip()
                        if (pd.notna(c) and str(c).strip() and not str(c).startswith("Unnamed"))
                        else f"Col_{i}"
                        for i, c in enumerate(cols)
                    ]
                    if hdr:
                        logger.info(
                            "Excel '%s' sheet '%s': header detected at row %d (skipped %d banner row(s))",
                            source,
                            sheet_name,
                            hdr,
                            hdr,
                        )
                    # Skip empty sheets
                    if df.empty or (df.shape[0] < 2 and df.shape[1] < 2):
                        logger.warning(
                            "Skipping empty/near-empty sheet '%s' (shape=%s) in %s",
                            sheet_name,
                            df.shape,
                            source,
                        )
                        continue
                    sheets.append((sheet_name, df))
                except Exception as e:
                    logger.warning("Failed to read sheet '%s' from %s: %s", sheet_name, source, e)

        for sheet_name, df in sheets:
            section_type = _detect_sheet_section_type(df, sheet_name)
            md = _df_to_markdown(df)

            if not md.strip():
                logger.warning(
                    "Skipping sheet '%s' (shape=%s) in %s: produced blank markdown",
                    sheet_name,
                    df.shape,
                    source,
                )
                continue

            meta = {
                "source_file": source,
                "sheet_name": sheet_name,
                "file_type": "excel",
                "section_type": section_type,
                "row_count": len(df),
                "col_count": len(df.columns),
            }

            sheet_chunks = chunk_excel_sheet(
                md,
                source=source,
                sheet_name=sheet_name,
                section_type=section_type,
                metadata=meta,
            )
            all_chunks.extend(sheet_chunks)

        if sheets and not all_chunks:
            logger.warning(
                "Ingested Excel '%s': %d sheets present but produced 0 chunks",
                source,
                len(sheets),
            )
        else:
            logger.info(
                "Ingested Excel '%s': %d sheets -> %d chunks",
                source,
                len(sheets),
                len(all_chunks),
            )

    except Exception as e:
        logger.error("Failed to ingest Excel file %s: %s", source, e)

    return all_chunks


def ingest_pdf(
    file_path: Path,
    child_token_target: int = 250,
    parent_token_target: int = 1200,
) -> List[RAGChunk]:
    """Ingest a PDF file into RAGChunks.

    Uses pdf_parser for structured extraction, then document_chunker
    for financial-aware chunking.

    Args:
        file_path: Path to the PDF file.
        child_token_target: Target child chunk size in tokens.
        parent_token_target: Target parent chunk size in tokens.

    Returns:
        List of RAGChunk objects ready for embedding.
    """
    from pdf_parser import parse_pdf

    file_path = Path(file_path)
    source = file_path.name
    all_chunks: List[RAGChunk] = []

    try:
        parsed = parse_pdf(file_path)

        doc_meta = {
            "source_file": source,
            "file_type": "pdf",
            "company": parsed.company,
            "period": parsed.period,
            "total_pages": parsed.total_pages,
            **parsed.metadata,
        }

        for section in parsed.sections:
            # Handle tables as atomic chunks
            for table in section.tables:
                table_chunk = chunk_table(
                    table,
                    source=source,
                    section_type=section.section_type,
                    section_title=section.title,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    metadata=doc_meta,
                )
                all_chunks.append(table_chunk)

            # Chunk the non-table text content
            # Remove tables from section content to avoid double-indexing
            text_content = section.content
            for table in section.tables:
                text_content = text_content.replace(table, "")

            text_content = text_content.strip()
            if text_content and len(text_content) > 50:
                text_chunks = chunk_text_content(
                    text_content,
                    source=source,
                    section_type=section.section_type,
                    section_title=section.title,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    child_token_target=child_token_target,
                    parent_token_target=parent_token_target,
                    metadata=doc_meta,
                )
                all_chunks.extend(text_chunks)

        logger.info(
            "Ingested PDF '%s': %d sections -> %d chunks",
            source,
            len(parsed.sections),
            len(all_chunks),
        )

    except Exception as e:
        logger.error("Failed to ingest PDF file %s: %s", source, e, exc_info=True)

    return all_chunks


def ingest_text(
    file_path: Path,
    child_token_target: int = 250,
    parent_token_target: int = 1200,
) -> List[RAGChunk]:
    """Ingest a text/markdown/docx file into RAGChunks.

    Args:
        file_path: Path to the text file.
        child_token_target: Target child chunk size.
        parent_token_target: Target parent chunk size.

    Returns:
        List of RAGChunk objects.
    """
    file_path = Path(file_path)
    source = file_path.name
    suffix = file_path.suffix.lower()

    # Size guard: reject files larger than 50 MB to prevent OOM
    _MAX_TEXT_BYTES = 50 * 1024 * 1024  # 50 MB
    try:
        file_size = file_path.stat().st_size
    except OSError:
        file_size = 0
    if file_size > _MAX_TEXT_BYTES:
        logger.warning(
            "Text file %s is %d bytes (limit %d). Skipping.",
            source,
            file_size,
            _MAX_TEXT_BYTES,
        )
        return []

    try:
        if suffix == ".docx":
            from docx import Document as DocxDocument

            doc = DocxDocument(file_path)
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            text = file_path.read_text(encoding="utf-8", errors="replace")

        if not text.strip():
            return []

        meta = {
            "source_file": source,
            "file_type": "text",
        }

        chunks = chunk_text_content(
            text,
            source=source,
            section_type="general",
            section_title=file_path.stem,
            child_token_target=child_token_target,
            parent_token_target=parent_token_target,
            metadata=meta,
        )

        logger.info("Ingested text '%s': %d chunks", source, len(chunks))
        return chunks

    except Exception as e:
        logger.error("Failed to ingest text file %s: %s", source, e)
        return []


def ingest_file(file_path: Path) -> List[RAGChunk]:
    """Ingest a single file into RAGChunks.

    Auto-detects file type and dispatches to the appropriate handler.

    Args:
        file_path: Path to the file.

    Returns:
        List of RAGChunk objects ready for embedding.
    """
    file_path = Path(file_path).resolve()
    if not file_path.is_file():
        logger.warning("File not found or not a file: %s", file_path)
        return []
    suffix = file_path.suffix.lower()

    if suffix in EXCEL_EXTENSIONS:
        return ingest_excel(file_path)
    elif suffix in PDF_EXTENSIONS:
        return ingest_pdf(file_path)
    elif suffix in TEXT_EXTENSIONS:
        return ingest_text(file_path)
    else:
        logger.warning("Unsupported file type: %s", suffix)
        return []


def chunks_to_documents(chunks: List[RAGChunk]) -> List[Dict[str, Any]]:
    """Convert RAGChunks to the document dict format used by SimpleRAG.

    The text used for embedding includes the NL description prefix
    when available, improving retrieval quality for numeric content.

    Args:
        chunks: List of RAGChunk objects.

    Returns:
        List of document dicts compatible with SimpleRAG.documents.
    """
    documents: List[Dict[str, Any]] = []

    for chunk in chunks:
        # Build the text to embed: NL description + actual content
        embed_text = chunk.text
        if chunk.nl_description:
            embed_text = chunk.nl_description + "\n\n" + chunk.text

        doc = {
            "source": chunk.source,
            "content": embed_text,
            "type": chunk.metadata.get("file_type", "unknown"),
            "metadata": {
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "parent_text": chunk.parent_text,
                "section_type": chunk.section_type,
                "section_title": chunk.section_title,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "is_table": chunk.is_table,
                "source": chunk.source,
                **chunk.metadata,
            },
        }
        documents.append(doc)

    return documents
