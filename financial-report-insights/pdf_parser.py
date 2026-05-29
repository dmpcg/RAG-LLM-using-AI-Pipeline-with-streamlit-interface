"""
PDF Parser for Financial Documents.

Converts PDF files to structured markdown with table preservation,
section boundary detection, and metadata extraction.

Uses pymupdf4llm for high-quality markdown output.
Falls back to PyMuPDF (fitz) raw text extraction if pymupdf4llm unavailable.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedSection:
    """A detected section within a PDF document."""
    title: str
    content: str
    section_type: str  # e.g. "income_statement", "balance_sheet", etc.
    page_start: int
    page_end: int
    tables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Complete parsed PDF document."""
    source_path: str
    title: str
    company: Optional[str]
    period: Optional[str]
    total_pages: int
    sections: List[ParsedSection]
    raw_markdown: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# Section header patterns — ordered from most specific to most generic
SECTION_PATTERNS: Dict[str, List[re.Pattern]] = {
    # Financial statements
    "income_statement": [
        re.compile(r"(?i)\b(consolidated\s+)?statements?\s+of\s+(income|operations|earnings|profit\s+(?:and|&)\s+loss)"),
        re.compile(r"(?i)\b(income|profit\s+(?:and|&)\s+loss|p\s*&\s*l)\s+statement"),
        re.compile(r"(?i)\bincome\s+statement\b"),
    ],
    "balance_sheet": [
        re.compile(r"(?i)\b(consolidated\s+)?balance\s+sheets?"),
        re.compile(r"(?i)\b(consolidated\s+)?statements?\s+of\s+financial\s+(position|condition)"),
    ],
    "cash_flow_statement": [
        re.compile(r"(?i)\b(consolidated\s+)?statements?\s+of\s+cash\s+flows?"),
        re.compile(r"(?i)\bcash\s+flow\s+statement"),
    ],
    "equity_statement": [
        re.compile(r"(?i)\b(consolidated\s+)?statements?\s+of\s+(stockholders|shareholders|owners)[\'\u2019]?\s+equity"),
        re.compile(r"(?i)\bchanges\s+in\s+(stockholders|shareholders)[\'\u2019]?\s+equity"),
    ],
    "comprehensive_income": [
        re.compile(r"(?i)\b(consolidated\s+)?statements?\s+of\s+comprehensive\s+(income|loss)"),
    ],

    # Regulatory/SEC filings
    "mda": [
        re.compile(r"(?i)\bmanagement[\'\u2019]?s?\s+discussion\s+(and|&)\s+analysis"),
        re.compile(r"(?i)\bmd\s*&\s*a\b"),
    ],
    "risk_factors": [
        re.compile(r"(?i)\brisk\s+factors\b"),
    ],
    "notes_to_financial_statements": [
        re.compile(r"(?i)\bnotes?\s+to\s+(the\s+)?(consolidated\s+)?financial\s+statements"),
    ],
    "auditor_report": [
        re.compile(r"(?i)\b(independent\s+)?auditor[\'\u2019]?s?\s+report"),
        re.compile(r"(?i)\breport\s+of\s+independent\s+(registered\s+)?public\s+accounting\s+firm"),
    ],

    # Valuation / M&A
    "dcf": [
        re.compile(r"(?i)\bdiscounted\s+cash\s+flow"),
        re.compile(r"(?i)\bDCF\s+(analysis|valuation|model)\b"),
    ],
    "lbo": [
        re.compile(r"(?i)\bleveraged\s+buyout"),
        re.compile(r"(?i)\bLBO\s+(analysis|model|returns)\b"),
    ],
    "comparable_companies": [
        re.compile(r"(?i)\bcomparable\s+(companies|company)\s+(analysis|table)"),
        re.compile(r"(?i)\btrading\s+comps?\b"),
        re.compile(r"(?i)\bpublic\s+comps?\b"),
    ],
    "precedent_transactions": [
        re.compile(r"(?i)\bprecedent\s+transactions?"),
        re.compile(r"(?i)\btransaction\s+comps?\b"),
    ],
    "merger_model": [
        re.compile(r"(?i)\bmerger\s+(model|analysis)"),
        re.compile(r"(?i)\baccretion[\s/]+dilution"),
    ],

    # Debt / Credit
    "debt_schedule": [
        re.compile(r"(?i)\bdebt\s+(schedule|summary|detail|waterfall)"),
        re.compile(r"(?i)\bcapital\s+structure\s+(summary|detail)"),
    ],
    "covenant_analysis": [
        re.compile(r"(?i)\bcovenant\s+(analysis|compliance|summary)"),
        re.compile(r"(?i)\bfinancial\s+covenants?\b"),
    ],

    # Cannabis-specific
    "280e_tax": [
        re.compile(r"(?i)\b280\s*e\s+(tax|classification|analysis)"),
        re.compile(r"(?i)\btax\s+rate.*280\s*e"),
    ],
    "production_schedule": [
        re.compile(r"(?i)\bproduction\s+schedule"),
        re.compile(r"(?i)\bcultivation\s+(schedule|plan)"),
    ],
    "license_inventory": [
        re.compile(r"(?i)\blicense\s+(inventory|summary|status)"),
        re.compile(r"(?i)\bregulatory\s+license"),
    ],

    # Real estate
    "rent_roll": [
        re.compile(r"(?i)\brent\s+roll\b"),
    ],
    "cap_rate_analysis": [
        re.compile(r"(?i)\bcap(italization)?\s+rate\s+(analysis|summary)"),
    ],
    "construction_budget": [
        re.compile(r"(?i)\bconstruction\s+budget"),
    ],

    # Fund reporting
    "fund_performance": [
        re.compile(r"(?i)\bfund\s+performance"),
        re.compile(r"(?i)\bpartner\s+capital\s+(statement|account)"),
    ],
    "capital_account": [
        re.compile(r"(?i)\bcapital\s+account\s+(statement|summary)"),
    ],

    # KPIs / Operating metrics
    "kpi_dashboard": [
        re.compile(r"(?i)\bkpi\s+(dashboard|summary|metrics)"),
        re.compile(r"(?i)\boperating\s+metrics\b"),
        re.compile(r"(?i)\bkey\s+performance\s+indicators\b"),
    ],

    # Budget / Forecast
    "budget": [
        re.compile(r"(?i)\b(annual\s+)?budget\b"),
        re.compile(r"(?i)\bbudget\s+vs\.?\s+actual"),
    ],
    "forecast": [
        re.compile(r"(?i)\b(cash\s+)?forecast\b"),
        re.compile(r"(?i)\bprojections?\b"),
    ],

    # General sections
    "executive_summary": [
        re.compile(r"(?i)\bexecutive\s+summary\b"),
    ],
    "table_of_contents": [
        re.compile(r"(?i)\btable\s+of\s+contents\b"),
    ],
    "assumptions": [
        re.compile(r"(?i)\b(key\s+)?assumptions?\b"),
    ],
    "sensitivity_analysis": [
        re.compile(r"(?i)\bsensitivity\s+(analysis|table)"),
    ],
    "scenario_analysis": [
        re.compile(r"(?i)\bscenario\s+analysis"),
    ],
}


def _detect_section_type(text: str) -> Optional[str]:
    """Detect the financial section type from text content."""
    # Check first 500 chars for section headers
    header_text = text[:500]
    for section_type, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(header_text):
                return section_type
    return None


def _extract_metadata(markdown: str, source_path: str) -> Dict[str, Any]:
    """Extract company name, period, and other metadata from document content."""
    metadata: Dict[str, Any] = {}
    first_page = markdown[:3000]

    # Company name heuristics — look for common patterns
    company_patterns = [
        re.compile(r"(?i)(?:^|\n)\s*([A-Z][A-Za-z\s&,\.]+(?:Inc|Corp|LLC|LP|Ltd|Co|Group|Holdings|Partners)\.?)\s*(?:\n|$)"),
        re.compile(r"(?i)annual\s+report\s+(?:of|for)\s+(.+?)(?:\n|$)"),
        re.compile(r"(?i)(?:^|\n)\s*([A-Z][A-Z\s&]+)\s*\n\s*(?:consolidated|financial|annual|quarterly)", re.MULTILINE),
    ]
    for pat in company_patterns:
        m = pat.search(first_page)
        if m:
            name = m.group(1).strip()
            if 5 < len(name) < 100:
                metadata["company"] = name
                break

    # Period detection
    period_patterns = [
        re.compile(r"(?i)(?:for\s+the\s+)?(?:fiscal\s+)?year\s+ended?\s+(\w+\s+\d{1,2},?\s+\d{4})"),
        re.compile(r"(?i)(?:for\s+the\s+)?(?:three|six|nine|twelve)\s+months?\s+ended?\s+(\w+\s+\d{1,2},?\s+\d{4})"),
        re.compile(r"(?i)(?:Q[1-4]\s+)?(?:FY\s*)?(\d{4})\s*(?:annual|quarterly|10-[kq])", re.IGNORECASE),
        re.compile(r"(?i)(?:^|\n)\s*(?:as\s+of\s+)?(\w+\s+\d{1,2},?\s+\d{4})\s*(?:\n|$)"),
    ]
    for pat in period_patterns:
        m = pat.search(first_page)
        if m:
            metadata["period"] = m.group(1).strip()
            break

    # Filing type
    filing_patterns = [
        (re.compile(r"(?i)\b10-K\b"), "10-K"),
        (re.compile(r"(?i)\b10-Q\b"), "10-Q"),
        (re.compile(r"(?i)\b8-K\b"), "8-K"),
        (re.compile(r"(?i)\bannual\s+report\b"), "annual_report"),
        (re.compile(r"(?i)\bquarterly\s+report\b"), "quarterly_report"),
        (re.compile(r"(?i)\bprospectus\b"), "prospectus"),
        (re.compile(r"(?i)\boffering\s+memorandum\b"), "offering_memorandum"),
    ]
    for pat, ftype in filing_patterns:
        if pat.search(first_page):
            metadata["filing_type"] = ftype
            break

    return metadata


def _split_into_sections(markdown: str) -> List[Dict[str, Any]]:
    """Split markdown into sections based on heading patterns.

    Identifies markdown headings (# ## ###) and financial section headers.
    """
    # Match markdown headings or all-caps lines that look like section titles
    heading_pattern = re.compile(
        r"(?:^|\n)(#{1,4}\s+.+|[A-Z][A-Z\s&/\-]{5,}(?:\n|$))",
        re.MULTILINE,
    )

    matches = list(heading_pattern.finditer(markdown))
    if not matches:
        return [{"title": "Document", "content": markdown, "start_pos": 0}]

    sections = []
    for i, match in enumerate(matches):
        title = match.group(1).strip().lstrip("#").strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        content = markdown[start:end].strip()
        if len(content) > 20:  # Skip trivially small sections
            sections.append({
                "title": title,
                "content": content,
                "start_pos": start,
            })

    # If the document starts before the first heading, capture that preamble
    if matches and matches[0].start() > 100:
        preamble = markdown[:matches[0].start()].strip()
        if preamble:
            sections.insert(0, {
                "title": "Preamble",
                "content": preamble,
                "start_pos": 0,
            })

    return sections


def _detect_tables_in_section(content: str) -> List[str]:
    """Extract markdown tables from section content."""
    tables = []
    lines = content.split("\n")
    table_lines: List[str] = []
    in_table = False

    for line in lines:
        # Markdown table row: starts with | or has | separators
        is_table_line = bool(re.match(r"\s*\|", line)) or ("|" in line and line.strip().startswith("|"))
        if is_table_line:
            if not in_table:
                in_table = True
            table_lines.append(line)
        else:
            if in_table and len(table_lines) >= 2:
                tables.append("\n".join(table_lines))
            table_lines = []
            in_table = False

    # Flush last table
    if in_table and len(table_lines) >= 2:
        tables.append("\n".join(table_lines))

    return tables


def _empty_parsed_document(file_path: Path) -> ParsedDocument:
    """Return a zero-content ParsedDocument for unreadable/corrupt PDFs."""
    return ParsedDocument(
        source_path=str(file_path),
        title=file_path.stem,
        company=None,
        period=None,
        total_pages=0,
        sections=[],
        raw_markdown="",
    )


def parse_pdf(file_path: Path) -> ParsedDocument:
    """Parse a PDF file into structured sections with metadata.

    Attempts pymupdf4llm first for high-quality markdown output,
    falls back to raw PyMuPDF text extraction.

    Returns an empty ParsedDocument (no raise) when the file is corrupt
    or unreadable (fitz.FileDataError, RuntimeError).

    Args:
        file_path: Path to the PDF file.

    Returns:
        ParsedDocument with sections, metadata, and raw markdown.
        An empty ParsedDocument is returned for corrupt/unreadable files.
    """
    file_path = Path(file_path)
    markdown = ""
    total_pages = 0

    _MAX_PDF_PAGES = 500

    # Resolve fitz.FileDataError once so we can reference it in except clauses
    # without re-importing inside every branch.
    try:
        import fitz as _fitz_mod
        _fitz_file_data_error: type = _fitz_mod.FileDataError
    except ImportError:
        # fitz not installed; define a dummy so the except clause compiles safely
        _fitz_file_data_error = type("_FitzFileDataErrorStub", (Exception,), {})

    # Try pymupdf4llm first (best quality)
    try:
        import pymupdf4llm
        import fitz as _fitz_check
        _doc_check = _fitz_check.open(file_path)
        total_pages = len(_doc_check)
        _doc_check.close()
        if total_pages > _MAX_PDF_PAGES:
            logger.warning(
                "PDF %s has %d pages (limit %d); truncating to first %d pages.",
                file_path.name, total_pages, _MAX_PDF_PAGES, _MAX_PDF_PAGES,
            )
        page_list = list(range(min(total_pages, _MAX_PDF_PAGES)))
        markdown = pymupdf4llm.to_markdown(str(file_path), pages=page_list)
        logger.info("Parsed PDF with pymupdf4llm: %s (%d pages)", file_path.name, total_pages)
    except ImportError:
        logger.info("pymupdf4llm not available, falling back to fitz")
        try:
            import fitz
            doc = fitz.open(file_path)
            total_pages = len(doc)
            if total_pages > _MAX_PDF_PAGES:
                logger.warning(
                    "PDF %s has %d pages (limit %d); truncating to first %d pages.",
                    file_path.name, total_pages, _MAX_PDF_PAGES, _MAX_PDF_PAGES,
                )
            pages = []
            for page_idx, page in enumerate(doc):
                if page_idx >= _MAX_PDF_PAGES:
                    break
                pages.append(page.get_text("text"))
            doc.close()
            markdown = "\n\n".join(pages)
        except ImportError:
            raise ImportError(
                "Neither pymupdf4llm nor PyMuPDF (fitz) is installed. "
                "Install with: pip install pymupdf4llm"
            )
        except (_fitz_file_data_error, RuntimeError) as exc:
            logger.warning(
                "PDF '%s' is corrupt or unreadable (fitz fallback): %s",
                file_path.name, exc, exc_info=True,
            )
            return _empty_parsed_document(file_path)
    except (_fitz_file_data_error, RuntimeError) as exc:
        logger.warning(
            "PDF '%s' is corrupt or unreadable (pymupdf4llm path): %s",
            file_path.name, exc, exc_info=True,
        )
        return _empty_parsed_document(file_path)

    # Extract metadata
    meta = _extract_metadata(markdown, str(file_path))

    # Split into sections
    raw_sections = _split_into_sections(markdown)

    # Build ParsedSection objects with type detection
    sections: List[ParsedSection] = []
    chars_per_page = max(1, len(markdown) // max(1, total_pages))

    for raw in raw_sections:
        section_type = _detect_section_type(raw["content"]) or "general"
        page_start = max(1, raw["start_pos"] // chars_per_page + 1)
        page_end = min(total_pages, (raw["start_pos"] + len(raw["content"])) // chars_per_page + 1)
        tables = _detect_tables_in_section(raw["content"])

        sections.append(ParsedSection(
            title=raw["title"],
            content=raw["content"],
            section_type=section_type,
            page_start=page_start,
            page_end=page_end,
            tables=tables,
        ))

    return ParsedDocument(
        source_path=str(file_path),
        title=meta.get("company", file_path.stem),
        company=meta.get("company"),
        period=meta.get("period"),
        total_pages=total_pages,
        sections=sections,
        raw_markdown=markdown,
        metadata=meta,
    )
