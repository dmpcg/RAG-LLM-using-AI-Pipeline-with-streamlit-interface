"""PDF export module for financial analysis reports.

Generates multi-page PDF documents with formatted financial data,
ratio tables, health scores, and executive summaries.
"""

import io
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fpdf import FPDF

from export_utils import (
    _categorize,
    _is_dollar_key,
    _is_percent_key,
    _is_ratio_key,
)
from financial_analyzer import (
    AltmanZScore,
    AltmanZScoreResult,
    CompositeHealthScore,
    FinancialData,
    FinancialReport,
    PiotroskiFScore,
    PiotroskiFScoreResult,
)

# ---------------------------------------------------------------------------
# Unicode sanitization (P1-E3)
# ---------------------------------------------------------------------------

# fpdf2's core "Helvetica" font is latin-1 only; emitting non-latin-1 glyphs
# either crashes (FPDFUnicodeEncodingException) or renders a corrupt glyph.
# Map the common offenders to ASCII, then backstop with a latin-1 round-trip
# so any remaining out-of-range char becomes "?" instead of surviving.
_SANITIZE_MAP = {
    "—": "-",  # em dash
    "–": "-",  # en dash
    "µ": "u",  # micro sign
    "μ": "u",  # Greek small letter mu
    "‘": "'",  # left single quote
    "’": "'",  # right single quote
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "≥": ">=",  # greater-than or equal
    "≤": "<=",  # less-than or equal
    "…": "...",  # horizontal ellipsis
}

_SANITIZE_TRANSLATION = str.maketrans(_SANITIZE_MAP)


def _sanitize_text(s: Any) -> str:
    """Map common non-latin-1 characters to ASCII and backstop with latin-1.

    Idempotent: applying it twice yields the same result. Every string sink in
    the PDF exporter funnels through this so the latin-1 core font never sees
    an unsupported glyph.
    """
    text = s if isinstance(s, str) else str(s)
    text = text.translate(_SANITIZE_TRANSLATION)
    return text.encode("latin-1", "replace").decode("latin-1")


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

_DARK_BLUE: Tuple[int, int, int] = (31, 78, 121)  # #1F4E79
_WHITE: Tuple[int, int, int] = (255, 255, 255)
_LIGHT_GRAY: Tuple[int, int, int] = (242, 242, 242)  # #F2F2F2
_GREEN: Tuple[int, int, int] = (198, 239, 206)  # #C6EFCE
_RED: Tuple[int, int, int] = (255, 199, 206)  # #FFC7CE
_BLACK: Tuple[int, int, int] = (0, 0, 0)


# ---------------------------------------------------------------------------
# Main exporter
# ---------------------------------------------------------------------------


class FinancialPDFExporter:
    """Generates multi-page PDF reports for financial analysis."""

    def __init__(self) -> None:
        self.title_size = 16
        self.section_size = 12
        self.body_size = 10
        self.margin = 15

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_full_report(
        self,
        data: FinancialData,
        analysis_results: Dict[str, Any],
        report: Optional[FinancialReport] = None,
    ) -> bytes:
        """Return a multi-page PDF report as *bytes*.

        Pages:
          1. Cover page
          2. Executive summary (if report provided)
          3+. Ratio tables by category
          Scoring section (Z-Score, F-Score, Health if present)
        """
        pdf = self._create_pdf()

        # Page 1 - Cover
        self._add_cover_page(pdf, data, report)

        # Page 2 - Executive Summary
        if report and report.executive_summary:
            pdf.add_page()
            self._add_header(pdf, "Executive Summary")
            pdf.set_font("Helvetica", "", self.body_size)
            pdf.multi_cell(0, 6, _sanitize_text(report.executive_summary))

        # Key financial data page
        pdf.add_page()
        self._add_header(pdf, "Key Financial Data")
        key_items = [
            ("Total Assets", data.total_assets, True),
            ("Total Liabilities", data.total_liabilities, True),
            ("Total Equity", data.total_equity, True),
            ("Revenue", data.revenue, True),
            ("Net Income", data.net_income, True),
            ("EBITDA", data.ebitda, True),
            ("Operating Cash Flow", data.operating_cash_flow, True),
        ]
        rows = []
        for label, value, is_dollar in key_items:
            rows.append([label, self._format_value(label.lower().replace(" ", "_"), value)])
        self._add_table(pdf, ["Item", "Value"], rows, col_widths=[90, 60])

        # Ratio pages by category
        numeric = {k: v for k, v in analysis_results.items() if isinstance(v, (int, float)) or v is None}
        _MAX_RATIO_ENTRIES = 500
        if len(numeric) > _MAX_RATIO_ENTRIES:
            numeric = dict(list(numeric.items())[:_MAX_RATIO_ENTRIES])
        if numeric:
            grouped: Dict[str, List[tuple]] = {}
            for key, value in numeric.items():
                cat = _categorize(key)
                grouped.setdefault(cat, []).append((key, value))

            ordered = ["Liquidity", "Profitability", "Leverage", "Efficiency"]
            cats = [c for c in ordered if c in grouped]
            cats += [c for c in sorted(grouped) if c not in ordered]

            pdf.add_page()
            self._add_header(pdf, "Financial Ratios")

            for cat in cats:
                # Check if we need a new page (leave at least 40mm)
                if pdf.get_y() > 240:
                    pdf.add_page()
                    self._add_header(pdf, "Financial Ratios (continued)")

                pdf.set_font("Helvetica", "B", self.section_size)
                pdf.cell(0, 8, cat, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)

                rows = []
                for key, value in grouped[cat]:
                    label = key.replace("_", " ").title()
                    rows.append([label, self._format_value(key, value)])
                self._add_table(pdf, ["Metric", "Value"], rows, col_widths=[90, 60])
                pdf.ln(4)

        # Scoring section
        self._add_scoring_section(pdf, analysis_results)

        # Health score section
        self._add_health_section(pdf, analysis_results)

        buf = io.BytesIO()
        buf.write(pdf.output())
        return buf.getvalue()

    def export_executive_summary(
        self,
        report: FinancialReport,
        health: Optional[CompositeHealthScore] = None,
    ) -> bytes:
        """Single-page executive summary with optional health score."""
        pdf = self._create_pdf()
        pdf.add_page()
        self._add_header(pdf, "Executive Summary")

        # Summary text
        pdf.set_font("Helvetica", "", self.body_size)
        pdf.multi_cell(0, 6, _sanitize_text(report.executive_summary or "No summary available."))
        pdf.ln(6)

        # Health gauge
        if health is not None:
            pdf.set_font("Helvetica", "B", self.section_size)
            pdf.cell(0, 8, "Composite Health Score", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

            # Score and grade
            rows = [
                ["Overall Score", str(health.score)],
                ["Grade", health.grade],
            ]
            if health.component_scores:
                for comp, comp_score in health.component_scores.items():
                    label = comp.replace("_", " ").title()
                    rows.append([label, str(comp_score) if comp_score is not None else "N/A"])
            self._add_table(pdf, ["Component", "Score"], rows, col_widths=[90, 60])

            if health.interpretation:
                pdf.ln(4)
                pdf.set_font("Helvetica", "I", self.body_size)
                pdf.multi_cell(0, 6, _sanitize_text(health.interpretation))

        buf = io.BytesIO()
        buf.write(pdf.output())
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_pdf(self) -> FPDF:
        """Create a base FPDF instance with standard settings."""
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=self.margin)
        pdf.set_margins(self.margin, self.margin, self.margin)
        pdf.set_font("Helvetica", "", self.body_size)
        return pdf

    def _add_header(self, pdf: FPDF, title: str) -> None:
        """Add a section header with title, date, and separator line."""
        pdf.set_font("Helvetica", "B", self.title_size)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, f"Generated: {datetime.now():%Y-%m-%d %H:%M}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*_BLACK)
        # Separator line
        y = pdf.get_y()
        pdf.line(self.margin, y, 210 - self.margin, y)
        pdf.ln(4)

    def _add_table(
        self,
        pdf: FPDF,
        headers: List[str],
        rows: List[List[str]],
        col_widths: Optional[List[int]] = None,
    ) -> None:
        """Render a table with alternating row colors and a dark header.

        Tables that overflow the page repeat the header row at the top of each
        new page. `_add_table` owns page breaks for its OWN rows: it disables
        the global auto-page-break on entry (so fpdf does not insert an
        unmanaged break mid-row) and restores the caller's original setting on
        exit, leaving callers that rely on auto-break unaffected.
        """
        if col_widths is None:
            n_cols = len(headers)
            available = 210 - 2 * self.margin
            col_widths = [available // n_cols] * n_cols

        row_height = 6

        # Save the caller's auto-page-break state and take manual control so a
        # mid-row break never fires; restore in the finally block below.
        original_auto = pdf.auto_page_break
        original_b_margin = pdf.b_margin
        pdf.set_auto_page_break(False)
        try:
            # The page-break trigger is the bottom of the printable area
            # (page height minus the original bottom margin).
            page_break_trigger = pdf.h - original_b_margin

            self._draw_table_header(pdf, headers, col_widths)

            # Data rows
            pdf.set_text_color(*_BLACK)
            pdf.set_font("Helvetica", "", self.body_size)
            for row_idx, row in enumerate(rows):
                # Break before drawing the row if it would overflow the page,
                # then redraw the header at the top of the new page.
                if pdf.get_y() + row_height > page_break_trigger:
                    pdf.add_page()
                    self._draw_table_header(pdf, headers, col_widths)
                    pdf.set_text_color(*_BLACK)
                    pdf.set_font("Helvetica", "", self.body_size)

                if row_idx % 2 == 1:
                    pdf.set_fill_color(*_LIGHT_GRAY)
                    fill = True
                else:
                    pdf.set_fill_color(*_WHITE)
                    fill = True  # always fill for clean look

                for i, cell_val in enumerate(row):
                    w = col_widths[i] if i < len(col_widths) else (col_widths[-1] if col_widths else 30)
                    # Sanitize BEFORE truncation so the latin-1 font never sees
                    # an unsupported glyph and substitutions stay within 50 chars.
                    text = _sanitize_text(cell_val)
                    display = text[:50] if len(text) > 50 else text
                    pdf.cell(w, row_height, display, border=1, fill=fill)
                pdf.ln()
        finally:
            # Restore the caller's auto-page-break setting unconditionally.
            pdf.set_auto_page_break(original_auto, margin=original_b_margin)

    def _draw_table_header(
        self,
        pdf: FPDF,
        headers: List[str],
        col_widths: List[int],
    ) -> None:
        """Draw the dark header row for a table at the current position."""
        pdf.set_fill_color(*_DARK_BLUE)
        pdf.set_text_color(*_WHITE)
        pdf.set_font("Helvetica", "B", self.body_size)
        for i, header in enumerate(headers):
            w = col_widths[i] if i < len(col_widths) else (col_widths[-1] if col_widths else 30)
            pdf.cell(w, 7, _sanitize_text(header), border=1, fill=True)
        pdf.ln()

    def _format_value(self, key: str, value: Any) -> str:
        """Format a value based on the key name."""
        if value is None:
            return "N/A"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if not isinstance(value, (int, float)):
            return _sanitize_text(value)

        # Check ratio BEFORE percent -- multiplier (1.5x) vs percentage (150%).
        if _is_ratio_key(key):
            return f"{value:.2f}x"
        if _is_percent_key(key):
            return f"{value:.2%}"
        if _is_dollar_key(key):
            if abs(value) >= 1_000_000:
                return f"${value / 1_000_000:,.1f}M"
            if abs(value) >= 1_000:
                return f"${value / 1_000:,.1f}K"
            return f"${value:,.0f}"
        # Score / generic
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _add_cover_page(
        self,
        pdf: FPDF,
        data: FinancialData,
        report: Optional[FinancialReport],
    ) -> None:
        """Render the cover page with title and company info."""
        pdf.add_page()

        # Centered title block
        pdf.ln(60)
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 15, "Financial Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # Date
        pdf.set_font("Helvetica", "", 14)
        pdf.cell(0, 10, datetime.now().strftime("%B %d, %Y"), align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)

        # Company info
        if data.revenue is not None:
            pdf.set_font("Helvetica", "", 11)
            info_lines = []
            if data.revenue:
                info_lines.append(f"Revenue: {self._format_value('revenue', data.revenue)}")
            if data.total_assets:
                info_lines.append(f"Total Assets: {self._format_value('total_assets', data.total_assets)}")
            if data.net_income:
                info_lines.append(f"Net Income: {self._format_value('net_income', data.net_income)}")

            for line in info_lines:
                pdf.cell(0, 7, _sanitize_text(line), align="C", new_x="LMARGIN", new_y="NEXT")

        # Separator
        pdf.ln(20)
        y = pdf.get_y()
        pdf.line(60, y, 150, y)

        # Period
        if data.period:
            period_str = str(data.period)[:100] if data.period else "N/A"
            pdf.ln(5)
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 7, _sanitize_text(f"Period: {period_str}"), align="C", new_x="LMARGIN", new_y="NEXT")

        if report and report.generated_at:
            generated_str = str(report.generated_at)[:100]
            pdf.ln(2)
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(
                0, 7, _sanitize_text(f"Report generated: {generated_str}"), align="C", new_x="LMARGIN", new_y="NEXT"
            )

    def _add_scoring_section(
        self,
        pdf: FPDF,
        results: Dict[str, Any],
    ) -> None:
        """Add Z-Score and F-Score details if present."""
        z_data = results.get("altman_z_score") or results.get("z_score")
        f_data = results.get("piotroski_f_score") or results.get("f_score")

        if z_data is None and f_data is None:
            return

        pdf.add_page()
        self._add_header(pdf, "Financial Scoring Models")

        # Altman Z-Score
        if z_data is not None:
            pdf.set_font("Helvetica", "B", self.section_size)
            pdf.cell(0, 8, "Altman Z-Score", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

            if isinstance(z_data, (AltmanZScore, AltmanZScoreResult)):
                z_dict = asdict(z_data)
            elif isinstance(z_data, dict):
                z_dict = z_data
            else:
                z_dict = {"z_score": z_data}

            rows = []
            for key, value in z_dict.items():
                if isinstance(value, (int, float)) and value is not None:
                    label = key.replace("_", " ").title()
                    rows.append([label, f"{value:.2f}"])
                elif isinstance(value, str) and value:
                    label = key.replace("_", " ").title()
                    rows.append([label, value])
            if rows:
                self._add_table(pdf, ["Metric", "Value"], rows, col_widths=[90, 60])
            pdf.ln(6)

        # Piotroski F-Score
        if f_data is not None:
            pdf.set_font("Helvetica", "B", self.section_size)
            pdf.cell(0, 8, "Piotroski F-Score", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

            if isinstance(f_data, (PiotroskiFScore, PiotroskiFScoreResult)):
                f_dict = asdict(f_data)
            elif isinstance(f_data, dict):
                f_dict = f_data
            else:
                f_dict = {"f_score": f_data}

            rows = []
            for key, value in f_dict.items():
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        label = sub_key.replace("_", " ").title()
                        if isinstance(sub_val, bool):
                            rows.append([f"  {label}", "Pass" if sub_val else "Fail"])
                        else:
                            rows.append([f"  {label}", str(sub_val)])
                elif isinstance(value, bool):
                    label = key.replace("_", " ").title()
                    rows.append([label, "Pass" if value else "Fail"])
                elif isinstance(value, (int, float)) and value is not None:
                    label = key.replace("_", " ").title()
                    rows.append([label, f"{value:.2f}" if isinstance(value, float) else str(value)])
                elif isinstance(value, str) and value:
                    label = key.replace("_", " ").title()
                    rows.append([label, value])
            if rows:
                self._add_table(pdf, ["Metric", "Value"], rows, col_widths=[90, 60])

    def _add_health_section(
        self,
        pdf: FPDF,
        results: Dict[str, Any],
    ) -> None:
        """Add composite health score section if present."""
        health = results.get("composite_health") or results.get("health_score")
        if not isinstance(health, (CompositeHealthScore, dict)):
            return

        # Ensure we have space or start new page
        if pdf.get_y() > 200:
            pdf.add_page()
            self._add_header(pdf, "Health Score")
        else:
            pdf.ln(6)
            pdf.set_font("Helvetica", "B", self.section_size)
            pdf.cell(0, 8, "Composite Health Score", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        if isinstance(health, CompositeHealthScore):
            score = health.score
            grade = health.grade
            components = health.component_scores
            interpretation = health.interpretation or ""
        else:
            score = health.get("score", 0)
            grade = health.get("grade", "")
            components = health.get("component_scores", {})
            interpretation = health.get("interpretation", "")

        rows = [
            ["Overall Score", str(score)],
            ["Grade", grade],
        ]
        if components:
            for comp, comp_score in components.items():
                label = comp.replace("_", " ").title()
                rows.append([label, str(comp_score) if comp_score is not None else "N/A"])

        self._add_table(pdf, ["Component", "Score"], rows, col_widths=[90, 60])

        if interpretation:
            pdf.ln(4)
            pdf.set_font("Helvetica", "I", self.body_size)
            pdf.multi_cell(0, 6, _sanitize_text(interpretation))
