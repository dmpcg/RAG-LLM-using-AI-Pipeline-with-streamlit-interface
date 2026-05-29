"""Excel export module for financial analysis reports.

Generates multi-sheet XLSX workbooks with formatted financial data,
ratio tables, health scores, and scenario comparisons.
"""

import io
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import xlsxwriter

logger = logging.getLogger(__name__)

# Hard cap on rows written to a single scenario-comparison sheet. The row
# counter spans MULTIPLE scenarios in one sheet; the cap is enforced at a
# whole-scenario boundary so no partial/corrupt scenario block is ever emitted.
_MAX_EXPORT_ROWS = 10_000

# Hard cap on the number of ratio/metric entries written to a non-scenario
# tabular sheet (Ratios sheet + single-sheet export_ratios). When the input
# exceeds this, the export is capped and SURFACED -- a truncation note row is
# written and the drop is logged (never a silent slice).
_MAX_RATIO_ENTRIES = 500

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
    ScenarioResult,
)

# ---------------------------------------------------------------------------
# Column-width helper (xlsx-only)
# ---------------------------------------------------------------------------


def _auto_col_width(text: str, minimum: int = 12, maximum: int = 40) -> int:
    """Estimate column width from string length."""
    return max(minimum, min(len(str(text)) + 4, maximum))


# ---------------------------------------------------------------------------
# Format dict builders (applied when workbook exists)
# ---------------------------------------------------------------------------


def _make_header_fmt(wb: xlsxwriter.Workbook) -> xlsxwriter.format.Format:
    return wb.add_format(
        {
            "bold": True,
            "bg_color": "#1F4E79",
            "font_color": "#FFFFFF",
            "border": 1,
            "text_wrap": True,
            "valign": "vcenter",
        }
    )


def _make_pct_fmt(wb: xlsxwriter.Workbook) -> xlsxwriter.format.Format:
    return wb.add_format({"num_format": "0.00%", "border": 1})


def _make_ratio_fmt(wb: xlsxwriter.Workbook) -> xlsxwriter.format.Format:
    """Multiplier format -- e.g. 1.5 renders as '1.50x'."""
    return wb.add_format({"num_format": '0.00"x"', "border": 1})


def _make_dollar_fmt(wb: xlsxwriter.Workbook) -> xlsxwriter.format.Format:
    return wb.add_format({"num_format": "$#,##0", "border": 1})


def _make_score_fmt(wb: xlsxwriter.Workbook) -> xlsxwriter.format.Format:
    return wb.add_format({"num_format": "0.0", "border": 1})


def _make_text_fmt(wb: xlsxwriter.Workbook) -> xlsxwriter.format.Format:
    return wb.add_format({"border": 1, "text_wrap": True, "valign": "vcenter"})


def _make_green_fmt(wb: xlsxwriter.Workbook) -> xlsxwriter.format.Format:
    return wb.add_format({"bg_color": "#C6EFCE", "border": 1})


def _make_red_fmt(wb: xlsxwriter.Workbook) -> xlsxwriter.format.Format:
    return wb.add_format({"bg_color": "#FFC7CE", "border": 1})


def _make_title_fmt(wb: xlsxwriter.Workbook) -> xlsxwriter.format.Format:
    return wb.add_format({"bold": True, "font_size": 14})


def _make_section_fmt(wb: xlsxwriter.Workbook) -> xlsxwriter.format.Format:
    return wb.add_format({"bold": True, "font_size": 11, "bottom": 1})


# ---------------------------------------------------------------------------
# Workbook format container
# ---------------------------------------------------------------------------


class _Formats:
    """Lazily built format collection tied to a workbook."""

    def __init__(self, wb: xlsxwriter.Workbook):
        self.header = _make_header_fmt(wb)
        self.pct = _make_pct_fmt(wb)
        self.ratio = _make_ratio_fmt(wb)
        self.dollar = _make_dollar_fmt(wb)
        self.score = _make_score_fmt(wb)
        self.text = _make_text_fmt(wb)
        self.green = _make_green_fmt(wb)
        self.red = _make_red_fmt(wb)
        self.title = _make_title_fmt(wb)
        self.section = _make_section_fmt(wb)

    def value_fmt(self, key: str) -> xlsxwriter.format.Format:
        """Select the best numeric format for *key*."""
        # Check ratio BEFORE percent -- a "ratio" is a multiplier (1.5x), not a percentage (150%).
        if _is_ratio_key(key):
            return self.ratio
        if _is_percent_key(key):
            return self.pct
        if _is_dollar_key(key):
            return self.dollar
        return self.score


# ---------------------------------------------------------------------------
# Main exporter
# ---------------------------------------------------------------------------


class FinancialExcelExporter:
    """Generates XLSX workbooks for financial analysis exports."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_full_report(
        self,
        data: FinancialData,
        analysis_results: Dict[str, Any],
        report: Optional[FinancialReport] = None,
    ) -> bytes:
        """Return a multi-sheet XLSX workbook as *bytes*.

        Sheets created:
          - Summary: company overview and key metrics
          - Ratios: all ratios grouped by category
          - Health Score: composite health breakdown (if present)
          - Scoring: Z-Score and F-Score details (if present)
        """
        buf = io.BytesIO()
        wb = xlsxwriter.Workbook(buf, {"in_memory": True})
        fmt = _Formats(wb)

        self._write_summary_sheet(wb, fmt, data, analysis_results, report)
        self._write_ratios_sheet(wb, fmt, analysis_results)
        self._write_health_sheet(wb, fmt, analysis_results)
        self._write_scoring_sheet(wb, fmt, analysis_results)

        wb.close()
        return buf.getvalue()

    def export_ratios(
        self,
        ratios: Dict[str, Optional[float]],
        company_name: str = "",
    ) -> bytes:
        """Single-sheet export of ratio name / value pairs."""
        buf = io.BytesIO()
        wb = xlsxwriter.Workbook(buf, {"in_memory": True})
        fmt = _Formats(wb)
        ws = wb.add_worksheet("Ratios")

        # Title
        row = 0
        title = f"Financial Ratios - {company_name}" if company_name else "Financial Ratios"
        ws.write(row, 0, title, fmt.title)
        row += 1
        ws.write(row, 0, f"Generated: {datetime.now():%Y-%m-%d %H:%M}", fmt.text)
        row += 2

        # Headers
        ws.write(row, 0, "Metric", fmt.header)
        ws.write(row, 1, "Value", fmt.header)
        ws.freeze_panes(row + 1, 0)
        row += 1

        # Cap + surface oversized inputs (log + truncation note row), so the
        # sheet size stays bounded without a silent slice.
        truncated = len(ratios) > _MAX_RATIO_ENTRIES
        if truncated:
            omitted = len(ratios) - _MAX_RATIO_ENTRIES
            logger.warning(
                "Ratios export entry cap (%d) reached; truncating %d ratio entry(ies) to avoid runaway sheet size.",
                _MAX_RATIO_ENTRIES,
                omitted,
            )
            ratios = dict(list(ratios.items())[:_MAX_RATIO_ENTRIES])

        col0_width = 12
        for key, value in ratios.items():
            label = key.replace("_", " ").title()
            ws.write(row, 0, label, fmt.text)
            col0_width = max(col0_width, _auto_col_width(label))
            if value is None:
                ws.write(row, 1, "N/A", fmt.text)
            else:
                ws.write_number(row, 1, value, fmt.value_fmt(key))
            row += 1

        if truncated:
            ws.write(
                row,
                0,
                f"... {omitted} ratio entries truncated (entry cap {_MAX_RATIO_ENTRIES} reached) ...",
                fmt.text,
            )
            row += 1

        ws.set_column(0, 0, col0_width)
        ws.set_column(1, 1, 18)
        wb.close()
        return buf.getvalue()

    def export_scenario_comparison(
        self,
        scenarios: List[ScenarioResult],
    ) -> bytes:
        """Export scenario comparison with base vs scenario columns."""
        buf = io.BytesIO()
        wb = xlsxwriter.Workbook(buf, {"in_memory": True})
        fmt = _Formats(wb)
        ws = wb.add_worksheet("Scenario Comparison")

        row = 0
        ws.write(row, 0, "Scenario Comparison", fmt.title)
        row += 1
        ws.write(row, 0, f"Generated: {datetime.now():%Y-%m-%d %H:%M}", fmt.text)
        row += 2

        for idx, scenario in enumerate(scenarios):
            # Enforce the row cap at a WHOLE-SCENARIO boundary: project this
            # scenario's full row footprint up front and stop before writing
            # it if it would push the sheet past the cap. This guarantees no
            # scenario is ever truncated mid-block (no corrupt partial block).
            all_keys = sorted(set(list(scenario.base_ratios.keys()) + list(scenario.scenario_ratios.keys())))
            # Worst-case projected rows for this scenario block:
            #   1 scenario header
            # + adjustments: 1 header + N rows + 1 gap (when present)
            # + ratio table: 1 header + len(all_keys) rows (when present)
            # + impact: 1 gap + 1 label + 1 text (when present)
            # + 2 trailing gap rows
            projected_scenario_rows = 1
            if scenario.adjustments:
                projected_scenario_rows += 1 + len(scenario.adjustments) + 1
            if all_keys:
                projected_scenario_rows += 1 + len(all_keys)
            if scenario.impact_summary:
                projected_scenario_rows += 3
            projected_scenario_rows += 2

            if row + projected_scenario_rows > _MAX_EXPORT_ROWS:
                omitted = len(scenarios) - idx
                logger.warning(
                    "Scenario export row cap (%d) reached; omitting %d scenario(s) to avoid memory blow-up.",
                    _MAX_EXPORT_ROWS,
                    omitted,
                )
                ws.write(
                    row,
                    0,
                    f"... {omitted} scenarios omitted (row cap {_MAX_EXPORT_ROWS} reached) ...",
                    fmt.text,
                )
                row += 1
                break

            # Scenario header
            ws.write(row, 0, scenario.scenario_name, fmt.section)
            row += 1

            # Adjustments
            if scenario.adjustments:
                ws.write(row, 0, "Adjustments", fmt.header)
                ws.write(row, 1, "Value", fmt.header)
                row += 1
                for adj_key, adj_val in scenario.adjustments.items():
                    label = adj_key.replace("_", " ").title()
                    ws.write(row, 0, label, fmt.text)
                    ws.write_number(row, 1, adj_val, fmt.score)
                    row += 1
                row += 1

            # Ratio comparison table (all_keys computed above for projection)
            if all_keys:
                ws.write(row, 0, "Metric", fmt.header)
                ws.write(row, 1, "Base", fmt.header)
                ws.write(row, 2, "Scenario", fmt.header)
                ws.write(row, 3, "Change", fmt.header)
                ws.freeze_panes(row + 1, 0)
                row += 1

                for key in all_keys:
                    base_val = scenario.base_ratios.get(key)
                    scen_val = scenario.scenario_ratios.get(key)
                    label = key.replace("_", " ").title()
                    vf = fmt.value_fmt(key)
                    ws.write(row, 0, label, fmt.text)

                    if base_val is not None:
                        ws.write_number(row, 1, base_val, vf)
                    else:
                        ws.write(row, 1, "N/A", fmt.text)

                    if scen_val is not None:
                        ws.write_number(row, 2, scen_val, vf)
                    else:
                        ws.write(row, 2, "N/A", fmt.text)

                    # Delta with conditional color
                    if base_val is not None and scen_val is not None:
                        delta = scen_val - base_val
                        color_fmt = fmt.green if delta >= 0 else fmt.red
                        ws.write_number(row, 3, delta, color_fmt)
                    else:
                        ws.write(row, 3, "N/A", fmt.text)
                    row += 1

            # Impact summary
            if scenario.impact_summary:
                row += 1
                ws.write(row, 0, "Impact Summary:", fmt.section)
                row += 1
                ws.write(row, 0, scenario.impact_summary, fmt.text)

            row += 2  # gap between scenarios

        ws.set_column(0, 0, 30)
        ws.set_column(1, 3, 18)
        wb.close()
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Private sheet writers
    # ------------------------------------------------------------------

    def _write_summary_sheet(
        self,
        wb: xlsxwriter.Workbook,
        fmt: _Formats,
        data: FinancialData,
        results: Dict[str, Any],
        report: Optional[FinancialReport],
    ) -> None:
        ws = wb.add_worksheet("Summary")
        row = 0

        # Title
        ws.write(row, 0, "Financial Analysis Summary", fmt.title)
        row += 1
        ws.write(row, 0, f"Generated: {datetime.now():%Y-%m-%d %H:%M}", fmt.text)
        row += 2

        # Executive summary
        if report and report.executive_summary:
            ws.write(row, 0, "Executive Summary", fmt.section)
            row += 1
            ws.write(row, 0, report.executive_summary, fmt.text)
            row += 2

        # Key financial data
        ws.write(row, 0, "Key Financial Data", fmt.section)
        row += 1
        ws.write(row, 0, "Item", fmt.header)
        ws.write(row, 1, "Value", fmt.header)
        ws.freeze_panes(row + 1, 0)
        row += 1

        key_items = [
            ("Total Assets", data.total_assets, True),
            ("Total Liabilities", data.total_liabilities, True),
            ("Total Equity", data.total_equity, True),
            ("Revenue", data.revenue, True),
            ("Net Income", data.net_income, True),
            ("EBITDA", data.ebitda, True),
            ("Operating Cash Flow", data.operating_cash_flow, True),
        ]
        for label, value, is_dollar in key_items:
            ws.write(row, 0, label, fmt.text)
            if value is not None:
                ws.write_number(row, 1, value, fmt.dollar if is_dollar else fmt.score)
            else:
                ws.write(row, 1, "N/A", fmt.text)
            row += 1

        # Key metrics from results
        row += 1
        ws.write(row, 0, "Key Metrics", fmt.section)
        row += 1
        highlight_keys = [
            "current_ratio",
            "quick_ratio",
            "gross_margin",
            "net_margin",
            "roe",
            "roa",
            "debt_to_equity",
            "interest_coverage",
        ]
        for key in highlight_keys:
            if key in results:
                label = key.replace("_", " ").title()
                ws.write(row, 0, label, fmt.text)
                val = results[key]
                if val is not None:
                    ws.write_number(row, 1, val, fmt.value_fmt(key))
                else:
                    ws.write(row, 1, "N/A", fmt.text)
                row += 1

        ws.set_column(0, 0, 30)
        ws.set_column(1, 1, 20)

    def _write_ratios_sheet(
        self,
        wb: xlsxwriter.Workbook,
        fmt: _Formats,
        results: Dict[str, Any],
    ) -> None:
        """Write all numeric ratios grouped by category."""
        ws = wb.add_worksheet("Ratios")

        # Filter to numeric values only
        numeric = {k: v for k, v in results.items() if isinstance(v, (int, float)) or v is None}
        truncated = len(numeric) > _MAX_RATIO_ENTRIES
        if truncated:
            omitted = len(numeric) - _MAX_RATIO_ENTRIES
            logger.warning(
                "Ratios sheet entry cap (%d) reached; truncating %d ratio entry(ies) to avoid runaway sheet size.",
                _MAX_RATIO_ENTRIES,
                omitted,
            )
            numeric = dict(list(numeric.items())[:_MAX_RATIO_ENTRIES])
        if not numeric:
            ws.write(0, 0, "No ratio data available.", fmt.text)
            return

        # Group by category
        grouped: Dict[str, List[tuple]] = {}
        for key, value in numeric.items():
            cat = _categorize(key)
            grouped.setdefault(cat, []).append((key, value))

        row = 0
        ws.write(row, 0, "Financial Ratios", fmt.title)
        row += 2

        # Sort categories: known first, Other last
        ordered = ["Liquidity", "Profitability", "Leverage", "Efficiency"]
        cats = [c for c in ordered if c in grouped]
        cats += [c for c in sorted(grouped) if c not in ordered]

        ws.write(row, 0, "Category", fmt.header)
        ws.write(row, 1, "Metric", fmt.header)
        ws.write(row, 2, "Value", fmt.header)
        ws.freeze_panes(row + 1, 0)
        row += 1

        for cat in cats:
            for key, value in grouped[cat]:
                label = key.replace("_", " ").title()
                ws.write(row, 0, cat, fmt.text)
                ws.write(row, 1, label, fmt.text)
                if value is not None:
                    ws.write_number(row, 2, value, fmt.value_fmt(key))
                else:
                    ws.write(row, 2, "N/A", fmt.text)
                row += 1

        if truncated:
            ws.write(
                row,
                0,
                f"... {omitted} ratio entries truncated (entry cap {_MAX_RATIO_ENTRIES} reached) ...",
                fmt.text,
            )
            row += 1

        ws.set_column(0, 0, 18)
        ws.set_column(1, 1, 30)
        ws.set_column(2, 2, 18)

    def _write_health_sheet(
        self,
        wb: xlsxwriter.Workbook,
        fmt: _Formats,
        results: Dict[str, Any],
    ) -> None:
        """Write composite health score breakdown."""
        health = results.get("composite_health") or results.get("health_score")
        if not isinstance(health, (CompositeHealthScore, dict)):
            return  # skip sheet entirely if no health data

        ws = wb.add_worksheet("Health Score")
        row = 0
        ws.write(row, 0, "Composite Health Score", fmt.title)
        row += 2

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

        ws.write(row, 0, "Overall Score", fmt.section)
        ws.write(row, 1, score, fmt.score)
        row += 1
        ws.write(row, 0, "Grade", fmt.section)
        ws.write(row, 1, grade, fmt.text)
        row += 2

        if components:
            ws.write(row, 0, "Component", fmt.header)
            ws.write(row, 1, "Score", fmt.header)
            row += 1
            for comp, comp_score in components.items():
                label = comp.replace("_", " ").title()
                ws.write(row, 0, label, fmt.text)
                if comp_score is not None:
                    color = fmt.green if comp_score >= 50 else fmt.red
                    ws.write_number(row, 1, comp_score, color)
                else:
                    ws.write(row, 1, "N/A", fmt.text)
                row += 1

        if interpretation:
            row += 1
            ws.write(row, 0, "Interpretation:", fmt.section)
            row += 1
            ws.write(row, 0, interpretation, fmt.text)

        ws.set_column(0, 0, 25)
        ws.set_column(1, 1, 18)

    def _write_scoring_sheet(
        self,
        wb: xlsxwriter.Workbook,
        fmt: _Formats,
        results: Dict[str, Any],
    ) -> None:
        """Write Z-Score and F-Score details."""
        z_data = results.get("altman_z_score") or results.get("z_score")
        f_data = results.get("piotroski_f_score") or results.get("f_score")

        if z_data is None and f_data is None:
            return  # skip sheet

        ws = wb.add_worksheet("Scoring")
        row = 0
        ws.write(row, 0, "Financial Scoring Models", fmt.title)
        row += 2

        # Altman Z-Score section
        if z_data is not None:
            ws.write(row, 0, "Altman Z-Score", fmt.section)
            row += 1

            if isinstance(z_data, (AltmanZScore, AltmanZScoreResult)):
                z_dict = asdict(z_data)
            elif isinstance(z_data, dict):
                z_dict = z_data
            else:
                z_dict = {"z_score": z_data}

            ws.write(row, 0, "Metric", fmt.header)
            ws.write(row, 1, "Value", fmt.header)
            row += 1
            for key, value in z_dict.items():
                if isinstance(value, (int, float)) and value is not None:
                    label = key.replace("_", " ").title()
                    ws.write(row, 0, label, fmt.text)
                    ws.write_number(row, 1, value, fmt.score)
                    row += 1
                elif isinstance(value, str) and value:
                    label = key.replace("_", " ").title()
                    ws.write(row, 0, label, fmt.text)
                    ws.write(row, 1, value, fmt.text)
                    row += 1
            row += 1

        # Piotroski F-Score section
        if f_data is not None:
            ws.write(row, 0, "Piotroski F-Score", fmt.section)
            row += 1

            if isinstance(f_data, (PiotroskiFScore, PiotroskiFScoreResult)):
                f_dict = asdict(f_data)
            elif isinstance(f_data, dict):
                f_dict = f_data
            else:
                f_dict = {"f_score": f_data}

            ws.write(row, 0, "Metric", fmt.header)
            ws.write(row, 1, "Value", fmt.header)
            row += 1
            for key, value in f_dict.items():
                if isinstance(value, dict):
                    # criteria dict - expand
                    for sub_key, sub_val in value.items():
                        label = sub_key.replace("_", " ").title()
                        ws.write(row, 0, f"  {label}", fmt.text)
                        if isinstance(sub_val, bool):
                            ws.write(row, 1, "Pass" if sub_val else "Fail", fmt.green if sub_val else fmt.red)
                        else:
                            ws.write(row, 1, str(sub_val), fmt.text)
                        row += 1
                elif isinstance(value, bool):
                    label = key.replace("_", " ").title()
                    ws.write(row, 0, label, fmt.text)
                    ws.write(row, 1, "Pass" if value else "Fail", fmt.green if value else fmt.red)
                    row += 1
                elif isinstance(value, (int, float)) and value is not None:
                    label = key.replace("_", " ").title()
                    ws.write(row, 0, label, fmt.text)
                    ws.write_number(row, 1, value, fmt.score)
                    row += 1
                elif isinstance(value, str) and value:
                    label = key.replace("_", " ").title()
                    ws.write(row, 0, label, fmt.text)
                    ws.write(row, 1, value, fmt.text)
                    row += 1

        ws.set_column(0, 0, 30)
        ws.set_column(1, 1, 18)
