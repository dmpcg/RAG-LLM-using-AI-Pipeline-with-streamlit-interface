"""Tests for line_item_mapper module."""

from line_item_mapper import (
    FIELD_MAPPINGS,
    _normalize_label,
    detect_periods,
    map_label,
    map_labels_batch,
)


class TestNormalizeLabel:
    def test_lowercase(self):
        assert _normalize_label("Revenue") == "revenue"

    def test_strip_whitespace(self):
        assert _normalize_label("  Revenue  ") == "revenue"

    def test_collapse_whitespace(self):
        assert _normalize_label("Cost  of   Goods   Sold") == "cost of goods sold"

    def test_remove_parenthetical(self):
        # Nested parens: (Prism(formula)) -> strips inner then outer
        result = _normalize_label("Lab Testing (Prism(formula))")
        assert result == "lab testing"

    def test_remove_via_parenthetical(self):
        assert _normalize_label("Insurance (via credit card)") == "insurance"

    def test_empty(self):
        assert _normalize_label("") == ""

    def test_non_string(self):
        assert _normalize_label(42) == ""
        assert _normalize_label(None) == ""


class TestMapLabel:
    # ── Income Statement ─────────────────────────────────────────────
    def test_revenue(self):
        result = map_label("Total Revenue")
        assert result is not None
        assert result.canonical_name == "revenue"
        assert result.category == "income_statement"

    def test_net_sales(self):
        result = map_label("Net Sales")
        assert result is not None
        assert result.canonical_name == "revenue"

    def test_cogs(self):
        result = map_label("Cost of Goods Sold")
        assert result is not None
        assert result.canonical_name == "cogs"

    def test_cogs_direct(self):
        result = map_label("Cost of Goods Sold- Direct")
        assert result is not None
        assert result.canonical_name == "cogs"

    def test_sga(self):
        result = map_label("Selling, General and Administrative")
        assert result is not None
        assert result.canonical_name == "sga"

    def test_sga_abbreviation(self):
        result = map_label("SG&A")
        assert result is not None
        assert result.canonical_name == "sga"

    def test_da(self):
        result = map_label("D&A")
        assert result is not None
        assert result.canonical_name == "depreciation_amortization"

    def test_ebitda(self):
        result = map_label("Adjusted EBITDA")
        assert result is not None
        assert result.canonical_name == "ebitda"

    def test_net_income(self):
        result = map_label("Net Income (Loss)")
        assert result is not None
        assert result.canonical_name == "net_income"

    def test_operating_income(self):
        result = map_label("Income from Operations")
        assert result is not None
        assert result.canonical_name == "operating_income"

    def test_interest_expense(self):
        result = map_label("Interest Expense, Net")
        assert result is not None
        assert result.canonical_name == "interest_expense"

    def test_tax_expense(self):
        result = map_label("Provision for Income Taxes")
        assert result is not None
        assert result.canonical_name == "tax_expense"

    # ── Balance Sheet ────────────────────────────────────────────────
    def test_cash(self):
        result = map_label("Cash and Cash Equivalents")
        assert result is not None
        assert result.canonical_name == "cash"

    def test_accounts_receivable(self):
        result = map_label("Accounts Receivable")
        assert result is not None
        assert result.canonical_name == "accounts_receivable"

    def test_total_assets(self):
        result = map_label("Total Assets")
        assert result is not None
        assert result.canonical_name == "total_assets"

    def test_long_term_debt(self):
        result = map_label("Long-term Debt")
        assert result is not None
        assert result.canonical_name == "long_term_debt"

    def test_total_equity(self):
        result = map_label("Total Stockholders' Equity")
        assert result is not None
        assert result.canonical_name == "total_equity"

    def test_retained_earnings(self):
        result = map_label("Retained Earnings (Accumulated Deficit)")
        assert result is not None
        assert result.canonical_name == "retained_earnings"

    # ── Cash Flow ────────────────────────────────────────────────────
    def test_cfo(self):
        result = map_label("Net Cash Provided by Operating Activities")
        assert result is not None
        assert result.canonical_name == "cfo"

    def test_capex(self):
        result = map_label("Capital Expenditures")
        assert result is not None
        assert result.canonical_name == "capex"

    def test_fcf(self):
        result = map_label("Free Cash Flow")
        assert result is not None
        assert result.canonical_name == "fcf"

    # ── Valuation / LBO ──────────────────────────────────────────────
    def test_enterprise_value(self):
        result = map_label("Enterprise Value")
        assert result is not None
        assert result.canonical_name == "enterprise_value"

    def test_irr(self):
        result = map_label("IRR")
        assert result is not None
        assert result.canonical_name == "irr"

    def test_moic(self):
        result = map_label("MOIC")
        assert result is not None
        assert result.canonical_name == "moic"

    def test_wacc(self):
        result = map_label("WACC")
        assert result is not None
        assert result.canonical_name == "discount_rate"

    # ── Cannabis ─────────────────────────────────────────────────────
    def test_wholesale_revenue(self):
        result = map_label("Wholesale Revenue - Bigs")
        assert result is not None
        assert result.canonical_name == "wholesale_revenue"

    def test_cost_per_lb(self):
        result = map_label("Cost per LB")
        assert result is not None
        assert result.canonical_name == "cost_per_lb"

    def test_metrc_tags(self):
        result = map_label("Metrc Tags")
        assert result is not None
        assert result.canonical_name == "metrc_tags"

    def test_280e(self):
        result = map_label("280E Tax Rate")
        assert result is not None
        assert result.canonical_name == "tax_280e_rate"

    # ── Cash Forecast ────────────────────────────────────────────────
    def test_beginning_cash(self):
        result = map_label("Beg Cash")
        assert result is not None
        assert result.canonical_name == "beginning_cash"

    def test_ar_collections(self):
        result = map_label("AR Collections")
        assert result is not None
        assert result.canonical_name == "ar_collections"

    # ── SaaS ─────────────────────────────────────────────────────────
    def test_mrr(self):
        result = map_label("MRR")
        assert result is not None
        assert result.canonical_name == "mrr"

    def test_cac(self):
        result = map_label("Customer Acquisition Cost")
        assert result is not None
        assert result.canonical_name == "cac"

    # ── Payroll ──────────────────────────────────────────────────────
    def test_payroll(self):
        result = map_label("Payroll and Related Payroll Taxes")
        assert result is not None
        assert result.canonical_name == "payroll"

    def test_contract_labor(self):
        result = map_label("Contract Labor")
        assert result is not None
        assert result.canonical_name == "contract_labor"

    # ── No match ─────────────────────────────────────────────────────
    def test_unrecognized(self):
        result = map_label("Xyzzy Blqrst Zymotic Plugh")
        assert result is None

    # ── Section context boost ────────────────────────────────────────
    def test_section_context_boost(self):
        result = map_label("Revenue", section_context="income_statement")
        assert result is not None
        assert result.confidence == 0.95

    def test_section_context_no_boost(self):
        result = map_label("Revenue", section_context="balance_sheet")
        assert result is not None
        assert result.confidence == 0.9

    # ── Parenthetical stripping ──────────────────────────────────────
    def test_parenthetical_formula(self):
        result = map_label("Revenue (formula)")
        assert result is not None
        assert result.canonical_name == "revenue"

    def test_parenthetical_see_below(self):
        result = map_label("COGS (see below)")
        assert result is not None
        assert result.canonical_name == "cogs"


class TestMapLabelsBatch:
    def test_batch_mapping(self):
        labels = [
            "Total Revenue",
            "Cost of Goods Sold",
            "Gross Profit",
            "Net Income",
            "Unknown Item XYZ",
        ]
        results = map_labels_batch(labels)
        assert len(results) == 4  # 4 matched, 1 unrecognized
        assert "Total Revenue" in results
        assert "Unknown Item XYZ" not in results

    def test_dedup_canonical(self):
        labels = ["Revenue", "Total Revenue", "Net Sales"]
        results = map_labels_batch(labels)
        # All map to "revenue" but only first should win
        assert len(results) == 1

    def test_empty_batch(self):
        results = map_labels_batch([])
        assert len(results) == 0


class TestDetectPeriods:
    def test_year_columns(self):
        cols = ["Label", "2022", "2023", "2024"]
        periods = detect_periods(cols)
        assert "2022" in periods
        assert "2023" in periods
        assert "2024" in periods

    def test_year_with_suffix(self):
        cols = ["Label", "2024A", "2025E"]
        periods = detect_periods(cols)
        assert len(periods) == 2

    def test_quarter_columns(self):
        cols = ["Label", "Q1 2024", "Q2 2024"]
        periods = detect_periods(cols)
        assert len(periods) == 2

    def test_month_columns(self):
        cols = ["Label", "Jan", "Feb", "Mar"]
        periods = detect_periods(cols)
        assert len(periods) == 3

    def test_ytd_mtd(self):
        cols = ["Label", "YTD", "MTD"]
        periods = detect_periods(cols)
        assert len(periods) == 2

    def test_week_columns(self):
        cols = ["Label", "Week 1", "Week 2", "Week 3"]
        periods = detect_periods(cols)
        assert len(periods) == 3

    def test_mixed_non_period(self):
        cols = ["Item", "Description", "Category"]
        periods = detect_periods(cols)
        assert len(periods) == 0

    def test_actual_forecast(self):
        cols = ["Label", "Actual", "Forecast"]
        periods = detect_periods(cols)
        assert len(periods) == 2

    def test_fy_prefix(self):
        cols = ["Label", "FY2024"]
        periods = detect_periods(cols)
        assert len(periods) == 1


class TestFieldMappingsCoverage:
    def test_minimum_field_count(self):
        assert len(FIELD_MAPPINGS) >= 70, f"Expected 70+ fields, got {len(FIELD_MAPPINGS)}"

    def test_all_fields_have_category(self):
        for name, info in FIELD_MAPPINGS.items():
            assert "category" in info, f"Field {name} missing category"

    def test_all_fields_have_patterns(self):
        for name, info in FIELD_MAPPINGS.items():
            assert "patterns" in info, f"Field {name} missing patterns"
            assert len(info["patterns"]) > 0, f"Field {name} has no patterns"

    def test_total_pattern_count(self):
        total = sum(len(info["patterns"]) for info in FIELD_MAPPINGS.values())
        assert total >= 300, f"Expected 300+ patterns, got {total}"

    def test_categories_covered(self):
        categories = {info["category"] for info in FIELD_MAPPINGS.values()}
        expected = {
            "income_statement",
            "balance_sheet",
            "cash_flow",
            "valuation",
            "debt",
            "cannabis",
            "cash_forecast",
            "kpi",
            "saas",
            "construction",
            "fund",
        }
        assert expected.issubset(categories), f"Missing categories: {expected - categories}"
