"""WS-3 edge-case regression tests (WP-7cd).

Covers:
- (c) Negative-CCC insight wording in working_capital_analysis.
- (d) DCF terminal value None when discount_rate <= terminal_growth or
      discount_rate <= -100%; DSCR None when annual debt service <= 0.
"""

import math

import pytest

from financial_analyzer import CharlieAnalyzer, FinancialData


@pytest.fixture
def analyzer():
    return CharlieAnalyzer()


# --- (c) Negative-CCC wording ------------------------------------------


def test_negative_ccc_wording(analyzer):
    data = FinancialData(
        revenue=1000,
        accounts_receivable=50,
        inventory=20,
        cogs=1000,
        accounts_payable=400,
    )
    result = analyzer.working_capital_analysis(data)

    # Numeric snapshot is unchanged: CCC == round(DSO + DIO - DPO, 1).
    dso = round(50 / 1000 * 365, 1)
    dio = round(20 / 1000 * 365, 1)
    dpo = round(400 / 1000 * 365, 1)
    expected_ccc = round(dso + dio - dpo, 1)
    assert result.ccc == expected_ccc
    assert result.ccc < 0

    joined = " ".join(result.insights)
    # New ASCII-only wording present; old double-signed wording absent.
    assert "company collects cash before paying suppliers" in joined
    assert "generates cash before paying suppliers. Excellent." not in joined
    # ASCII-only source string -- no unicode dashes leaked into insight.
    assert all(ord(ch) < 128 for ch in joined)


# --- (d) DCF terminal value guards -------------------------------------


def test_dcf_terminal_value_none_when_growth_ge_discount(analyzer):
    data = FinancialData(revenue=1000, cogs=400, operating_expenses=200)
    # terminal_growth == discount_rate -> GGM undefined -> None (not inf).
    result = analyzer.forecast_cashflow(data, periods=5, discount_rate=0.05, terminal_growth=0.05)
    assert result.terminal_value is None
    assert result.dcf_value is not None
    assert not math.isinf(result.dcf_value)

    # terminal_growth > discount_rate -> still None.
    result2 = analyzer.forecast_cashflow(data, periods=5, discount_rate=0.03, terminal_growth=0.08)
    assert result2.terminal_value is None
    assert not math.isinf(result2.dcf_value)


def test_dcf_none_when_discount_rate_at_or_below_neg_100pct(analyzer):
    data = FinancialData(revenue=1000, cogs=400, operating_expenses=200)
    result = analyzer.forecast_cashflow(data, periods=5, discount_rate=-1.0, terminal_growth=0.02)
    # discount_rate <= -100% makes (1 + r) ** i non-positive -> undefined DCF.
    # Guard returns an empty forecast: no terminal value, no DCF value, no inf.
    assert result.terminal_value is None
    assert result.dcf_value is None


# --- (d) DSCR zero/negative debt service -------------------------------


def test_dscr_none_when_debt_service_nonpositive(analyzer):
    # Zero interest expense -> annual debt service <= 0 -> DSCR None.
    data_zero = FinancialData(revenue=1000, ebitda=500, interest_expense=0)
    assert analyzer.debt_service_coverage_analysis(data_zero).dscr is None

    # Negative interest expense -> still None.
    data_neg = FinancialData(revenue=1000, ebitda=500, interest_expense=-10)
    assert analyzer.debt_service_coverage_analysis(data_neg).dscr is None
