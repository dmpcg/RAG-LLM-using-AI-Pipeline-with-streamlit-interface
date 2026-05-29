"""Tests for viz_utils.py financial visualization utilities."""

import math

import pandas as pd
import plotly.graph_objects as go
import pytest


# ---------------------------------------------------------------------------
# format_currency
# ---------------------------------------------------------------------------


class TestFormatCurrency:
    def test_billions(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_currency(5_000_000_000) == "$5B"

    def test_millions(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_currency(2_500_000) == "$2M"

    def test_thousands(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_currency(75_000) == "$75K"

    def test_small_value(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_currency(500) == "$500"

    def test_negative_value(self):
        from viz_utils import FinancialVizUtils

        result = FinancialVizUtils.format_currency(-3_000_000)
        assert result == "-$3M"

    def test_zero(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_currency(0) == "$0"

    def test_none_returns_na(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_currency(None) == "N/A"

    def test_nan_returns_na(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_currency(float("nan")) == "N/A"

    def test_custom_currency(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_currency(1_000_000, currency="€") == "€1M"

    def test_decimals(self):
        from viz_utils import FinancialVizUtils

        result = FinancialVizUtils.format_currency(1_500_000, decimals=1)
        assert result == "$1.5M"


# ---------------------------------------------------------------------------
# format_percent
# ---------------------------------------------------------------------------


class TestFormatPercent:
    def test_basic(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_percent(0.125) == "12.5%"

    def test_zero(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_percent(0) == "0.0%"

    def test_negative(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_percent(-0.05) == "-5.0%"

    def test_none(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_percent(None) == "N/A"

    def test_nan(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_percent(float("nan")) == "N/A"

    def test_custom_decimals(self):
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_percent(0.12345, decimals=2) == "12.35%"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_palette_has_colors(self):
        from viz_utils import FinancialVizUtils

        assert len(FinancialVizUtils.PALETTE) >= 8

    def test_color_constants_are_hex(self):
        from viz_utils import FinancialVizUtils

        for attr in ["POSITIVE_COLOR", "NEGATIVE_COLOR", "NEUTRAL_COLOR", "WARNING_COLOR"]:
            val = getattr(FinancialVizUtils, attr)
            assert val.startswith("#"), f"{attr} should be hex color"


# ---------------------------------------------------------------------------
# create_kpi_card
# ---------------------------------------------------------------------------


class TestCreateKpiCard:
    def test_returns_figure(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_kpi_card(1_000_000, title="Revenue")
        assert isinstance(fig, go.Figure)

    def test_currency_format(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_kpi_card(
            500_000, title="Revenue", format_type="currency"
        )
        assert fig.data[0].value == 500_000

    def test_percent_format(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_kpi_card(
            0.125, title="Margin", format_type="percent"
        )
        # Value should be multiplied by 100 for percent display
        assert fig.data[0].value == 12.5

    def test_number_format(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_kpi_card(42, format_type="number")
        assert fig.data[0].value == 42

    def test_with_delta(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_kpi_card(
            1_000_000, delta=100_000, format_type="currency"
        )
        assert "delta" in fig.data[0].mode


# ---------------------------------------------------------------------------
# create_gauge_chart
# ---------------------------------------------------------------------------


class TestCreateGaugeChart:
    def test_returns_figure(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_gauge_chart(75, title="Score")
        assert isinstance(fig, go.Figure)

    def test_custom_thresholds(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_gauge_chart(
            50, thresholds={"warning": 30, "good": 70}
        )
        assert fig.data[0].value == 50

    def test_default_thresholds(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_gauge_chart(50, max_val=100)
        assert fig.data[0].gauge.steps is not None


# ---------------------------------------------------------------------------
# create_waterfall
# ---------------------------------------------------------------------------


class TestCreateWaterfall:
    def test_returns_figure(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_waterfall(
            ["Revenue", "COGS"], [1000, -600]
        )
        assert isinstance(fig, go.Figure)

    def test_with_total(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_waterfall(
            ["A", "B"], [100, -30], show_total=True
        )
        # Should have 3 entries (A, B, Total)
        assert len(fig.data[0].x) == 3

    def test_without_total(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_waterfall(
            ["A", "B"], [100, -30], show_total=False
        )
        assert len(fig.data[0].x) == 2


# ---------------------------------------------------------------------------
# create_bullet_chart
# ---------------------------------------------------------------------------


class TestCreateBulletChart:
    def test_returns_figure(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_bullet_chart(
            actual=80, target=100, ranges=[50, 75, 100], title="Sales"
        )
        assert isinstance(fig, go.Figure)

    def test_has_target_shape(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_bullet_chart(
            actual=80, target=100, ranges=[50, 75, 100]
        )
        # Should have a line shape for the target
        assert len(fig.layout.shapes) >= 1


# ---------------------------------------------------------------------------
# create_sparkline
# ---------------------------------------------------------------------------


class TestCreateSparkline:
    def test_returns_figure(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_sparkline([1, 2, 3, 4, 5])
        assert isinstance(fig, go.Figure)

    def test_with_area(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_sparkline([1, 2, 3], show_area=True)
        # Area fill trace + line trace + marker trace
        assert len(fig.data) >= 2

    def test_without_area(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_sparkline(
            [1, 2, 3], show_area=False, highlight_last=False
        )
        assert len(fig.data) == 1  # Just the line

    def test_highlight_positive_trend(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_sparkline(
            [1, 2, 3], highlight_last=True, show_area=False
        )
        # Last trace is the highlight marker
        marker = fig.data[-1]
        assert marker.marker.color == FinancialVizUtils.POSITIVE_COLOR

    def test_highlight_negative_trend(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_sparkline(
            [3, 2, 1], highlight_last=True, show_area=False
        )
        marker = fig.data[-1]
        assert marker.marker.color == FinancialVizUtils.NEGATIVE_COLOR

    def test_empty_values(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_sparkline([], highlight_last=True)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# create_heatmap
# ---------------------------------------------------------------------------


class TestCreateHeatmap:
    def test_returns_figure(self):
        from viz_utils import FinancialVizUtils

        df = pd.DataFrame(
            {"A": [1.0, 0.5], "B": [0.5, 1.0]}, index=["X", "Y"]
        )
        fig = FinancialVizUtils.create_heatmap(df, title="Correlation")
        assert isinstance(fig, go.Figure)

    def test_without_values(self):
        from viz_utils import FinancialVizUtils

        df = pd.DataFrame({"A": [1.0], "B": [0.5]}, index=["X"])
        fig = FinancialVizUtils.create_heatmap(df, show_values=False)
        assert fig.data[0].texttemplate is None


# ---------------------------------------------------------------------------
# create_time_series
# ---------------------------------------------------------------------------


class TestCreateTimeSeries:
    def test_returns_figure(self):
        from viz_utils import FinancialVizUtils

        df = pd.DataFrame({
            "Period": ["Q1", "Q2", "Q3"],
            "Revenue": [100, 110, 120],
            "Profit": [20, 25, 30],
        })
        fig = FinancialVizUtils.create_time_series(
            df, x_col="Period", y_cols=["Revenue", "Profit"]
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Two y_cols

    def test_single_series(self):
        from viz_utils import FinancialVizUtils

        df = pd.DataFrame({"Period": ["Q1", "Q2"], "Revenue": [100, 110]})
        fig = FinancialVizUtils.create_time_series(
            df, x_col="Period", y_cols=["Revenue"]
        )
        assert len(fig.data) == 1


# ---------------------------------------------------------------------------
# create_comparison_bar
# ---------------------------------------------------------------------------


class TestCreateComparisonBar:
    def test_returns_figure(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_comparison_bar(
            ["Sales", "Marketing"], [100, 80], [90, 85]
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Two bar traces

    def test_horizontal(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_comparison_bar(
            ["A", "B"], [10, 20], [15, 25], horizontal=True
        )
        assert fig.data[0].orientation == "h"


# ---------------------------------------------------------------------------
# create_donut_chart
# ---------------------------------------------------------------------------


class TestCreateDonutChart:
    def test_returns_figure(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_donut_chart(
            ["Revenue", "Expenses"], [700, 300]
        )
        assert isinstance(fig, go.Figure)
        assert fig.data[0].hole == 0.4

    def test_custom_hole_size(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_donut_chart(
            ["A", "B"], [50, 50], hole_size=0.6
        )
        assert fig.data[0].hole == 0.6


# ---------------------------------------------------------------------------
# create_ratio_dashboard
# ---------------------------------------------------------------------------


class TestCreateRatioDashboard:
    def test_returns_figure(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_ratio_dashboard(
            {"ROA": 0.08, "ROE": 0.15, "Current": 1.5}
        )
        assert isinstance(fig, go.Figure)

    def test_with_benchmarks(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_ratio_dashboard(
            {"ROA": 0.08},
            benchmarks={"ROA": {"target": 0.10, "good": 0.08}},
        )
        assert isinstance(fig, go.Figure)

    def test_handles_none_values(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_ratio_dashboard(
            {"ROA": 0.08, "ROE": None}
        )
        # Should not crash on None values
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# create_trend_chart
# ---------------------------------------------------------------------------


class TestCreateTrendChart:
    def test_without_forecast(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_trend_chart(
            ["Q1", "Q2", "Q3"], [100, 110, 120]
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Just historical

    def test_with_forecast(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_trend_chart(
            ["Q1", "Q2", "Q3"],
            [100, 110, 120],
            show_forecast=True,
            forecast_periods=["Q4", "Q5"],
            forecast_values=[130, 140],
        )
        assert len(fig.data) == 2  # Historical + Forecast

    def test_forecast_connects_to_last_historical(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_trend_chart(
            ["Q1", "Q2"],
            [100, 110],
            show_forecast=True,
            forecast_periods=["Q3"],
            forecast_values=[120],
        )
        # Forecast line should start from last historical point
        forecast_trace = fig.data[1]
        assert forecast_trace.x[0] == "Q2"
        assert forecast_trace.y[0] == 110


# ---------------------------------------------------------------------------
# create_variance_table_chart
# ---------------------------------------------------------------------------


class TestCreateVarianceTableChart:
    def test_returns_figure(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_variance_table_chart(
            ["Sales", "Marketing"],
            [100, 80],
            [90, 85],
        )
        assert isinstance(fig, go.Figure)

    def test_variance_calculation(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_variance_table_chart(
            ["Item"],
            [110],   # actual
            [100],   # budget
        )
        # Variance = 110 - 100 = 10
        assert fig.data[0].x == (10,)

    def test_zero_budget_no_crash(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_variance_table_chart(
            ["Item"], [100], [0]
        )
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# create_simple_bar (Wave 3 VIZ migration helper)
# ---------------------------------------------------------------------------


class TestCreateSimpleBar:
    """
    Unit tests for FinancialVizUtils.create_simple_bar.

    Verify: trace count, trace type, x/y data, marker colors, title,
    y-axis label, height, showlegend.  These are the figure-equivalence
    assertions required by WS5-UI-PLAN.md Decision Log D7.
    """

    def test_returns_go_figure(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_simple_bar(
            labels=["A", "B"],
            values=[1.0, 2.0],
            colors=None,
            title="T",
            y_title="Y",
        )
        assert isinstance(fig, go.Figure)

    def test_exactly_one_bar_trace(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_simple_bar(
            labels=["X", "Y", "Z"],
            values=[10.0, 20.0, 30.0],
            colors=["#111111", "#222222", "#333333"],
            title="Test",
            y_title="Units",
        )
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Bar)

    def test_x_data_matches_labels(self):
        from viz_utils import FinancialVizUtils

        labels = ["ROE", "ROA", "ROIC"]
        fig = FinancialVizUtils.create_simple_bar(
            labels=labels,
            values=[15.0, 8.0, 12.0],
            colors=None,
            title="Returns",
            y_title="%",
        )
        assert list(fig.data[0].x) == labels

    def test_y_data_matches_values(self):
        from viz_utils import FinancialVizUtils

        values = [5.0, 10.0, 15.0]
        fig = FinancialVizUtils.create_simple_bar(
            labels=["A", "B", "C"],
            values=values,
            colors=None,
            title="Test",
            y_title="",
        )
        assert list(fig.data[0].y) == values

    def test_marker_colors_applied(self):
        from viz_utils import FinancialVizUtils

        colors = ["#636EFA", "#00CC96", "#FFA15A"]
        fig = FinancialVizUtils.create_simple_bar(
            labels=["A", "B", "C"],
            values=[1.0, 2.0, 3.0],
            colors=colors,
            title="T",
            y_title="",
        )
        assert list(fig.data[0].marker.color) == colors

    def test_title_in_layout(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_simple_bar(
            labels=["A"],
            values=[1.0],
            colors=None,
            title="My Chart Title",
            y_title="",
        )
        assert fig.layout.title.text == "My Chart Title"

    def test_y_axis_title_in_layout(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_simple_bar(
            labels=["A"],
            values=[1.0],
            colors=None,
            title="T",
            y_title="Return (%)",
        )
        assert fig.layout.yaxis.title.text == "Return (%)"

    def test_default_height_is_350(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_simple_bar(
            labels=["A"],
            values=[1.0],
            colors=None,
            title="T",
            y_title="",
        )
        assert fig.layout.height == 350

    def test_custom_height_applied(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_simple_bar(
            labels=["A"],
            values=[1.0],
            colors=None,
            title="T",
            y_title="",
            height=500,
        )
        assert fig.layout.height == 500

    def test_showlegend_false_by_default(self):
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_simple_bar(
            labels=["A"],
            values=[1.0],
            colors=None,
            title="T",
            y_title="",
        )
        assert fig.layout.showlegend is False

    def test_none_colors_uses_palette(self):
        """When colors=None, PALETTE is used; no AttributeError."""
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_simple_bar(
            labels=["A", "B"],
            values=[1.0, 2.0],
            colors=None,
            title="Palette test",
            y_title="",
        )
        # Colors list should have 2 entries from PALETTE
        assert len(fig.data[0].marker.color) == 2

    def test_empty_labels_no_crash(self):
        """Empty input must not crash (returns an empty bar chart)."""
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_simple_bar(
            labels=[],
            values=[],
            colors=[],
            title="Empty",
            y_title="",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_figure_equivalence_valuation_multiples(self):
        """
        Figure-equivalence assertion for the valuation multiples migration
        (insights_page.py _render_valuation_indicators).

        Pre-migration: go.Figure(data=[go.Bar(x=..., y=..., marker_color=...)])
                       + update_layout(title="Valuation Multiples", yaxis_title="Multiple", height=350)

        Post-migration: create_simple_bar(labels, values, colors, "Valuation Multiples", "Multiple")

        Assert: same trace type, same x/y data, same title, same y-axis title, same height.
        """
        import plotly.graph_objects as _go
        from viz_utils import FinancialVizUtils

        labels = ["EV/EBITDA", "EV/Revenue", "EV/EBIT"]
        values = [12.5, 2.3, 10.1]
        colors = ["#3498db", "#2ecc71", "#e67e22"]

        # PRE-migration reference figure
        ref = _go.Figure(data=[_go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
        )])
        ref.update_layout(title="Valuation Multiples", yaxis_title="Multiple", height=350)

        # POST-migration helper figure
        got = FinancialVizUtils.create_simple_bar(
            labels=labels,
            values=values,
            colors=colors,
            title="Valuation Multiples",
            y_title="Multiple",
        )

        assert len(got.data) == len(ref.data)
        assert isinstance(got.data[0], _go.Bar)
        assert list(got.data[0].x) == list(ref.data[0].x)
        assert list(got.data[0].y) == list(ref.data[0].y)
        assert list(got.data[0].marker.color) == list(ref.data[0].marker.color)
        assert got.layout.title.text == ref.layout.title.text
        assert got.layout.yaxis.title.text == ref.layout.yaxis.title.text
        assert got.layout.height == ref.layout.height
