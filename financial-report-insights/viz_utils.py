"""
Visualization Utilities for Financial Insights Dashboard.
Reusable Plotly components for CFO-grade financial visualizations.
"""

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class FinancialVizUtils:
    """Reusable financial visualization components."""

    # Color schemes
    POSITIVE_COLOR = "#00cc96"
    NEGATIVE_COLOR = "#ef553b"
    NEUTRAL_COLOR = "#636efa"
    WARNING_COLOR = "#ffa15a"

    # Color palette for multiple series
    PALETTE = ["#636efa", "#00cc96", "#ffa15a", "#ef553b", "#ab63fa", "#19d3f3", "#e763fa", "#b6e880"]

    # Theme colors
    BACKGROUND_COLOR = "#fafafa"
    GRID_COLOR = "#e0e0e0"
    TEXT_COLOR = "#2d2d2d"

    @staticmethod
    def format_currency(value: float, currency: str = "$", decimals: int = 0) -> str:
        """Format number as currency with K/M/B suffixes."""
        if value is None or pd.isna(value):
            return "N/A"

        abs_value = abs(value)
        sign = "-" if value < 0 else ""

        if abs_value >= 1e9:
            return f"{sign}{currency}{abs_value / 1e9:.{decimals}f}B"
        elif abs_value >= 1e6:
            return f"{sign}{currency}{abs_value / 1e6:.{decimals}f}M"
        elif abs_value >= 1e3:
            return f"{sign}{currency}{abs_value / 1e3:.{decimals}f}K"
        return f"{sign}{currency}{abs_value:.{decimals}f}"

    @staticmethod
    def format_percent(value: float, decimals: int = 1) -> str:
        """Format number as percentage."""
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value * 100:.{decimals}f}%"

    @classmethod
    def create_kpi_card(
        cls,
        value: float,
        title: str = "",
        delta: Optional[float] = None,
        format_type: str = "currency",
        reference: Optional[float] = None,
    ) -> go.Figure:
        """
        Create a KPI indicator card.

        Args:
            value: The main value to display
            title: Title for the KPI
            delta: Change from previous period
            format_type: 'currency', 'percent', or 'number'
            reference: Reference value for delta comparison
        """
        # Format the value
        if format_type == "currency":
            formatted_value = cls.format_currency(value)
        elif format_type == "percent":
            formatted_value = cls.format_percent(value)
        else:
            formatted_value = f"{value:,.0f}"

        # Create indicator
        fig = go.Figure()

        indicator_args = {
            "mode": "number",
            "value": value,
            "title": {"text": title, "font": {"size": 14}},
            "number": {
                "font": {"size": 36, "color": cls.TEXT_COLOR},
                "valueformat": ",.0f" if format_type == "number" else None,
            },
        }

        if format_type == "currency":
            indicator_args["number"]["prefix"] = "$"
            indicator_args["number"]["valueformat"] = ",.0f"

        if format_type == "percent":
            indicator_args["number"]["suffix"] = "%"
            indicator_args["number"]["valueformat"] = ".1f"
            indicator_args["value"] = value * 100

        if delta is not None:
            indicator_args["mode"] = "number+delta"
            indicator_args["delta"] = {
                "reference": reference if reference else value - delta,
                "relative": True,
                "valueformat": ".1%",
                "increasing": {"color": cls.POSITIVE_COLOR},
                "decreasing": {"color": cls.NEGATIVE_COLOR},
            }

        fig.add_trace(go.Indicator(**indicator_args))

        fig.update_layout(height=150, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="white", plot_bgcolor="white")

        return fig

    @classmethod
    def create_gauge_chart(
        cls,
        value: float,
        title: str = "",
        min_val: float = 0,
        max_val: float = 100,
        thresholds: Optional[Dict[str, float]] = None,
        format_type: str = "number",
    ) -> go.Figure:
        """
        Create a gauge chart for ratio visualization.

        Args:
            value: Current value
            title: Chart title
            min_val: Minimum value for the gauge
            max_val: Maximum value for the gauge
            thresholds: Dict with 'warning' and 'good' threshold values
            format_type: 'number', 'percent', or 'ratio'
        """
        # Default thresholds
        if thresholds is None:
            thresholds = {"warning": max_val * 0.3, "good": max_val * 0.7}

        # Define gauge steps (color bands)
        steps = [
            {"range": [min_val, thresholds["warning"]], "color": "#ffcdd2"},
            {"range": [thresholds["warning"], thresholds["good"]], "color": "#fff9c4"},
            {"range": [thresholds["good"], max_val], "color": "#c8e6c9"},
        ]

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": title, "font": {"size": 16}},
                number={"font": {"size": 28}},
                gauge={
                    "axis": {"range": [min_val, max_val], "tickwidth": 1},
                    "bar": {"color": cls.NEUTRAL_COLOR},
                    "steps": steps,
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": thresholds["warning"],
                    },
                },
            )
        )

        fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=30), paper_bgcolor="white")

        return fig

    @classmethod
    def create_waterfall(
        cls, categories: List[str], values: List[float], title: str = "Variance Analysis", show_total: bool = True
    ) -> go.Figure:
        """
        Create a waterfall chart for variance analysis.

        Args:
            categories: List of category names
            values: List of values (positive or negative)
            title: Chart title
            show_total: Whether to show total bar
        """
        # Determine measure types
        measures = ["relative"] * len(values)
        if show_total:
            categories = categories + ["Total"]
            values = values + [sum(values)]
            measures = measures + ["total"]

        fig = go.Figure(
            go.Waterfall(
                name="Variance",
                orientation="v",
                x=categories,
                y=values,
                measure=measures,
                textposition="outside",
                text=[cls.format_currency(v) for v in values],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": cls.NEGATIVE_COLOR}},
                increasing={"marker": {"color": cls.POSITIVE_COLOR}},
                totals={"marker": {"color": cls.NEUTRAL_COLOR}},
            )
        )

        fig.update_layout(
            title=title,
            showlegend=False,
            height=400,
            margin=dict(l=50, r=50, t=60, b=50),
            paper_bgcolor="white",
            plot_bgcolor="white",
            yaxis_title="Amount",
        )

        return fig

    @classmethod
    def create_bullet_chart(
        cls, actual: float, target: float, ranges: List[float], title: str = "", format_type: str = "currency"
    ) -> go.Figure:
        """
        Create a bullet chart for target comparison.

        Args:
            actual: Actual value
            target: Target value
            ranges: List of range values [poor, average, good]
            title: Chart title
            format_type: 'currency', 'percent', or 'number'
        """
        fig = go.Figure()

        max_range = max(ranges + [actual, target]) * 1.1

        # Background ranges
        colors = ["#ffcdd2", "#fff9c4", "#c8e6c9"]
        for i, (r, color) in enumerate(zip(sorted(ranges), colors)):
            fig.add_trace(
                go.Bar(x=[r], y=[title], orientation="h", marker_color=color, showlegend=False, hoverinfo="skip")
            )

        # Actual bar
        fig.add_trace(
            go.Bar(
                x=[actual],
                y=[title],
                orientation="h",
                marker_color=cls.NEUTRAL_COLOR,
                width=0.3,
                name="Actual",
                text=cls.format_currency(actual) if format_type == "currency" else f"{actual:,.0f}",
                textposition="outside",
            )
        )

        # Target line
        fig.add_shape(type="line", x0=target, x1=target, y0=-0.3, y1=0.3, yref="paper", line=dict(color="red", width=3))

        fig.update_layout(
            barmode="overlay",
            height=100,
            margin=dict(l=100, r=50, t=30, b=30),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(range=[0, max_range]),
            showlegend=False,
        )

        return fig

    @classmethod
    def create_sparkline(
        cls, values: List[float], highlight_last: bool = True, show_area: bool = True, height: int = 60
    ) -> go.Figure:
        """
        Create a mini sparkline chart.

        Args:
            values: List of values
            highlight_last: Whether to highlight the last point
            show_area: Whether to fill area under line
            height: Chart height in pixels
        """
        x = list(range(len(values)))

        fig = go.Figure()

        # Area fill
        if show_area:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=values,
                    fill="tozeroy",
                    fillcolor="rgba(99, 110, 250, 0.2)",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines",
                line=dict(color=cls.NEUTRAL_COLOR, width=2),
                showlegend=False,
                hoverinfo="y",
            )
        )

        # Highlight last point
        if highlight_last and values:
            color = cls.POSITIVE_COLOR if len(values) > 1 and values[-1] >= values[-2] else cls.NEGATIVE_COLOR
            fig.add_trace(
                go.Scatter(
                    x=[x[-1]],
                    y=[values[-1]],
                    mode="markers",
                    marker=dict(size=8, color=color),
                    showlegend=False,
                    hoverinfo="y",
                )
            )

        fig.update_layout(
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )

        return fig

    @classmethod
    def create_heatmap(
        cls, data: pd.DataFrame, title: str = "", color_scale: str = "RdYlGn", show_values: bool = True
    ) -> go.Figure:
        """
        Create a heatmap for correlation or period comparison.

        Args:
            data: DataFrame with numeric values
            title: Chart title
            color_scale: Plotly color scale name
            show_values: Whether to show values in cells
        """
        fig = go.Figure(
            data=go.Heatmap(
                z=data.values,
                x=data.columns.tolist(),
                y=data.index.tolist(),
                colorscale=color_scale,
                text=data.values.round(2) if show_values else None,
                texttemplate="%{text}" if show_values else None,
                textfont={"size": 10},
                hoverongaps=False,
            )
        )

        fig.update_layout(title=title, height=400, margin=dict(l=100, r=50, t=60, b=50), paper_bgcolor="white")

        return fig

    @classmethod
    def create_time_series(
        cls,
        df: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        title: str = "",
        y_title: str = "",
        show_markers: bool = True,
    ) -> go.Figure:
        """
        Create a multi-line time series chart.

        Args:
            df: DataFrame with time series data
            x_col: Column name for x-axis (usually dates)
            y_cols: List of column names for y-axis
            title: Chart title
            y_title: Y-axis title
            show_markers: Whether to show data point markers
        """
        fig = go.Figure()

        for i, col in enumerate(y_cols):
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    name=col,
                    mode="lines+markers" if show_markers else "lines",
                    line=dict(color=cls.PALETTE[i % len(cls.PALETTE)], width=2),
                    marker=dict(size=6),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_title,
            height=400,
            margin=dict(l=50, r=50, t=60, b=50),
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=cls.GRID_COLOR)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=cls.GRID_COLOR)

        return fig

    @classmethod
    def create_comparison_bar(
        cls,
        categories: List[str],
        values1: List[float],
        values2: List[float],
        name1: str = "Actual",
        name2: str = "Budget",
        title: str = "",
        horizontal: bool = False,
    ) -> go.Figure:
        """
        Create a grouped bar chart for comparison.

        Args:
            categories: Category labels
            values1: First series values
            values2: Second series values
            name1: First series name
            name2: Second series name
            title: Chart title
            horizontal: Whether bars should be horizontal
        """
        fig = go.Figure()

        orientation = "h" if horizontal else "v"
        x_data, y_data = (values1, categories) if horizontal else (categories, values1)
        x_data2, y_data2 = (values2, categories) if horizontal else (categories, values2)

        fig.add_trace(
            go.Bar(
                x=x_data if not horizontal else values1,
                y=y_data if not horizontal else categories,
                name=name1,
                marker_color=cls.NEUTRAL_COLOR,
                orientation=orientation,
            )
        )

        fig.add_trace(
            go.Bar(
                x=x_data2 if not horizontal else values2,
                y=y_data2 if not horizontal else categories,
                name=name2,
                marker_color=cls.WARNING_COLOR,
                orientation=orientation,
            )
        )

        fig.update_layout(
            title=title,
            barmode="group",
            height=400,
            margin=dict(l=50, r=50, t=60, b=50),
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig

    @classmethod
    def create_donut_chart(
        cls, labels: List[str], values: List[float], title: str = "", hole_size: float = 0.4
    ) -> go.Figure:
        """
        Create a donut chart.

        Args:
            labels: Slice labels
            values: Slice values
            title: Chart title
            hole_size: Size of center hole (0-1)
        """
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=hole_size,
                    marker_colors=cls.PALETTE[: len(labels)],
                    textinfo="percent+label",
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title=title,
            height=400,
            margin=dict(l=50, r=50, t=60, b=50),
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
        )

        return fig

    @classmethod
    def create_ratio_dashboard(
        cls,
        ratios: Dict[str, float],
        benchmarks: Optional[Dict[str, Dict[str, float]]] = None,
        title: str = "Financial Ratios",
    ) -> go.Figure:
        """
        Create a comprehensive ratio dashboard with multiple gauges.

        Args:
            ratios: Dict of ratio names to values
            benchmarks: Dict of ratio names to benchmark dicts
            title: Dashboard title
        """
        n_ratios = len(ratios)
        cols = min(3, n_ratios)
        rows = (n_ratios + cols - 1) // cols

        fig = make_subplots(
            rows=rows, cols=cols, specs=[[{"type": "indicator"}] * cols for _ in range(rows)], vertical_spacing=0.3
        )

        for i, (name, value) in enumerate(ratios.items()):
            row = i // cols + 1
            col = i % cols + 1

            if value is None:
                continue

            # Get benchmark if available
            benchmark = benchmarks.get(name, {}) if benchmarks else {}
            reference = benchmark.get("target", benchmark.get("good"))

            fig.add_trace(
                go.Indicator(
                    mode="number+delta" if reference else "number",
                    value=value,
                    title={"text": name.replace("_", " ").title(), "font": {"size": 12}},
                    number={"font": {"size": 24}},
                    delta={"reference": reference, "relative": True} if reference else None,
                ),
                row=row,
                col=col,
            )

        fig.update_layout(
            title=title, height=150 * rows + 60, margin=dict(l=30, r=30, t=60, b=30), paper_bgcolor="white"
        )

        return fig

    @classmethod
    def create_trend_chart(
        cls,
        periods: List[str],
        values: List[float],
        title: str = "",
        show_forecast: bool = False,
        forecast_periods: Optional[List[str]] = None,
        forecast_values: Optional[List[float]] = None,
    ) -> go.Figure:
        """
        Create a trend chart with optional forecast.

        Args:
            periods: Time period labels
            values: Historical values
            title: Chart title
            show_forecast: Whether to show forecast
            forecast_periods: Forecast period labels
            forecast_values: Forecasted values
        """
        fig = go.Figure()

        # Historical line
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=values,
                name="Historical",
                mode="lines+markers",
                line=dict(color=cls.NEUTRAL_COLOR, width=2),
                marker=dict(size=8),
            )
        )

        # Forecast line
        if show_forecast and forecast_periods and forecast_values:
            # Connect to last historical point
            all_periods = [periods[-1]] + forecast_periods
            all_values = [values[-1]] + forecast_values

            fig.add_trace(
                go.Scatter(
                    x=all_periods,
                    y=all_values,
                    name="Forecast",
                    mode="lines+markers",
                    line=dict(color=cls.WARNING_COLOR, width=2, dash="dash"),
                    marker=dict(size=8, symbol="diamond"),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Period",
            yaxis_title="Value",
            height=400,
            margin=dict(l=50, r=50, t=60, b=50),
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=cls.GRID_COLOR)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=cls.GRID_COLOR)

        return fig

    @classmethod
    def create_variance_table_chart(
        cls, categories: List[str], actual: List[float], budget: List[float], title: str = "Budget vs Actual"
    ) -> go.Figure:
        """
        Create a table-style chart showing variances.
        """
        variance = [a - b for a, b in zip(actual, budget)]
        variance_pct = [(a - b) / b * 100 if b != 0 else 0 for a, b in zip(actual, budget)]

        # Colors based on variance (favorable/unfavorable)
        colors = [cls.POSITIVE_COLOR if v >= 0 else cls.NEGATIVE_COLOR for v in variance]

        fig = go.Figure()

        # Variance bars
        fig.add_trace(
            go.Bar(
                y=categories,
                x=variance,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+,.0f} ({p:+.1f}%)" for v, p in zip(variance, variance_pct)],
                textposition="outside",
                name="Variance",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Variance",
            height=max(300, len(categories) * 40 + 100),
            margin=dict(l=150, r=100, t=60, b=50),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
        )

        # Add zero line
        fig.add_vline(x=0, line_width=2, line_color="gray")

        return fig

    @classmethod
    def create_simple_bar(
        cls,
        labels: List[str],
        values: List[float],
        colors: Optional[List[str]],
        title: str,
        y_title: str = "",
        height: int = 350,
        show_legend: bool = False,
    ) -> go.Figure:
        """
        Create a single-series vertical bar chart.

        This is the shared helper for the high-frequency inline pattern:
            fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
            fig.update_layout(title=title, yaxis_title=y_title,
                              height=height, showlegend=False)

        Used in place of that boilerplate across profitability, returns,
        coverage, and working-capital render methods.

        Args:
            labels: X-axis bar labels.
            values: Y-axis bar heights.
            colors: Per-bar colour strings (len must match labels). If None,
                    the class PALETTE is used cyclically.
            title:  Chart title.
            y_title: Y-axis label.
            height: Chart height in pixels (default 350).
            show_legend: Whether to show the Plotly legend (default False).

        Returns:
            go.Figure with a single Bar trace.
        """
        bar_colors = colors if colors is not None else [cls.PALETTE[i % len(cls.PALETTE)] for i in range(len(labels))]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=bar_colors,
                )
            ]
        )
        fig.update_layout(
            title=title,
            yaxis_title=y_title,
            height=height,
            showlegend=show_legend,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=50, r=50, t=60, b=50),
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=cls.GRID_COLOR)
        return fig
