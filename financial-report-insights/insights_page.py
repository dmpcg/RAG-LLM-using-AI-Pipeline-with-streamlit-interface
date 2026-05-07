"""
Dynamic Financial Insights Dashboard Page.
Interactive Streamlit page with Plotly visualizations for CFO-grade financial insights.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np

from config import settings
from excel_processor import ExcelProcessor, WorkbookData
from financial_analyzer import CharlieAnalyzer, FinancialData, CustomKPIDefinition, PeerCompanyData
from viz_utils import FinancialVizUtils

logger = logging.getLogger(__name__)


class FinancialInsightsPage:
    """
    Dynamic financial insights dashboard with:
    - Executive summary cards
    - Interactive charts
    - Drill-down capabilities
    - Natural language query integration
    """

    # (label, method_name, needs_workbook)
    CATEGORY_TABS = {
        "Executive & Overview": [
            ("Executive Summary", "_render_executive_summary", True),
            ("Financial Ratios", "_render_ratios_dashboard", False),
            ("Health Score", "_render_health_score", False),
            ("Financial Health", "_render_financial_health_score", False),
            ("Credit Rating", "_render_credit_rating", False),
            ("AI Narrative", "_render_narrative", False),
            ("Custom KPIs", "_render_custom_kpis", False),
            ("Industry Benchmark", "_render_industry_benchmark", False),
        ],
        "Profitability": [
            ("Gross Margin", "_render_gross_margin_stability", False),
            ("EBITDA Quality", "_render_ebitda_margin_quality", False),
            ("Net Profit Margin", "_render_net_profit_margin", False),
            ("Operating Margin", "_render_operating_margin", False),
            ("Earnings Quality", "_render_earnings_quality", False),
            ("Income Quality", "_render_income_quality", False),
            ("Op Income Quality", "_render_operating_income_quality", False),
            ("Income Stability", "_render_income_stability", False),
            ("Income Resilience", "_render_income_resilience", False),
            ("Income Retention", "_render_income_retention", False),
            ("Profitability Decomp", "_render_profitability_decomp", False),
            ("Profitability Depth", "_render_profitability_depth", False),
            ("Profit Retention", "_render_profit_retention", False),
            ("Profit Sustainability", "_render_profit_sustainability", False),
            ("Profit Conversion", "_render_profit_conversion", False),
        ],
        "Returns & Efficiency": [
            ("ROE Analysis", "_render_roe_analysis", False),
            ("ROA Quality", "_render_roa_quality", False),
            ("ROIC Analysis", "_render_roic_analysis", False),
            ("Asset Efficiency", "_render_asset_efficiency", False),
            ("Revenue Efficiency", "_render_revenue_efficiency", False),
            ("Tax Efficiency", "_render_tax_efficiency", False),
            ("Fixed Asset Eff", "_render_fixed_asset_efficiency", False),
            ("Operational Eff", "_render_operational_efficiency", False),
            ("Capital Efficiency", "_render_capital_efficiency", False),
            ("Risk-Adjusted", "_render_risk_adjusted", False),
            ("Asset Deployment", "_render_asset_deployment_efficiency", False),
            ("Financial Prod", "_render_financial_productivity", False),
            ("Resource Optim", "_render_resource_optimization", False),
            ("Revenue Quality", "_render_revenue_quality_index", False),
        ],
        "Cash Flow & Liquidity": [
            ("Cash Flow", "_render_cashflow_dashboard", False),
            ("CF Quality", "_render_cash_flow_quality", False),
            ("CF Forecast", "_render_cashflow_forecast", False),
            ("Cash Conversion", "_render_cash_conversion", False),
            ("Cash Conv Eff", "_render_cash_conversion_efficiency", False),
            ("Cash Burn", "_render_cash_burn", False),
            ("Defensive Interval", "_render_defensive_interval", False),
            ("Defensive Posture", "_render_defensive_posture", False),
            ("CF Stability", "_render_cash_flow_stability", False),
            ("Liquidity Stress", "_render_liquidity_stress", False),
            ("Operating CF Ratio", "_render_operating_cash_flow_ratio", False),
            ("Revenue Cash Real", "_render_revenue_cash_realization", False),
        ],
        "Leverage & Debt": [
            ("DuPont Analysis", "_render_dupont_analysis", False),
            ("Operating Leverage", "_render_operating_leverage", False),
            ("Op Leverage Depth", "_render_operational_leverage_depth", False),
            ("Fixed Cost Leverage", "_render_fixed_cost_leverage_ratio", False),
            ("Debt Composition", "_render_debt_composition", False),
            ("Debt Quality", "_render_debt_quality", False),
            ("Debt Svc Coverage", "_render_debt_service_coverage", False),
            ("Debt to Capital", "_render_debt_to_capital", False),
            ("Debt to Equity", "_render_debt_to_equity", False),
            ("Debt Discipline", "_render_debt_discipline", False),
            ("Debt Management", "_render_debt_management", False),
            ("Debt Burden Index", "_render_debt_burden_index", False),
            ("Interest Coverage", "_render_interest_coverage", False),
        ],
        "Working Capital & Turnover": [
            ("Working Capital", "_render_working_capital", False),
            ("Receivables Mgmt", "_render_receivables_management", False),
            ("Receivables Turn", "_render_receivables_turnover", False),
            ("Payables Turnover", "_render_payables_turnover", False),
            ("Inventory Turnover", "_render_inventory_turnover", False),
            ("Cash Cycle", "_render_cash_conversion_cycle", False),
            ("Inventory Holding", "_render_inventory_holding_cost", False),
            ("Inventory Coverage", "_render_inventory_coverage", False),
            ("Liability Coverage", "_render_liability_coverage_strength", False),
            ("Liability Mgmt", "_render_liability_management", False),
        ],
        "Valuation & Growth": [
            ("Valuation Indicators", "_render_valuation_indicators", False),
            ("Sustainable Growth", "_render_sustainable_growth", False),
            ("Internal Growth Rate", "_render_internal_growth_rate", False),
            ("Internal Growth Cap", "_render_internal_growth_capacity", False),
            ("FCF Yield", "_render_fcf_yield", False),
            ("Net Worth Growth", "_render_net_worth_growth", False),
            ("Valuation Signal", "_render_valuation_signal", False),
            ("Revenue Growth", "_render_revenue_growth", False),
            ("Equity Reinvest", "_render_equity_reinvestment", False),
            ("Revenue Predict", "_render_revenue_predictability", False),
            ("Margin of Safety", "_render_margin_of_safety", False),
        ],
        "Scoring Models & Risk": [
            ("Scoring Models", "_render_scoring_models", False),
            ("Altman Z-Score", "_render_altman_z_score", False),
            ("Piotroski F-Score", "_render_piotroski_f_score", False),
            ("Beneish M-Score", "_render_beneish_m_score", False),
            ("Concentration Risk", "_render_concentration_risk", False),
            ("Operational Risk", "_render_operational_risk", False),
            ("Covenant Monitor", "_render_covenant_monitor", False),
            ("Financial Resilience", "_render_financial_resilience", False),
            ("Structural Strength", "_render_structural_strength", False),
        ],
        "Capital Structure": [
            ("Capital Allocation", "_render_capital_allocation", False),
            ("Capital Adequacy", "_render_capital_adequacy", False),
            ("Capital Discipline", "_render_capital_discipline", False),
            ("Capital Preserv", "_render_capital_preservation", False),
            ("Solvency Depth", "_render_solvency_depth", False),
            ("Financial Flex", "_render_financial_flexibility", False),
            ("Equity Multiplier", "_render_equity_multiplier", False),
            ("Equity Preserv", "_render_equity_preservation", False),
            ("Asset Quality", "_render_asset_quality", False),
            ("Noncurrent Assets", "_render_noncurrent_asset_ratio", False),
            ("Net Debt Position", "_render_net_debt_position", False),
            ("Earnings to Debt", "_render_earnings_to_debt", False),
            ("Profit Ret Power", "_render_profit_retention_power", False),
            ("Cost Control", "_render_cost_control", False),
        ],
        "Trends & Comparison": [
            ("Trends Dashboard", "_render_trends_dashboard", True),
            ("Trend Forecast", "_render_trend_forecast", False),
            ("Period Comparison", "_render_period_comparison", True),
            ("Peer Comparison", "_render_peer_comparison", False),
            ("Budget Analysis", "_render_budget_dashboard", True),
            ("Variance Waterfall", "_render_variance_waterfall", True),
        ],
        "Simulation & Scenarios": [
            ("What-If Analysis", "_render_what_if_analysis", False),
            ("Monte Carlo", "_render_monte_carlo", False),
            ("Driver Analysis", "_render_driver_analysis", False),
            ("Ratio Decomposition", "_render_ratio_decomposition", False),
        ],
        "Advanced Metrics": [
            ("WACC Analysis", "_render_wacc_analysis", False),
            ("EVA Analysis", "_render_eva_analysis", False),
            ("Depreciation Burden", "_render_depreciation_burden", False),
            ("CapEx to Revenue", "_render_capex_to_revenue", False),
            ("OpEx Ratio", "_render_operating_expense_ratio", False),
            ("Expense Discipline", "_render_expense_ratio_discipline", False),
            ("Funding Mix", "_render_funding_mix_balance", False),
            ("Funding Efficiency", "_render_funding_efficiency", False),
            ("Obligation Coverage", "_render_obligation_coverage", False),
            ("Payout Discipline", "_render_payout_discipline", False),
            ("Payout Resilience", "_render_payout_resilience", False),
            ("Dividend Payout", "_render_dividend_payout", False),
            ("CF to Debt", "_render_cash_flow_to_debt", False),
            ("Asset Lightness", "_render_asset_lightness", False),
            ("Operating Momentum", "_render_operating_momentum", False),
        ],
        "Underwriting": [
            ("Credit Scorecard", "_render_credit_scorecard", False),
            ("Debt Capacity", "_render_debt_capacity", False),
            ("Covenant Package", "_render_covenant_package", False),
        ],
        "Startup Modeling": [
            ("SaaS Metrics", "_render_saas_metrics", False),
            ("Unit Economics", "_render_unit_economics", False),
            ("Burn & Runway", "_render_burn_runway", False),
            ("Funding Scenarios", "_render_funding_scenarios", False),
        ],
        "Portfolio Analysis": [
            ("Portfolio Overview", "_render_portfolio_overview", True),
            ("Correlation Matrix", "_render_portfolio_correlation", True),
            ("Diversification", "_render_portfolio_diversification", True),
        ],
        "Regulatory & Compliance": [
            ("SOX Compliance", "_render_sox_compliance", False),
            ("SEC Filing Quality", "_render_sec_filing_quality", False),
            ("Regulatory Thresholds", "_render_regulatory_thresholds", False),
            ("Audit Risk", "_render_audit_risk", False),
        ],
    }

    def __init__(self, docs_folder: str = "./documents"):
        """Initialize the insights page."""
        self.docs_folder = Path(docs_folder)
        self.analyzer = CharlieAnalyzer()
        self.viz = FinancialVizUtils()
        self.processor = ExcelProcessor(docs_folder)

    def _get_analyzer(self):
        """Backwards-compat alias for self.analyzer."""
        return self.analyzer

    def render(self):
        """Main page render method."""
        st.title("Financial Insights Dashboard")
        st.markdown("*Powered by Charlie-style Analysis*")

        # Load data
        excel_files = self.processor.scan_for_excel_files()

        if not excel_files:
            st.warning("No Excel files found in the documents folder.")
            st.info("Upload Excel files (.xlsx, .xlsm, .csv) to the documents folder to begin analysis.")

            # Show upload option
            self._render_upload_section()
            return

        # Sidebar: Data selection
        with st.sidebar:
            selected_file, selected_sheet = self._render_data_selector(excel_files)
            analysis_options = self._render_analysis_options()

        # Load selected data
        if selected_file:
            try:
                workbook = self.processor.load_workbook(selected_file)
                combined = self.processor.combine_sheets_intelligently(workbook)

                # Get the selected sheet's data
                df = self._get_sheet_data(workbook, selected_sheet)

                if df is not None and not df.empty:
                    # Store in session state for cross-tab access
                    st.session_state['current_df'] = df
                    st.session_state['current_workbook'] = workbook
                    cache_key = f"analysis_{hash(str(df.to_dict()))}"
                    if cache_key not in st.session_state:
                        st.session_state[cache_key] = self.analyzer.analyze(df)
                    st.session_state['analysis_results'] = st.session_state[cache_key]

                    # Category-based navigation (replaces 132 flat tabs)
                    categories = list(self.CATEGORY_TABS.keys())
                    selected_cat = st.selectbox(
                        "Analysis Category", categories, index=0,
                        key="insights_category"
                    )

                    # Sub-tabs for selected category
                    entries = self.CATEGORY_TABS[selected_cat]
                    sub_tabs = st.tabs([label for label, _, _ in entries])


                    for tab, (label, method, needs_wb) in zip(sub_tabs, entries):
                        with tab:
                            if needs_wb:
                                getattr(self, method)(df, workbook)
                            else:
                                getattr(self, method)(df)

                    # Report download & data explorer (always visible)
                    self._render_report_download(df)
                    self._render_data_explorer(df, workbook)

                else:
                    st.warning("No data available in the selected sheet.")

            except Exception as e:
                st.error(f"Error loading data: {e}")
                logger.exception("Failed to load Excel data")

    def _render_data_selector(self, excel_files: List[Path]) -> tuple:
        """Render the data source selector in sidebar."""
        st.header("Data Source")

        # File selection
        file_names = [f.name for f in excel_files]
        selected_name = st.selectbox("Select File", file_names)
        selected_file = excel_files[file_names.index(selected_name)] if selected_name else None

        # Sheet selection (reuse cached workbook to avoid double load)
        selected_sheet = None
        if selected_file:
            try:
                cache_key = f"_wb_{selected_file}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = self.processor.load_workbook(selected_file)
                workbook = st.session_state[cache_key]
                sheet_names = [s.name for s in workbook.sheets]
                if sheet_names:
                    selected_sheet = st.selectbox("Select Sheet", ["All Sheets"] + sheet_names)
            except Exception as e:
                st.error(f"Error reading sheets: {e}")
                logger.exception("Failed to read sheet names")

        # File info
        if selected_file:
            st.caption(f"Size: {selected_file.stat().st_size / 1024:.1f} KB")

        return selected_file, selected_sheet

    def _render_analysis_options(self) -> Dict[str, Any]:
        """Render analysis options in sidebar."""
        st.header("Analysis Options")

        options = {
            'show_insights': st.checkbox("Show AI Insights", value=True),
            'show_benchmarks': st.checkbox("Show Benchmarks", value=True),
            'currency': st.selectbox("Currency", ["$", "€", "£", "¥"], index=0),
            'decimal_places': st.slider("Decimal Places", 0, 4, 2)
        }

        # Refresh button
        if st.button("Refresh Analysis"):
            # Clear cached data
            for key in ['current_df', 'current_workbook', 'analysis_results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        return options

    def _get_sheet_data(self, workbook: WorkbookData, selected_sheet: Optional[str]) -> pd.DataFrame:
        """Get data for the selected sheet."""
        if not workbook.sheets:
            return pd.DataFrame()

        if selected_sheet == "All Sheets" or selected_sheet is None:
            # Combine all sheets
            dfs = [s.df for s in workbook.sheets]
            if dfs:
                # Add source column
                for i, (df, sheet) in enumerate(zip(dfs, workbook.sheets)):
                    dfs[i] = df.copy()
                    dfs[i]['_source_sheet'] = sheet.name
                return pd.concat(dfs, ignore_index=True)
            return pd.DataFrame()

        # Find specific sheet
        for sheet in workbook.sheets:
            if sheet.name == selected_sheet:
                return sheet.df

        return pd.DataFrame()

    def _render_upload_section(self):
        """Render file upload section."""
        st.subheader("Upload Financial Data")

        uploaded_files = st.file_uploader(
            "Upload Excel or CSV files",
            type=['xlsx', 'xlsm', 'xls', 'csv'],
            accept_multiple_files=True
        )

        if uploaded_files:
            max_size = settings.max_file_size_mb * 1024 * 1024
            for file in uploaded_files:
                safe_name = os.path.basename(file.name).strip()
                if not safe_name or safe_name in (".", ".."):
                    st.error(f"Invalid filename: {file.name}")
                    continue
                save_path = (self.docs_folder / safe_name).resolve()
                if not str(save_path).startswith(str(self.docs_folder.resolve())):
                    st.error(f"Rejected path traversal attempt: {file.name}")
                    continue
                file.seek(0, 2)
                size = file.tell()
                file.seek(0)
                if size > max_size:
                    st.error(f"File too large: {safe_name} ({size / 1024 / 1024:.1f} MB)")
                    continue
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
                st.success(f"Uploaded: {safe_name}")

            if st.button("Load Uploaded Files"):
                st.rerun()

    def _render_executive_summary(self, df: pd.DataFrame, workbook: WorkbookData):
        """Render executive summary with KPI cards and key insights."""
        st.subheader("Executive Summary")

        # Extract key metrics from analysis
        analysis = st.session_state.get('analysis_results', {})

        # KPI Cards row
        col1, col2, col3, col4 = st.columns(4)

        # Try to find key financial metrics
        metrics = self._extract_key_metrics(df)

        with col1:
            if metrics.get('revenue'):
                st.metric(
                    label="Revenue",
                    value=self.viz.format_currency(metrics['revenue']),
                    delta=f"{metrics.get('revenue_growth', 0):.1%}" if metrics.get('revenue_growth') else None
                )
            else:
                st.metric(label="Total Records", value=f"{len(df):,}")

        with col2:
            if metrics.get('net_income'):
                st.metric(
                    label="Net Income",
                    value=self.viz.format_currency(metrics['net_income']),
                    delta=f"{metrics.get('net_income_growth', 0):.1%}" if metrics.get('net_income_growth') else None
                )
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.metric(label=f"Avg {numeric_cols[0]}", value=f"{df[numeric_cols[0]].mean():,.2f}")

        with col3:
            prof_ratios = analysis.get('profitability_ratios', {})
            if prof_ratios.get('operating_margin'):
                st.metric(
                    label="Operating Margin",
                    value=f"{prof_ratios['operating_margin']:.1%}"
                )
            elif prof_ratios.get('net_margin'):
                st.metric(
                    label="Net Margin",
                    value=f"{prof_ratios['net_margin']:.1%}"
                )
            else:
                st.metric(label="Columns", value=len(df.columns))

        with col4:
            cf_analysis = analysis.get('cash_flow')
            if cf_analysis and cf_analysis.free_cash_flow:
                st.metric(
                    label="Free Cash Flow",
                    value=self.viz.format_currency(cf_analysis.free_cash_flow)
                )
            else:
                st.metric(label="Sheets", value=len(workbook.sheets))

        st.divider()

        # AI-generated insights
        insights = analysis.get('insights', [])
        if insights:
            st.subheader("AI-Generated Insights")

            for insight in insights[:5]:
                severity_color = {
                    'info': 'blue',
                    'warning': 'orange',
                    'critical': 'red'
                }.get(insight.severity, 'gray')

                with st.container():
                    st.markdown(f"**{insight.category.upper()}**: {insight.message}")
                    if insight.recommendation:
                        st.caption(f"*Recommendation: {insight.recommendation}*")

        # Data overview
        st.subheader("Data Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Detected Financial Types:**")
            for sheet in workbook.sheets:
                st.caption(f"• {sheet.name}: {sheet.detected_type or 'Custom data'}")

        with col2:
            st.markdown("**Numeric Columns Available:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:10]
            for col in numeric_cols:
                st.caption(f"• {col}")

    def _render_ratios_dashboard(self, df: pd.DataFrame):
        """Render financial ratios visualization."""
        st.subheader("Financial Ratios Analysis")

        analysis = st.session_state.get('analysis_results', {})

        col1, col2 = st.columns(2)

        with col1:
            # Liquidity ratios
            st.markdown("**Liquidity Ratios**")
            liquidity = analysis.get('liquidity_ratios', {})

            if any(v is not None for v in liquidity.values()):
                fig = self.viz.create_ratio_dashboard(
                    {k: v for k, v in liquidity.items() if v is not None},
                    title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Liquidity ratios require balance sheet data (Current Assets, Current Liabilities, Inventory, Cash)")

        with col2:
            # Profitability ratios
            st.markdown("**Profitability Ratios**")
            profitability = analysis.get('profitability_ratios', {})

            if any(v is not None for v in profitability.values()):
                # Filter out None values and format for display
                profit_display = {k: v for k, v in profitability.items() if v is not None}
                fig = self.viz.create_ratio_dashboard(profit_display, title="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Profitability ratios require income statement data (Revenue, COGS, Operating Income, Net Income)")

        # Leverage and Efficiency
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Leverage Ratios**")
            leverage = analysis.get('leverage_ratios', {})

            if any(v is not None for v in leverage.values()):
                lev_display = {k: v for k, v in leverage.items() if v is not None}
                fig = self.viz.create_ratio_dashboard(lev_display, title="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Leverage ratios require debt and equity data")

        with col4:
            st.markdown("**Efficiency Ratios**")
            efficiency = analysis.get('efficiency_ratios', {})

            if any(v is not None for v in efficiency.values()):
                eff_display = {k: v for k, v in efficiency.items() if v is not None}
                fig = self.viz.create_ratio_dashboard(eff_display, title="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Efficiency ratios require revenue, inventory, and receivables data")

        # Ratio definitions
        with st.expander("Ratio Definitions"):
            for name, defn in self.analyzer.ratio_definitions.items():
                st.markdown(f"**{defn['name']}**: {defn['formula']}")
                st.caption(defn['interpretation'])

    def _render_trends_dashboard(self, df: pd.DataFrame, workbook: WorkbookData):
        """Render trends and forecasts visualization."""
        st.subheader("Trends & Forecasts")

        # Select metric for trend analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.info("No numeric columns available for trend analysis.")
            return

        col1, col2 = st.columns([1, 2])

        with col1:
            selected_metric = st.selectbox("Select Metric", numeric_cols)
            forecast_periods = st.slider("Forecast Periods", 1, 12, 3)
            forecast_method = st.selectbox(
                "Forecast Method",
                ["linear", "moving_average", "growth_rate"]
            )

        # Check if there's a time/period column
        potential_time_cols = [col for col in df.columns if any(
            pattern in col.lower() for pattern in ['date', 'period', 'month', 'year', 'quarter']
        )]

        period_col = None
        if potential_time_cols:
            period_col = st.selectbox("Period Column (optional)", ["None"] + potential_time_cols)
            if period_col == "None":
                period_col = None

        with col2:
            # Create trend visualization
            values = df[selected_metric].dropna().tolist()

            if len(values) >= 2:
                # Trend analysis
                if period_col:
                    periods = df[period_col].astype(str).tolist()
                else:
                    periods = [f"Period {i+1}" for i in range(len(values))]

                # Generate forecast
                forecast = self.analyzer.forecast_simple(values, forecast_periods, forecast_method)

                fig = self.viz.create_trend_chart(
                    periods=periods,
                    values=values,
                    title=f"{selected_metric} Trend",
                    show_forecast=True,
                    forecast_periods=forecast.forecast_periods,
                    forecast_values=forecast.forecasted_values
                )
                st.plotly_chart(fig, use_container_width=True)

                # Trend statistics
                trend = self.analyzer.analyze_trends(
                    pd.DataFrame({selected_metric: values}),
                    selected_metric
                )

                st.markdown("**Trend Statistics:**")
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

                with stats_col1:
                    if trend.cagr is not None:
                        st.metric("CAGR", f"{trend.cagr:.1%}")
                with stats_col2:
                    if trend.yoy_growth is not None:
                        st.metric("YoY Growth", f"{trend.yoy_growth:.1%}")
                with stats_col3:
                    st.metric("Trend Direction", trend.trend_direction.title())
                with stats_col4:
                    st.metric("Seasonality", "Detected" if trend.seasonality_detected else "Not detected")

            else:
                st.warning("Need at least 2 data points for trend analysis.")

        # Multi-metric comparison
        st.subheader("Multi-Metric Comparison")

        selected_metrics = st.multiselect(
            "Select metrics to compare",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )

        if selected_metrics and len(df) > 1:
            # Create comparison chart
            chart_df = df[selected_metrics].copy()
            chart_df['Period'] = range(len(chart_df))

            fig = self.viz.create_time_series(
                chart_df,
                'Period',
                selected_metrics,
                title="Metric Comparison",
                show_markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_budget_dashboard(self, df: pd.DataFrame, workbook: WorkbookData):
        """Render budget vs actual analysis."""
        st.subheader("Budget vs Actual Analysis")

        # Try to detect budget and actual columns
        cols = df.columns.tolist()

        actual_cols = [c for c in cols if 'actual' in c.lower()]
        budget_cols = [c for c in cols if 'budget' in c.lower() or 'plan' in c.lower() or 'target' in c.lower()]
        item_cols = [c for c in cols if any(x in c.lower() for x in ['item', 'category', 'account', 'description', 'name'])]

        if not (actual_cols and budget_cols):
            st.info("""
            Budget analysis requires columns with 'Actual' and 'Budget' (or 'Plan'/'Target') in their names.

            **Tips for budget analysis:**
            - Ensure your data has columns like 'Actual', 'Budget', 'Variance'
            - Include a category/item column for line-item analysis
            - Upload a budget vs actual report for full functionality
            """)

            # Show general variance if numeric columns exist
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                st.subheader("Manual Variance Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    actual_col = st.selectbox("Select Actual Column", numeric_cols)
                with col2:
                    budget_col = st.selectbox("Select Budget Column", [c for c in numeric_cols if c != actual_col])

                if st.button("Calculate Variance"):
                    variance = df[actual_col].sum() - df[budget_col].sum()
                    variance_pct = variance / df[budget_col].sum() if df[budget_col].sum() != 0 else 0

                    st.metric(
                        "Total Variance",
                        self.viz.format_currency(variance),
                        f"{variance_pct:.1%}"
                    )
            return

        # Auto-detected budget analysis
        col1, col2, col3 = st.columns(3)

        with col1:
            actual_col = st.selectbox("Actual Column", actual_cols)
        with col2:
            budget_col = st.selectbox("Budget Column", budget_cols)
        with col3:
            item_col = st.selectbox("Item Column", item_cols if item_cols else cols[:5])

        # Calculate variances
        df_analysis = df[[item_col, actual_col, budget_col]].dropna()
        df_analysis['Variance'] = df_analysis[actual_col] - df_analysis[budget_col]
        df_analysis['Variance_Pct'] = df_analysis['Variance'] / df_analysis[budget_col].abs() * 100

        # Summary metrics
        total_actual = df_analysis[actual_col].sum()
        total_budget = df_analysis[budget_col].sum()
        total_variance = total_actual - total_budget
        variance_pct = total_variance / total_budget if total_budget != 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Actual", self.viz.format_currency(total_actual))
        with m2:
            st.metric("Total Budget", self.viz.format_currency(total_budget))
        with m3:
            st.metric("Total Variance", self.viz.format_currency(total_variance))
        with m4:
            st.metric("Variance %", f"{variance_pct:.1%}")

        # Waterfall chart
        st.subheader("Variance Waterfall")

        # Get top variances for waterfall
        top_variances = df_analysis.nlargest(10, 'Variance', keep='first')

        fig = self.viz.create_waterfall(
            categories=top_variances[item_col].tolist(),
            values=top_variances['Variance'].tolist(),
            title="Top Variances by Category"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Favorable Variances (Under Budget)**")
            favorable = df_analysis[df_analysis['Variance'] < 0].sort_values('Variance')
            if not favorable.empty:
                st.dataframe(favorable.head(10), use_container_width=True)
            else:
                st.caption("No favorable variances")

        with col2:
            st.markdown("**Unfavorable Variances (Over Budget)**")
            unfavorable = df_analysis[df_analysis['Variance'] > 0].sort_values('Variance', ascending=False)
            if not unfavorable.empty:
                st.dataframe(unfavorable.head(10), use_container_width=True)
            else:
                st.caption("No unfavorable variances")

    def _render_cashflow_dashboard(self, df: pd.DataFrame):
        """Render cash flow and working capital analysis."""
        st.subheader("Cash Flow & Working Capital")

        analysis = st.session_state.get('analysis_results', {})
        cf_analysis = analysis.get('cash_flow')
        wc_analysis = analysis.get('working_capital')

        # Cash Flow Metrics
        st.markdown("**Cash Flow Metrics**")

        if cf_analysis:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if cf_analysis.operating_cf is not None:
                    st.metric("Operating CF", self.viz.format_currency(cf_analysis.operating_cf))
            with col2:
                if cf_analysis.free_cash_flow is not None:
                    st.metric("Free Cash Flow", self.viz.format_currency(cf_analysis.free_cash_flow))
            with col3:
                if cf_analysis.dso is not None:
                    st.metric("DSO", f"{cf_analysis.dso:.0f} days")
            with col4:
                if cf_analysis.cash_conversion_cycle is not None:
                    st.metric("Cash Conversion Cycle", f"{cf_analysis.cash_conversion_cycle:.0f} days")

            # Cash Conversion Cycle gauge
            if cf_analysis.cash_conversion_cycle is not None:
                fig = self.viz.create_gauge_chart(
                    value=cf_analysis.cash_conversion_cycle,
                    title="Cash Conversion Cycle (Days)",
                    min_val=0,
                    max_val=120,
                    thresholds={'warning': 60, 'good': 30}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Working capital cycle components
            if cf_analysis.dso is not None or cf_analysis.dio is not None or cf_analysis.dpo is not None:
                st.markdown("**Working Capital Cycle Components**")
                components = []
                values = []

                if cf_analysis.dso:
                    components.append("DSO\n(Days Sales Outstanding)")
                    values.append(cf_analysis.dso)
                if cf_analysis.dio:
                    components.append("DIO\n(Days Inventory Outstanding)")
                    values.append(cf_analysis.dio)
                if cf_analysis.dpo:
                    components.append("DPO\n(Days Payables Outstanding)")
                    values.append(cf_analysis.dpo)

                if components:
                    fig = self.viz.create_comparison_bar(
                        categories=components,
                        values1=values,
                        values2=[45] * len(values),  # Benchmark
                        name1="Actual",
                        name2="Benchmark",
                        title="Working Capital Cycle Days"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("""
            Cash flow analysis requires specific data:
            - Operating Cash Flow
            - Investing Cash Flow
            - Financing Cash Flow
            - Accounts Receivable
            - Inventory
            - Accounts Payable
            - Revenue and COGS

            Upload a cash flow statement or balance sheet for full functionality.
            """)

        # Working Capital Analysis
        st.markdown("**Working Capital Analysis**")

        if wc_analysis and wc_analysis.net_working_capital is not None:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Current Assets", self.viz.format_currency(wc_analysis.current_assets or 0))
            with col2:
                st.metric("Current Liabilities", self.viz.format_currency(wc_analysis.current_liabilities or 0))
            with col3:
                st.metric("Net Working Capital", self.viz.format_currency(wc_analysis.net_working_capital))

    def _render_scoring_models(self, df: pd.DataFrame):
        """Render Scoring Models dashboard: DuPont, Z-Score, F-Score, Composite Health."""
        st.subheader("Scoring Models")

        analysis = st.session_state.get('analysis_results', {})
        dupont = analysis.get('dupont')
        z_result = analysis.get('altman_z_score')
        f_result = analysis.get('piotroski_f_score')
        health = analysis.get('composite_health')

        # --- Row 1: Composite Health Score ---
        if health is not None:
            st.markdown("**Composite Financial Health**")
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                st.metric("Health Score", f"{health.score}/100")
            with col2:
                grade_colors = {'A': 'green', 'B': 'blue', 'C': 'orange', 'D': 'red', 'F': 'red'}
                color = grade_colors.get(health.grade, 'gray')
                st.markdown(f"**Grade:** :{color}[**{health.grade}**]")
            with col3:
                if health.interpretation:
                    st.info(health.interpretation)

            # Component breakdown bar chart
            if health.component_scores:
                import plotly.graph_objects as go
                components = list(health.component_scores.keys())
                values = list(health.component_scores.values())
                max_pts = {'z_score': 25, 'f_score': 25, 'profitability': 20,
                           'liquidity': 15, 'leverage': 15}
                maxes = [max_pts.get(c, 25) for c in components]
                labels = [c.replace('_', ' ').title() for c in components]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=labels, y=values, name='Score',
                    marker_color='steelblue', text=values, textposition='auto',
                ))
                fig.add_trace(go.Bar(
                    x=labels, y=[m - v for m, v in zip(maxes, values)],
                    name='Remaining', marker_color='lightgray',
                ))
                fig.update_layout(
                    barmode='stack', title='Health Score Components',
                    yaxis_title='Points', height=300,
                    showlegend=False, margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

        # --- Row 2: Altman Z-Score and Piotroski F-Score side by side ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Altman Z-Score**")
            if z_result and z_result.z_score is not None:
                zone_colors = {'safe': 'green', 'grey': 'orange', 'distress': 'red', 'partial': 'gray'}
                zone_color = zone_colors.get(z_result.zone, 'gray')
                st.metric("Z-Score", f"{z_result.z_score:.2f}")
                st.markdown(f"Zone: :{zone_color}[**{z_result.zone.title()}**]")

                # Z-Score gauge
                fig = self.viz.create_gauge_chart(
                    value=z_result.z_score,
                    title="Altman Z-Score",
                    min_val=0, max_val=5,
                    thresholds={'warning': 1.81, 'good': 2.99},
                )
                st.plotly_chart(fig, use_container_width=True)

                # Components table
                if z_result.components:
                    st.markdown("**Components:**")
                    comp_labels = {
                        'x1': 'Working Capital / Assets',
                        'x2': 'Retained Earnings / Assets',
                        'x3': 'EBIT / Assets',
                        'x4': 'Equity / Liabilities',
                        'x5': 'Sales / Assets',
                    }
                    for k, v in z_result.components.items():
                        if v is not None:
                            label = comp_labels.get(k, k)
                            st.caption(f"{k.upper()}: {label} = {v:.4f}")
            else:
                st.info("Insufficient data for Z-Score calculation. Requires total assets, "
                        "current assets/liabilities, retained earnings, EBIT, equity, "
                        "liabilities, and revenue.")

        with col_right:
            st.markdown("**Piotroski F-Score**")
            if f_result:
                st.metric("F-Score", f"{f_result.score}/{f_result.max_score}")

                if f_result.score >= 7:
                    strength = "Strong"
                    strength_color = "green"
                elif f_result.score >= 4:
                    strength = "Moderate"
                    strength_color = "orange"
                else:
                    strength = "Weak"
                    strength_color = "red"
                st.markdown(f"Strength: :{strength_color}[**{strength}**]")

                # Criteria checklist
                if f_result.criteria:
                    st.markdown("**Criteria:**")
                    for criterion, passed in f_result.criteria.items():
                        icon = "+" if passed else "-"
                        label = criterion.replace('_', ' ').title()
                        st.caption(f"{icon} {label}")
            else:
                st.info("Insufficient data for F-Score calculation.")

        st.divider()

        # --- Row 3: DuPont Decomposition ---
        st.markdown("**DuPont ROE Decomposition**")
        if dupont and dupont.roe is not None:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("ROE", f"{dupont.roe:.1%}")
            with col2:
                if dupont.net_margin is not None:
                    st.metric("Net Margin", f"{dupont.net_margin:.1%}")
            with col3:
                if dupont.asset_turnover is not None:
                    st.metric("Asset Turnover", f"{dupont.asset_turnover:.2f}x")
            with col4:
                if dupont.equity_multiplier is not None:
                    st.metric("Equity Multiplier", f"{dupont.equity_multiplier:.2f}x")

            if dupont.primary_driver:
                st.caption(f"Primary ROE driver: {dupont.primary_driver.replace('_', ' ').title()}")

            # 5-factor extensions
            if dupont.tax_burden is not None or dupont.interest_burden is not None:
                st.markdown("**5-Factor Extensions:**")
                ext_col1, ext_col2 = st.columns(2)
                with ext_col1:
                    if dupont.tax_burden is not None:
                        st.metric("Tax Burden (NI/EBT)", f"{dupont.tax_burden:.1%}")
                with ext_col2:
                    if dupont.interest_burden is not None:
                        st.metric("Interest Burden (EBT/EBIT)", f"{dupont.interest_burden:.1%}")

            if dupont.interpretation:
                st.info(dupont.interpretation)
        else:
            st.info("DuPont analysis requires net income, revenue, total assets, and total equity.")

        # --- Row 4: Industry Benchmark Comparison ---
        st.divider()
        self._render_industry_benchmarks(analysis)

    # Industry benchmarks (general cross-industry averages)
    INDUSTRY_BENCHMARKS = {
        'current_ratio': {'label': 'Current Ratio', 'benchmark': 1.5, 'good': 2.0, 'unit': 'x'},
        'quick_ratio': {'label': 'Quick Ratio', 'benchmark': 1.0, 'good': 1.5, 'unit': 'x'},
        'net_margin': {'label': 'Net Margin', 'benchmark': 0.08, 'good': 0.15, 'unit': '%'},
        'roe': {'label': 'Return on Equity', 'benchmark': 0.12, 'good': 0.20, 'unit': '%'},
        'roa': {'label': 'Return on Assets', 'benchmark': 0.06, 'good': 0.10, 'unit': '%'},
        'debt_to_equity': {'label': 'Debt to Equity', 'benchmark': 1.0, 'good': 0.5, 'unit': 'x', 'lower_is_better': True},
        'interest_coverage': {'label': 'Interest Coverage', 'benchmark': 3.0, 'good': 6.0, 'unit': 'x'},
        'asset_turnover': {'label': 'Asset Turnover', 'benchmark': 0.8, 'good': 1.2, 'unit': 'x'},
        'gross_margin': {'label': 'Gross Margin', 'benchmark': 0.35, 'good': 0.50, 'unit': '%'},
        'operating_margin': {'label': 'Operating Margin', 'benchmark': 0.10, 'good': 0.20, 'unit': '%'},
    }

    def _render_industry_benchmarks(self, analysis: Dict[str, Any]):
        """Render industry benchmark comparisons for key financial ratios."""
        st.markdown("**Industry Benchmark Comparison**")

        # Collect company ratios from analysis results
        company_ratios = {}
        for category_key in ('liquidity_ratios', 'profitability_ratios', 'leverage_ratios', 'efficiency_ratios'):
            ratios = analysis.get(category_key, {})
            for key, value in ratios.items():
                if value is not None:
                    company_ratios[key] = value

        if not company_ratios:
            st.info("No ratio data available for benchmark comparison.")
            return

        # Build comparison data
        rows = []
        for ratio_key, bench in self.INDUSTRY_BENCHMARKS.items():
            company_val = company_ratios.get(ratio_key)
            if company_val is None:
                continue

            lower_is_better = bench.get('lower_is_better', False)

            if lower_is_better:
                if company_val <= bench['good']:
                    status = "Above Average"
                elif company_val <= bench['benchmark']:
                    status = "Average"
                else:
                    status = "Below Average"
            else:
                if company_val >= bench['good']:
                    status = "Above Average"
                elif company_val >= bench['benchmark']:
                    status = "Average"
                else:
                    status = "Below Average"

            if bench['unit'] == '%':
                fmt_company = f"{company_val:.1%}"
                fmt_bench = f"{bench['benchmark']:.1%}"
            else:
                fmt_company = f"{company_val:.2f}x"
                fmt_bench = f"{bench['benchmark']:.2f}x"

            rows.append({
                'Metric': bench['label'],
                'Company': fmt_company,
                'Industry Avg': fmt_bench,
                'Status': status,
            })

        if not rows:
            st.info("No matching benchmarks for available ratios.")
            return

        bench_df = pd.DataFrame(rows)

        # Color the status column
        def color_status(val):
            if val == "Above Average":
                return "color: green"
            elif val == "Below Average":
                return "color: red"
            return "color: orange"

        styled = bench_df.style.map(color_status, subset=['Status'])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Plotly comparison chart
        available_benchmarks = [r for r in rows if r['Status'] != '']
        if available_benchmarks:
            import plotly.graph_objects as go

            labels = [r['Metric'] for r in available_benchmarks]
            # Re-extract raw values for charting
            company_vals = []
            bench_vals = []
            for r in available_benchmarks:
                ratio_key = next(
                    k for k, b in self.INDUSTRY_BENCHMARKS.items()
                    if b['label'] == r['Metric']
                )
                cv = company_ratios[ratio_key]
                bv = self.INDUSTRY_BENCHMARKS[ratio_key]['benchmark']
                if self.INDUSTRY_BENCHMARKS[ratio_key]['unit'] == '%':
                    cv *= 100
                    bv *= 100
                company_vals.append(round(cv, 2))
                bench_vals.append(round(bv, 2))

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels, y=company_vals,
                name='Company', marker_color='steelblue',
            ))
            fig.add_trace(go.Bar(
                x=labels, y=bench_vals,
                name='Industry Avg', marker_color='lightcoral',
            ))
            fig.update_layout(
                barmode='group', title='Company vs Industry Benchmarks',
                yaxis_title='Value', height=350,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_report_download(self, df: pd.DataFrame):
        """Render a downloadable financial analysis report."""
        st.divider()
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Download Financial Report")
        with col2:
            try:
                financial_data = self.analyzer._dataframe_to_financial_data(df)
                report = self.analyzer.generate_report(financial_data)

                # Build full report text
                report_lines = [
                    "=" * 60,
                    "FINANCIAL ANALYSIS REPORT",
                    f"Generated: {report.generated_at}",
                    "=" * 60,
                    "",
                    "EXECUTIVE SUMMARY",
                    "-" * 40,
                    report.executive_summary,
                    "",
                ]

                for section_key, section_title in [
                    ('ratio_analysis', 'RATIO ANALYSIS'),
                    ('scoring_models', 'SCORING MODELS'),
                    ('risk_assessment', 'RISK ASSESSMENT'),
                    ('recommendations', 'RECOMMENDATIONS'),
                    ('period_comparison', 'PERIOD COMPARISON'),
                ]:
                    if section_key in report.sections:
                        report_lines.extend([
                            section_title,
                            "-" * 40,
                            report.sections[section_key],
                            "",
                        ])

                report_lines.append("=" * 60)
                report_text = "\n".join(report_lines)

                st.download_button(
                    "Download Report",
                    report_text,
                    "financial_report.txt",
                    "text/plain",
                    type="primary",
                )
            except Exception as e:
                logger.debug(f"Could not generate report: {e}")
                st.caption("Report unavailable for this data.")

        # XLSX and PDF export buttons
        try:
            financial_data = self.analyzer._dataframe_to_financial_data(df)
            export_cache_key = f"analysis_{hash(str(financial_data))}"
            if export_cache_key not in st.session_state:
                st.session_state[export_cache_key] = self.analyzer.analyze(financial_data)
            analysis = st.session_state[export_cache_key]
            report = self.analyzer.generate_report(financial_data)

            xlsx_col, pdf_col = st.columns(2)
            with xlsx_col:
                try:
                    from export_xlsx import FinancialExcelExporter
                    exporter = FinancialExcelExporter()
                    xlsx_bytes = exporter.export_full_report(financial_data, analysis, report=report)
                    st.download_button(
                        "Export as Excel",
                        xlsx_bytes,
                        "financial_report.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_xlsx",
                    )
                except Exception as e:
                    logger.debug("XLSX export unavailable: %s", e)
            with pdf_col:
                try:
                    from export_pdf import FinancialPDFExporter
                    exporter = FinancialPDFExporter()
                    pdf_bytes = exporter.export_full_report(financial_data, analysis, report=report)
                    st.download_button(
                        "Export as PDF",
                        pdf_bytes,
                        "financial_report.pdf",
                        "application/pdf",
                        key="dl_pdf",
                    )
                except Exception as e:
                    logger.debug("PDF export unavailable: %s", e)
        except Exception as e:
            logger.debug("Export buttons unavailable: %s", e)

    # ===== PHASE 5: WHAT-IF ANALYSIS =====

    def _render_what_if_analysis(self, df: pd.DataFrame):
        """Render what-if scenario analysis with interactive sliders and sensitivity tables."""
        import plotly.graph_objects as go

        st.subheader("What-If Scenario Analysis")
        st.caption("Adjust financial inputs to see how key metrics respond.")

        financial_data = self.analyzer._dataframe_to_financial_data(df)

        # Check if we have enough data
        if financial_data.revenue is None and financial_data.total_assets is None:
            st.info("What-If analysis requires financial data with revenue, assets, or other key metrics. "
                    "Upload a balance sheet or income statement for full functionality.")
            return

        # --- Scenario Sliders ---
        st.markdown("**Adjust Scenario Inputs**")

        slider_fields = []
        if financial_data.revenue is not None:
            slider_fields.append(('revenue', 'Revenue'))
        if financial_data.cogs is not None:
            slider_fields.append(('cogs', 'Cost of Goods Sold'))
        if financial_data.total_assets is not None:
            slider_fields.append(('total_assets', 'Total Assets'))
        if financial_data.total_liabilities is not None:
            slider_fields.append(('total_liabilities', 'Total Liabilities'))
        if financial_data.current_assets is not None:
            slider_fields.append(('current_assets', 'Current Assets'))
        if financial_data.current_liabilities is not None:
            slider_fields.append(('current_liabilities', 'Current Liabilities'))
        if financial_data.net_income is not None:
            slider_fields.append(('net_income', 'Net Income'))
        if financial_data.total_equity is not None:
            slider_fields.append(('total_equity', 'Total Equity'))

        if not slider_fields:
            st.info("No adjustable financial fields detected in the data.")
            return

        # Render sliders in columns
        adjustments: dict = {}
        cols = st.columns(min(len(slider_fields), 4))
        for i, (field_name, label) in enumerate(slider_fields):
            with cols[i % len(cols)]:
                pct = st.slider(
                    f"{label} Change %",
                    min_value=-50,
                    max_value=50,
                    value=0,
                    step=5,
                    key=f"whatif_{field_name}",
                )
                if pct != 0:
                    adjustments[field_name] = 1 + pct / 100.0

        # Run scenario if any adjustments made
        if adjustments:
            adj_labels = [f"{k.replace('_', ' ').title()} {(v-1)*100:+.0f}%"
                          for k, v in adjustments.items()]
            scenario_name = " + ".join(adj_labels)

            result = self.analyzer.scenario_analysis(financial_data, adjustments, scenario_name)

            # Impact summary
            st.markdown("---")
            st.markdown(f"**Scenario:** {scenario_name}")

            # Metrics comparison
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                delta = result.scenario_health.score - result.base_health.score
                st.metric("Health Score",
                          f"{result.scenario_health.score}/100",
                          f"{delta:+d} pts")
            with col2:
                st.metric("Grade",
                          result.scenario_health.grade,
                          f"was {result.base_health.grade}" if result.base_health.grade != result.scenario_health.grade else "unchanged")
            with col3:
                if result.base_z_score and result.scenario_z_score:
                    z_delta = result.scenario_z_score - result.base_z_score
                    st.metric("Z-Score",
                              f"{result.scenario_z_score:.2f}",
                              f"{z_delta:+.2f}")
            with col4:
                if result.base_f_score is not None and result.scenario_f_score is not None:
                    f_delta = result.scenario_f_score - result.base_f_score
                    st.metric("F-Score",
                              f"{result.scenario_f_score}/9",
                              f"{f_delta:+d}")

            # Ratio comparison table
            st.markdown("**Ratio Impact**")
            comparison_rows = []
            for key in result.base_ratios:
                base_val = result.base_ratios.get(key)
                scen_val = result.scenario_ratios.get(key)
                if base_val is not None and scen_val is not None:
                    delta = scen_val - base_val
                    is_pct = any(x in key for x in ('margin', 'roe', 'roa', 'roic'))
                    if is_pct:
                        comparison_rows.append({
                            'Metric': key.replace('_', ' ').title(),
                            'Base': f"{base_val:.1%}",
                            'Scenario': f"{scen_val:.1%}",
                            'Change': f"{delta:+.1%}",
                        })
                    else:
                        comparison_rows.append({
                            'Metric': key.replace('_', ' ').title(),
                            'Base': f"{base_val:.2f}",
                            'Scenario': f"{scen_val:.2f}",
                            'Change': f"{delta:+.2f}",
                        })

            if comparison_rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

            st.info(result.impact_summary)

        else:
            st.caption("Move the sliders above to create a what-if scenario.")

        # --- Sensitivity Analysis ---
        st.divider()
        st.subheader("Sensitivity Analysis")
        st.caption("See how varying a single input affects key metrics across a range.")

        sens_fields = [(f, l) for f, l in slider_fields]
        if sens_fields:
            sens_col1, sens_col2 = st.columns(2)
            with sens_col1:
                selected_var_idx = st.selectbox(
                    "Variable to Analyze",
                    range(len(sens_fields)),
                    format_func=lambda i: sens_fields[i][1],
                    key="sensitivity_var",
                )
            with sens_col2:
                sens_range_max = st.slider("Range (+/-%)", 5, 50, 20, step=5, key="sens_range")

            selected_field = sens_fields[selected_var_idx][0]
            selected_label = sens_fields[selected_var_idx][1]
            pct_steps = list(range(-sens_range_max, sens_range_max + 1, 5))

            sens_result = self.analyzer.sensitivity_analysis(
                financial_data, selected_field, pct_steps
            )

            # Render sensitivity table
            table_data = {'Change': sens_result.variable_labels}
            for metric_name, values in sens_result.metric_results.items():
                if any(v is not None for v in values):
                    display_name = metric_name.replace('_', ' ').title()
                    formatted = []
                    for v in values:
                        if v is None:
                            formatted.append("N/A")
                        elif metric_name in ('net_margin', 'roe'):
                            formatted.append(f"{v:.1%}")
                        elif metric_name == 'f_score':
                            formatted.append(f"{int(v)}/9")
                        else:
                            formatted.append(f"{v:.2f}")
                    table_data[display_name] = formatted

            import pandas as pd
            sens_df = pd.DataFrame(table_data)
            st.dataframe(sens_df, use_container_width=True, hide_index=True)

            # Sensitivity chart - Health Score and Z-Score
            health_vals = sens_result.metric_results.get('health_score', [])
            z_vals = sens_result.metric_results.get('z_score', [])

            if any(v is not None for v in health_vals):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sens_result.variable_labels,
                    y=health_vals,
                    name='Health Score',
                    mode='lines+markers',
                    line=dict(color='steelblue', width=2),
                ))
                if any(v is not None for v in z_vals):
                    fig.add_trace(go.Scatter(
                        x=sens_result.variable_labels,
                        y=[v * 20 if v else None for v in z_vals],  # Scale Z-Score for dual axis
                        name='Z-Score (x20)',
                        mode='lines+markers',
                        line=dict(color='orange', width=2, dash='dash'),
                    ))
                fig.update_layout(
                    title=f'Sensitivity: {selected_label} Impact on Key Metrics',
                    xaxis_title=f'{selected_label} Change',
                    yaxis_title='Score',
                    height=350,
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ===== PHASE 5: PERIOD COMPARISON UI =====

    def _render_period_comparison(self, df: pd.DataFrame, workbook):
        """Render multi-period comparison using existing compare_periods() method."""
        import plotly.graph_objects as go

        st.subheader("Period Comparison")
        st.caption("Compare financial metrics across two time periods from different sheets.")

        if not workbook.sheets or len(workbook.sheets) < 2:
            st.info("Period comparison requires at least 2 sheets in your workbook "
                    "(e.g., 'Q1' and 'Q2', or '2024' and '2025'). "
                    "Each sheet should contain comparable financial data.")
            return

        sheet_names = [s.name for s in workbook.sheets]

        col1, col2 = st.columns(2)
        with col1:
            current_sheet = st.selectbox("Current Period", sheet_names, index=len(sheet_names) - 1,
                                         key="period_current")
        with col2:
            prior_idx = max(0, len(sheet_names) - 2)
            prior_sheet = st.selectbox("Prior Period", sheet_names, index=prior_idx,
                                       key="period_prior")

        if current_sheet == prior_sheet:
            st.warning("Please select two different sheets/periods to compare.")
            return

        # Get data for each period
        current_df = None
        prior_df = None
        for sheet in workbook.sheets:
            if sheet.name == current_sheet:
                current_df = sheet.df
            if sheet.name == prior_sheet:
                prior_df = sheet.df

        if current_df is None or prior_df is None or current_df.empty or prior_df.empty:
            st.warning("Could not load data for the selected periods.")
            return

        current_data = self.analyzer._dataframe_to_financial_data(current_df)
        prior_data = self.analyzer._dataframe_to_financial_data(prior_df)

        comparison = self.analyzer.compare_periods(current_data, prior_data)

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Improvements", len(comparison.improvements))
        with m2:
            st.metric("Deteriorations", len(comparison.deteriorations))
        with m3:
            st.metric("Metrics Compared", len(comparison.deltas))

        # Improvements & Deteriorations
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Improvements**")
            if comparison.improvements:
                for metric in comparison.improvements:
                    delta = comparison.deltas[metric]
                    label = metric.replace('_', ' ').title()
                    st.caption(f"+ {label}: {delta:+.4f}")
            else:
                st.caption("No improvements detected.")

        with col_right:
            st.markdown("**Deteriorations**")
            if comparison.deteriorations:
                for metric in comparison.deteriorations:
                    delta = comparison.deltas[metric]
                    label = metric.replace('_', ' ').title()
                    st.caption(f"- {label}: {delta:+.4f}")
            else:
                st.caption("No deteriorations detected.")

        # Full comparison table
        st.divider()
        st.markdown("**Full Ratio Comparison**")

        import pandas as pd
        rows = []
        for key in sorted(comparison.current_ratios.keys()):
            cv = comparison.current_ratios.get(key)
            pv = comparison.prior_ratios.get(key)
            delta = comparison.deltas.get(key)

            if cv is not None or pv is not None:
                is_pct = any(x in key for x in ('margin', 'roe', 'roa', 'roic'))
                fmt = lambda v: f"{v:.1%}" if is_pct and v is not None else (f"{v:.2f}" if v is not None else "N/A")

                status = ""
                if delta is not None:
                    lower_better = key in ('leverage_debt_to_equity', 'leverage_debt_to_assets')
                    if lower_better:
                        status = "Improved" if delta < -1e-6 else ("Declined" if delta > 1e-6 else "Unchanged")
                    else:
                        status = "Improved" if delta > 1e-6 else ("Declined" if delta < -1e-6 else "Unchanged")

                rows.append({
                    'Metric': key.replace('_', ' ').title(),
                    f'{current_sheet}': fmt(cv),
                    f'{prior_sheet}': fmt(pv),
                    'Delta': f"{delta:+.4f}" if delta is not None else "N/A",
                    'Status': status,
                })

        if rows:
            comp_df = pd.DataFrame(rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Delta bar chart
        if comparison.deltas:
            sorted_deltas = sorted(comparison.deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
            if sorted_deltas:
                labels = [k.replace('_', ' ').title() for k, _ in sorted_deltas]
                values = [v for _, v in sorted_deltas]
                colors = ['green' if v > 0 else 'red' for v in values]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=labels, y=values,
                    marker_color=colors,
                    text=[f"{v:+.4f}" for v in values],
                    textposition='auto',
                ))
                fig.update_layout(
                    title=f'Top Changes: {current_sheet} vs {prior_sheet}',
                    yaxis_title='Delta',
                    height=350,
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

    def _render_monte_carlo(self, df: pd.DataFrame):
        """Render Monte Carlo simulation tab."""
        st.subheader("Monte Carlo Simulation")
        st.markdown("Model uncertainty in financial outcomes using randomized scenario generation.")

        data = self.analyzer._dataframe_to_financial_data(df)

        # Let user configure assumptions
        available_fields = []
        for fld in ['revenue', 'cogs', 'operating_expenses', 'total_assets',
                     'current_assets', 'current_liabilities', 'total_debt']:
            if getattr(data, fld, None) is not None:
                available_fields.append(fld)

        if not available_fields:
            st.info("Insufficient financial data for Monte Carlo simulation.")
            return

        st.markdown("**Configure Variable Uncertainty**")
        assumptions = {}
        cols = st.columns(min(len(available_fields), 3))
        for i, fld in enumerate(available_fields):
            with cols[i % len(cols)]:
                std = st.slider(
                    f"{fld.replace('_', ' ').title()} Std Dev %",
                    min_value=1, max_value=50, value=10,
                    key=f"mc_std_{fld}",
                )
                assumptions[fld] = {'mean_pct': 0.0, 'std_pct': float(std)}

        n_sims = st.select_slider(
            "Number of Simulations",
            options=[100, 250, 500, 1000, 2500],
            value=500,
        )

        result = self.analyzer.monte_carlo_simulation(data, assumptions, n_simulations=n_sims)

        if result.n_simulations == 0:
            st.warning(result.summary)
            return

        st.success(result.summary)

        # Percentile table
        st.markdown("**Distribution Percentiles**")
        rows = []
        for metric, pcts in result.percentiles.items():
            rows.append({
                'Metric': metric.replace('_', ' ').title(),
                'P10': f"{pcts['p10']:.2f}",
                'P25': f"{pcts['p25']:.2f}",
                'Median': f"{pcts['p50']:.2f}",
                'P75': f"{pcts['p75']:.2f}",
                'P90': f"{pcts['p90']:.2f}",
                'Mean': f"{pcts['mean']:.2f}",
                'Std Dev': f"{pcts['std']:.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Histogram for selected metric
        metric_choice = st.selectbox(
            "View Distribution",
            list(result.metric_distributions.keys()),
            format_func=lambda x: x.replace('_', ' ').title(),
        )

        if metric_choice and metric_choice in result.metric_distributions:
            import plotly.graph_objects as go
            values = result.metric_distributions[metric_choice]
            fig = go.Figure(data=[go.Histogram(x=values, nbinsx=40,
                                               marker_color='steelblue',
                                               opacity=0.75)])
            pcts = result.percentiles[metric_choice]
            fig.add_vline(x=pcts['p50'], line_dash="dash", line_color="red",
                          annotation_text="Median")
            fig.add_vline(x=pcts['p10'], line_dash="dot", line_color="orange",
                          annotation_text="P10")
            fig.add_vline(x=pcts['p90'], line_dash="dot", line_color="orange",
                          annotation_text="P90")
            fig.update_layout(
                title=f'{metric_choice.replace("_", " ").title()} Distribution ({n_sims} sims)',
                xaxis_title=metric_choice.replace('_', ' ').title(),
                yaxis_title='Frequency',
                height=350,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_cashflow_forecast(self, df: pd.DataFrame):
        """Render cash flow forecasting and DCF valuation tab."""
        st.subheader("Cash Flow Forecast & DCF Valuation")

        data = self.analyzer._dataframe_to_financial_data(df)

        if not data.revenue or data.revenue <= 0:
            st.info("Revenue data required for cash flow forecasting.")
            return

        # User controls
        col1, col2, col3 = st.columns(3)
        with col1:
            n_periods = st.slider("Forecast Periods", 3, 24, 12, key="cf_periods")
            rev_growth = st.slider("Revenue Growth %", -10.0, 30.0, 5.0, 0.5,
                                   key="cf_rev_growth") / 100.0
        with col2:
            capex_pct = st.slider("CapEx % of Revenue", 1.0, 20.0, 5.0, 0.5,
                                  key="cf_capex") / 100.0
            discount = st.slider("Discount Rate (WACC) %", 5.0, 20.0, 10.0, 0.5,
                                 key="cf_discount") / 100.0
        with col3:
            terminal_g = st.slider("Terminal Growth %", 0.0, 5.0, 2.0, 0.25,
                                   key="cf_terminal") / 100.0

        result = self.analyzer.forecast_cashflow(
            data, periods=n_periods, revenue_growth=rev_growth,
            capex_ratio=capex_pct, discount_rate=discount,
            terminal_growth=terminal_g,
        )

        if not result.periods:
            st.warning("Could not generate forecast with current data.")
            return

        # DCF headline
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("DCF Enterprise Value",
                       f"${result.dcf_value:,.0f}" if result.dcf_value else "N/A")
        with col_b:
            st.metric("Terminal Value",
                       f"${result.terminal_value:,.0f}" if result.terminal_value else "N/A")
        with col_c:
            total_fcf = sum(result.fcf_forecast)
            st.metric("Total Forecast FCF", f"${total_fcf:,.0f}")

        # Forecast table
        st.markdown("**Period-by-Period Projections**")
        forecast_df = pd.DataFrame({
            'Period': result.periods,
            'Revenue': [f"${v:,.0f}" for v in result.revenue_forecast],
            'Expenses': [f"${v:,.0f}" for v in result.expense_forecast],
            'Net Cash Flow': [f"${v:,.0f}" for v in result.net_cash_flow],
            'FCF': [f"${v:,.0f}" for v in result.fcf_forecast],
            'Cumulative Cash': [f"${v:,.0f}" for v in result.cumulative_cash],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        # Chart
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=result.periods, y=result.revenue_forecast,
            name='Revenue', marker_color='steelblue',
        ))
        fig.add_trace(go.Bar(
            x=result.periods, y=result.expense_forecast,
            name='Expenses', marker_color='salmon',
        ))
        fig.add_trace(go.Scatter(
            x=result.periods, y=result.fcf_forecast,
            name='Free Cash Flow', mode='lines+markers',
            line=dict(color='green', width=2),
        ))
        fig.add_trace(go.Scatter(
            x=result.periods, y=result.cumulative_cash,
            name='Cumulative Cash', mode='lines+markers',
            line=dict(color='purple', width=2, dash='dash'),
        ))
        fig.update_layout(
            title='Cash Flow Forecast',
            yaxis_title='Amount ($)',
            barmode='group',
            height=400,
            margin=dict(t=40, b=20),
            legend=dict(orientation='h', y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_driver_analysis(self, df: pd.DataFrame):
        """Render tornado / driver analysis and breakeven tab."""
        st.subheader("Driver Impact Analysis")
        st.markdown("Identify which financial variables have the greatest impact on key metrics.")

        data = self.analyzer._dataframe_to_financial_data(df)

        # Detect available variables
        candidate_fields = [
            'revenue', 'cogs', 'operating_expenses', 'net_income',
            'total_assets', 'total_liabilities', 'total_equity',
            'current_assets', 'current_liabilities', 'total_debt',
            'ebit', 'ebitda', 'interest_expense',
        ]
        available = [f for f in candidate_fields if getattr(data, f, None) is not None]

        if len(available) < 2:
            st.info("Need at least 2 financial variables for driver analysis.")
            return

        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox(
                "Target Metric",
                ['health_score', 'z_score', 'f_score', 'net_margin',
                 'current_ratio', 'roe', 'debt_to_equity'],
                key="tornado_target",
            )
        with col2:
            swing = st.slider(
                "Swing (%)", min_value=5, max_value=30, value=10,
                key="tornado_swing",
            )

        result = self.analyzer.tornado_analysis(data, target_metric=target, pct_swing=float(swing))

        if not result.drivers:
            st.warning("No drivers could be computed.")
            return

        # Show top driver highlight
        st.success(f"**Top Driver:** {result.top_driver.replace('_', ' ').title()} "
                   f"(spread: {result.drivers[0].spread:.4f})")

        # Tornado chart (horizontal bar)
        import plotly.graph_objects as go
        top_n = min(10, len(result.drivers))
        drivers = result.drivers[:top_n]
        # Reverse for plotly horizontal bars (top = most impactful)
        drivers_rev = list(reversed(drivers))
        var_names = [d.variable.replace('_', ' ').title() for d in drivers_rev]
        low_deltas = [d.low_value - d.base_value for d in drivers_rev]
        high_deltas = [d.high_value - d.base_value for d in drivers_rev]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=var_names, x=low_deltas, name=f'-{swing}%',
            orientation='h', marker_color='salmon',
        ))
        fig.add_trace(go.Bar(
            y=var_names, x=high_deltas, name=f'+{swing}%',
            orientation='h', marker_color='steelblue',
        ))
        fig.update_layout(
            title=f'Tornado Chart: Impact on {target.replace("_", " ").title()}',
            xaxis_title=f'Change in {target.replace("_", " ").title()}',
            barmode='overlay',
            height=max(300, top_n * 35),
            margin=dict(t=40, b=20, l=150),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Driver table
        st.markdown("**Driver Details**")
        driver_data = []
        for d in result.drivers:
            driver_data.append({
                'Variable': d.variable.replace('_', ' ').title(),
                f'-{swing}% Value': f"{d.low_value:.4f}",
                'Base Value': f"{d.base_value:.4f}",
                f'+{swing}% Value': f"{d.high_value:.4f}",
                'Spread': f"{d.spread:.4f}",
            })
        st.dataframe(pd.DataFrame(driver_data), use_container_width=True, hide_index=True)

        # ===== BREAKEVEN ANALYSIS =====
        st.markdown("---")
        st.subheader("Breakeven Analysis")

        be = self.analyzer.breakeven_analysis(data)

        if be.breakeven_revenue is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Breakeven Revenue", f"${be.breakeven_revenue:,.0f}")
            with col2:
                st.metric("Current Revenue", f"${be.current_revenue:,.0f}")
            with col3:
                mos_pct = (be.margin_of_safety or 0) * 100
                st.metric("Margin of Safety", f"{mos_pct:.1f}%",
                          delta=f"{'Above' if mos_pct > 0 else 'Below'} breakeven")

            st.markdown(f"- **Fixed Costs:** ${be.fixed_costs:,.0f}")
            st.markdown(f"- **Variable Cost Ratio:** {(be.variable_cost_ratio or 0)*100:.1f}%")
            st.markdown(f"- **Contribution Margin:** {(be.contribution_margin_ratio or 0)*100:.1f}%")
        else:
            st.info("Insufficient data for breakeven analysis (need revenue, COGS, and operating expenses).")

    def _render_covenant_monitor(self, df: pd.DataFrame):
        """Render covenant / KPI monitoring tab."""
        st.subheader("Covenant & KPI Monitor")
        st.markdown("Track financial covenants and KPI thresholds with traffic-light alerts.")

        data = self.analyzer._dataframe_to_financial_data(df)

        # Let user configure custom covenants or use defaults
        use_custom = st.checkbox("Customize Covenants", value=False, key="cov_custom")

        covenants = None
        if use_custom:
            st.markdown("**Define Covenants**")
            n_covenants = st.number_input("Number of covenants", min_value=1, max_value=10,
                                          value=3, key="cov_n")
            covenants = []
            metric_options = ['current_ratio', 'debt_to_equity', 'net_margin',
                              'roe', 'roa', 'interest_coverage', 'health_score', 'dscr']
            for i in range(int(n_covenants)):
                cols = st.columns(4)
                with cols[0]:
                    name = st.text_input(f"Name #{i+1}", value=f"Covenant {i+1}",
                                         key=f"cov_name_{i}")
                with cols[1]:
                    metric = st.selectbox(f"Metric #{i+1}", metric_options,
                                          key=f"cov_metric_{i}")
                with cols[2]:
                    threshold = st.number_input(f"Threshold #{i+1}", value=1.5,
                                                step=0.1, key=f"cov_thresh_{i}")
                with cols[3]:
                    direction = st.selectbox(f"Direction #{i+1}", ['above', 'below'],
                                             key=f"cov_dir_{i}")
                covenants.append({
                    'name': name, 'metric': metric,
                    'threshold': threshold, 'direction': direction,
                })

        result = self.analyzer.covenant_monitor(data, covenants)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Covenants", len(result.checks))
        with col2:
            st.metric("Passing", result.passes)
        with col3:
            st.metric("Warnings", result.warnings)
        with col4:
            st.metric("Breaches", result.breaches)

        if result.breaches > 0:
            st.error(result.summary)
        elif result.warnings > 0:
            st.warning(result.summary)
        else:
            st.success(result.summary)

        # Covenant detail table with color coding
        if result.checks:
            check_data = []
            for c in result.checks:
                status_icon = {'pass': 'PASS', 'warning': 'WARN',
                               'breach': 'BREACH', 'unknown': 'N/A'}.get(c.status, '?')
                check_data.append({
                    'Covenant': c.name,
                    'Current': f"{c.current_value:.4f}" if c.current_value is not None else "N/A",
                    'Threshold': f"{c.threshold:.4f}",
                    'Direction': c.direction.title(),
                    'Headroom': f"{c.headroom:+.4f}" if c.headroom is not None else "N/A",
                    'Status': status_icon,
                })

            cov_df = pd.DataFrame(check_data)

            def color_status(val):
                if val == 'PASS':
                    return 'background-color: #c6efce; color: #006100'
                elif val == 'WARN':
                    return 'background-color: #ffeb9c; color: #9c6500'
                elif val == 'BREACH':
                    return 'background-color: #ffc7ce; color: #9c0006'
                return ''

            styled = cov_df.style.map(color_status, subset=['Status'])
            st.dataframe(styled, use_container_width=True, hide_index=True)

    def _render_working_capital(self, df: pd.DataFrame):
        """Render working capital analysis tab."""
        st.subheader("Working Capital Analysis")
        st.markdown("Analyze cash conversion efficiency and working capital optimization opportunities.")

        data = self.analyzer._dataframe_to_financial_data(df)
        result = self.analyzer.working_capital_analysis(data)

        # Days metrics in a row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            val = f"{result.dso:.0f} days" if result.dso is not None else "N/A"
            st.metric("DSO (Days Sales Outstanding)", val)
        with col2:
            val = f"{result.dio:.0f} days" if result.dio is not None else "N/A"
            st.metric("DIO (Days Inventory Outstanding)", val)
        with col3:
            val = f"{result.dpo:.0f} days" if result.dpo is not None else "N/A"
            st.metric("DPO (Days Payables Outstanding)", val)
        with col4:
            if result.ccc is not None:
                color = "normal" if result.ccc > 0 else "inverse"
                st.metric("Cash Conversion Cycle", f"{result.ccc:.0f} days",
                          delta=f"{'Efficient' if result.ccc < 30 else 'Monitor'}")
            else:
                st.metric("Cash Conversion Cycle", "N/A")

        # NWC and ratio
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if result.net_working_capital is not None:
                st.metric("Net Working Capital", f"${result.net_working_capital:,.0f}")
        with col2:
            if result.working_capital_ratio is not None:
                st.metric("Working Capital Ratio", f"{result.working_capital_ratio:.2f}x")

        # CCC waterfall chart
        if result.dso is not None and result.dio is not None and result.dpo is not None:
            import plotly.graph_objects as go
            fig = go.Figure(go.Waterfall(
                name="Cash Conversion Cycle",
                orientation="v",
                measure=["relative", "relative", "relative", "total"],
                x=["DSO", "DIO", "DPO", "CCC"],
                y=[result.dso, result.dio, -result.dpo, 0],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "salmon"}},
                decreasing={"marker": {"color": "steelblue"}},
                totals={"marker": {"color": "gold"}},
            ))
            fig.update_layout(
                title="Cash Conversion Cycle Breakdown",
                yaxis_title="Days",
                height=350,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Insights
        if result.insights:
            st.markdown("**Insights & Recommendations**")
            for insight in result.insights:
                st.markdown(f"- {insight}")

    def _render_narrative(self, df: pd.DataFrame):
        """Render AI narrative intelligence tab."""
        st.subheader("AI Narrative Intelligence")
        st.markdown("Automated SWOT-style financial narrative generated from all analysis modules.")

        data = self.analyzer._dataframe_to_financial_data(df)
        report = self.analyzer.generate_narrative(data)

        # Headline
        st.markdown(f"### {report.headline}")

        # SWOT grid
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Strengths**")
            if report.strengths:
                for s in report.strengths:
                    st.markdown(f"- {s}")
            else:
                st.markdown("_No notable strengths identified._")

            st.markdown("**Opportunities**")
            if report.opportunities:
                for o in report.opportunities:
                    st.markdown(f"- {o}")
            else:
                st.markdown("_No immediate opportunities identified._")

        with col2:
            st.markdown("**Weaknesses**")
            if report.weaknesses:
                for w in report.weaknesses:
                    st.markdown(f"- {w}")
            else:
                st.markdown("_No notable weaknesses identified._")

            st.markdown("**Risks**")
            if report.risks:
                for r in report.risks:
                    st.markdown(f"- {r}")
            else:
                st.markdown("_No significant risks identified._")

        # Recommendation
        st.markdown("---")
        st.markdown(f"**Recommendation:** {report.recommendation}")

        # Summary stats
        st.markdown("---")
        st.markdown("**Analysis Summary**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Strengths", len(report.strengths))
        with col2:
            st.metric("Weaknesses", len(report.weaknesses))
        with col3:
            st.metric("Opportunities", len(report.opportunities))
        with col4:
            st.metric("Risks", len(report.risks))

    def _render_trend_forecast(self, df: pd.DataFrame):
        """Render regression-based trend forecasting tab."""
        st.header("Trend Forecast")
        st.markdown("Regression-based extrapolation with confidence bands.")

        analyzer = CharlieAnalyzer()
        fd = analyzer._dataframe_to_financial_data(df)

        # Let user pick metric and method
        metric_options = {
            "Revenue": fd.revenue,
            "Net Income": fd.net_income,
            "EBITDA": fd.ebitda,
            "Total Assets": fd.total_assets,
            "Operating Cash Flow": fd.operating_cash_flow,
        }

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_metric = st.selectbox("Metric", list(metric_options.keys()), key="tf_metric")
        with col2:
            method = st.selectbox("Method", ["linear", "exponential", "polynomial"], key="tf_method")
        with col3:
            periods = st.slider("Periods Ahead", 1, 12, 4, key="tf_periods")

        # Build sample historical from the single data point (simulate trend)
        base_val = metric_options.get(selected_metric)
        if base_val and base_val > 0:
            # Create synthetic history: use base_val as latest, simulate 6 historical points
            rng = np.random.default_rng(42)
            noise = rng.normal(0, 0.03, 5)
            growth = np.linspace(0.85, 1.0, 5)
            historical = [base_val * g * (1 + n) for g, n in zip(growth, noise)]
            historical.append(base_val)

            result = analyzer.regression_forecast(
                values=historical, periods_ahead=periods,
                method=method, metric_name=selected_metric,
            )

            if result.forecast_values:
                import plotly.graph_objects as go
                fig = go.Figure()
                x_hist = list(range(len(historical)))
                x_fore = list(range(len(historical), len(historical) + periods))

                fig.add_trace(go.Scatter(x=x_hist, y=historical, mode='lines+markers',
                                         name='Historical', line=dict(color='#2196F3')))
                fig.add_trace(go.Scatter(x=x_fore, y=result.forecast_values,
                                         mode='lines+markers', name='Forecast',
                                         line=dict(color='#FF9800', dash='dash')))
                if result.confidence_upper:
                    fig.add_trace(go.Scatter(x=x_fore, y=result.confidence_upper,
                                             mode='lines', name='Upper Band',
                                             line=dict(color='#FF9800', width=0.5)))
                    fig.add_trace(go.Scatter(x=x_fore, y=result.confidence_lower,
                                             mode='lines', name='Lower Band',
                                             fill='tonexty', line=dict(color='#FF9800', width=0.5)))
                fig.update_layout(title=f"{selected_metric} Forecast ({method})",
                                  xaxis_title="Period", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)

                # Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R-squared", f"{result.r_squared:.4f}" if result.r_squared else "N/A")
                with col2:
                    st.metric("Slope", f"{result.slope:.2f}" if result.slope else "N/A")
                with col3:
                    st.metric("Next Period", f"{result.forecast_values[0]:,.0f}")
            else:
                st.info("Not enough data for regression forecast.")
        else:
            st.info(f"No positive value available for {selected_metric}.")

    def _render_industry_benchmark(self, df: pd.DataFrame):
        """Render industry benchmarking tab."""
        st.header("Industry Benchmark")
        st.markdown("Compare company metrics against industry percentile benchmarks.")

        analyzer = CharlieAnalyzer()
        fd = analyzer._dataframe_to_financial_data(df)

        industry = st.selectbox("Industry", ["general", "technology", "manufacturing", "retail", "healthcare"],
                                key="ib_industry")

        result = analyzer.industry_benchmark(fd, industry=industry)

        if result.comparisons:
            st.markdown(f"**{result.summary}**")

            if result.overall_percentile is not None:
                st.progress(min(result.overall_percentile / 100, 1.0),
                            text=f"Overall Percentile: {result.overall_percentile:.0f}th")

            # Build comparison table
            rows = []
            for c in result.comparisons:
                rows.append({
                    "Metric": c.metric_name.replace("_", " ").title(),
                    "Company": f"{c.company_value:.2f}" if c.company_value is not None else "N/A",
                    "Industry Median": f"{c.industry_median:.2f}" if c.industry_median is not None else "N/A",
                    "P25": f"{c.industry_p25:.2f}" if c.industry_p25 is not None else "N/A",
                    "P75": f"{c.industry_p75:.2f}" if c.industry_p75 is not None else "N/A",
                    "Percentile": f"{c.percentile_rank:.0f}th" if c.percentile_rank is not None else "N/A",
                    "Rating": c.rating,
                })
            bench_df = pd.DataFrame(rows)

            def color_rating(val):
                if val == "above average":
                    return "background-color: #C6EFCE; color: #006100"
                elif val == "below average":
                    return "background-color: #FFC7CE; color: #9C0006"
                return ""

            styled = bench_df.style.map(color_rating, subset=["Rating"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Bar chart of percentile ranks
            import plotly.express as px
            chart_data = pd.DataFrame({
                "Metric": [c.metric_name.replace("_", " ").title() for c in result.comparisons],
                "Percentile": [c.percentile_rank or 0 for c in result.comparisons],
            })
            fig = px.bar(chart_data, x="Percentile", y="Metric", orientation='h',
                         color="Percentile", color_continuous_scale=["#FFC7CE", "#FFEB9C", "#C6EFCE"],
                         range_color=[0, 100])
            fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="Median")
            fig.update_layout(title=f"Percentile Ranking vs {industry.title()} Peers",
                              xaxis_title="Percentile", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for industry benchmarking.")

    def _render_custom_kpis(self, df: pd.DataFrame):
        """Render custom KPI builder tab."""
        st.header("Custom KPI Builder")
        st.markdown("Define your own financial metrics using field names and arithmetic operators.")

        analyzer = CharlieAnalyzer()
        fd = analyzer._dataframe_to_financial_data(df)

        st.markdown("**Available fields:** `revenue`, `cogs`, `gross_profit`, `ebit`, `ebitda`, "
                     "`net_income`, `total_assets`, `total_equity`, `total_liabilities`, "
                     "`current_assets`, `current_liabilities`, `total_debt`, `operating_cash_flow`, "
                     "`capex`, `interest_expense`, `depreciation`, `inventory`, etc.")

        # Default KPI examples
        default_kpis = [
            ("Asset Turnover", "revenue / total_assets", "", ""),
            ("Gross Margin %", "(revenue - cogs) / revenue * 100", "30", ""),
            ("EBITDA - CapEx", "ebitda - capex", "0", ""),
        ]

        num_kpis = st.number_input("Number of KPIs", 1, 10, 3, key="kpi_count")

        kpi_defs: list = []
        for i in range(int(num_kpis)):
            cols = st.columns([2, 3, 1, 1])
            defaults = default_kpis[i] if i < len(default_kpis) else ("", "", "", "")
            with cols[0]:
                name = st.text_input("Name", value=defaults[0], key=f"kpi_name_{i}")
            with cols[1]:
                formula = st.text_input("Formula", value=defaults[1], key=f"kpi_formula_{i}")
            with cols[2]:
                tmin = st.text_input("Min Target", value=defaults[2], key=f"kpi_min_{i}")
            with cols[3]:
                tmax = st.text_input("Max Target", value=defaults[3], key=f"kpi_max_{i}")

            if name and formula:
                kpi_defs.append(CustomKPIDefinition(
                    name=name,
                    formula=formula,
                    target_min=float(tmin) if tmin else None,
                    target_max=float(tmax) if tmax else None,
                ))

        if kpi_defs and st.button("Evaluate KPIs", key="eval_kpis"):
            report = analyzer.evaluate_custom_kpis(fd, kpi_defs)
            st.markdown(f"**{report.summary}**")

            rows = []
            for r in report.results:
                status = ""
                if r.error:
                    status = f"Error: {r.error}"
                elif r.meets_target is True:
                    status = "On Target"
                elif r.meets_target is False:
                    status = "Off Target"
                else:
                    status = "No Target"

                rows.append({
                    "KPI": r.name,
                    "Formula": r.formula,
                    "Value": f"{r.value:.4f}" if r.value is not None else "N/A",
                    "Status": status,
                })

            kpi_df = pd.DataFrame(rows)

            def color_status(val):
                if val == "On Target":
                    return "background-color: #C6EFCE; color: #006100"
                elif val == "Off Target":
                    return "background-color: #FFC7CE; color: #9C0006"
                elif "Error" in val:
                    return "background-color: #FFC7CE; color: #9C0006"
                return ""

            styled = kpi_df.style.map(color_status, subset=["Status"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

    def _render_peer_comparison(self, df: pd.DataFrame):
        """Render Peer Comparison tab."""
        st.subheader("Peer Comparison")
        st.markdown("Compare financial metrics across multiple entities.")

        financial_data = self.analyzer._dataframe_to_financial_data(df)

        # Build synthetic peers by adjusting the current company's data
        st.markdown("**Simulated peers based on your data with adjustments:**")

        num_peers = st.slider("Number of peers", 2, 5, 3, key="peer_count")

        peer_labels = ["Your Company"]
        adjustments = [1.0]
        for i in range(1, num_peers):
            peer_labels.append(f"Peer {i}")
            adj = st.slider(
                f"Peer {i} revenue multiplier",
                0.5, 2.0, 0.8 + i * 0.2, 0.1,
                key=f"peer_adj_{i}",
            )
            adjustments.append(adj)

        if st.button("Compare Peers", key="peer_compare_btn"):
            peers = []
            for idx, (label, adj) in enumerate(zip(peer_labels, adjustments)):
                if idx == 0:
                    peers.append(PeerCompanyData(name=label, data=financial_data))
                else:
                    adj_data = FinancialData(
                        revenue=(financial_data.revenue or 0) * adj,
                        cogs=(financial_data.cogs or 0) * adj * (1 + (adj - 1) * 0.3),
                        gross_profit=(financial_data.gross_profit or 0) * adj * (1 - (adj - 1) * 0.1),
                        operating_income=(financial_data.operating_income or 0) * adj * 0.9,
                        operating_expenses=(financial_data.operating_expenses or 0) * adj,
                        net_income=(financial_data.net_income or 0) * adj * 0.85,
                        ebit=(financial_data.ebit or 0) * adj * 0.9,
                        ebitda=(financial_data.ebitda or 0) * adj * 0.95,
                        total_assets=(financial_data.total_assets or 0) * adj * 1.1,
                        total_liabilities=(financial_data.total_liabilities or 0) * adj * 1.2,
                        total_equity=(financial_data.total_equity or 0) * adj * 0.9,
                        current_assets=(financial_data.current_assets or 0) * adj,
                        current_liabilities=(financial_data.current_liabilities or 0) * adj * 1.1,
                        total_debt=(financial_data.total_debt or 0) * adj * 1.15,
                        interest_expense=(financial_data.interest_expense or 0) * adj,
                        accounts_receivable=(financial_data.accounts_receivable or 0) * adj,
                        inventory=(financial_data.inventory or 0) * adj,
                    )
                    peers.append(PeerCompanyData(name=label, data=adj_data))

            report = self.analyzer.peer_comparison(peers)

            st.markdown(f"**{report.summary}**")

            # Rankings
            st.markdown("#### Rankings")
            rank_cols = st.columns(len(report.peer_names))
            for i, name in enumerate(report.peer_names):
                with rank_cols[i]:
                    rank = report.rankings.get(name, 0)
                    medal = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
                    st.metric(name, medal)

            # Comparison table
            rows = []
            for comp in report.comparisons:
                row = {"Metric": comp.metric_name}
                for name in report.peer_names:
                    v = comp.values.get(name)
                    row[name] = f"{v:.4f}" if v is not None else "N/A"
                row["Average"] = f"{comp.average:.4f}" if comp.average is not None else "N/A"
                row["Best"] = comp.best_performer
                rows.append(row)

            comp_df = pd.DataFrame(rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Radar chart
            try:
                import plotly.graph_objects as go
                categories = [c.metric_name for c in report.comparisons]
                fig = go.Figure()
                for name in report.peer_names:
                    vals = []
                    for comp in report.comparisons:
                        v = comp.values.get(name)
                        vals.append(v if v is not None else 0)
                    # Normalize to 0-1 range for radar
                    max_vals = [max(abs(comp.values.get(n) or 0) for n in report.peer_names) for comp in report.comparisons]
                    norm = [v / m if m and m > 0 else 0 for v, m in zip(vals, max_vals)]
                    fig.add_trace(go.Scatterpolar(r=norm + [norm[0]], theta=categories + [categories[0]], name=name))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1.2])), title="Peer Radar Chart")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

    def _render_ratio_decomposition(self, df: pd.DataFrame):
        """Render Ratio Decomposition tab."""
        st.subheader("ROE Decomposition Tree")
        st.markdown("Break down Return on Equity into its DuPont components and sub-drivers.")

        financial_data = self.analyzer._dataframe_to_financial_data(df)
        tree = self.analyzer.ratio_decomposition(financial_data)

        if tree.root is None:
            st.warning("Insufficient data for ratio decomposition.")
            return

        st.markdown(f"**{tree.summary}**")

        # Level 0: ROE
        root = tree.root
        st.markdown(f"### {root.name}: {root.value:.2%}" if root.value is not None else f"### {root.name}: N/A")

        # Level 1: DuPont factors
        if root.children:
            cols = st.columns(len(root.children))
            for i, child in enumerate(root.children):
                with cols[i]:
                    val_str = f"{child.value:.4f}" if child.value is not None else "N/A"
                    st.metric(child.name, val_str)
                    st.caption(child.formula)

                    # Level 2: Sub-drivers
                    if child.children:
                        for sub in child.children:
                            sub_val = f"{sub.value:.4f}" if sub.value is not None else "N/A"
                            st.markdown(f"- **{sub.name}**: {sub_val}")

        # Waterfall data table
        rows = []
        if root.children:
            for child in root.children:
                rows.append({
                    "Component": child.name,
                    "Value": f"{child.value:.4f}" if child.value is not None else "N/A",
                    "Formula": child.formula,
                    "Level": "DuPont Factor",
                })
                for sub in child.children:
                    rows.append({
                        "Component": f"  {sub.name}",
                        "Value": f"{sub.value:.4f}" if sub.value is not None else "N/A",
                        "Formula": sub.formula,
                        "Level": "Sub-Driver",
                    })

        if rows:
            tree_df = pd.DataFrame(rows)

            def color_level(val):
                if val == "DuPont Factor":
                    return "background-color: #DCE6F1"
                return ""

            styled = tree_df.style.map(color_level, subset=["Level"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # Bar chart of DuPont factors
        try:
            import plotly.graph_objects as go
            if root.children:
                names = [c.name for c in root.children]
                vals = [c.value if c.value is not None else 0 for c in root.children]
                fig = go.Figure(data=[go.Bar(x=names, y=vals, marker_color=["#4472C4", "#ED7D31", "#70AD47"])])
                fig.update_layout(title="DuPont Factor Values", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            logger.debug("Chart render skipped", exc_info=True)

    def _render_credit_rating(self, df: pd.DataFrame):
        """Render Credit Rating tab."""
        st.subheader("Financial Credit Rating")
        st.markdown("Composite letter-grade rating based on liquidity, profitability, leverage, efficiency, and cash flow.")

        financial_data = self.analyzer._dataframe_to_financial_data(df)
        rating = self.analyzer.financial_rating(financial_data)

        # Overall grade display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            grade_color = {
                "AAA": "#006100", "AA": "#006100", "A": "#4472C4",
                "BBB": "#4472C4", "BB": "#ED7D31", "B": "#ED7D31",
                "CCC": "#9C0006", "CC": "#9C0006", "C": "#9C0006",
            }
            color = grade_color.get(rating.overall_grade, "#000000")
            st.markdown(
                f"<h1 style='text-align:center; color:{color}; font-size:64px'>"
                f"{rating.overall_grade}</h1>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<p style='text-align:center'>Score: {rating.overall_score:.1f} / 10</p>", unsafe_allow_html=True)

        st.markdown(f"**{rating.summary}**")

        # Category breakdown
        for cat in rating.categories:
            col_l, col_r = st.columns([3, 1])
            with col_l:
                st.progress(min(1.0, cat.score / 10.0))
                st.caption(f"{cat.name}: {cat.details}")
            with col_r:
                st.metric(cat.name, f"{cat.grade} ({cat.score:.1f})")

        # Bar chart
        try:
            import plotly.graph_objects as go
            names = [c.name for c in rating.categories]
            scores = [c.score for c in rating.categories]
            colors = ["#006100" if s >= 7 else "#4472C4" if s >= 5 else "#ED7D31" if s >= 3 else "#9C0006" for s in scores]
            fig = go.Figure(data=[go.Bar(x=names, y=scores, marker_color=colors)])
            fig.update_layout(title="Category Scores", yaxis_title="Score (0-10)", yaxis_range=[0, 10])
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            logger.debug("Chart render skipped", exc_info=True)

    def _render_variance_waterfall(self, df: pd.DataFrame, workbook):
        """Render Variance Waterfall tab."""
        st.subheader("Net Income Variance Waterfall")
        st.markdown("Decompose what drove the change in net income between two periods.")

        if not hasattr(workbook, 'sheets') or len(workbook.sheets) < 2:
            st.info("Upload a workbook with at least 2 sheets (periods) to compare.")
            # Use simulated prior period
            financial_data = self.analyzer._dataframe_to_financial_data(df)
            adj = st.slider("Prior period revenue multiplier", 0.5, 1.5, 0.85, 0.05, key="wf_adj")

            if st.button("Generate Waterfall", key="wf_btn"):
                prior = FinancialData(
                    revenue=(financial_data.revenue or 0) * adj,
                    cogs=(financial_data.cogs or 0) * adj * 1.05,
                    operating_expenses=(financial_data.operating_expenses or 0) * adj * 0.95,
                    interest_expense=(financial_data.interest_expense or 0) * adj,
                    net_income=(financial_data.net_income or 0) * adj * 0.8,
                )
                waterfall = self.analyzer.variance_waterfall(financial_data, prior)
                self._display_waterfall(waterfall)
        else:
            financial_data = self.analyzer._dataframe_to_financial_data(df)
            # Use second sheet as prior period
            sheet_names = [s.name for s in workbook.sheets]
            prior_sheet = st.selectbox("Prior period sheet", sheet_names[1:], key="wf_prior")
            if st.button("Generate Waterfall", key="wf_btn2"):
                prior_s = next((s for s in workbook.sheets if s.name == prior_sheet), None)
                if prior_s and prior_s.dataframe is not None:
                    prior_data = self.analyzer._dataframe_to_financial_data(prior_s.dataframe)
                    waterfall = self.analyzer.variance_waterfall(financial_data, prior_data)
                    self._display_waterfall(waterfall)

    def _display_waterfall(self, waterfall):
        """Display waterfall chart and table."""
        st.markdown(f"**{waterfall.summary}**")

        # Table
        rows = []
        for item in waterfall.items:
            rows.append({
                "Component": item.label,
                "Value": f"{item.value:+,.0f}" if item.item_type != "start" else f"{item.value:,.0f}",
                "Cumulative": f"{item.cumulative:,.0f}",
                "Type": item.item_type.title(),
            })
        wf_df = pd.DataFrame(rows)
        st.dataframe(wf_df, use_container_width=True, hide_index=True)

        # Waterfall chart
        try:
            import plotly.graph_objects as go
            labels = [i.label for i in waterfall.items]
            measures = []
            values = []
            for item in waterfall.items:
                if item.item_type == "start":
                    measures.append("absolute")
                    values.append(item.value)
                elif item.item_type == "total":
                    measures.append("total")
                    values.append(0)
                else:
                    measures.append("relative")
                    values.append(item.value)

            fig = go.Figure(go.Waterfall(
                x=labels, y=values, measure=measures,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#006100"}},
                decreasing={"marker": {"color": "#9C0006"}},
                totals={"marker": {"color": "#4472C4"}},
            ))
            fig.update_layout(title="Net Income Bridge", yaxis_title="Amount")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            logger.debug("Chart render skipped", exc_info=True)

    def _render_earnings_quality(self, df: pd.DataFrame):
        """Render Earnings Quality tab."""
        st.subheader("Earnings Quality Assessment")
        st.markdown("Evaluate how well reported earnings are supported by actual cash flows.")

        financial_data = self.analyzer._dataframe_to_financial_data(df)
        result = self.analyzer.earnings_quality(financial_data)

        if result.quality_grade == "N/A":
            st.warning(result.summary)
            return

        # Grade display
        color_map = {"High": "#006100", "Moderate": "#ED7D31", "Low": "#9C0006"}
        color = color_map.get(result.quality_grade, "#333333")
        st.markdown(
            f"<h1 style='text-align:center;color:{color};'>{result.quality_grade}</h1>"
            f"<p style='text-align:center;font-size:1.2em;'>Score: {result.quality_score:.1f} / 10</p>",
            unsafe_allow_html=True,
        )

        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            if result.accrual_ratio is not None:
                st.metric("Accrual Ratio", f"{result.accrual_ratio:.4f}")
            else:
                st.metric("Accrual Ratio", "N/A")
        with col2:
            if result.cash_to_income_ratio is not None:
                st.metric("Cash-to-Income Ratio", f"{result.cash_to_income_ratio:.2f}")
            else:
                st.metric("Cash-to-Income Ratio", "N/A")

        # Indicators
        st.markdown("**Indicators:**")
        for ind in result.indicators:
            st.markdown(f"- {ind}")

        # Bar chart of score
        try:
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=[result.quality_score],
                y=["Quality Score"],
                orientation="h",
                marker_color=color,
                text=[f"{result.quality_score:.1f}"],
                textposition="auto",
            ))
            fig.update_layout(
                xaxis=dict(range=[0, 10], title="Score"),
                yaxis=dict(showticklabels=False),
                height=150,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            logger.debug("Chart render skipped", exc_info=True)

        st.caption(result.summary)

    def _render_capital_efficiency(self, df: pd.DataFrame):
        """Render Capital Efficiency tab."""
        st.subheader("Capital Efficiency & Value Creation")
        st.markdown("Assess how effectively the company deploys capital to generate returns.")

        financial_data = self.analyzer._dataframe_to_financial_data(df)

        wacc = st.slider("Cost of Capital (WACC %)", 4.0, 20.0, 10.0, 0.5, key="ce_wacc") / 100.0
        result = self.analyzer.capital_efficiency(financial_data, cost_of_capital=wacc)

        if result.roic is None and result.eva is None:
            st.warning(result.summary)
            return

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if result.roic is not None:
                st.metric("ROIC", f"{result.roic:.1%}")
            else:
                st.metric("ROIC", "N/A")
        with col2:
            if result.eva is not None:
                st.metric("EVA", f"${result.eva:,.0f}")
            else:
                st.metric("EVA", "N/A")
        with col3:
            if result.capital_turnover is not None:
                st.metric("Capital Turnover", f"{result.capital_turnover:.2f}x")
            else:
                st.metric("Capital Turnover", "N/A")
        with col4:
            if result.reinvestment_rate is not None:
                st.metric("Reinvestment Rate", f"{result.reinvestment_rate:.1%}")
            else:
                st.metric("Reinvestment Rate", "N/A")

        # ROIC vs WACC comparison
        if result.roic is not None:
            spread = result.roic - wacc
            if spread > 0:
                st.success(f"ROIC exceeds WACC by {spread:.1%} — creating shareholder value")
            else:
                st.error(f"ROIC below WACC by {abs(spread):.1%} — destroying shareholder value")

        # Detail table
        rows = []
        if result.nopat is not None:
            rows.append({"Metric": "NOPAT", "Value": f"${result.nopat:,.0f}"})
        if result.invested_capital is not None:
            rows.append({"Metric": "Invested Capital", "Value": f"${result.invested_capital:,.0f}"})
        if result.roic is not None:
            rows.append({"Metric": "ROIC", "Value": f"{result.roic:.2%}"})
        rows.append({"Metric": "WACC (assumed)", "Value": f"{wacc:.2%}"})
        if result.eva is not None:
            rows.append({"Metric": "Economic Value Added", "Value": f"${result.eva:,.0f}"})
        if result.capital_turnover is not None:
            rows.append({"Metric": "Capital Turnover", "Value": f"{result.capital_turnover:.2f}x"})
        if result.reinvestment_rate is not None:
            rows.append({"Metric": "Reinvestment Rate", "Value": f"{result.reinvestment_rate:.1%}"})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Chart: ROIC vs WACC bar
        try:
            import plotly.graph_objects as go
            if result.roic is not None:
                fig = go.Figure(data=[
                    go.Bar(name="ROIC", x=["Return vs Cost"], y=[result.roic * 100],
                           marker_color="#006100"),
                    go.Bar(name="WACC", x=["Return vs Cost"], y=[wacc * 100],
                           marker_color="#9C0006"),
                ])
                fig.update_layout(
                    title="ROIC vs WACC",
                    yaxis_title="Percentage (%)",
                    barmode="group",
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            logger.debug("Chart render skipped", exc_info=True)

        st.caption(result.summary)

    def _render_liquidity_stress(self, df: pd.DataFrame):
        """Render Liquidity Stress Test tab."""
        st.subheader("Liquidity Stress Test")
        st.markdown("Simulate revenue shocks and assess cash runway survival.")

        financial_data = self.analyzer._dataframe_to_financial_data(df)
        result = self.analyzer.liquidity_stress_test(financial_data)

        # Risk display
        color_map = {"Low": "#006100", "Moderate": "#ED7D31", "High": "#9C0006", "Critical": "#660000"}
        color = color_map.get(result.risk_level, "#333333")
        st.markdown(
            f"<h2 style='text-align:center;color:{color};'>Liquidity Risk: {result.risk_level}</h2>",
            unsafe_allow_html=True,
        )

        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cash (liquid)", f"${result.current_cash:,.0f}")
        with col2:
            st.metric("Monthly Burn", f"${result.monthly_burn:,.0f}")
        with col3:
            if result.months_of_cash is not None:
                st.metric("Months of Cash", f"{result.months_of_cash:.1f}")
            else:
                st.metric("Months of Cash", "N/A")

        if result.stressed_quick_ratio is not None:
            st.metric("Quick Ratio (stressed)", f"{result.stressed_quick_ratio:.2f}")

        # Scenario table
        if result.stress_scenarios:
            rows = []
            for s in result.stress_scenarios:
                rows.append({
                    "Scenario": s["label"],
                    "Monthly CF": f"${s['monthly_cash_flow']:,.0f}",
                    "Survival (mo)": f"{s['survival_months']:.1f}" if s["survival_months"] is not None else "Indefinite",
                    "Survives 12m": "Yes" if s["survives_12m"] else "No",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Chart
        try:
            import plotly.graph_objects as go
            labels = [s["label"] for s in result.stress_scenarios]
            survivals = [s["survival_months"] if s["survival_months"] is not None else 36 for s in result.stress_scenarios]
            colors = ["#006100" if s >= 12 else "#ED7D31" if s >= 6 else "#9C0006" for s in survivals]
            fig = go.Figure(go.Bar(
                x=labels, y=survivals,
                marker_color=colors,
                text=[f"{v:.1f}m" if v < 36 else "36+m" for v in survivals],
                textposition="auto",
            ))
            fig.add_hline(y=12, line_dash="dash", annotation_text="12-month threshold")
            fig.update_layout(title="Cash Survival Under Revenue Shocks", yaxis_title="Months")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            logger.debug("Chart render skipped", exc_info=True)

        st.caption(result.summary)

    def _render_health_score(self, df: pd.DataFrame):
        """Render the Comprehensive Health Score tab."""
        from financial_analyzer import ComprehensiveHealthResult, HealthDimension
        import plotly.graph_objects as go

        st.subheader("Comprehensive Financial Health Score")
        data = self.analyzer._dataframe_to_financial_data(df)
        result = self.analyzer.comprehensive_health_score(data)

        # --- Overall grade banner ---
        grade_colors = {
            "A+": "#15803d", "A": "#16a34a", "B+": "#65a30d",
            "B": "#ca8a04", "C+": "#d97706", "C": "#ea580c",
            "D": "#dc2626", "F": "#991b1b",
        }
        color = grade_colors.get(result.grade, "#6b7280")
        st.markdown(
            f"<h1 style='text-align:center;color:{color}'>"
            f"{result.grade} &mdash; {result.overall_score:.0f}/100</h1>",
            unsafe_allow_html=True,
        )

        # --- Dimension breakdown table ---
        st.markdown("#### Dimension Breakdown")
        rows = []
        for d in result.dimensions:
            status_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(d.status, "⚪")
            rows.append({
                "Dimension": d.name,
                "Score": f"{d.score:.0f}/100",
                "Weight": f"{d.weight:.0%}",
                "Weighted": f"{d.score * d.weight:.1f}",
                "Status": status_icon,
                "Detail": d.detail,
            })
        if rows:
            st.table(pd.DataFrame(rows))

        # --- Radar chart ---
        try:
            categories = [d.name for d in result.dimensions]
            values = [d.score for d in result.dimensions]
            # Close the radar polygon
            categories_closed = categories + [categories[0]]
            values_closed = values + [values[0]]

            fig = go.Figure(data=go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                line=dict(color=color),
                fillcolor=color,
                opacity=0.3,
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Health Dimension Radar",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            logger.debug("Chart render skipped", exc_info=True)

        # --- Horizontal bar chart ---
        try:
            bar_colors = [
                {"green": "#22c55e", "yellow": "#eab308", "red": "#ef4444"}.get(d.status, "#9ca3af")
                for d in result.dimensions
            ]
            fig = go.Figure(go.Bar(
                x=[d.score for d in result.dimensions],
                y=[d.name for d in result.dimensions],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{d.score:.0f}" for d in result.dimensions],
                textposition="auto",
            ))
            fig.update_layout(
                title="Scores by Dimension",
                xaxis=dict(title="Score", range=[0, 100]),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            logger.debug("Chart render skipped", exc_info=True)

        st.caption(result.summary)

    def _render_operating_leverage(self, df: pd.DataFrame):
        """Render the Operating Leverage & Break-Even tab."""
        from financial_analyzer import OperatingLeverageResult
        import plotly.graph_objects as go

        st.subheader("Operating Leverage & Break-Even Analysis")
        data = self.analyzer._dataframe_to_financial_data(df)
        result = self.analyzer.operating_leverage_analysis(data)

        # --- Cost structure badge ---
        struct_colors = {
            "High Fixed": "#dc2626",
            "Balanced": "#ca8a04",
            "High Variable": "#16a34a",
            "N/A": "#6b7280",
        }
        color = struct_colors.get(result.cost_structure, "#6b7280")
        st.markdown(
            f"<h3 style='color:{color}'>Cost Structure: {result.cost_structure}</h3>",
            unsafe_allow_html=True,
        )

        # --- Key metrics ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            val = f"{result.degree_of_operating_leverage:.2f}x" if result.degree_of_operating_leverage is not None else "N/A"
            st.metric("DOL", val)
        with col2:
            val = f"{result.contribution_margin_ratio:.1%}" if result.contribution_margin_ratio is not None else "N/A"
            st.metric("CM Ratio", val)
        with col3:
            val = f"${result.break_even_revenue:,.0f}" if result.break_even_revenue is not None else "N/A"
            st.metric("Break-Even Revenue", val)
        with col4:
            val = f"{result.margin_of_safety_pct:.1%}" if result.margin_of_safety_pct is not None else "N/A"
            st.metric("Margin of Safety", val)

        # --- Detail table ---
        rows = []
        if result.estimated_fixed_costs is not None:
            rows.append({"Metric": "Estimated Fixed Costs", "Value": f"${result.estimated_fixed_costs:,.0f}"})
        if result.estimated_variable_costs is not None:
            rows.append({"Metric": "Estimated Variable Costs", "Value": f"${result.estimated_variable_costs:,.0f}"})
        if result.contribution_margin is not None:
            rows.append({"Metric": "Contribution Margin", "Value": f"${result.contribution_margin:,.0f}"})
        if result.break_even_revenue is not None:
            rows.append({"Metric": "Break-Even Revenue", "Value": f"${result.break_even_revenue:,.0f}"})
        if result.margin_of_safety is not None:
            rows.append({"Metric": "Margin of Safety ($)", "Value": f"${result.margin_of_safety:,.0f}"})
        if result.degree_of_operating_leverage is not None:
            rows.append({"Metric": "Degree of Operating Leverage", "Value": f"{result.degree_of_operating_leverage:.2f}x"})
        if rows:
            st.table(pd.DataFrame(rows))

        # --- Cost structure pie chart ---
        try:
            if result.estimated_fixed_costs and result.estimated_variable_costs:
                fig = go.Figure(data=[go.Pie(
                    labels=["Fixed Costs", "Variable Costs"],
                    values=[result.estimated_fixed_costs, result.estimated_variable_costs],
                    marker_colors=["#3b82f6", "#f97316"],
                    hole=0.4,
                )])
                fig.update_layout(title="Cost Structure Split", height=350)
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            logger.debug("Chart render skipped", exc_info=True)

        # --- Break-even chart ---
        try:
            if result.break_even_revenue is not None and (data.revenue or 0) > 0:
                rev = data.revenue
                # Generate points from 0 to 1.5x revenue
                x_vals = [rev * i / 20 for i in range(31)]
                cm_ratio = result.contribution_margin_ratio or 0
                fc = result.estimated_fixed_costs or 0
                total_rev_line = x_vals
                total_cost_line = [fc + x * (1 - cm_ratio) for x in x_vals]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_vals, y=total_rev_line,
                    name="Revenue", line=dict(color="#22c55e"),
                ))
                fig.add_trace(go.Scatter(
                    x=x_vals, y=total_cost_line,
                    name="Total Cost", line=dict(color="#ef4444"),
                ))
                # Break-even point
                fig.add_vline(
                    x=result.break_even_revenue,
                    line_dash="dash", line_color="#6b7280",
                    annotation_text=f"Break-Even: ${result.break_even_revenue:,.0f}",
                )
                # Current revenue marker
                fig.add_vline(
                    x=rev,
                    line_dash="dot", line_color="#3b82f6",
                    annotation_text=f"Current: ${rev:,.0f}",
                )
                fig.update_layout(
                    title="Break-Even Chart",
                    xaxis_title="Revenue",
                    yaxis_title="Amount ($)",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            logger.debug("Chart render skipped", exc_info=True)

        st.caption(result.summary)

    def _render_cash_flow_quality(self, df: pd.DataFrame):
        """Render the Cash Flow Quality tab."""
        from financial_analyzer import CashFlowQualityResult
        import plotly.graph_objects as go

        st.subheader("Cash Flow Quality & Free Cash Flow Analysis")
        data = self.analyzer._dataframe_to_financial_data(df)
        result = self.analyzer.cash_flow_quality(data)

        # --- Grade banner ---
        grade_colors = {
            "Strong": "#15803d", "Adequate": "#ca8a04",
            "Weak": "#ea580c", "Poor": "#dc2626",
        }
        color = grade_colors.get(result.quality_grade, "#6b7280")
        st.markdown(
            f"<h3 style='color:{color}'>Quality Grade: {result.quality_grade}</h3>",
            unsafe_allow_html=True,
        )

        # --- Key metrics ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            val = f"${result.fcf:,.0f}" if result.fcf is not None else "N/A"
            st.metric("Free Cash Flow", val)
        with col2:
            val = f"{result.fcf_yield:.1%}" if result.fcf_yield is not None else "N/A"
            st.metric("FCF Yield", val)
        with col3:
            val = f"{result.fcf_margin:.1%}" if result.fcf_margin is not None else "N/A"
            st.metric("FCF Margin", val)
        with col4:
            val = f"{result.cash_conversion_efficiency:.0%}" if result.cash_conversion_efficiency is not None else "N/A"
            st.metric("Cash Conversion", val)

        # --- Indicators ---
        if result.indicators:
            st.markdown("#### Quality Indicators")
            for ind in result.indicators:
                st.markdown(f"- {ind}")

        # --- Detail table ---
        rows = []
        if result.ocf_to_net_income is not None:
            rows.append({"Metric": "OCF / Net Income", "Value": f"{result.ocf_to_net_income:.2f}x"})
        if result.capex_to_revenue is not None:
            rows.append({"Metric": "CapEx / Revenue", "Value": f"{result.capex_to_revenue:.1%}"})
        if result.capex_to_ocf is not None:
            rows.append({"Metric": "CapEx / OCF", "Value": f"{result.capex_to_ocf:.1%}"})
        if result.fcf_margin is not None:
            rows.append({"Metric": "FCF Margin", "Value": f"{result.fcf_margin:.1%}"})
        if result.fcf_yield is not None:
            rows.append({"Metric": "FCF Yield", "Value": f"{result.fcf_yield:.1%}"})
        if rows:
            st.table(pd.DataFrame(rows))

        # --- Waterfall chart: OCF -> FCF ---
        try:
            if data.operating_cash_flow is not None:
                ocf = data.operating_cash_flow
                capex_val = data.capex or 0
                fcf_val = result.fcf or 0

                fig = go.Figure(go.Waterfall(
                    x=["Operating CF", "CapEx", "Free CF"],
                    y=[ocf, -capex_val, fcf_val],
                    measure=["absolute", "relative", "total"],
                    connector=dict(line=dict(color="#6b7280")),
                    increasing=dict(marker_color="#22c55e"),
                    decreasing=dict(marker_color="#ef4444"),
                    totals=dict(marker_color="#3b82f6"),
                    text=[f"${ocf:,.0f}", f"-${capex_val:,.0f}", f"${fcf_val:,.0f}"],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Operating Cash Flow to Free Cash Flow",
                    yaxis_title="Amount ($)",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            logger.debug("Chart render skipped", exc_info=True)

        st.caption(result.summary)

    def _render_asset_efficiency(self, df: pd.DataFrame):
        """Render asset efficiency and turnover analysis tab."""
        from financial_analyzer import AssetEfficiencyResult

        analyzer = CharlieAnalyzer()
        data = analyzer._dataframe_to_financial_data(df)
        result = analyzer.asset_efficiency_analysis(data)

        # --- Grade badge ---
        grade_colors = {
            "Excellent": "#4CAF50",
            "Good": "#2196F3",
            "Average": "#FF9800",
            "Below Average": "#F44336",
        }
        color = grade_colors.get(result.efficiency_grade, "#9E9E9E")
        st.markdown(
            f"<div style='text-align:center;padding:12px;border-radius:8px;"
            f"background:{color};color:white;font-size:1.3em;font-weight:bold;'>"
            f"Asset Efficiency: {result.efficiency_grade} ({result.efficiency_score:.1f}/10)</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        # --- Key metrics ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            val = f"{result.total_asset_turnover:.2f}x" if result.total_asset_turnover is not None else "N/A"
            st.metric("Asset Turnover", val)
        with c2:
            val = f"{result.inventory_turnover:.1f}x" if result.inventory_turnover is not None else "N/A"
            st.metric("Inventory Turnover", val)
        with c3:
            val = f"{result.receivables_turnover:.1f}x" if result.receivables_turnover is not None else "N/A"
            st.metric("AR Turnover", val)
        with c4:
            val = f"{result.equity_turnover:.2f}x" if result.equity_turnover is not None else "N/A"
            st.metric("Equity Turnover", val)

        # --- Detail table ---
        detail_rows = []
        metrics = [
            ("Total Asset Turnover", result.total_asset_turnover, ".2f"),
            ("Fixed Asset Turnover", result.fixed_asset_turnover, ".2f"),
            ("Inventory Turnover", result.inventory_turnover, ".1f"),
            ("Receivables Turnover", result.receivables_turnover, ".1f"),
            ("Payables Turnover", result.payables_turnover, ".1f"),
            ("Working Capital Turnover", result.working_capital_turnover, ".2f"),
            ("Equity Turnover", result.equity_turnover, ".2f"),
        ]
        for name, val, fmt in metrics:
            if val is not None:
                try:
                    formatted = f"{val:{fmt}}x"
                except (ValueError, TypeError):
                    formatted = str(round(val, 4))
                detail_rows.append({"Metric": name, "Value": formatted})
        if detail_rows:
            st.subheader("Turnover Ratios")
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

        # --- Bar chart of turnover ratios ---
        chart_data = {}
        if result.total_asset_turnover is not None:
            chart_data["Asset"] = result.total_asset_turnover
        if result.inventory_turnover is not None:
            chart_data["Inventory"] = result.inventory_turnover
        if result.receivables_turnover is not None:
            chart_data["AR"] = result.receivables_turnover
        if result.payables_turnover is not None:
            chart_data["AP"] = result.payables_turnover
        if result.equity_turnover is not None:
            chart_data["Equity"] = result.equity_turnover
        if chart_data:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Bar(
                x=list(chart_data.keys()),
                y=list(chart_data.values()),
                marker_color=["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"][:len(chart_data)],
                text=[f"{v:.1f}x" for v in chart_data.values()],
                textposition="auto",
            )])
            fig.update_layout(
                title="Turnover Ratios Comparison",
                yaxis_title="Times (x)",
                showlegend=False,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_profitability_decomp(self, df: pd.DataFrame):
        """Render profitability decomposition tab."""
        from financial_analyzer import ProfitabilityDecompResult

        analyzer = CharlieAnalyzer()
        data = analyzer._dataframe_to_financial_data(df)
        result = analyzer.profitability_decomposition(data)

        # --- Grade badge ---
        grade_colors = {
            "Elite": "#4CAF50",
            "Strong": "#2196F3",
            "Adequate": "#FF9800",
            "Poor": "#F44336",
        }
        color = grade_colors.get(result.profitability_grade, "#9E9E9E")
        st.markdown(
            f"<div style='text-align:center;padding:12px;border-radius:8px;"
            f"background:{color};color:white;font-size:1.3em;font-weight:bold;'>"
            f"Profitability: {result.profitability_grade} ({result.profitability_score:.1f}/10)</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        # --- Key returns ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            val = f"{result.roe:.1%}" if result.roe is not None else "N/A"
            st.metric("ROE", val)
        with c2:
            val = f"{result.roa:.1%}" if result.roa is not None else "N/A"
            st.metric("ROA", val)
        with c3:
            val = f"{result.roic:.1%}" if result.roic is not None else "N/A"
            st.metric("ROIC", val)
        with c4:
            val = f"{result.spread:.1%}" if result.spread is not None else "N/A"
            st.metric("Economic Spread", val)

        # --- Detail table ---
        detail_rows = []
        metrics = [
            ("Return on Equity (ROE)", result.roe, ".1%"),
            ("Return on Assets (ROA)", result.roa, ".1%"),
            ("Return on Invested Capital (ROIC)", result.roic, ".1%"),
            ("NOPAT", result.nopat, ",.0f"),
            ("Invested Capital", result.invested_capital, ",.0f"),
            ("Economic Spread", result.spread, ".2%"),
            ("Economic Profit", result.economic_profit, ",.0f"),
            ("Asset Turnover", result.asset_turnover, ".2f"),
            ("Financial Leverage (TA/Equity)", result.financial_leverage, ".2f"),
            ("Tax Efficiency (NI/EBT)", result.tax_efficiency, ".1%"),
            ("Capital Intensity (TA/Revenue)", result.capital_intensity, ".2f"),
        ]
        for name, val, fmt in metrics:
            if val is not None:
                try:
                    formatted = f"{val:{fmt}}"
                except (ValueError, TypeError):
                    formatted = str(round(val, 4))
                detail_rows.append({"Metric": name, "Value": formatted})
        if detail_rows:
            st.subheader("Detailed Metrics")
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

        # --- Returns comparison bar chart ---
        returns_data = {}
        if result.roe is not None:
            returns_data["ROE"] = result.roe * 100
        if result.roa is not None:
            returns_data["ROA"] = result.roa * 100
        if result.roic is not None:
            returns_data["ROIC"] = result.roic * 100
        if returns_data:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Bar(
                x=list(returns_data.keys()),
                y=list(returns_data.values()),
                marker_color=["#4CAF50", "#2196F3", "#FF9800"][:len(returns_data)],
                text=[f"{v:.1f}%" for v in returns_data.values()],
                textposition="auto",
            )])
            fig.update_layout(
                title="Return Metrics Comparison",
                yaxis_title="Return (%)",
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_risk_adjusted(self, df: pd.DataFrame):
        """Render risk-adjusted performance tab."""
        from financial_analyzer import RiskAdjustedResult

        analyzer = CharlieAnalyzer()
        data = analyzer._dataframe_to_financial_data(df)
        result = analyzer.risk_adjusted_performance(data)

        # --- Grade badge ---
        grade_colors = {
            "Superior": "#4CAF50",
            "Favorable": "#2196F3",
            "Neutral": "#FF9800",
            "Elevated": "#F44336",
        }
        color = grade_colors.get(result.risk_grade, "#9E9E9E")
        st.markdown(
            f"<div style='text-align:center;padding:12px;border-radius:8px;"
            f"background:{color};color:white;font-size:1.3em;font-weight:bold;'>"
            f"Risk-Adjusted: {result.risk_grade} ({result.risk_score:.1f}/10)</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        # --- Key metrics ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            val = f"{result.return_on_risk:.1%}" if result.return_on_risk is not None else "N/A"
            st.metric("Return on Risk", val)
        with c2:
            val = f"{result.risk_adjusted_roe:.1%}" if result.risk_adjusted_roe is not None else "N/A"
            st.metric("Risk-Adj ROE", val)
        with c3:
            val = f"{result.margin_of_safety:.1f}x" if result.margin_of_safety is not None else "N/A"
            st.metric("Safety Margin", val)
        with c4:
            val = f"{result.downside_risk:.1f}x" if result.downside_risk is not None else "N/A"
            st.metric("Downside Risk (D/EBITDA)", val)

        # --- Detail table ---
        detail_rows = []
        metrics = [
            ("Return on Risk (ROE/Leverage)", result.return_on_risk, ".2%"),
            ("Risk-Adjusted ROE", result.risk_adjusted_roe, ".2%"),
            ("Risk-Adjusted ROA", result.risk_adjusted_roa, ".2%"),
            ("Debt-Adjusted Return", result.debt_adjusted_return, ".2%"),
            ("Volatility Proxy", result.volatility_proxy, ".2%"),
            ("Downside Risk (Debt/EBITDA)", result.downside_risk, ".2f"),
            ("Margin of Safety (Interest)", result.margin_of_safety, ".1f"),
            ("Return per Unit Risk", result.return_per_unit_risk, ".2f"),
        ]
        for name, val, fmt in metrics:
            if val is not None:
                try:
                    formatted = f"{val:{fmt}}"
                except (ValueError, TypeError):
                    formatted = str(round(val, 4))
                detail_rows.append({"Metric": name, "Value": formatted})
        if detail_rows:
            st.subheader("Detailed Metrics")
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

        # --- Risk-return comparison chart ---
        chart_data = {}
        if result.return_on_risk is not None:
            chart_data["Return/Risk"] = result.return_on_risk * 100
        if result.risk_adjusted_roe is not None:
            chart_data["Risk-Adj ROE"] = result.risk_adjusted_roe * 100
        if result.risk_adjusted_roa is not None:
            chart_data["Risk-Adj ROA"] = result.risk_adjusted_roa * 100
        if result.debt_adjusted_return is not None:
            chart_data["Debt-Adj Return"] = result.debt_adjusted_return * 100
        if chart_data:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Bar(
                x=list(chart_data.keys()),
                y=list(chart_data.values()),
                marker_color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"][:len(chart_data)],
                text=[f"{v:.1f}%" for v in chart_data.values()],
                textposition="auto",
            )])
            fig.update_layout(
                title="Risk-Adjusted Return Metrics",
                yaxis_title="Return (%)",
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_valuation_indicators(self, df: pd.DataFrame):
        """Render Valuation Indicators analysis tab."""
        from financial_analyzer import ValuationIndicatorsResult
        data = self.analyzer._dataframe_to_financial_data(df)
        result: ValuationIndicatorsResult = self.analyzer.valuation_indicators(data)

        # Grade badge
        grade_colors = {
            "Undervalued": "green", "Fair Value": "blue",
            "Fully Valued": "orange", "Overvalued": "red",
        }
        color = grade_colors.get(result.valuation_grade, "gray")
        st.markdown(
            f"### Valuation Grade: "
            f"<span style='color:{color};font-weight:bold'>{result.valuation_grade}</span> "
            f"({result.valuation_score:.1f}/10)",
            unsafe_allow_html=True,
        )

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EV/EBITDA", f"{result.ev_to_ebitda:.1f}x" if result.ev_to_ebitda is not None else "N/A")
        c2.metric("Earnings Yield", f"{result.earnings_yield:.1%}" if result.earnings_yield is not None else "N/A")
        c3.metric("EV/Revenue", f"{result.ev_to_revenue:.2f}x" if result.ev_to_revenue is not None else "N/A")
        c4.metric("ROIC", f"{result.return_on_invested_capital:.1%}" if result.return_on_invested_capital is not None else "N/A")

        # Detail table
        details = {
            "Earnings Yield": result.earnings_yield,
            "Book Value (Equity)": result.book_value_per_share_proxy,
            "Enterprise Value Proxy": result.ev_proxy,
            "EV / EBITDA": result.ev_to_ebitda,
            "EV / Revenue": result.ev_to_revenue,
            "EV / EBIT": result.ev_to_ebit,
            "Price/Book Proxy": result.price_to_book_proxy,
            "ROIC": result.return_on_invested_capital,
            "Graham Number Proxy": result.graham_number_proxy,
            "Intrinsic Value Proxy": result.intrinsic_value_proxy,
            "Margin of Safety %": result.margin_of_safety_pct,
        }
        rows = []
        for k, v in details.items():
            if v is not None:
                if "Value" in k or "Book Value" in k or "Graham" in k:
                    rows.append({"Metric": k, "Value": f"${v:,.0f}"})
                elif "Yield" in k or "ROIC" in k or "Safety" in k:
                    rows.append({"Metric": k, "Value": f"{v:.2%}"})
                else:
                    rows.append({"Metric": k, "Value": f"{v:.2f}"})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Valuation multiples bar chart
        multiples = {}
        if result.ev_to_ebitda is not None:
            multiples["EV/EBITDA"] = result.ev_to_ebitda
        if result.ev_to_revenue is not None:
            multiples["EV/Revenue"] = result.ev_to_revenue
        if result.ev_to_ebit is not None:
            multiples["EV/EBIT"] = result.ev_to_ebit
        if result.price_to_book_proxy is not None:
            multiples["P/B Proxy"] = result.price_to_book_proxy

        if multiples:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Bar(
                x=list(multiples.keys()),
                y=list(multiples.values()),
                marker_color=["#3498db", "#2ecc71", "#e67e22", "#9b59b6"][:len(multiples)],
            )])
            fig.update_layout(title="Valuation Multiples", yaxis_title="Multiple", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_sustainable_growth(self, df: pd.DataFrame):
        """Render Sustainable Growth analysis tab."""
        from financial_analyzer import SustainableGrowthResult
        data = self.analyzer._dataframe_to_financial_data(df)
        result: SustainableGrowthResult = self.analyzer.sustainable_growth_analysis(data)

        # Grade badge
        grade_colors = {
            "High Growth": "green", "Sustainable": "blue",
            "Moderate": "orange", "Constrained": "red",
        }
        color = grade_colors.get(result.growth_grade, "gray")
        st.markdown(
            f"### Growth Grade: "
            f"<span style='color:{color};font-weight:bold'>{result.growth_grade}</span> "
            f"({result.growth_score:.1f}/10)",
            unsafe_allow_html=True,
        )

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SGR", f"{result.sustainable_growth_rate:.1%}" if result.sustainable_growth_rate is not None else "N/A")
        c2.metric("IGR", f"{result.internal_growth_rate:.1%}" if result.internal_growth_rate is not None else "N/A")
        c3.metric("Retention Ratio", f"{result.retention_ratio:.0%}" if result.retention_ratio is not None else "N/A")
        c4.metric("ROE", f"{result.roe:.1%}" if result.roe is not None else "N/A")

        # Detail table
        details = {
            "Sustainable Growth Rate": result.sustainable_growth_rate,
            "Internal Growth Rate": result.internal_growth_rate,
            "Retention Ratio": result.retention_ratio,
            "Payout Ratio": result.payout_ratio,
            "Plowback Capacity": result.plowback_capacity,
            "ROE": result.roe,
            "ROA": result.roa,
            "Equity Growth Rate": result.equity_growth_rate,
            "Reinvestment Rate": result.reinvestment_rate,
            "Growth-Profitability Balance": result.growth_profitability_balance,
        }
        rows = []
        for k, v in details.items():
            if v is not None:
                if "Capacity" in k or "Balance" in k:
                    rows.append({"Metric": k, "Value": f"{v:.2f}x"})
                elif "Rate" in k or "Ratio" in k or "ROE" in k or "ROA" in k:
                    rows.append({"Metric": k, "Value": f"{v:.2%}"})
                else:
                    rows.append({"Metric": k, "Value": f"{v:.2f}"})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Growth rates comparison bar chart
        rates = {}
        if result.sustainable_growth_rate is not None:
            rates["SGR"] = result.sustainable_growth_rate * 100
        if result.internal_growth_rate is not None:
            rates["IGR"] = result.internal_growth_rate * 100
        if result.roe is not None:
            rates["ROE"] = result.roe * 100
        if result.roa is not None:
            rates["ROA"] = result.roa * 100

        if rates:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Bar(
                x=list(rates.keys()),
                y=list(rates.values()),
                marker_color=["#27ae60", "#2ecc71", "#3498db", "#5dade2"][:len(rates)],
            )])
            fig.update_layout(title="Growth & Return Rates (%)", yaxis_title="%", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_concentration_risk(self, df: pd.DataFrame):
        """Render Concentration Risk analysis tab."""
        from financial_analyzer import ConcentrationRiskResult
        data = self.analyzer._dataframe_to_financial_data(df)
        result: ConcentrationRiskResult = self.analyzer.concentration_risk_analysis(data)

        # Grade badge
        grade_colors = {
            "Well Diversified": "green", "Balanced": "blue",
            "Concentrated": "orange", "Highly Concentrated": "red",
        }
        color = grade_colors.get(result.concentration_grade, "gray")
        st.markdown(
            f"### Concentration Grade: "
            f"<span style='color:{color};font-weight:bold'>{result.concentration_grade}</span> "
            f"({result.concentration_score:.1f}/10)",
            unsafe_allow_html=True,
        )

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Asset Turnover", f"{result.revenue_asset_intensity:.2f}x" if result.revenue_asset_intensity is not None else "N/A")
        c2.metric("Operating Margin", f"{result.operating_dependency:.1%}" if result.operating_dependency is not None else "N/A")
        c3.metric("Cash Conversion", f"{result.cash_conversion_efficiency:.2f}x" if result.cash_conversion_efficiency is not None else "N/A")
        c4.metric("Interest Burden", f"{result.interest_burden:.1%}" if result.interest_burden is not None else "N/A")

        # Detail table
        details = {
            "Revenue/Asset Intensity": result.revenue_asset_intensity,
            "Operating Dependency": result.operating_dependency,
            "Current Asset Ratio": result.asset_composition_current,
            "Fixed Asset Ratio": result.asset_composition_fixed,
            "Current Liability Ratio": result.liability_structure_current,
            "Earnings Retention (NI/EBITDA)": result.earnings_retention_ratio,
            "Working Capital Intensity": result.working_capital_intensity,
            "Capex Intensity": result.capex_intensity,
            "Interest Burden": result.interest_burden,
            "Cash Conversion (OCF/NI)": result.cash_conversion_efficiency,
            "Fixed Assets / Equity": result.fixed_asset_ratio,
            "Debt / Total Liabilities": result.debt_concentration,
        }
        rows = []
        for k, v in details.items():
            if v is not None:
                if "Intensity" in k and "Revenue" in k:
                    rows.append({"Metric": k, "Value": f"{v:.2f}x"})
                elif "Conversion" in k or "Equity" in k:
                    rows.append({"Metric": k, "Value": f"{v:.2f}x"})
                else:
                    rows.append({"Metric": k, "Value": f"{v:.2%}"})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Asset composition pie chart
        if result.asset_composition_current is not None:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Pie(
                labels=["Current Assets", "Fixed Assets"],
                values=[result.asset_composition_current, result.asset_composition_fixed or 0],
                marker_colors=["#3498db", "#e67e22"],
                hole=0.4,
            )])
            fig.update_layout(title="Asset Composition", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_margin_of_safety(self, df: pd.DataFrame):
        """Render Margin of Safety analysis tab."""
        from financial_analyzer import MarginOfSafetyResult
        data = self.analyzer._dataframe_to_financial_data(df)
        result: MarginOfSafetyResult = self.analyzer.margin_of_safety_analysis(data)

        grade_colors = {"Wide Margin": "green", "Adequate": "blue", "Thin": "orange", "No Margin": "red"}
        color = grade_colors.get(result.safety_grade, "gray")
        st.markdown(f"### Margin of Safety &nbsp; :{color}[{result.safety_grade}] &nbsp; ({result.safety_score:.1f}/10)")

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            val = f"{result.earnings_yield:.1%}" if result.earnings_yield is not None else "N/A"
            st.metric("Earnings Yield", val)
        with c2:
            val = f"{result.price_to_book:.2f}x" if result.price_to_book is not None else "N/A"
            st.metric("Price / Book", val)
        with c3:
            val = f"{result.intrinsic_margin:.1%}" if result.intrinsic_margin is not None else "N/A"
            st.metric("Intrinsic Margin", val)
        with c4:
            val = f"${result.net_current_asset_value:,.0f}" if result.net_current_asset_value is not None else "N/A"
            st.metric("NCAV", val)

        # Detail table
        detail = {}
        if result.earnings_yield is not None:
            detail["Earnings Yield"] = f"{result.earnings_yield:.2%}"
        if result.book_value_per_share is not None:
            detail["Book Value / Share"] = f"${result.book_value_per_share:,.2f}"
        if result.price_to_book is not None:
            detail["Price / Book"] = f"{result.price_to_book:.2f}x"
        if result.book_value_discount is not None:
            detail["Book Value Discount"] = f"{result.book_value_discount:.2%}"
        if result.intrinsic_value_estimate is not None:
            detail["Intrinsic Value (Cap. Earnings)"] = f"${result.intrinsic_value_estimate:,.0f}"
        if result.market_cap is not None:
            detail["Market Cap"] = f"${result.market_cap:,.0f}"
        if result.intrinsic_margin is not None:
            detail["Intrinsic Value Margin"] = f"{result.intrinsic_margin:.2%}"
        if result.tangible_bv is not None:
            detail["Tangible Book Value"] = f"${result.tangible_bv:,.0f}"
        if result.liquidation_value is not None:
            detail["Liquidation Value (70% TBV)"] = f"${result.liquidation_value:,.0f}"
        if result.net_current_asset_value is not None:
            detail["Net Current Asset Value"] = f"${result.net_current_asset_value:,.0f}"
        if result.ncav_per_share is not None:
            detail["NCAV / Share"] = f"${result.ncav_per_share:,.2f}"
        if detail:
            st.table(pd.DataFrame.from_dict(detail, orient="index", columns=["Value"]))

        # Chart: IV vs Market Cap waterfall
        if result.intrinsic_value_estimate is not None and result.market_cap is not None:
            import plotly.graph_objects as go
            iv = result.intrinsic_value_estimate
            mc = result.market_cap
            margin = iv - mc
            fig = go.Figure(data=[
                go.Bar(
                    x=["Intrinsic Value", "Market Cap", "Margin of Safety"],
                    y=[iv, mc, margin],
                    marker_color=["#636EFA", "#EF553B", "#00CC96" if margin >= 0 else "#FFA15A"],
                    text=[f"${iv:,.0f}", f"${mc:,.0f}", f"${margin:,.0f}"],
                    textposition="outside",
                )
            ])
            fig.update_layout(title="Intrinsic Value vs Market Cap", yaxis_title="$", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_earnings_quality(self, df: pd.DataFrame):
        """Render Earnings Quality analysis tab."""
        from financial_analyzer import EarningsQualityResult
        data = self.analyzer._dataframe_to_financial_data(df)
        result: EarningsQualityResult = self.analyzer.earnings_quality_analysis(data)

        grade_colors = {"High": "green", "Adequate": "blue", "Questionable": "orange", "Poor": "red"}
        color = grade_colors.get(result.earnings_quality_grade, "gray")
        st.markdown(f"### Earnings Quality &nbsp; :{color}[{result.earnings_quality_grade}] &nbsp; ({result.earnings_quality_score:.1f}/10)")

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            val = f"{result.cash_to_income:.2f}x" if result.cash_to_income is not None else "N/A"
            st.metric("Cash / Income", val)
        with c2:
            val = f"{result.accruals_ratio:.2%}" if result.accruals_ratio is not None else "N/A"
            st.metric("Accruals Ratio", val)
        with c3:
            val = f"{result.operating_cash_quality:.1%}" if result.operating_cash_quality is not None else "N/A"
            st.metric("OCF / EBITDA", val)
        with c4:
            val = f"{result.revenue_cash_ratio:.1%}" if result.revenue_cash_ratio is not None else "N/A"
            st.metric("OCF / Revenue", val)

        # Detail table
        detail = {}
        if result.accruals_ratio is not None:
            detail["Accruals Ratio (NI-OCF)/TA"] = f"{result.accruals_ratio:.2%}"
        if result.cash_to_income is not None:
            detail["Cash / Income (OCF/NI)"] = f"{result.cash_to_income:.2f}x"
        if result.earnings_persistence is not None:
            detail["Earnings Persistence (NI/Rev)"] = f"{result.earnings_persistence:.2%}"
        if result.cash_return_on_assets is not None:
            detail["Cash ROA (OCF/TA)"] = f"{result.cash_return_on_assets:.2%}"
        if result.operating_cash_quality is not None:
            detail["Operating Cash Quality (OCF/EBITDA)"] = f"{result.operating_cash_quality:.1%}"
        if result.revenue_cash_ratio is not None:
            detail["Revenue Cash Ratio (OCF/Rev)"] = f"{result.revenue_cash_ratio:.1%}"
        if result.depreciation_to_capex is not None:
            detail["Depreciation / Capex"] = f"{result.depreciation_to_capex:.2f}x"
        if result.capex_to_revenue is not None:
            detail["Capex / Revenue"] = f"{result.capex_to_revenue:.1%}"
        if result.reinvestment_rate is not None:
            detail["Reinvestment Rate (Capex/Dep)"] = f"{result.reinvestment_rate:.2f}x"
        if detail:
            st.table(pd.DataFrame.from_dict(detail, orient="index", columns=["Value"]))

        # Chart: Cash vs Accrual earnings components
        chart_data = {}
        if result.cash_to_income is not None and result.accruals_ratio is not None:
            import plotly.graph_objects as go
            ni = data.net_income or 0
            ocf_val = data.operating_cash_flow or 0
            accrual_portion = ni - ocf_val
            fig = go.Figure(data=[
                go.Bar(
                    x=["Cash Earnings (OCF)", "Accrual Component", "Reported NI"],
                    y=[ocf_val, accrual_portion, ni],
                    marker_color=["#00CC96", "#FFA15A" if accrual_portion > 0 else "#636EFA", "#636EFA"],
                    text=[f"${ocf_val:,.0f}", f"${accrual_portion:,.0f}", f"${ni:,.0f}"],
                    textposition="outside",
                )
            ])
            fig.update_layout(title="Earnings Decomposition: Cash vs Accrual", yaxis_title="$", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_financial_flexibility(self, df: pd.DataFrame):
        """Render Financial Flexibility analysis tab."""
        from financial_analyzer import FinancialFlexibilityResult
        data = self.analyzer._dataframe_to_financial_data(df)
        result: FinancialFlexibilityResult = self.analyzer.financial_flexibility_analysis(data)

        grade_colors = {"Highly Flexible": "green", "Flexible": "blue", "Constrained": "orange", "Rigid": "red"}
        color = grade_colors.get(result.flexibility_grade, "gray")
        st.markdown(f"### Financial Flexibility &nbsp; :{color}[{result.flexibility_grade}] &nbsp; ({result.flexibility_score:.1f}/10)")

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            val = f"{result.cash_to_assets:.1%}" if result.cash_to_assets is not None else "N/A"
            st.metric("Cash / Assets", val)
        with c2:
            val = f"{result.fcf_to_revenue:.1%}" if result.fcf_to_revenue is not None else "N/A"
            st.metric("FCF Margin", val)
        with c3:
            val = f"${result.debt_headroom:,.0f}" if result.debt_headroom is not None else "N/A"
            st.metric("Debt Headroom", val)
        with c4:
            val = f"{result.financial_slack:.2f}x" if result.financial_slack is not None else "N/A"
            st.metric("Financial Slack", val)

        # Detail table
        detail = {}
        if result.cash_to_assets is not None:
            detail["Cash / Assets"] = f"{result.cash_to_assets:.2%}"
        if result.cash_to_debt is not None:
            detail["Cash / Debt"] = f"{result.cash_to_debt:.2%}"
        if result.cash_to_revenue is not None:
            detail["Cash / Revenue"] = f"{result.cash_to_revenue:.2%}"
        if result.fcf_to_revenue is not None:
            detail["FCF / Revenue"] = f"{result.fcf_to_revenue:.2%}"
        if result.spare_borrowing_capacity is not None:
            detail["Spare Borrowing Capacity"] = f"{result.spare_borrowing_capacity:.2%}"
        if result.unencumbered_assets is not None:
            detail["Unencumbered Assets"] = f"${result.unencumbered_assets:,.0f}"
        if result.financial_slack is not None:
            detail["Financial Slack (Cash+FCF)/Debt"] = f"{result.financial_slack:.2f}x"
        if result.debt_headroom is not None:
            detail["Debt Headroom (3x EBITDA - Debt)"] = f"${result.debt_headroom:,.0f}"
        if result.retained_earnings_ratio is not None:
            detail["Retained Earnings / Equity"] = f"{result.retained_earnings_ratio:.2%}"
        if detail:
            st.table(pd.DataFrame.from_dict(detail, orient="index", columns=["Value"]))

        # Chart: Flexibility components gauge
        gauge_data = {}
        if result.cash_to_assets is not None:
            gauge_data["Cash/Assets"] = min(result.cash_to_assets * 100, 50)
        if result.fcf_to_revenue is not None:
            gauge_data["FCF Margin %"] = min(max(result.fcf_to_revenue * 100, -10), 50)
        if result.spare_borrowing_capacity is not None:
            gauge_data["Spare Capacity %"] = min(max(result.spare_borrowing_capacity * 100, -20), 60)
        if gauge_data:
            import plotly.graph_objects as go
            labels = list(gauge_data.keys())
            values = list(gauge_data.values())
            colors = ["#00CC96" if v > 10 else "#FFA15A" if v > 0 else "#EF553B" for v in values]
            fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
            fig.update_layout(title="Flexibility Components (%)", yaxis_title="%", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_dupont_analysis(self, df: pd.DataFrame):
        """Render DuPont decomposition of ROE."""
        from financial_analyzer import CharlieAnalyzer, DupontAnalysisResult
        analyzer = CharlieAnalyzer()
        fd = analyzer._dataframe_to_financial_data(df)
        result = analyzer.dupont_analysis(fd)

        grade_colors = {
            "Excellent": "#00CC96", "Good": "#636EFA",
            "Fair": "#FFA15A", "Weak": "#EF553B",
        }
        color = grade_colors.get(result.dupont_grade, "#888")
        st.markdown(
            f"<span style='background:{color};color:white;padding:4px 12px;"
            f"border-radius:8px;font-weight:bold'>{result.dupont_grade} "
            f"({result.dupont_score:.1f}/10)</span>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROE", f"{result.roe:.1%}" if result.roe is not None else "N/A")
        c2.metric("Net Margin", f"{result.net_profit_margin:.1%}" if result.net_profit_margin is not None else "N/A")
        c3.metric("Asset Turnover", f"{result.asset_turnover:.2f}x" if result.asset_turnover is not None else "N/A")
        c4.metric("Equity Multiplier", f"{result.equity_multiplier:.2f}x" if result.equity_multiplier is not None else "N/A")

        detail = {}
        if result.roe is not None:
            detail["ROE"] = f"{result.roe:.2%}"
        if result.net_profit_margin is not None:
            detail["Net Profit Margin"] = f"{result.net_profit_margin:.2%}"
        if result.asset_turnover is not None:
            detail["Asset Turnover"] = f"{result.asset_turnover:.3f}x"
        if result.equity_multiplier is not None:
            detail["Equity Multiplier"] = f"{result.equity_multiplier:.3f}x"
        if result.operating_margin is not None:
            detail["Operating Margin (EBIT/Rev)"] = f"{result.operating_margin:.2%}"
        if result.tax_burden is not None:
            detail["Tax Burden (NI/EBT)"] = f"{result.tax_burden:.3f}"
        if result.interest_burden is not None:
            detail["Interest Burden (EBT/EBIT)"] = f"{result.interest_burden:.3f}"
        if result.roe_3factor is not None:
            detail["ROE (3-Factor Recon)"] = f"{result.roe_3factor:.2%}"
        if result.roe_5factor is not None:
            detail["ROE (5-Factor Recon)"] = f"{result.roe_5factor:.2%}"
        if detail:
            st.table(pd.DataFrame(list(detail.items()), columns=["Metric", "Value"]))

        # DuPont decomposition bar chart
        bar_data = {}
        if result.net_profit_margin is not None:
            bar_data["Net Margin"] = result.net_profit_margin * 100
        if result.asset_turnover is not None:
            bar_data["Asset Turnover"] = result.asset_turnover * 100
        if result.equity_multiplier is not None:
            bar_data["Equity Multiplier"] = result.equity_multiplier * 100
        if bar_data:
            import plotly.graph_objects as go
            labels = list(bar_data.keys())
            values = list(bar_data.values())
            colors = ["#636EFA", "#00CC96", "#FFA15A"][:len(values)]
            fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
            fig.update_layout(title="3-Factor DuPont Components", yaxis_title="Value (%/100x)", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_altman_z_score(self, df: pd.DataFrame):
        """Render Altman Z-Score bankruptcy prediction."""
        from financial_analyzer import CharlieAnalyzer, AltmanZScoreResult
        analyzer = CharlieAnalyzer()
        fd = analyzer._dataframe_to_financial_data(df)
        result = analyzer.altman_z_score_analysis(fd)

        grade_colors = {
            "Strong": "#00CC96", "Adequate": "#636EFA",
            "Watch": "#FFA15A", "Critical": "#EF553B",
        }
        color = grade_colors.get(result.altman_grade, "#888")
        st.markdown(
            f"<span style='background:{color};color:white;padding:4px 12px;"
            f"border-radius:8px;font-weight:bold'>{result.altman_grade} "
            f"({result.altman_score:.1f}/10)</span>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Z-Score", f"{result.z_score:.2f}" if result.z_score is not None else "N/A")
        c2.metric("Zone", result.z_zone or "N/A")
        c3.metric("WC/TA", f"{result.working_capital_to_assets:.1%}" if result.working_capital_to_assets is not None else "N/A")
        c4.metric("EBIT/TA", f"{result.ebit_to_assets:.1%}" if result.ebit_to_assets is not None else "N/A")

        detail = {}
        if result.working_capital_to_assets is not None:
            detail["X1: WC / Total Assets"] = f"{result.working_capital_to_assets:.3f}"
        if result.retained_earnings_to_assets is not None:
            detail["X2: Retained Earnings / TA"] = f"{result.retained_earnings_to_assets:.3f}"
        if result.ebit_to_assets is not None:
            detail["X3: EBIT / TA"] = f"{result.ebit_to_assets:.3f}"
        if result.equity_to_liabilities is not None:
            detail["X4: Equity / Total Liabilities"] = f"{result.equity_to_liabilities:.3f}"
        if result.revenue_to_assets is not None:
            detail["X5: Revenue / TA"] = f"{result.revenue_to_assets:.3f}"
        if result.z_score is not None:
            detail["Z-Score"] = f"{result.z_score:.3f}"
            detail["Zone"] = result.z_zone
        if detail:
            st.table(pd.DataFrame(list(detail.items()), columns=["Component", "Value"]))

        # Weighted components bar chart
        bar_data = {}
        if result.x1_weighted is not None:
            bar_data["1.2 x WC/TA"] = result.x1_weighted
        if result.x2_weighted is not None:
            bar_data["1.4 x RE/TA"] = result.x2_weighted
        if result.x3_weighted is not None:
            bar_data["3.3 x EBIT/TA"] = result.x3_weighted
        if result.x4_weighted is not None:
            bar_data["0.6 x Eq/TL"] = result.x4_weighted
        if result.x5_weighted is not None:
            bar_data["1.0 x Rev/TA"] = result.x5_weighted
        if bar_data:
            import plotly.graph_objects as go
            labels = list(bar_data.keys())
            values = list(bar_data.values())
            colors = ["#00CC96" if v > 0 else "#EF553B" for v in values]
            fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
            fig.update_layout(title="Z-Score Weighted Components", yaxis_title="Contribution", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_piotroski_f_score(self, df: pd.DataFrame):
        """Render Piotroski F-Score value screen."""
        from financial_analyzer import CharlieAnalyzer, PiotroskiFScoreResult
        analyzer = CharlieAnalyzer()
        fd = analyzer._dataframe_to_financial_data(df)
        result = analyzer.piotroski_f_score_analysis(fd)

        grade_colors = {
            "Strong Value": "#00CC96", "Moderate Value": "#636EFA",
            "Weak": "#FFA15A", "Avoid": "#EF553B",
        }
        color = grade_colors.get(result.piotroski_grade, "#888")
        st.markdown(
            f"<span style='background:{color};color:white;padding:4px 12px;"
            f"border-radius:8px;font-weight:bold'>{result.piotroski_grade} "
            f"(F-Score: {result.f_score}/{result.f_score_max})</span>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F-Score", f"{result.f_score}/{result.f_score_max}")
        c2.metric("ROA", f"{result.roa:.1%}" if result.roa is not None else "N/A")
        c3.metric("Current Ratio", f"{result.current_ratio:.2f}x" if result.current_ratio is not None else "N/A")
        c4.metric("Gross Margin", f"{result.gross_margin:.1%}" if result.gross_margin is not None else "N/A")

        # Signal checklist
        signals = [
            ("ROA > 0 (Profitability)", result.roa_positive),
            ("OCF > 0 (Cash Generation)", result.ocf_positive),
            ("OCF > NI (Accrual Quality)", result.accruals_negative),
            ("Current Ratio > 1 (Liquidity)", result.current_ratio_above_1),
            ("Debt/TA < 50% (Low Leverage)", result.low_leverage),
            ("Gross Margin > 20%", result.gross_margin_healthy),
            ("Asset Turnover > 0.5x", result.asset_turnover_adequate),
        ]
        rows = []
        for name, val in signals:
            if val is True:
                rows.append((name, "PASS", "+1"))
            elif val is False:
                rows.append((name, "FAIL", "0"))
            else:
                rows.append((name, "N/A", "—"))
        st.table(pd.DataFrame(rows, columns=["Signal", "Result", "Score"]))

        # Pass/Fail bar chart
        pass_count = sum(1 for _, v in signals if v is True)
        fail_count = sum(1 for _, v in signals if v is False)
        na_count = sum(1 for _, v in signals if v is None)
        if pass_count + fail_count > 0:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Bar(
                x=["Pass", "Fail", "N/A"],
                y=[pass_count, fail_count, na_count],
                marker_color=["#00CC96", "#EF553B", "#888"],
            )])
            fig.update_layout(title="F-Score Signal Results", yaxis_title="Count", height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_interest_coverage(self, df: pd.DataFrame):
        """Render interest coverage and debt capacity analysis."""
        from financial_analyzer import CharlieAnalyzer, InterestCoverageResult
        analyzer = CharlieAnalyzer()
        fd = analyzer._dataframe_to_financial_data(df)
        result = analyzer.interest_coverage_analysis(fd)

        grade_colors = {
            "Excellent": "#00CC96", "Adequate": "#636EFA",
            "Strained": "#FFA15A", "Critical": "#EF553B",
        }
        color = grade_colors.get(result.coverage_grade, "#888")
        st.markdown(
            f"<span style='background:{color};color:white;padding:4px 12px;"
            f"border-radius:8px;font-weight:bold'>{result.coverage_grade} "
            f"({result.coverage_score:.1f}/10)</span>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EBIT Coverage", f"{result.ebit_coverage:.1f}x" if result.ebit_coverage is not None else "N/A")
        c2.metric("EBITDA Coverage", f"{result.ebitda_coverage:.1f}x" if result.ebitda_coverage is not None else "N/A")
        c3.metric("Debt/EBITDA", f"{result.debt_to_ebitda:.1f}x" if result.debt_to_ebitda is not None else "N/A")
        c4.metric("Spare Capacity", f"${result.spare_debt_capacity:,.0f}" if result.spare_debt_capacity is not None else "N/A")

        detail = {}
        if result.ebit_coverage is not None:
            detail["EBIT / Interest"] = f"{result.ebit_coverage:.2f}x"
        if result.ebitda_coverage is not None:
            detail["EBITDA / Interest"] = f"{result.ebitda_coverage:.2f}x"
        if result.debt_to_ebitda is not None:
            detail["Debt / EBITDA"] = f"{result.debt_to_ebitda:.2f}x"
        if result.ocf_to_debt is not None:
            detail["OCF / Total Debt"] = f"{result.ocf_to_debt:.2%}"
        if result.fcf_to_debt is not None:
            detail["FCF / Total Debt"] = f"{result.fcf_to_debt:.2%}"
        if result.interest_to_revenue is not None:
            detail["Interest / Revenue"] = f"{result.interest_to_revenue:.2%}"
        if result.debt_to_equity is not None:
            detail["Debt / Equity"] = f"{result.debt_to_equity:.2f}x"
        if result.max_debt_capacity is not None:
            detail["Max Debt Capacity (3x EBITDA)"] = f"${result.max_debt_capacity:,.0f}"
        if result.spare_debt_capacity is not None:
            detail["Spare Debt Capacity"] = f"${result.spare_debt_capacity:,.0f}"
        if detail:
            st.table(pd.DataFrame(list(detail.items()), columns=["Metric", "Value"]))

        bar_data = {}
        if result.ebit_coverage is not None:
            bar_data["EBIT Coverage"] = result.ebit_coverage
        if result.ebitda_coverage is not None:
            bar_data["EBITDA Coverage"] = result.ebitda_coverage
        if result.ocf_to_debt is not None:
            bar_data["OCF/Debt %"] = result.ocf_to_debt * 100
        if bar_data:
            import plotly.graph_objects as go
            labels = list(bar_data.keys())
            values = list(bar_data.values())
            colors = ["#00CC96" if v > 3 else "#FFA15A" if v > 1 else "#EF553B" for v in values]
            fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
            fig.update_layout(title="Coverage Ratios", yaxis_title="Value", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_wacc_analysis(self, df: pd.DataFrame):
        """Render WACC & Cost of Capital analysis tab."""
        from financial_analyzer import CharlieAnalyzer, WACCResult
        analyzer = CharlieAnalyzer()
        data = analyzer._dataframe_to_financial_data(df)
        result = analyzer.wacc_analysis(data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Fair": "orange", "Expensive": "red"}
        color = grade_colors.get(result.wacc_grade, "gray")
        st.markdown(f"### WACC Analysis &mdash; :{color}[{result.wacc_grade}]")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("WACC", f"{result.wacc * 100:.1f}%" if result.wacc is not None else "N/A")
        c2.metric("Cost of Debt", f"{result.cost_of_debt * 100:.1f}%" if result.cost_of_debt is not None else "N/A")
        c3.metric("Cost of Equity", f"{result.implied_cost_of_equity * 100:.1f}%" if result.implied_cost_of_equity is not None else "N/A")
        c4.metric("Debt Weight", f"{result.debt_weight * 100:.0f}%" if result.debt_weight is not None else "N/A")

        rows = []
        if result.wacc is not None:
            rows.append({"Metric": "WACC", "Value": f"{result.wacc * 100:.2f}%"})
        if result.cost_of_debt is not None:
            rows.append({"Metric": "Pre-Tax Cost of Debt", "Value": f"{result.cost_of_debt * 100:.2f}%"})
        if result.after_tax_cost_of_debt is not None:
            rows.append({"Metric": "After-Tax Cost of Debt", "Value": f"{result.after_tax_cost_of_debt * 100:.2f}%"})
        if result.implied_cost_of_equity is not None:
            rows.append({"Metric": "Implied Cost of Equity", "Value": f"{result.implied_cost_of_equity * 100:.2f}%"})
        if result.effective_tax_rate is not None:
            rows.append({"Metric": "Effective Tax Rate", "Value": f"{result.effective_tax_rate * 100:.1f}%"})
        if result.debt_weight is not None:
            rows.append({"Metric": "Debt Weight", "Value": f"{result.debt_weight * 100:.1f}%"})
        if result.equity_weight is not None:
            rows.append({"Metric": "Equity Weight", "Value": f"{result.equity_weight * 100:.1f}%"})
        if result.total_capital is not None:
            rows.append({"Metric": "Total Capital", "Value": f"${result.total_capital:,.0f}"})

        if rows:
            st.table(pd.DataFrame(rows))

        # Capital structure pie chart
        if result.debt_weight is not None and result.equity_weight is not None:
            fig = go.Figure(data=[go.Pie(
                labels=["Debt", "Equity"],
                values=[result.debt_weight, result.equity_weight],
                marker_colors=["#EF553B", "#636EFA"],
                hole=0.4,
            )])
            fig.update_layout(title="Capital Structure", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_eva_analysis(self, df: pd.DataFrame):
        """Render EVA (Economic Value Added) analysis tab."""
        from financial_analyzer import CharlieAnalyzer, EVAResult
        analyzer = CharlieAnalyzer()
        data = analyzer._dataframe_to_financial_data(df)
        result = analyzer.eva_analysis(data)

        grade_colors = {"Value Creator": "green", "Adequate": "blue", "Marginal": "orange", "Value Destroyer": "red"}
        color = grade_colors.get(result.eva_grade, "gray")
        st.markdown(f"### EVA Analysis &mdash; :{color}[{result.eva_grade}]")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EVA", f"${result.eva:,.0f}" if result.eva is not None else "N/A")
        c2.metric("NOPAT", f"${result.nopat:,.0f}" if result.nopat is not None else "N/A")
        c3.metric("ROIC", f"{result.roic * 100:.1f}%" if result.roic is not None else "N/A")
        c4.metric("ROIC-WACC Spread", f"{result.roic_wacc_spread * 100:.1f}%" if result.roic_wacc_spread is not None else "N/A")

        rows = []
        if result.nopat is not None:
            rows.append({"Metric": "NOPAT", "Value": f"${result.nopat:,.0f}"})
        if result.invested_capital is not None:
            rows.append({"Metric": "Invested Capital", "Value": f"${result.invested_capital:,.0f}"})
        if result.wacc_used is not None:
            rows.append({"Metric": "WACC Used", "Value": f"{result.wacc_used * 100:.2f}%"})
        if result.capital_charge is not None:
            rows.append({"Metric": "Capital Charge", "Value": f"${result.capital_charge:,.0f}"})
        if result.eva is not None:
            rows.append({"Metric": "EVA", "Value": f"${result.eva:,.0f}"})
        if result.eva_margin is not None:
            rows.append({"Metric": "EVA Margin", "Value": f"{result.eva_margin * 100:.2f}%"})
        if result.roic is not None:
            rows.append({"Metric": "ROIC", "Value": f"{result.roic * 100:.2f}%"})
        if result.roic_wacc_spread is not None:
            rows.append({"Metric": "ROIC-WACC Spread", "Value": f"{result.roic_wacc_spread * 100:.2f}%"})

        if rows:
            st.table(pd.DataFrame(rows))

        # EVA waterfall chart
        if result.nopat is not None and result.capital_charge is not None and result.eva is not None:
            fig = go.Figure(go.Waterfall(
                name="EVA",
                orientation="v",
                measure=["absolute", "relative", "total"],
                x=["NOPAT", "Capital Charge", "EVA"],
                y=[result.nopat, -result.capital_charge, result.eva],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#2CA02C"}},
                decreasing={"marker": {"color": "#D62728"}},
                totals={"marker": {"color": "#636EFA"}},
            ))
            fig.update_layout(title="EVA Waterfall", yaxis_title="$", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_fcf_yield(self, df: pd.DataFrame):
        """Render Free Cash Flow Yield analysis tab."""
        from financial_analyzer import CharlieAnalyzer, FCFYieldResult
        analyzer = CharlieAnalyzer()
        data = analyzer._dataframe_to_financial_data(df)
        result = analyzer.fcf_yield_analysis(data)

        grade_colors = {"Strong": "green", "Healthy": "blue", "Weak": "orange", "Negative": "red"}
        color = grade_colors.get(result.fcf_grade, "gray")
        st.markdown(f"### FCF Yield &mdash; :{color}[{result.fcf_grade}]")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Free Cash Flow", f"${result.fcf:,.0f}" if result.fcf is not None else "N/A")
        c2.metric("FCF Margin", f"{result.fcf_margin * 100:.1f}%" if result.fcf_margin is not None else "N/A")
        c3.metric("FCF Conversion", f"{result.fcf_to_net_income:.2f}x" if result.fcf_to_net_income is not None else "N/A")
        c4.metric("FCF Yield on Capital", f"{result.fcf_yield_on_capital * 100:.1f}%" if result.fcf_yield_on_capital is not None else "N/A")

        rows = []
        if result.fcf is not None:
            rows.append({"Metric": "Free Cash Flow", "Value": f"${result.fcf:,.0f}"})
        if result.fcf_margin is not None:
            rows.append({"Metric": "FCF Margin", "Value": f"{result.fcf_margin * 100:.2f}%"})
        if result.fcf_to_net_income is not None:
            rows.append({"Metric": "FCF / Net Income", "Value": f"{result.fcf_to_net_income:.2f}x"})
        if result.fcf_yield_on_capital is not None:
            rows.append({"Metric": "FCF Yield on Capital", "Value": f"{result.fcf_yield_on_capital * 100:.2f}%"})
        if result.fcf_yield_on_equity is not None:
            rows.append({"Metric": "FCF Yield on Equity", "Value": f"{result.fcf_yield_on_equity * 100:.2f}%"})
        if result.fcf_to_debt is not None:
            rows.append({"Metric": "FCF / Total Debt", "Value": f"{result.fcf_to_debt:.2f}x"})
        if result.capex_to_ocf is not None:
            rows.append({"Metric": "CapEx / OCF", "Value": f"{result.capex_to_ocf * 100:.1f}%"})
        if result.capex_to_revenue is not None:
            rows.append({"Metric": "CapEx / Revenue", "Value": f"{result.capex_to_revenue * 100:.1f}%"})

        if rows:
            st.table(pd.DataFrame(rows))

        # FCF components bar chart
        if result.fcf is not None:
            ocf_val = data.operating_cash_flow or 0
            capex_val = data.capex or 0
            fig = go.Figure(data=[go.Bar(
                x=["Operating CF", "CapEx", "Free CF"],
                y=[ocf_val, -capex_val, result.fcf],
                marker_color=["#636EFA", "#EF553B", "#2CA02C" if result.fcf >= 0 else "#D62728"],
            )])
            fig.update_layout(title="FCF Components", yaxis_title="$", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_operating_leverage(self, df: pd.DataFrame):
        """Render Operating Leverage Analysis tab (Phase 40)."""
        from financial_analyzer import OperatingLeverageResult
        analyzer = CharlieAnalyzer()
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.operating_leverage_analysis(data)

        grade_colors = {
            "Low Risk": "green",
            "Moderate": "blue",
            "Elevated": "orange",
            "High Risk": "red",
        }
        color = grade_colors.get(result.operating_risk_grade, "gray")
        st.markdown(
            f"### Operating Leverage &nbsp; "
            f"<span style='background:{color};color:white;padding:4px 12px;"
            f"border-radius:8px;font-size:0.9em'>"
            f"{result.operating_risk_grade} ({result.operating_risk_score:.1f}/10)</span>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        dol = result.degree_of_operating_leverage
        c1.metric("DOL", f"{dol:.2f}x" if dol is not None else "N/A")
        cm = result.contribution_margin_ratio
        c2.metric("CM Ratio", f"{cm:.1%}" if cm is not None else "N/A")
        mos = result.margin_of_safety_pct
        c3.metric("Margin of Safety", f"{mos:.1%}" if mos is not None else "N/A")
        bep = result.breakeven_revenue
        c4.metric("Breakeven Revenue", f"${bep:,.0f}" if bep is not None else "N/A")

        rows = []
        if dol is not None:
            rows.append({"Metric": "Degree of Operating Leverage", "Value": f"{dol:.2f}x"})
        if result.contribution_margin is not None:
            rows.append({"Metric": "Contribution Margin", "Value": f"${result.contribution_margin:,.0f}"})
        if cm is not None:
            rows.append({"Metric": "Contribution Margin Ratio", "Value": f"{cm:.1%}"})
        if result.variable_cost_ratio is not None:
            rows.append({"Metric": "Variable Cost Ratio", "Value": f"{result.variable_cost_ratio:.1%}"})
        if result.fixed_cost_ratio is not None:
            rows.append({"Metric": "Fixed Cost Ratio", "Value": f"{result.fixed_cost_ratio:.1%}"})
        if bep is not None:
            rows.append({"Metric": "Breakeven Revenue", "Value": f"${bep:,.0f}"})
        if mos is not None:
            rows.append({"Metric": "Margin of Safety", "Value": f"{mos:.1%}"})
        rows.append({"Metric": "Risk Score", "Value": f"{result.operating_risk_score:.1f} / 10"})

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Cost structure pie chart
        if result.variable_cost_ratio is not None and result.fixed_cost_ratio is not None:
            import plotly.graph_objects as go
            vc = result.variable_cost_ratio
            fc = result.fixed_cost_ratio
            op_margin = 1.0 - vc - fc if (vc + fc) <= 1.0 else 0.0
            labels = ["Variable Costs", "Fixed Costs", "Operating Income"]
            values = [vc, fc, max(0, op_margin)]
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=["#EF553B", "#636EFA", "#00CC96"],
            )])
            fig.update_layout(title="Cost Structure Breakdown", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_cash_conversion(self, df: pd.DataFrame):
        """Render Cash Conversion Efficiency tab (Phase 41)."""
        from financial_analyzer import CashConversionResult
        analyzer = CharlieAnalyzer()
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.cash_conversion_analysis(data)

        grade_colors = {
            "Excellent": "green",
            "Good": "blue",
            "Fair": "orange",
            "Poor": "red",
        }
        color = grade_colors.get(result.cash_conversion_grade, "gray")
        st.markdown(
            f"### Cash Conversion &nbsp; "
            f"<span style='background:{color};color:white;padding:4px 12px;"
            f"border-radius:8px;font-size:0.9em'>"
            f"{result.cash_conversion_grade} ({result.cash_conversion_score:.1f}/10)</span>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        ccc = result.ccc
        c1.metric("CCC", f"{ccc:.0f} days" if ccc is not None else "N/A")
        dso = result.dso
        c2.metric("DSO", f"{dso:.0f} days" if dso is not None else "N/A")
        ocf_rev = result.ocf_to_revenue
        c3.metric("OCF/Revenue", f"{ocf_rev:.1%}" if ocf_rev is not None else "N/A")
        ocf_eb = result.ocf_to_ebitda
        c4.metric("OCF/EBITDA", f"{ocf_eb:.1%}" if ocf_eb is not None else "N/A")

        rows = []
        if dso is not None:
            rows.append({"Metric": "Days Sales Outstanding", "Value": f"{dso:.0f} days"})
        if result.dio is not None:
            rows.append({"Metric": "Days Inventory Outstanding", "Value": f"{result.dio:.0f} days"})
        if result.dpo is not None:
            rows.append({"Metric": "Days Payable Outstanding", "Value": f"{result.dpo:.0f} days"})
        if ccc is not None:
            rows.append({"Metric": "Cash Conversion Cycle", "Value": f"{ccc:.0f} days"})
        if result.cash_to_revenue is not None:
            rows.append({"Metric": "Cash / Revenue", "Value": f"{result.cash_to_revenue:.1%}"})
        if ocf_rev is not None:
            rows.append({"Metric": "OCF / Revenue", "Value": f"{ocf_rev:.1%}"})
        if ocf_eb is not None:
            rows.append({"Metric": "OCF / EBITDA", "Value": f"{ocf_eb:.1%}"})
        rows.append({"Metric": "Cash Conversion Score", "Value": f"{result.cash_conversion_score:.1f} / 10"})

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # CCC components bar chart
        if dso is not None and result.dio is not None and result.dpo is not None:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Bar(
                x=["DSO", "DIO", "DPO", "CCC"],
                y=[dso, result.dio, result.dpo, ccc if ccc else 0],
                marker_color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
            )])
            fig.update_layout(
                title="Cash Conversion Cycle Components (Days)",
                yaxis_title="Days",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_beneish_m_score(self, df: pd.DataFrame):
        """Render Beneish M-Score analysis tab."""
        st.subheader("Beneish M-Score Analysis")

        data = self.analyzer._dataframe_to_financial_data(df)
        result = self.analyzer.beneish_m_score_analysis(data)

        grade_colors = {
            "Unlikely": "green",
            "Possible": "orange",
            "Likely": "red",
            "Highly Likely": "darkred",
        }
        color = grade_colors.get(result.manipulation_grade, "gray")
        st.markdown(
            f"**Manipulation Risk:** <span style='color:{color}; font-size:1.3em;'>"
            f"{result.manipulation_grade}</span> &nbsp; (Score: {result.manipulation_score:.1f}/10)",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("M-Score", f"{result.m_score:.2f}" if result.m_score is not None else "N/A")
        with c2:
            st.metric("TATA", f"{result.tata:.3f}" if result.tata is not None else "N/A")
        with c3:
            st.metric("DSRI", f"{result.dsri:.2f}" if result.dsri is not None else "N/A")
        with c4:
            st.metric("GMI", f"{result.gmi:.2f}" if result.gmi is not None else "N/A")

        detail_rows = [
            ("M-Score", result.m_score, ".2f"),
            ("DSRI (Days Sales Receivables)", result.dsri, ".3f"),
            ("GMI (Gross Margin Index)", result.gmi, ".3f"),
            ("AQI (Asset Quality Index)", result.aqi, ".3f"),
            ("SGI (Sales Growth Index)", result.sgi, ".3f"),
            ("DEPI (Depreciation Index)", result.depi, ".3f"),
            ("SGAI (SGA Index)", result.sgai, ".3f"),
            ("LVGI (Leverage Index)", result.lvgi, ".3f"),
            ("TATA (Total Accruals)", result.tata, ".4f"),
            ("Manipulation Score", result.manipulation_score, ".1f"),
        ]
        detail_data = []
        for label, val, fmt in detail_rows:
            detail_data.append({
                "Metric": label,
                "Value": f"{val:{fmt}}" if val is not None else "N/A",
            })
        st.table(pd.DataFrame(detail_data).set_index("Metric"))

        if result.m_score is not None:
            import plotly.graph_objects as go
            threshold = -1.78
            fig = go.Figure()
            bar_color = "#00CC96" if result.m_score < threshold else "#EF553B"
            fig.add_trace(go.Bar(
                x=["M-Score"],
                y=[result.m_score],
                marker_color=bar_color,
                name="M-Score",
            ))
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({threshold})",
            )
            fig.update_layout(
                title="Beneish M-Score vs. Threshold",
                yaxis_title="M-Score",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_defensive_posture(self, df: pd.DataFrame):
        """Render Phase 133: Defensive Posture Analysis."""
        from financial_analyzer import CharlieAnalyzer, DefensivePostureResult
        analyzer = CharlieAnalyzer()
        rows = df.to_dict("records")
        if not rows:
            st.warning("No data available for Defensive Posture analysis.")
            return
        for row in rows:
            period = row.get("Period", "N/A")
            fd = self._row_to_financial_data(row)
            result = analyzer.defensive_posture_analysis(fd)
            color = "green" if result.dp_grade == "Excellent" else "blue" if result.dp_grade == "Good" else "orange" if result.dp_grade == "Adequate" else "red"
            st.markdown(f"### {period} — Defensive Posture: :{color}[{result.dp_grade}] ({result.dp_score}/10)")

            c1, c2, c3, c4 = st.columns(4)
            _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
            _r0 = lambda v: f"{v:.0f}" if v is not None else "N/A"
            _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
            c1.metric("Defensive Interval", f"{_r0(result.defensive_interval)} days")
            c2.metric("Cash Ratio", _r2(result.cash_ratio))
            c3.metric("Quick Ratio", _r2(result.quick_ratio))
            c4.metric("Equity Buffer", _pct(result.equity_buffer))

            details = {"Metric": [], "Value": []}
            details["Metric"].append("CF Coverage")
            details["Value"].append(_pct(result.cash_flow_coverage))
            details["Metric"].append("Debt Shield")
            details["Value"].append(f"{_r2(result.debt_shield)}x")
            details["Metric"].append("Score")
            details["Value"].append(f"{result.dp_score}/10")
            details["Metric"].append("Grade")
            details["Value"].append(result.dp_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_income_stability(self, df: pd.DataFrame):
        """Render Phase 134: Income Stability Analysis."""
        from financial_analyzer import CharlieAnalyzer, IncomeStabilityResult
        analyzer = CharlieAnalyzer()
        rows = df.to_dict("records")
        if not rows:
            st.warning("No data available for Income Stability analysis.")
            return
        for row in rows:
            period = row.get("Period", "N/A")
            fd = self._row_to_financial_data(row)
            result = analyzer.income_stability_analysis(fd)
            color = "green" if result.is_grade == "Excellent" else "blue" if result.is_grade == "Good" else "orange" if result.is_grade == "Adequate" else "red"
            st.markdown(f"### {period} — Income Stability: :{color}[{result.is_grade}] ({result.is_score}/10)")

            c1, c2, c3, c4 = st.columns(4)
            _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
            _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
            c1.metric("OI Cushion", f"{_r2(result.operating_income_cushion)}x")
            c2.metric("NI Margin", _pct(result.net_income_margin))
            c3.metric("EBITDA Margin", _pct(result.ebitda_margin))
            c4.metric("Resilience", f"{_r2(result.income_resilience)}x")

            details = {"Metric": [], "Value": []}
            details["Metric"].append("Retained Earn Ratio")
            details["Value"].append(_pct(result.retained_earnings_ratio))
            details["Metric"].append("Net/Gross Ratio")
            details["Value"].append(_pct(result.net_to_gross_ratio))
            details["Metric"].append("Score")
            details["Value"].append(f"{result.is_score}/10")
            details["Metric"].append("Grade")
            details["Value"].append(result.is_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_profit_retention_power(self, df: pd.DataFrame):
        """Phase 356: Profit Retention Power Analysis."""
        from financial_analyzer import CharlieAnalyzer, ProfitRetentionPowerResult

        analyzer = CharlieAnalyzer()
        fin = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.profit_retention_power_analysis(fin)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.prp_grade, "gray")
        st.markdown(f"### Profit Retention Power &nbsp; :{color}[{result.prp_grade}]")

        c1, c2, c3, c4 = st.columns(4)
        prp_str = f"{result.prp_ratio:.4f}" if result.prp_ratio is not None else "N/A"
        rr_str = f"{result.retention_rate:.1%}" if result.retention_rate is not None else "N/A"
        rer_str = f"{result.re_to_equity:.1%}" if result.re_to_equity is not None else "N/A"
        c1.metric("RE/Assets", prp_str)
        c2.metric("Retention Rate", rr_str)
        c3.metric("RE/Equity", rer_str)
        c4.metric("Score", f"{result.prp_score:.1f}/10")

        with st.expander("Profit Retention Power Details"):
            details = {"Metric": [], "Value": []}
            for label, val in [
                ("RE/Total Assets", result.prp_ratio),
                ("RE/Total Equity", result.re_to_equity),
                ("RE/Revenue", result.re_to_revenue),
                ("RE Growth Capacity", result.re_growth_capacity),
                ("Retention Rate", result.retention_rate),
                ("PRP Spread vs 0.20", result.prp_spread),
            ]:
                details["Metric"].append(label)
                details["Value"].append(f"{val:.4f}" if val is not None else "N/A")
            details["Metric"].append("Score")
            details["Value"].append(f"{result.prp_score:.1f}")
            details["Metric"].append("Grade")
            details["Value"].append(result.prp_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_earnings_to_debt(self, df: pd.DataFrame):
        """Phase 353: Earnings To Debt Analysis."""
        from financial_analyzer import CharlieAnalyzer, EarningsToDebtResult

        analyzer = CharlieAnalyzer()
        fin = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.earnings_to_debt_analysis(fin)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.etd_grade, "gray")
        st.markdown(f"### Earnings To Debt &nbsp; :{color}[{result.etd_grade}]")

        c1, c2, c3, c4 = st.columns(4)
        etd_str = f"{result.etd_ratio:.4f}" if result.etd_ratio is not None else "N/A"
        nti_str = f"{result.ni_to_interest:.2f}x" if result.ni_to_interest is not None else "N/A"
        dye_str = f"{result.debt_years_from_earnings:.2f} yrs" if result.debt_years_from_earnings is not None else "N/A"
        c1.metric("NI/Total Debt", etd_str)
        c2.metric("NI/Interest", nti_str)
        c3.metric("Payback", dye_str)
        c4.metric("Score", f"{result.etd_score:.1f}/10")

        with st.expander("Earnings To Debt Details"):
            details = {"Metric": [], "Value": []}
            for label, val in [
                ("NI/Total Debt", result.etd_ratio),
                ("NI/Interest", result.ni_to_interest),
                ("NI/Liabilities", result.ni_to_liabilities),
                ("Earnings Yield on Debt", result.earnings_yield_on_debt),
                ("Payback Years", result.debt_years_from_earnings),
                ("ETD Spread vs 0.20", result.etd_spread),
            ]:
                details["Metric"].append(label)
                details["Value"].append(f"{val:.4f}" if val is not None else "N/A")
            details["Metric"].append("Score")
            details["Value"].append(f"{result.etd_score:.1f}")
            details["Metric"].append("Grade")
            details["Value"].append(result.etd_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_revenue_growth(self, df: pd.DataFrame):
        """Phase 350: Revenue Growth Capacity Analysis."""
        from financial_analyzer import CharlieAnalyzer, RevenueGrowthResult

        analyzer = CharlieAnalyzer()
        fin = self.analyzer._dataframe_to_financial_data(df)
        result: RevenueGrowthResult = analyzer.revenue_growth_analysis(fin)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.rg_grade, "gray")
        st.markdown(f"**Revenue Growth Grade:** :{color}[{result.rg_grade}] ({result.rg_score}/10)")

        c1, c2, c3, c4 = st.columns(4)
        sgr_str = f"{result.sustainable_growth:.4f}" if result.sustainable_growth is not None else "N/A"
        roe_str = f"{result.roe:.4f}" if result.roe is not None else "N/A"
        pb_str = f"{result.plowback:.4f}" if result.plowback is not None else "N/A"
        ra_str = f"{result.revenue_per_asset:.4f}" if result.revenue_per_asset is not None else "N/A"
        c1.metric("Sust. Growth Rate", sgr_str)
        c2.metric("ROE", roe_str)
        c3.metric("Plowback Rate", pb_str)
        c4.metric("Revenue/Assets", ra_str)

        with st.expander("Revenue Growth Details"):
            details: dict = {"Metric": [], "Value": []}
            details["Metric"].append("RG Capacity")
            rg_str = f"{result.rg_capacity:.4f}" if result.rg_capacity is not None else "N/A"
            details["Value"].append(rg_str)
            details["Metric"].append("RG Spread")
            sp_str = f"{result.rg_spread:.4f}" if result.rg_spread is not None else "N/A"
            details["Value"].append(sp_str)
            details["Metric"].append("Score")
            details["Value"].append(f"{result.rg_score}/10")
            details["Metric"].append("Grade")
            details["Value"].append(result.rg_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_operating_margin(self, df: pd.DataFrame):
        """Phase 349: Operating Margin Analysis."""
        from financial_analyzer import CharlieAnalyzer, OperatingMarginResult

        analyzer = CharlieAnalyzer()
        fin = self.analyzer._dataframe_to_financial_data(df)
        result: OperatingMarginResult = analyzer.operating_margin_analysis(fin)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.opm_grade, "gray")
        st.markdown(f"**Operating Margin Grade:** :{color}[{result.opm_grade}] ({result.opm_score}/10)")

        c1, c2, c3, c4 = st.columns(4)
        oi_str = f"{result.oi_to_revenue:.4f}" if result.oi_to_revenue is not None else "N/A"
        ebit_str = f"{result.ebit_margin:.4f}" if result.ebit_margin is not None else "N/A"
        ebitda_str = f"{result.ebitda_margin:.4f}" if result.ebitda_margin is not None else "N/A"
        sp_str = f"{result.opm_spread:.4f}" if result.opm_spread is not None else "N/A"
        c1.metric("OI/Revenue", oi_str)
        c2.metric("EBIT Margin", ebit_str)
        c3.metric("EBITDA Margin", ebitda_str)
        c4.metric("Margin Spread", sp_str)

        with st.expander("Operating Margin Details"):
            details: dict = {"Metric": [], "Value": []}
            details["Metric"].append("Operating Margin")
            om_str = f"{result.operating_margin:.4f}" if result.operating_margin is not None else "N/A"
            details["Value"].append(om_str)
            details["Metric"].append("Score")
            details["Value"].append(f"{result.opm_score}/10")
            details["Metric"].append("Grade")
            details["Value"].append(result.opm_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_debt_to_equity(self, df: pd.DataFrame):
        """Phase 348: Debt To Equity Analysis."""
        from financial_analyzer import CharlieAnalyzer, DebtToEquityResult

        analyzer = CharlieAnalyzer()
        fin = self.analyzer._dataframe_to_financial_data(df)
        result: DebtToEquityResult = analyzer.debt_to_equity_analysis(fin)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.dte_grade, "gray")
        st.markdown(f"**Debt to Equity Grade:** :{color}[{result.dte_grade}] ({result.dte_score}/10)")

        c1, c2, c3, c4 = st.columns(4)
        td_te_str = f"{result.td_to_te:.4f}" if result.td_to_te is not None else "N/A"
        d_a_str = f"{result.debt_to_assets:.4f}" if result.debt_to_assets is not None else "N/A"
        em_str = f"{result.equity_multiplier:.2f}" if result.equity_multiplier is not None else "N/A"
        lt_str = f"{result.lt_debt_to_equity:.4f}" if result.lt_debt_to_equity is not None else "N/A"
        c1.metric("TD/Total Equity", td_te_str)
        c2.metric("Debt/Assets", d_a_str)
        c3.metric("Equity Multiplier", em_str)
        c4.metric("LT Debt/Equity", lt_str)

        with st.expander("Debt to Equity Details"):
            details: dict = {"Metric": [], "Value": []}
            details["Metric"].append("D/E Ratio")
            dte_str = f"{result.dte_ratio:.4f}" if result.dte_ratio is not None else "N/A"
            details["Value"].append(dte_str)
            details["Metric"].append("D/E Spread")
            sp_str = f"{result.dte_spread:.4f}" if result.dte_spread is not None else "N/A"
            details["Value"].append(sp_str)
            details["Metric"].append("Score")
            details["Value"].append(f"{result.dte_score}/10")
            details["Metric"].append("Grade")
            details["Value"].append(result.dte_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_cash_flow_to_debt(self, df: pd.DataFrame):
        """Phase 347: Cash Flow To Debt Analysis."""
        from financial_analyzer import CharlieAnalyzer, CashFlowToDebtResult

        analyzer = CharlieAnalyzer()
        fin = self.analyzer._dataframe_to_financial_data(df)
        result: CashFlowToDebtResult = analyzer.cash_flow_to_debt_analysis(fin)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.cfd_grade, "gray")
        st.markdown(f"**Cash Flow to Debt Grade:** :{color}[{result.cfd_grade}] ({result.cfd_score}/10)")

        c1, c2, c3, c4 = st.columns(4)
        ocf_td_str = f"{result.ocf_to_td:.4f}" if result.ocf_to_td is not None else "N/A"
        fcf_td_str = f"{result.fcf_to_td:.4f}" if result.fcf_to_td is not None else "N/A"
        dpb_str = f"{result.debt_payback_years:.2f}" if result.debt_payback_years is not None else "N/A"
        oi_str = f"{result.ocf_to_interest:.2f}" if result.ocf_to_interest is not None else "N/A"
        c1.metric("OCF/Total Debt", ocf_td_str)
        c2.metric("FCF/Total Debt", fcf_td_str)
        c3.metric("Debt Payback Yrs", dpb_str)
        c4.metric("OCF/Interest", oi_str)

        with st.expander("Cash Flow to Debt Details"):
            details: dict = {"Metric": [], "Value": []}
            details["Metric"].append("CF to Debt")
            cf_str = f"{result.cf_to_debt:.4f}" if result.cf_to_debt is not None else "N/A"
            details["Value"].append(cf_str)
            details["Metric"].append("CF Debt Spread")
            sp_str = f"{result.cf_debt_spread:.4f}" if result.cf_debt_spread is not None else "N/A"
            details["Value"].append(sp_str)
            details["Metric"].append("Score")
            details["Value"].append(f"{result.cfd_score}/10")
            details["Metric"].append("Grade")
            details["Value"].append(result.cfd_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_net_worth_growth(self, df: pd.DataFrame):
        """Phase 346: Net Worth Growth Analysis."""
        from financial_analyzer import CharlieAnalyzer, NetWorthGrowthResult

        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            data = self._row_to_financial_data(row)
            result = analyzer.net_worth_growth_analysis(data)

            grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
            color = grade_colors.get(result.nwg_grade, "gray")
            st.markdown(f"**Net Worth Growth Grade:** :{color}[{result.nwg_grade}]")

            col1, col2, col3, col4 = st.columns(4)
            rete_str = f"{result.re_to_equity:.4f}" if result.re_to_equity is not None else "N/A"
            col1.metric("RE/Equity", rete_str)
            col2.metric("NWG Score", f"{result.nwg_score:.1f}/10")
            ea_str = f"{result.equity_to_assets:.4f}" if result.equity_to_assets is not None else "N/A"
            col3.metric("Equity/Assets", ea_str)
            pb_str = f"{result.plowback_rate:.4f}" if result.plowback_rate is not None else "N/A"
            col4.metric("Plowback Rate", pb_str)

            with st.expander("Net Worth Growth Details"):
                details = {
                    "RE to Equity": result.re_to_equity,
                    "Equity to Assets": result.equity_to_assets,
                    "NI to Equity": result.ni_to_equity,
                    "Plowback Rate": result.plowback_rate,
                    "NW Spread": result.nw_spread,
                }
                for k, v in details.items():
                    val_str = f"{v:.4f}" if v is not None else "N/A"
                    st.write(f"- **{k}:** {val_str}")

            st.caption(result.summary)

    def _render_asset_lightness(self, df: pd.DataFrame):
        """Phase 341: Asset Lightness Analysis."""
        from financial_analyzer import CharlieAnalyzer, AssetLightnessResult
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = analyzer.asset_lightness_analysis(fd)
            grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
            color = grade_colors.get(result.alt_grade, "gray")
            st.markdown(f"**Asset Lightness Grade:** :{color}[{result.alt_grade}] ({result.alt_score:.1f}/10)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("CA/TA", f"{result.lightness_ratio:.4f}" if result.lightness_ratio is not None else "N/A")
            with col2:
                st.metric("Revenue/TA", f"{result.revenue_to_assets:.4f}" if result.revenue_to_assets is not None else "N/A")
            with col3:
                st.metric("Fixed Asset Ratio", f"{result.fixed_asset_ratio:.4f}" if result.fixed_asset_ratio is not None else "N/A")
            with col4:
                st.metric("Lightness Spread", f"{result.lightness_spread:+.4f}" if result.lightness_spread is not None else "N/A")
            with st.expander("Asset Lightness Details"):
                detail_data = {
                    "Metric": ["CA/TA", "Revenue/TA", "Fixed Asset Ratio", "Intangible Intensity", "Lightness Spread", "ALT Score", "ALT Grade"],
                    "Value": [
                        f"{result.ca_to_ta:.4f}" if result.ca_to_ta is not None else "N/A",
                        f"{result.revenue_to_assets:.4f}" if result.revenue_to_assets is not None else "N/A",
                        f"{result.fixed_asset_ratio:.4f}" if result.fixed_asset_ratio is not None else "N/A",
                        f"{result.intangible_intensity:.4f}" if result.intangible_intensity is not None else "N/A",
                        f"{result.lightness_spread:+.4f}" if result.lightness_spread is not None else "N/A",
                        f"{result.alt_score:.1f}/10",
                        result.alt_grade,
                    ],
                }
                st.table(detail_data)
            st.caption(result.summary)

    def _render_internal_growth_rate(self, df: pd.DataFrame):
        """Phase 337: Internal Growth Rate Analysis."""
        from financial_analyzer import CharlieAnalyzer, InternalGrowthRateResult
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = analyzer.internal_growth_rate_analysis(fd)
            grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
            color = grade_colors.get(result.igr_grade, "gray")
            st.markdown(f"**Internal Growth Rate Grade:** :{color}[{result.igr_grade}] ({result.igr_score:.1f}/10)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("IGR", f"{result.igr * 100:.2f}%" if result.igr is not None else "N/A")
            with col2:
                st.metric("ROA", f"{result.roa:.4f}" if result.roa is not None else "N/A")
            with col3:
                st.metric("Retention Ratio", f"{result.retention_ratio:.4f}" if result.retention_ratio is not None else "N/A")
            with col4:
                st.metric("Sust. Growth", f"{result.sustainable_growth * 100:.2f}%" if result.sustainable_growth is not None else "N/A")
            with st.expander("Internal Growth Rate Details"):
                detail_data = {
                    "Metric": ["IGR", "ROA", "Retention Ratio", "ROA*b", "Sustainable Growth", "Growth Capacity", "IGR Score", "IGR Grade"],
                    "Value": [
                        f"{result.igr * 100:.2f}%" if result.igr is not None else "N/A",
                        f"{result.roa:.4f}" if result.roa is not None else "N/A",
                        f"{result.retention_ratio:.4f}" if result.retention_ratio is not None else "N/A",
                        f"{result.roa_times_b:.4f}" if result.roa_times_b is not None else "N/A",
                        f"{result.sustainable_growth * 100:.2f}%" if result.sustainable_growth is not None else "N/A",
                        f"{result.growth_capacity * 100:.2f}%" if result.growth_capacity is not None else "N/A",
                        f"{result.igr_score:.1f}/10",
                        result.igr_grade,
                    ],
                }
                st.table(detail_data)
            st.caption(result.summary)

    def _render_operating_expense_ratio(self, df: pd.DataFrame):
        """Phase 330: Operating Expense Ratio Analysis."""
        from financial_analyzer import CharlieAnalyzer, OperatingExpenseRatioResult
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = analyzer.operating_expense_ratio_analysis(fd)
            grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
            color = grade_colors.get(result.oer_grade, "gray")
            st.markdown(f"**Operating Expense Ratio Grade:** :{color}[{result.oer_grade}] ({result.oer_score:.1f}/10)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("OpEx/Revenue", f"{result.opex_ratio:.4f}" if result.opex_ratio is not None else "N/A")
            with col2:
                st.metric("OpEx/GP", f"{result.opex_to_gross_profit:.4f}" if result.opex_to_gross_profit is not None else "N/A")
            with col3:
                st.metric("OpEx/EBITDA", f"{result.opex_to_ebitda:.4f}" if result.opex_to_ebitda is not None else "N/A")
            with col4:
                st.metric("OpEx Coverage", f"{result.opex_coverage:.2f}x" if result.opex_coverage is not None else "N/A")
            with st.expander("Operating Expense Ratio Details"):
                detail_data = {
                    "Metric": ["OpEx Ratio", "OpEx/GP", "OpEx/EBITDA", "OpEx Coverage", "Efficiency Gap", "OER Score", "OER Grade"],
                    "Value": [
                        f"{result.opex_ratio:.4f}" if result.opex_ratio is not None else "N/A",
                        f"{result.opex_to_gross_profit:.4f}" if result.opex_to_gross_profit is not None else "N/A",
                        f"{result.opex_to_ebitda:.4f}" if result.opex_to_ebitda is not None else "N/A",
                        f"{result.opex_coverage:.2f}x" if result.opex_coverage is not None else "N/A",
                        f"{result.efficiency_gap:.4f}" if result.efficiency_gap is not None else "N/A",
                        f"{result.oer_score:.1f}/10",
                        result.oer_grade,
                    ],
                }
                st.table(detail_data)
            st.caption(result.summary)

    def _render_noncurrent_asset_ratio(self, df: pd.DataFrame):
        """Phase 327: Noncurrent Asset Ratio Analysis."""
        from financial_analyzer import CharlieAnalyzer, NoncurrentAssetRatioResult
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = analyzer.noncurrent_asset_ratio_analysis(fd)

            grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
            color = grade_colors.get(result.nar_grade, "gray")
            st.markdown(f"**Noncurrent Asset Ratio Grade:** :{color}[{result.nar_grade}] ({result.nar_score:.1f}/10)")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("NCA Ratio", f"{result.nca_ratio:.4f}" if result.nca_ratio is not None else "N/A")
            col2.metric("Current Asset Ratio", f"{result.current_asset_ratio:.4f}" if result.current_asset_ratio is not None else "N/A")
            col3.metric("NCA/Equity", f"{result.nca_to_equity:.4f}" if result.nca_to_equity is not None else "N/A")
            col4.metric("Structure Spread", f"{result.asset_structure_spread:.4f}" if result.asset_structure_spread is not None else "N/A")

            with st.expander("Noncurrent Asset Ratio Details"):
                detail_data = {
                    "Metric": ["NCA Ratio", "Current Asset Ratio", "NCA/Equity", "NCA/Debt", "Structure Spread", "Liquidity Complement", "Score", "Grade"],
                    "Value": [
                        f"{result.nca_ratio:.4f}" if result.nca_ratio is not None else "N/A",
                        f"{result.current_asset_ratio:.4f}" if result.current_asset_ratio is not None else "N/A",
                        f"{result.nca_to_equity:.4f}" if result.nca_to_equity is not None else "N/A",
                        f"{result.nca_to_debt:.4f}" if result.nca_to_debt is not None else "N/A",
                        f"{result.asset_structure_spread:.4f}" if result.asset_structure_spread is not None else "N/A",
                        f"{result.liquidity_complement:.4f}" if result.liquidity_complement is not None else "N/A",
                        f"{result.nar_score:.1f}",
                        result.nar_grade,
                    ],
                }
                st.table(detail_data)

            st.caption(result.summary)

    def _render_payout_resilience(self, df: pd.DataFrame):
        """Phase 317: Payout Resilience Analysis."""
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = analyzer.payout_resilience_analysis(fd)

            grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.prs_grade, "gray")
            st.markdown(f"**Payout Resilience**: :{grade_color}[{result.prs_grade}] (Score: {result.prs_score:.1f}/10)")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Div/NI", f"{result.div_to_ni:.4f}" if result.div_to_ni is not None else "N/A")
            c2.metric("Div/OCF", f"{result.div_to_ocf:.4f}" if result.div_to_ocf is not None else "N/A")
            c3.metric("Payout Ratio", f"{result.payout_ratio:.4f}" if result.payout_ratio is not None else "N/A")
            c4.metric("Resilience Buffer", f"{result.resilience_buffer:.4f}" if result.resilience_buffer is not None else "N/A")

            with st.expander("Payout Resilience Details"):
                detail_data = {
                    "Metric": ["Div/NI", "Div/OCF", "Div/Revenue", "Div/EBITDA", "Payout Ratio", "Resilience Buffer"],
                    "Value": [
                        f"{result.div_to_ni:.4f}" if result.div_to_ni is not None else "N/A",
                        f"{result.div_to_ocf:.4f}" if result.div_to_ocf is not None else "N/A",
                        f"{result.div_to_revenue:.4f}" if result.div_to_revenue is not None else "N/A",
                        f"{result.div_to_ebitda:.4f}" if result.div_to_ebitda is not None else "N/A",
                        f"{result.payout_ratio:.4f}" if result.payout_ratio is not None else "N/A",
                        f"{result.resilience_buffer:.4f}" if result.resilience_buffer is not None else "N/A",
                    ],
                }
                st.table(detail_data)
            st.caption(result.summary)

    def _render_debt_burden_index(self, df: pd.DataFrame):
        """Phase 314: Debt Burden Index Analysis."""
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = analyzer.debt_burden_index_analysis(fd)

            grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.dbi_grade, "gray")
            st.markdown(f"**Debt Burden Index**: :{grade_color}[{result.dbi_grade}] (Score: {result.dbi_score:.1f}/10)")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Debt/EBITDA", f"{result.debt_to_ebitda:.4f}" if result.debt_to_ebitda is not None else "N/A")
            c2.metric("Debt/Assets", f"{result.debt_to_assets:.4f}" if result.debt_to_assets is not None else "N/A")
            c3.metric("Debt/Equity", f"{result.debt_to_equity:.4f}" if result.debt_to_equity is not None else "N/A")
            c4.metric("Debt/Revenue", f"{result.debt_to_revenue:.4f}" if result.debt_to_revenue is not None else "N/A")

            with st.expander("Debt Burden Index Details"):
                detail_data = {
                    "Metric": ["Debt/EBITDA", "Debt/Assets", "Debt/Equity", "Debt/Revenue", "Debt Ratio", "Burden Intensity"],
                    "Value": [
                        f"{result.debt_to_ebitda:.4f}" if result.debt_to_ebitda is not None else "N/A",
                        f"{result.debt_to_assets:.4f}" if result.debt_to_assets is not None else "N/A",
                        f"{result.debt_to_equity:.4f}" if result.debt_to_equity is not None else "N/A",
                        f"{result.debt_to_revenue:.4f}" if result.debt_to_revenue is not None else "N/A",
                        f"{result.debt_ratio:.4f}" if result.debt_ratio is not None else "N/A",
                        f"{result.burden_intensity:.4f}" if result.burden_intensity is not None else "N/A",
                    ],
                }
                st.table(detail_data)
            st.caption(result.summary)

    def _render_inventory_coverage(self, df: pd.DataFrame):
        """Phase 309: Inventory Coverage Analysis."""
        analyzer = CharlieAnalyzer()
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.inventory_coverage_analysis(data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.icv_grade, "gray")
        st.markdown(f"**Inventory Coverage Grade:** :{color}[{result.icv_grade}] ({result.icv_score:.1f}/10)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Inv/COGS", f"{result.inventory_to_cogs:.2%}" if result.inventory_to_cogs is not None else "N/A")
        col2.metric("Inv/Revenue", f"{result.inventory_to_revenue:.2%}" if result.inventory_to_revenue is not None else "N/A")
        col3.metric("Inv Days", f"{result.inventory_days:.1f}" if result.inventory_days is not None else "N/A")
        col4.metric("Inv/Assets", f"{result.inventory_to_assets:.2%}" if result.inventory_to_assets is not None else "N/A")

        with st.expander("Inventory Coverage Details"):
            details = {
                "Inv/COGS": result.inventory_to_cogs,
                "Inv/Revenue": result.inventory_to_revenue,
                "Inv/Assets": result.inventory_to_assets,
                "Inv/Current Assets": result.inventory_to_current_assets,
                "Inventory Days": result.inventory_days,
                "Inventory Buffer": result.inventory_buffer,
            }
            st.table({k: f"{v:.4f}" if v is not None else "N/A" for k, v in details.items()})

            st.caption(result.summary)

    def _render_capex_to_revenue(self, df: pd.DataFrame):
        """Render Phase 307: CapEx to Revenue Analysis."""
        from financial_analyzer import CharlieAnalyzer, CapexToRevenueResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.capex_to_revenue_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.ctr_grade, "gray")
        st.markdown(f"**CapEx to Revenue Grade:** :{color}[{result.ctr_grade}] ({result.ctr_score:.1f}/10)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CapEx/Rev", f"{result.capex_to_revenue:.2%}" if result.capex_to_revenue is not None else "N/A")
        col2.metric("CapEx/OCF", f"{result.capex_to_ocf:.2f}" if result.capex_to_ocf is not None else "N/A")
        col3.metric("CapEx/EBITDA", f"{result.capex_to_ebitda:.2f}" if result.capex_to_ebitda is not None else "N/A")
        col4.metric("CapEx Yield", f"{result.capex_yield:.1f}x" if result.capex_yield is not None else "N/A")

        with st.expander("CapEx to Revenue Details"):
            details = {
                "CapEx/Revenue": result.capex_to_revenue,
                "CapEx/OCF": result.capex_to_ocf,
                "CapEx/EBITDA": result.capex_to_ebitda,
                "CapEx/Assets": result.capex_to_assets,
                "Investment Intensity": result.investment_intensity,
                "CapEx Yield": result.capex_yield,
            }
            detail_df = pd.DataFrame([
                {"Metric": k, "Value": f"{v:.4f}" if v is not None else "N/A"}
                for k, v in details.items()
            ])
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

        st.caption(result.summary)

    def _render_inventory_holding_cost(self, df: pd.DataFrame):
        """Phase 294: Inventory Holding Cost tab."""
        from financial_analyzer import CharlieAnalyzer
        analyzer = CharlieAnalyzer()
        fd = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.inventory_holding_cost_analysis(fd)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.ihc_grade, "gray")
        st.markdown(f"**Inventory Holding Cost Grade:** :{color}[{result.ihc_grade}] ({result.ihc_score:.1f}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Inv/Revenue", f"{result.inventory_to_revenue:.2f}" if result.inventory_to_revenue is not None else "N/A")
        c2.metric("Inv/Current Assets", f"{result.inventory_to_current_assets:.2f}" if result.inventory_to_current_assets is not None else "N/A")
        c3.metric("Inventory Days", f"{result.inventory_days:.1f}" if result.inventory_days is not None else "N/A")
        c4.metric("Inv/Total Assets", f"{result.inventory_to_total_assets:.2f}" if result.inventory_to_total_assets is not None else "N/A")

        with st.expander("Details"):
            details = {"Metric": [], "Value": []}
            for field_name in ["inventory_to_revenue", "inventory_to_current_assets", "inventory_to_total_assets", "inventory_days", "inventory_intensity", "ihc_score", "ihc_grade"]:
                val = getattr(result, field_name)
                details["Metric"].append(field_name)
                if isinstance(val, float):
                    details["Value"].append(f"{val:.4f}")
                else:
                    details["Value"].append(str(val) if val is not None else "N/A")
            st.table(details)

        st.caption(result.summary)

    def _render_funding_mix_balance(self, df: pd.DataFrame):
        """Phase 293: Funding Mix Balance tab."""
        from financial_analyzer import CharlieAnalyzer
        analyzer = CharlieAnalyzer()
        fd = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.funding_mix_balance_analysis(fd)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.fmb_grade, "gray")
        st.markdown(f"**Funding Mix Balance Grade:** :{color}[{result.fmb_grade}] ({result.fmb_score:.1f}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equity/Total Capital", f"{result.equity_to_total_capital:.2f}" if result.equity_to_total_capital is not None else "N/A")
        c2.metric("Debt/Equity", f"{result.debt_to_equity:.2f}" if result.debt_to_equity is not None else "N/A")
        c3.metric("Debt/Total Capital", f"{result.debt_to_total_capital:.2f}" if result.debt_to_total_capital is not None else "N/A")
        c4.metric("Leverage Headroom", f"{result.leverage_headroom:.2f}" if result.leverage_headroom is not None else "N/A")

        with st.expander("Details"):
            details = {"Metric": [], "Value": []}
            for field_name in ["equity_to_total_capital", "debt_to_equity", "debt_to_total_capital", "equity_multiplier", "leverage_headroom", "funding_stability", "fmb_score", "fmb_grade"]:
                val = getattr(result, field_name)
                details["Metric"].append(field_name)
                if isinstance(val, float):
                    details["Value"].append(f"{val:.4f}")
                else:
                    details["Value"].append(str(val) if val is not None else "N/A")
            st.table(details)

        st.caption(result.summary)

    def _render_expense_ratio_discipline(self, df: pd.DataFrame):
        """Phase 292: Expense Ratio Discipline tab."""
        from financial_analyzer import CharlieAnalyzer
        analyzer = CharlieAnalyzer()
        fd = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.expense_ratio_discipline_analysis(fd)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.erd_grade, "gray")
        st.markdown(f"**Expense Ratio Discipline Grade:** :{color}[{result.erd_grade}] ({result.erd_score:.1f}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OpEx/Revenue", f"{result.opex_to_revenue:.2f}" if result.opex_to_revenue is not None else "N/A")
        c2.metric("COGS/Revenue", f"{result.cogs_to_revenue:.2f}" if result.cogs_to_revenue is not None else "N/A")
        c3.metric("Total Expense Ratio", f"{result.total_expense_ratio:.2f}" if result.total_expense_ratio is not None else "N/A")
        c4.metric("Operating Margin", f"{result.operating_margin:.2f}" if result.operating_margin is not None else "N/A")

        with st.expander("Details"):
            details = {"Metric": [], "Value": []}
            for field_name in ["opex_to_revenue", "cogs_to_revenue", "total_expense_ratio", "operating_margin", "expense_efficiency", "erd_score", "erd_grade"]:
                val = getattr(result, field_name)
                details["Metric"].append(field_name)
                if isinstance(val, float):
                    details["Value"].append(f"{val:.4f}")
                else:
                    details["Value"].append(str(val) if val is not None else "N/A")
            st.table(details)

        st.caption(result.summary)

    def _render_revenue_cash_realization(self, df: pd.DataFrame):
        """Phase 291: Revenue Cash Realization tab."""
        from financial_analyzer import CharlieAnalyzer
        analyzer = CharlieAnalyzer()
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.revenue_cash_realization_analysis(data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.rcr_grade, "gray")
        st.markdown(f"**Revenue Cash Realization Grade:** :{color}[{result.rcr_grade}] ({result.rcr_score:.1f}/10)")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("OCF/Revenue", f"{result.ocf_to_revenue:.1%}" if result.ocf_to_revenue is not None else "N/A")
        with col2:
            st.metric("Collection Rate", f"{result.collection_rate:.1%}" if result.collection_rate is not None else "N/A")
        with col3:
            st.metric("Cash/Revenue", f"{result.cash_to_revenue:.1%}" if result.cash_to_revenue is not None else "N/A")
        with col4:
            st.metric("Rev-Cash Gap", f"${result.revenue_cash_gap:,.0f}" if result.revenue_cash_gap is not None else "N/A")

        with st.expander("Revenue Cash Realization Details"):
            details = {"Metric": [], "Value": []}
            for label, val in [
                ("OCF/Revenue", result.ocf_to_revenue),
                ("Cash/Revenue", result.cash_to_revenue),
                ("Collection Rate", result.collection_rate),
                ("Revenue-Cash Gap", result.revenue_cash_gap),
                ("Cash Conversion Speed", result.cash_conversion_speed),
                ("Revenue Quality Ratio", result.revenue_quality_ratio),
            ]:
                details["Metric"].append(label)
                if val is None:
                    details["Value"].append("N/A")
                elif abs(val) >= 1000:
                    details["Value"].append(f"${val:,.0f}")
                else:
                    details["Value"].append(f"{val:.4f}")
            details["Metric"].append("Grade")
            details["Value"].append(result.rcr_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_net_debt_position(self, df: pd.DataFrame):
        """Phase 286: Net Debt Position tab."""
        from financial_analyzer import CharlieAnalyzer, NetDebtPositionResult
        analyzer = CharlieAnalyzer()
        fin = self._extract_financial_data(df)
        result = analyzer.net_debt_position_analysis(fin)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.ndp_grade, "gray")
        st.markdown(f"**Net Debt Position Grade:** :{color}[{result.ndp_grade}] (Score: {result.ndp_score:.1f}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net Debt", f"${result.net_debt:,.0f}" if result.net_debt is not None else "N/A")
        c2.metric("Net Debt/EBITDA", f"{result.net_debt_to_ebitda:.2f}x" if result.net_debt_to_ebitda is not None else "N/A")
        c3.metric("Cash/Debt", f"{result.cash_to_debt:.2%}" if result.cash_to_debt is not None else "N/A")
        c4.metric("Net Debt/Equity", f"{result.net_debt_to_equity:.2f}x" if result.net_debt_to_equity is not None else "N/A")

        with st.expander("Net Debt Position Details"):
            detail = {
                "Net Debt (Debt - Cash)": f"${result.net_debt:,.0f}" if result.net_debt is not None else "N/A",
                "Net Debt / EBITDA": f"{result.net_debt_to_ebitda:.4f}" if result.net_debt_to_ebitda is not None else "N/A",
                "Net Debt / Equity": f"{result.net_debt_to_equity:.4f}" if result.net_debt_to_equity is not None else "N/A",
                "Net Debt / Total Assets": f"{result.net_debt_to_assets:.4f}" if result.net_debt_to_assets is not None else "N/A",
                "Cash / Debt": f"{result.cash_to_debt:.4f}" if result.cash_to_debt is not None else "N/A",
                "Net Debt / OCF": f"{result.net_debt_to_ocf:.4f}" if result.net_debt_to_ocf is not None else "N/A",
            }
            st.table(detail)

        if result.summary:
            st.caption(result.summary)

    def _render_liability_coverage_strength(self, df: pd.DataFrame):
        """Phase 281: Liability Coverage Strength tab."""
        from financial_analyzer import LiabilityCoverageStrengthResult

        for _, row in df.iterrows():
            analyzer = row.get("analyzer")
            if not analyzer:
                continue
            data = row.get("financial_data")
            if not data:
                continue
            result = analyzer.liability_coverage_strength_analysis(data)
            if not isinstance(result, LiabilityCoverageStrengthResult):
                continue

            grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.lcs_grade, "gray")
            st.markdown(f"**Liability Coverage Strength** &mdash; :{grade_color}[{result.lcs_grade}] (Score: {result.lcs_score:.1f}/10)")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("OCF/Liabilities", f"{result.ocf_to_liabilities:.2%}" if result.ocf_to_liabilities is not None else "N/A")
            c2.metric("Assets/Liabilities", f"{result.assets_to_liabilities:.2f}x" if result.assets_to_liabilities is not None else "N/A")
            c3.metric("Equity/Liabilities", f"{result.equity_to_liabilities:.2f}x" if result.equity_to_liabilities is not None else "N/A")
            c4.metric("Liability Burden", f"{result.liability_burden:.2%}" if result.liability_burden is not None else "N/A")

            with st.expander("Liability Coverage Strength Details"):
                details = {"Metric": [], "Value": []}
                for label, val, fmt in [
                    ("OCF/Liabilities", result.ocf_to_liabilities, ".2%"),
                    ("EBITDA/Liabilities", result.ebitda_to_liabilities, ".2%"),
                    ("Assets/Liabilities", result.assets_to_liabilities, ".2f"),
                    ("Equity/Liabilities", result.equity_to_liabilities, ".2f"),
                    ("Liabilities/Revenue", result.liability_to_revenue, ".2f"),
                    ("Liability Burden", result.liability_burden, ".2%"),
                ]:
                    details["Metric"].append(label)
                    details["Value"].append(f"{val:{fmt}}" if val is not None else "N/A")
                details["Metric"].append("Score")
                details["Value"].append(f"{result.lcs_score:.1f}")
                details["Metric"].append("Grade")
                details["Value"].append(result.lcs_grade)
                st.table(details)

            st.caption(result.summary)

    def _render_capital_adequacy(self, df: pd.DataFrame):
        """Phase 279: Capital Adequacy tab."""
        from financial_analyzer import CapitalAdequacyResult

        for _, row in df.iterrows():
            analyzer = row.get("analyzer")
            if not analyzer:
                continue
            data = row.get("financial_data")
            if not data:
                continue
            result = analyzer.capital_adequacy_analysis(data)
            if not isinstance(result, CapitalAdequacyResult):
                continue

            grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.caq_grade, "gray")
            st.markdown(f"**Capital Adequacy** &mdash; :{grade_color}[{result.caq_grade}] (Score: {result.caq_score:.1f}/10)")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Equity Ratio", f"{result.equity_ratio:.2%}" if result.equity_ratio is not None else "N/A")
            c2.metric("Equity/Debt", f"{result.equity_to_debt:.2f}x" if result.equity_to_debt is not None else "N/A")
            c3.metric("RE/Equity", f"{result.retained_to_equity:.2%}" if result.retained_to_equity is not None else "N/A")
            c4.metric("Capital Buffer", f"{result.capital_buffer:.2%}" if result.capital_buffer is not None else "N/A")

            with st.expander("Capital Adequacy Details"):
                details = {"Metric": [], "Value": []}
                for label, val, fmt in [
                    ("Equity Ratio", result.equity_ratio, ".2%"),
                    ("Equity/Debt", result.equity_to_debt, ".2f"),
                    ("RE/Equity", result.retained_to_equity, ".2%"),
                    ("Equity/Liabilities", result.equity_to_liabilities, ".2f"),
                    ("Tangible Equity Ratio", result.tangible_equity_ratio, ".2%"),
                    ("Capital Buffer", result.capital_buffer, ".2%"),
                ]:
                    details["Metric"].append(label)
                    details["Value"].append(f"{val:{fmt}}" if val is not None else "N/A")
                details["Metric"].append("Score")
                details["Value"].append(f"{result.caq_score:.1f}")
                details["Metric"].append("Grade")
                details["Value"].append(result.caq_grade)
                st.table(details)

            st.caption(result.summary)

    def _render_operating_income_quality(self, df: pd.DataFrame):
        """Phase 275: Operating Income Quality tab."""
        from financial_analyzer import OperatingIncomeQualityResult

        for _, row in df.iterrows():
            data = self._row_to_financial_data(row)
            result = self.analyzer.operating_income_quality_analysis(data)

            if result.oi_to_revenue is None and result.oiq_score == 0.0:
                st.info("Insufficient data for Operating Income Quality analysis.")
                return

            badge_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.oiq_grade, "gray")
            st.markdown(f"**Operating Income Quality Grade:** :{badge_color}[{result.oiq_grade}] ({result.oiq_score:.1f}/10)")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("OI Margin", f"{result.oi_to_revenue:.2%}" if result.oi_to_revenue is not None else "N/A")
            c2.metric("OI/EBITDA", f"{result.oi_to_ebitda:.2f}" if result.oi_to_ebitda is not None else "N/A")
            c3.metric("OI/OCF", f"{result.oi_to_ocf:.2f}" if result.oi_to_ocf is not None else "N/A")
            c4.metric("OI/Assets", f"{result.oi_to_total_assets:.2f}" if result.oi_to_total_assets is not None else "N/A")

            with st.expander("Operating Income Quality Details"):
                details = {"Metric": [], "Value": []}
                for label, val, fmt in [
                    ("OI / Revenue", result.oi_to_revenue, ".4f"),
                    ("OI / EBITDA", result.oi_to_ebitda, ".4f"),
                    ("OI / OCF", result.oi_to_ocf, ".4f"),
                    ("OI / Total Assets", result.oi_to_total_assets, ".4f"),
                    ("Operating Spread", result.operating_spread, ".4f"),
                    ("OI Cash Backing (OCF/OI)", result.oi_cash_backing, ".4f"),
                ]:
                    details["Metric"].append(label)
                    details["Value"].append(f"{val:{fmt}}" if val is not None else "N/A")
                details["Metric"].append("Score")
                details["Value"].append(f"{result.oiq_score:.1f}")
                details["Metric"].append("Grade")
                details["Value"].append(result.oiq_grade)
                st.table(details)

            st.caption(result.summary)

    def _render_debt_quality(self, df: pd.DataFrame):
        """Phase 267: Debt Quality tab."""
        from financial_analyzer import DebtQualityResult
        analyzer = self.analyzer
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.debt_quality_analysis(data)
        if not isinstance(result, DebtQualityResult):
            st.warning("Debt Quality analysis unavailable.")
            return
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.dq_grade, "gray")
        st.markdown(f"### Debt Quality &mdash; :{color}[{result.dq_grade}] ({result.dq_score:.1f}/10)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("D/E Ratio", f"{result.debt_to_equity:.2f}" if result.debt_to_equity is not None else "N/A")
        c2.metric("D/Assets", f"{result.debt_to_assets:.2%}" if result.debt_to_assets is not None else "N/A")
        c3.metric("D/EBITDA", f"{result.debt_to_ebitda:.2f}x" if result.debt_to_ebitda is not None else "N/A")
        c4.metric("Interest Cov", f"{result.interest_coverage:.2f}x" if result.interest_coverage is not None else "N/A")
        with st.expander("Detail"):
            details = {"Metric": [], "Value": []}
            for label, val, fmt in [
                ("Debt-to-Equity", result.debt_to_equity, ".2f"),
                ("Debt-to-Assets", result.debt_to_assets, ".2%"),
                ("LT Debt Ratio", result.long_term_debt_ratio, ".2%"),
                ("Debt/EBITDA", result.debt_to_ebitda, ".2f"),
                ("Interest Coverage", result.interest_coverage, ".2f"),
                ("Debt Cost", result.debt_cost, ".2%"),
            ]:
                details["Metric"].append(label)
                details["Value"].append(f"{val:{fmt}}" if val is not None else "N/A")
            details["Metric"].append("Score / Grade")
            details["Value"].append(f"{result.dq_score:.1f} / {result.dq_grade}")
            st.table(details)
        st.caption(result.summary)

    def _render_depreciation_burden(self, df: pd.DataFrame):
        """Phase 259: Depreciation Burden tab."""
        from financial_analyzer import DepreciationBurdenResult
        st.subheader("Depreciation Burden Analysis")
        for _, row in df.iterrows():
            data = self._row_to_financial_data(row)
            result = self.analyzer.depreciation_burden_analysis(data)

            color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.db_grade, "gray")
            st.markdown(f"**Grade:** :{color}[{result.db_grade}] | **Score:** {result.db_score:.1f}/10")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("D&A/Revenue", f"{result.dep_to_revenue:.1%}" if result.dep_to_revenue is not None else "N/A")
            c2.metric("D&A/EBITDA", f"{result.dep_to_ebitda:.2f}" if result.dep_to_ebitda is not None else "N/A")
            c3.metric("D&A/Assets", f"{result.dep_to_assets:.2%}" if result.dep_to_assets is not None else "N/A")
            c4.metric("D&A/Gross Profit", f"{result.dep_to_gross_profit:.2f}" if result.dep_to_gross_profit is not None else "N/A")

            with st.expander("Details"):
                details = {
                    "D&A/Revenue": f"{result.dep_to_revenue:.2%}" if result.dep_to_revenue is not None else "N/A",
                    "D&A/EBITDA": f"{result.dep_to_ebitda:.2f}" if result.dep_to_ebitda is not None else "N/A",
                    "D&A/Assets": f"{result.dep_to_assets:.2%}" if result.dep_to_assets is not None else "N/A",
                    "EBITDA/EBIT Spread": f"{result.ebitda_to_ebit_spread:.2f}" if result.ebitda_to_ebit_spread is not None else "N/A",
                    "Asset Age Proxy": f"{result.asset_age_proxy:.4f}" if result.asset_age_proxy is not None else "N/A",
                }
                st.table(details)

            st.caption(result.summary)

    def _render_debt_to_capital(self, df: pd.DataFrame):
        """Phase 258: Debt-to-Capital tab."""
        from financial_analyzer import DebtToCapitalResult
        st.subheader("Debt-to-Capital Analysis")
        for _, row in df.iterrows():
            data = self._row_to_financial_data(row)
            result = self.analyzer.debt_to_capital_analysis(data)

            color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.dtc_grade, "gray")
            st.markdown(f"**Grade:** :{color}[{result.dtc_grade}] | **Score:** {result.dtc_score:.1f}/10")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Debt/Capital", f"{result.debt_to_capital:.1%}" if result.debt_to_capital is not None else "N/A")
            c2.metric("Debt/Equity", f"{result.debt_to_equity:.2f}" if result.debt_to_equity is not None else "N/A")
            c3.metric("Equity Ratio", f"{result.equity_ratio:.1%}" if result.equity_ratio is not None else "N/A")
            c4.metric("Net D/Capital", f"{result.net_debt_to_capital:.1%}" if result.net_debt_to_capital is not None else "N/A")

            with st.expander("Details"):
                details = {
                    "Debt/Capital": f"{result.debt_to_capital:.2%}" if result.debt_to_capital is not None else "N/A",
                    "Debt/Equity": f"{result.debt_to_equity:.2f}" if result.debt_to_equity is not None else "N/A",
                    "Equity Ratio": f"{result.equity_ratio:.2%}" if result.equity_ratio is not None else "N/A",
                    "Net Debt/Capital": f"{result.net_debt_to_capital:.2%}" if result.net_debt_to_capital is not None else "N/A",
                    "Financial Risk Index": f"{result.financial_risk_index:.4f}" if result.financial_risk_index is not None else "N/A",
                }
                st.table(details)

            st.caption(result.summary)

    def _render_operating_leverage(self, df: pd.DataFrame):
        """Phase 253: Operating Leverage tab."""
        from financial_analyzer import CharlieAnalyzer, OperatingLeverageResult
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = analyzer.operating_leverage_analysis(fd)
            grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.ol_grade, "gray")
            st.markdown(f"**Operating Leverage:** :{grade_color}[{result.ol_grade}] ({result.ol_score:.1f}/10)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("DOL", f"{result.dol:.2f}" if result.dol is not None else "N/A")
            c2.metric("Margin of Safety", f"{result.margin_of_safety:.1%}" if result.margin_of_safety is not None else "N/A")
            c3.metric("Fixed Cost Ratio", f"{result.fixed_cost_ratio:.1%}" if result.fixed_cost_ratio is not None else "N/A")
            c4.metric("Breakeven Rev", f"${result.breakeven_revenue:,.0f}" if result.breakeven_revenue is not None else "N/A")
            with st.expander("Operating Leverage Details"):
                detail = {
                    "DOL": f"{result.dol:.4f}" if result.dol is not None else "N/A",
                    "Variable Cost Ratio": f"{result.variable_cost_ratio:.4f}" if result.variable_cost_ratio is not None else "N/A",
                    "Contribution Margin": f"{result.contribution_margin_ratio:.4f}" if result.contribution_margin_ratio is not None else "N/A",
                    "Fixed Cost Ratio": f"{result.fixed_cost_ratio:.4f}" if result.fixed_cost_ratio is not None else "N/A",
                    "Breakeven Revenue": f"${result.breakeven_revenue:,.0f}" if result.breakeven_revenue is not None else "N/A",
                    "Margin of Safety": f"{result.margin_of_safety:.4f}" if result.margin_of_safety is not None else "N/A",
                }
                st.table(detail)
            st.caption(result.summary)

    def _render_dividend_payout(self, df: pd.DataFrame):
        """Phase 251: Dividend Payout tab."""
        from financial_analyzer import CharlieAnalyzer, DividendPayoutResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.dividend_payout_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.dpr_grade, "gray")
        st.markdown(f"**Dividend Payout** &mdash; :{color}[{result.dpr_grade}] ({result.dpr_score:.1f}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Div / NI", f"{result.div_to_ni:.1%}" if result.div_to_ni is not None else "N/A")
        c2.metric("Retention", f"{result.retention_ratio:.1%}" if result.retention_ratio is not None else "N/A")
        c3.metric("Div Coverage", f"{result.div_coverage:.2f}x" if result.div_coverage is not None else "N/A")
        c4.metric("Div / OCF", f"{result.div_to_ocf:.1%}" if result.div_to_ocf is not None else "N/A")

        with st.expander("Dividend Payout Details"):
            detail = {
                "Div / Net Income": result.div_to_ni,
                "Retention Ratio": result.retention_ratio,
                "Div / OCF": result.div_to_ocf,
                "Div / FCF": result.div_to_fcf,
                "Div / Revenue": result.div_to_revenue,
                "Dividend Coverage": result.div_coverage,
            }
            st.table({k: f"{v:.4f}" if v is not None else "N/A" for k, v in detail.items()})

        st.caption(result.summary)

    def _render_operating_cash_flow_ratio(self, df: pd.DataFrame):
        """Phase 249: Operating Cash Flow Ratio tab."""
        from financial_analyzer import CharlieAnalyzer, OperatingCashFlowRatioResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.operating_cash_flow_ratio_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.ocfr_grade, "gray")
        st.markdown(f"**Operating Cash Flow Ratio** &mdash; :{color}[{result.ocfr_grade}] ({result.ocfr_score:.1f}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OCF / CL", f"{result.ocf_to_cl:.2f}x" if result.ocf_to_cl is not None else "N/A")
        c2.metric("OCF / TL", f"{result.ocf_to_tl:.2f}x" if result.ocf_to_tl is not None else "N/A")
        c3.metric("OCF / Revenue", f"{result.ocf_to_revenue:.1%}" if result.ocf_to_revenue is not None else "N/A")
        c4.metric("OCF / NI", f"{result.ocf_to_ni:.2f}x" if result.ocf_to_ni is not None else "N/A")

        with st.expander("Operating Cash Flow Ratio Details"):
            detail = {
                "OCF / CL": result.ocf_to_cl,
                "OCF / TL": result.ocf_to_tl,
                "OCF / Revenue": result.ocf_to_revenue,
                "OCF / Net Income": result.ocf_to_ni,
                "OCF / Total Debt": result.ocf_to_debt,
                "OCF Margin": result.ocf_margin,
            }
            st.table({k: f"{v:.4f}" if v is not None else "N/A" for k, v in detail.items()})

        st.caption(result.summary)

    def _render_cash_conversion_cycle(self, df: pd.DataFrame):
        """Phase 248: Cash Conversion Cycle tab."""
        from financial_analyzer import CharlieAnalyzer, CashConversionCycleResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.cash_conversion_cycle_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.ccc_grade, "gray")
        st.markdown(f"**Cash Conversion Cycle** &mdash; :{color}[{result.ccc_grade}] ({result.ccc_score:.1f}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CCC (days)", f"{result.ccc:.1f}" if result.ccc is not None else "N/A")
        c2.metric("DSO (days)", f"{result.dso:.1f}" if result.dso is not None else "N/A")
        c3.metric("DIO (days)", f"{result.dio:.1f}" if result.dio is not None else "N/A")
        c4.metric("DPO (days)", f"{result.dpo:.1f}" if result.dpo is not None else "N/A")

        with st.expander("Cash Conversion Cycle Details"):
            detail = {
                "CCC (days)": result.ccc,
                "DSO (days)": result.dso,
                "DIO (days)": result.dio,
                "DPO (days)": result.dpo,
                "CCC / 365": result.ccc_to_revenue,
                "Working Capital Days": result.working_cap_days,
            }
            st.table({k: f"{v:.2f}" if v is not None else "N/A" for k, v in detail.items()})

        st.caption(result.summary)

    def _render_inventory_turnover(self, df: pd.DataFrame):
        """Phase 247: Inventory Turnover tab."""
        from financial_analyzer import CharlieAnalyzer, InventoryTurnoverResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.inventory_turnover_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.ito_grade, "gray")
        st.markdown(f"**Inventory Turnover Grade:** :{color}[{result.ito_grade}] ({result.ito_score:.1f}/10)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("COGS/Inv", f"{result.cogs_to_inv:.2f}x" if result.cogs_to_inv is not None else "N/A")
        col2.metric("DIO", f"{result.dio:.0f} days" if result.dio is not None else "N/A")
        col3.metric("Inv/CA", f"{result.inv_to_ca:.2%}" if result.inv_to_ca is not None else "N/A")
        col4.metric("Inv/Rev", f"{result.inv_to_revenue:.2%}" if result.inv_to_revenue is not None else "N/A")

        with st.expander("Inventory Turnover Details"):
            detail_data = {
                "Metric": ["COGS/Inv", "DIO (days)", "Inv/CA", "Inv/TA", "Inv/Revenue", "Inv Velocity"],
                "Value": [
                    f"{result.cogs_to_inv:.4f}" if result.cogs_to_inv is not None else "N/A",
                    f"{result.dio:.1f}" if result.dio is not None else "N/A",
                    f"{result.inv_to_ca:.4f}" if result.inv_to_ca is not None else "N/A",
                    f"{result.inv_to_ta:.4f}" if result.inv_to_ta is not None else "N/A",
                    f"{result.inv_to_revenue:.4f}" if result.inv_to_revenue is not None else "N/A",
                    f"{result.inv_velocity:.4f}" if result.inv_velocity is not None else "N/A",
                ],
            }
            st.table(detail_data)

        st.caption(result.summary)

    def _render_payables_turnover(self, df: pd.DataFrame):
        """Phase 246: Payables Turnover tab."""
        from financial_analyzer import CharlieAnalyzer, PayablesTurnoverResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.payables_turnover_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.pto_grade, "gray")
        st.markdown(f"**Payables Turnover Grade:** :{color}[{result.pto_grade}] ({result.pto_score:.1f}/10)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("COGS/AP", f"{result.cogs_to_ap:.2f}x" if result.cogs_to_ap is not None else "N/A")
        col2.metric("DPO", f"{result.dpo:.0f} days" if result.dpo is not None else "N/A")
        col3.metric("AP/CL", f"{result.ap_to_cl:.2%}" if result.ap_to_cl is not None else "N/A")
        col4.metric("AP/COGS", f"{result.ap_to_cogs:.2%}" if result.ap_to_cogs is not None else "N/A")

        with st.expander("Payables Turnover Details"):
            detail_data = {
                "Metric": ["COGS/AP", "DPO (days)", "AP/CL", "AP/TL", "AP/COGS", "Payment Velocity"],
                "Value": [
                    f"{result.cogs_to_ap:.4f}" if result.cogs_to_ap is not None else "N/A",
                    f"{result.dpo:.1f}" if result.dpo is not None else "N/A",
                    f"{result.ap_to_cl:.4f}" if result.ap_to_cl is not None else "N/A",
                    f"{result.ap_to_tl:.4f}" if result.ap_to_tl is not None else "N/A",
                    f"{result.ap_to_cogs:.4f}" if result.ap_to_cogs is not None else "N/A",
                    f"{result.payment_velocity:.4f}" if result.payment_velocity is not None else "N/A",
                ],
            }
            st.table(detail_data)

        st.caption(result.summary)

    def _render_receivables_turnover(self, df: pd.DataFrame):
        """Phase 245: Receivables Turnover tab."""
        from financial_analyzer import CharlieAnalyzer, ReceivablesTurnoverResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.receivables_turnover_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.rto_grade, "gray")
        st.markdown(f"**Receivables Turnover Grade:** :{color}[{result.rto_grade}] ({result.rto_score:.1f}/10)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rev/AR", f"{result.rev_to_ar:.2f}x" if result.rev_to_ar is not None else "N/A")
        col2.metric("DSO", f"{result.dso:.0f} days" if result.dso is not None else "N/A")
        col3.metric("AR/Revenue", f"{result.ar_to_revenue:.2%}" if result.ar_to_revenue is not None else "N/A")
        col4.metric("Collect Eff", f"{result.collection_efficiency:.2%}" if result.collection_efficiency is not None else "N/A")

        with st.expander("Receivables Turnover Details"):
            detail_data = {
                "Metric": ["Rev/AR", "DSO (days)", "AR/CA", "AR/TA", "AR/Revenue", "Collection Efficiency"],
                "Value": [
                    f"{result.rev_to_ar:.4f}" if result.rev_to_ar is not None else "N/A",
                    f"{result.dso:.1f}" if result.dso is not None else "N/A",
                    f"{result.ar_to_ca:.4f}" if result.ar_to_ca is not None else "N/A",
                    f"{result.ar_to_ta:.4f}" if result.ar_to_ta is not None else "N/A",
                    f"{result.ar_to_revenue:.4f}" if result.ar_to_revenue is not None else "N/A",
                    f"{result.collection_efficiency:.4f}" if result.collection_efficiency is not None else "N/A",
                ],
            }
            st.table(detail_data)

        st.caption(result.summary)

    def _render_cash_conversion_efficiency(self, df: pd.DataFrame):
        """Phase 237: Cash Conversion Efficiency tab."""
        from financial_analyzer import CharlieAnalyzer, CashConversionEfficiencyResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.cash_conversion_efficiency_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.cce_grade, "gray")
        st.markdown(f"**Cash Conversion Efficiency Grade:** :{color}[{result.cce_grade}] (Score: {result.cce_score:.1f}/10)")

        _pct = lambda v: f"{v:.4f}" if v is not None else "N/A"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OCF / OI", _pct(result.ocf_to_oi))
        c2.metric("OCF / NI", _pct(result.ocf_to_ni))
        c3.metric("OCF / Revenue", _pct(result.ocf_to_revenue))
        c4.metric("FCF / OI", _pct(result.fcf_to_oi))

        with st.expander("Cash Conversion Efficiency Details"):
            details = {"Metric": [], "Value": []}
            details["Metric"].append("OCF / OI")
            details["Value"].append(_pct(result.ocf_to_oi))
            details["Metric"].append("OCF / NI")
            details["Value"].append(_pct(result.ocf_to_ni))
            details["Metric"].append("OCF / Revenue")
            details["Value"].append(_pct(result.ocf_to_revenue))
            details["Metric"].append("OCF / EBITDA")
            details["Value"].append(_pct(result.ocf_to_ebitda))
            details["Metric"].append("FCF / OI")
            details["Value"].append(_pct(result.fcf_to_oi))
            details["Metric"].append("Cash / OI")
            details["Value"].append(_pct(result.cash_to_oi))
            details["Metric"].append("Score")
            details["Value"].append(f"{result.cce_score}/10")
            details["Metric"].append("Grade")
            details["Value"].append(result.cce_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_fixed_cost_leverage_ratio(self, df: pd.DataFrame):
        """Phase 236: Fixed Cost Leverage Ratio tab."""
        from financial_analyzer import CharlieAnalyzer, FixedCostLeverageRatioResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.fixed_cost_leverage_ratio_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.fclr_grade, "gray")
        st.markdown(f"**Fixed Cost Leverage Ratio Grade:** :{color}[{result.fclr_grade}] (Score: {result.fclr_score:.1f}/10)")

        _pct = lambda v: f"{v:.4f}" if v is not None else "N/A"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DOL", _pct(result.dol))
        c2.metric("Contribution Margin", _pct(result.contribution_margin))
        c3.metric("OI / Revenue", _pct(result.oi_to_revenue))
        c4.metric("COGS / Revenue", _pct(result.cogs_to_revenue))

        with st.expander("Fixed Cost Leverage Ratio Details"):
            details = {"Metric": [], "Value": []}
            details["Metric"].append("DOL (GP/OI)")
            details["Value"].append(_pct(result.dol))
            details["Metric"].append("Contribution Margin")
            details["Value"].append(_pct(result.contribution_margin))
            details["Metric"].append("OI / Revenue")
            details["Value"].append(_pct(result.oi_to_revenue))
            details["Metric"].append("COGS / Revenue")
            details["Value"].append(_pct(result.cogs_to_revenue))
            details["Metric"].append("OpEx / Revenue")
            details["Value"].append(_pct(result.opex_to_revenue))
            details["Metric"].append("Breakeven Proxy")
            details["Value"].append(_pct(result.breakeven_proxy))
            details["Metric"].append("Score")
            details["Value"].append(f"{result.fclr_score}/10")
            details["Metric"].append("Grade")
            details["Value"].append(result.fclr_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_revenue_quality_index(self, df: pd.DataFrame):
        """Phase 232: Revenue Quality Index tab."""
        from financial_analyzer import CharlieAnalyzer, RevenueQualityIndexResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.revenue_quality_index_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.rqi_grade, "gray")
        st.markdown(f"**Revenue Quality Index Grade:** :{color}[{result.rqi_grade}] (Score: {result.rqi_score:.1f}/10)")

        _pct = lambda v: f"{v:.4f}" if v is not None else "N/A"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OCF / Revenue", _pct(result.ocf_to_revenue))
        c2.metric("Gross Margin", _pct(result.gross_margin))
        c3.metric("AR / Revenue", _pct(result.ar_to_revenue))
        c4.metric("NI / Revenue", _pct(result.ni_to_revenue))

        with st.expander("Revenue Quality Index Details"):
            details = {"Metric": [], "Value": []}
            details["Metric"].append("OCF / Revenue")
            details["Value"].append(_pct(result.ocf_to_revenue))
            details["Metric"].append("Gross Margin")
            details["Value"].append(_pct(result.gross_margin))
            details["Metric"].append("NI / Revenue")
            details["Value"].append(_pct(result.ni_to_revenue))
            details["Metric"].append("EBITDA / Revenue")
            details["Value"].append(_pct(result.ebitda_to_revenue))
            details["Metric"].append("AR / Revenue")
            details["Value"].append(_pct(result.ar_to_revenue))
            details["Metric"].append("Cash / Revenue")
            details["Value"].append(_pct(result.cash_to_revenue))
            details["Metric"].append("Score")
            details["Value"].append(f"{result.rqi_score}/10")
            details["Metric"].append("Grade")
            details["Value"].append(result.rqi_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_cost_control(self, df: pd.DataFrame):
        """Phase 215: Cost Control tab."""
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.cost_control_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.cc_grade, "gray")
        st.markdown(f"**Cost Control Grade:** :{color}[{result.cc_grade}] ({result.cc_score:.1f}/10)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("OpEx/Revenue", f"{result.opex_to_revenue:.4f}" if result.opex_to_revenue is not None else "N/A")
        col2.metric("COGS/Revenue", f"{result.cogs_to_revenue:.4f}" if result.cogs_to_revenue is not None else "N/A")
        col3.metric("Op Margin", f"{result.operating_margin:.4f}" if result.operating_margin is not None else "N/A")
        col4.metric("CC Score", f"{result.cc_score:.1f}")

        with st.expander("Cost Control Details"):
            detail_data = {
                "Metric": ["OpEx/Revenue", "COGS/Revenue", "SGA/Revenue",
                           "Operating Margin (OI/Rev)", "OpEx/Gross Profit", "EBITDA Margin"],
                "Value": [
                    f"{result.opex_to_revenue:.4f}" if result.opex_to_revenue is not None else "N/A",
                    f"{result.cogs_to_revenue:.4f}" if result.cogs_to_revenue is not None else "N/A",
                    f"{result.sga_to_revenue:.4f}" if result.sga_to_revenue is not None else "N/A",
                    f"{result.operating_margin:.4f}" if result.operating_margin is not None else "N/A",
                    f"{result.opex_to_gross_profit:.4f}" if result.opex_to_gross_profit is not None else "N/A",
                    f"{result.ebitda_margin:.4f}" if result.ebitda_margin is not None else "N/A",
                ],
            }
            st.table(detail_data)

            st.caption(result.summary)

    def _render_valuation_signal(self, df: pd.DataFrame):
        """Phase 212: Valuation Signal tab."""
        st.subheader("Valuation Signal Analysis")
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = analyzer.valuation_signal_analysis(fd)
            source = row.get("source", "Unknown")

            grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.vsg_grade, "gray")
            st.markdown(f"**{source}** — :{grade_color}[{result.vsg_grade}] ({result.vsg_score:.1f}/10)")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("EV/EBITDA", f"{result.ev_to_ebitda:.2f}" if result.ev_to_ebitda is not None else "N/A")
            col2.metric("Earn Yield", f"{result.earnings_yield:.4f}" if result.earnings_yield is not None else "N/A")
            col3.metric("P/B", f"{result.price_to_book:.2f}" if result.price_to_book is not None else "N/A")
            col4.metric("FCF Yield", f"{result.fcf_yield:.4f}" if result.fcf_yield is not None else "N/A")

            with st.expander(f"Details — {source}"):
                details = {"Metric": [], "Value": []}
                for label, val in [
                    ("EV/EBITDA", result.ev_to_ebitda),
                    ("P/E Proxy", result.price_to_earnings),
                    ("P/B Proxy", result.price_to_book),
                    ("EV/Revenue", result.ev_to_revenue),
                    ("Earnings Yield", result.earnings_yield),
                    ("FCF Yield", result.fcf_yield),
                ]:
                    details["Metric"].append(label)
                    details["Value"].append(f"{val:.4f}" if val is not None else "N/A")
                details["Metric"].append("Score")
                details["Value"].append(f"{result.vsg_score}/10")
                details["Metric"].append("Grade")
                details["Value"].append(result.vsg_grade)
                st.table(details)

            st.caption(result.summary)

    def _render_capital_discipline(self, df: pd.DataFrame):
        """Phase 211: Capital Discipline tab."""
        st.subheader("Capital Discipline Analysis")
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = analyzer.capital_discipline_analysis(fd)
            source = row.get("source", "Unknown")

            grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.cd_grade, "gray")
            st.markdown(f"**{source}** — :{grade_color}[{result.cd_grade}] ({result.cd_score:.1f}/10)")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RE/Equity", f"{result.retained_to_equity:.4f}" if result.retained_to_equity is not None else "N/A")
            col2.metric("OCF/Debt", f"{result.ocf_to_debt:.4f}" if result.ocf_to_debt is not None else "N/A")
            col3.metric("CapEx/OCF", f"{result.capex_to_ocf:.4f}" if result.capex_to_ocf is not None else "N/A")
            col4.metric("D/E", f"{result.debt_to_equity:.4f}" if result.debt_to_equity is not None else "N/A")

            with st.expander(f"Details — {source}"):
                details = {"Metric": [], "Value": []}
                for label, val in [
                    ("RE/Equity", result.retained_to_equity),
                    ("RE/Assets", result.retained_to_assets),
                    ("Div Payout", result.dividend_payout),
                    ("CapEx/OCF", result.capex_to_ocf),
                    ("Debt/Equity", result.debt_to_equity),
                    ("OCF/Debt", result.ocf_to_debt),
                ]:
                    details["Metric"].append(label)
                    details["Value"].append(f"{val:.4f}" if val is not None else "N/A")
                details["Metric"].append("Score")
                details["Value"].append(f"{result.cd_score}/10")
                details["Metric"].append("Grade")
                details["Value"].append(result.cd_grade)
                st.table(details)

            st.caption(result.summary)

    def _render_resource_optimization(self, df: pd.DataFrame):
        """Phase 210: Resource Optimization tab."""
        st.subheader("Resource Optimization Analysis")
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = analyzer.resource_optimization_analysis(fd)
            source = row.get("source", "Unknown")

            grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.ro_grade, "gray")
            st.markdown(f"**{source}** — :{grade_color}[{result.ro_grade}] ({result.ro_score:.1f}/10)")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("FCF/Rev", f"{result.fcf_to_revenue:.4f}" if result.fcf_to_revenue is not None else "N/A")
            col2.metric("OCF/Rev", f"{result.ocf_to_revenue:.4f}" if result.ocf_to_revenue is not None else "N/A")
            col3.metric("CapEx/Rev", f"{result.capex_to_revenue:.4f}" if result.capex_to_revenue is not None else "N/A")
            col4.metric("FCF/Assets", f"{result.fcf_to_assets:.4f}" if result.fcf_to_assets is not None else "N/A")

            with st.expander(f"Details — {source}"):
                details = {"Metric": [], "Value": []}
                for label, val in [
                    ("FCF/Revenue", result.fcf_to_revenue),
                    ("OCF/Revenue", result.ocf_to_revenue),
                    ("CapEx/Revenue", result.capex_to_revenue),
                    ("OCF/Assets", result.ocf_to_assets),
                    ("FCF/Assets", result.fcf_to_assets),
                    ("Dividend Payout", result.dividend_payout_ratio),
                ]:
                    details["Metric"].append(label)
                    details["Value"].append(f"{val:.4f}" if val is not None else "N/A")
                details["Metric"].append("Score")
                details["Value"].append(f"{result.ro_score}/10")
                details["Metric"].append("Grade")
                details["Value"].append(result.ro_grade)
                st.table(details)

            st.caption(result.summary)

    def _render_financial_productivity(self, df: pd.DataFrame):
        """Phase 205: Financial Productivity tab."""
        st.subheader("Financial Productivity Analysis")
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = analyzer.financial_productivity_analysis(fd)
            source = row.get("source", "Unknown")

            grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.fp_grade, "gray")
            st.markdown(f"**{source}** — :{grade_color}[{result.fp_grade}] ({result.fp_score:.1f}/10)")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rev/Assets", f"{result.revenue_per_asset:.2f}" if result.revenue_per_asset is not None else "N/A")
            col2.metric("Rev/Equity", f"{result.revenue_per_equity:.2f}" if result.revenue_per_equity is not None else "N/A")
            col3.metric("EBITDA/OpEx", f"{result.ebitda_per_employee_proxy:.2f}" if result.ebitda_per_employee_proxy is not None else "N/A")
            col4.metric("OCF/Assets", f"{result.cash_flow_per_asset:.2f}" if result.cash_flow_per_asset is not None else "N/A")

            with st.expander(f"Details — {source}"):
                details = {"Metric": [], "Value": []}
                for label, val in [
                    ("Revenue/Assets", result.revenue_per_asset),
                    ("Revenue/Equity", result.revenue_per_equity),
                    ("EBITDA/OpEx", result.ebitda_per_employee_proxy),
                    ("OI/Assets", result.operating_income_per_asset),
                    ("NI/Revenue", result.net_income_per_revenue),
                    ("OCF/Assets", result.cash_flow_per_asset),
                ]:
                    details["Metric"].append(label)
                    details["Value"].append(f"{val:.4f}" if val is not None else "N/A")
                details["Metric"].append("Score")
                details["Value"].append(f"{result.fp_score}/10")
                details["Metric"].append("Grade")
                details["Value"].append(result.fp_grade)
                st.table(details)

            st.caption(result.summary)

    def _render_equity_preservation(self, df: pd.DataFrame):
        """Phase 198: Equity Preservation tab."""
        st.subheader("Equity Preservation Analysis")
        analyzer = CharlieAnalyzer()
        fin = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.equity_preservation_analysis(fin)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.ep_grade, "gray")
        st.markdown(f"**Grade:** :{color}[{result.ep_grade}] &emsp; **Score:** {result.ep_score:.1f}/10")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Equity/Assets", f"{result.equity_to_assets:.2%}" if result.equity_to_assets is not None else "N/A")
        col2.metric("Retained/Equity", f"{result.retained_to_equity:.2%}" if result.retained_to_equity is not None else "N/A")
        col3.metric("Equity Growth Cap", f"{result.equity_growth_capacity:.2%}" if result.equity_growth_capacity is not None else "N/A")
        col4.metric("Equity/Liabilities", f"{result.equity_to_liabilities:.2f}" if result.equity_to_liabilities is not None else "N/A")

        with st.expander("Details"):
            details = {
                "Equity-to-Assets (TE/TA)": f"{result.equity_to_assets:.4f}" if result.equity_to_assets is not None else "N/A",
                "Retained-to-Equity (RE/TE)": f"{result.retained_to_equity:.4f}" if result.retained_to_equity is not None else "N/A",
                "Equity Growth Capacity (NI/TE)": f"{result.equity_growth_capacity:.4f}" if result.equity_growth_capacity is not None else "N/A",
                "Equity-to-Liabilities (TE/TL)": f"{result.equity_to_liabilities:.4f}" if result.equity_to_liabilities is not None else "N/A",
                "Tangible Equity Ratio (TE/TA)": f"{result.tangible_equity_ratio:.4f}" if result.tangible_equity_ratio is not None else "N/A",
                "Equity per Revenue (TE/Rev)": f"{result.equity_per_revenue:.4f}" if result.equity_per_revenue is not None else "N/A",
            }
            st.table(details)

        st.caption(result.summary)

    def _render_debt_management(self, df: pd.DataFrame):
        """Phase 197: Debt Management tab."""
        st.subheader("Debt Management Analysis")
        analyzer = CharlieAnalyzer()
        fin = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.debt_management_analysis(fin)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.dm_grade, "gray")
        st.markdown(f"**Grade:** :{color}[{result.dm_grade}] &emsp; **Score:** {result.dm_score:.1f}/10")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Debt/OI", f"{result.debt_to_operating_income:.2f}" if result.debt_to_operating_income is not None else "N/A")
        col2.metric("Interest/Rev", f"{result.interest_to_revenue:.2%}" if result.interest_to_revenue is not None else "N/A")
        col3.metric("Debt Coverage", f"{result.debt_coverage_ratio:.2f}" if result.debt_coverage_ratio is not None else "N/A")
        col4.metric("Net Debt Ratio", f"{result.net_debt_ratio:.2%}" if result.net_debt_ratio is not None else "N/A")

        with st.expander("Details"):
            details = {
                "Debt-to-OI (TD/OI)": f"{result.debt_to_operating_income:.4f}" if result.debt_to_operating_income is not None else "N/A",
                "Debt-to-OCF (TD/OCF)": f"{result.debt_to_ocf:.4f}" if result.debt_to_ocf is not None else "N/A",
                "Interest-to-Revenue (IE/Rev)": f"{result.interest_to_revenue:.4f}" if result.interest_to_revenue is not None else "N/A",
                "Debt-to-Gross Profit (TD/GP)": f"{result.debt_to_gross_profit:.4f}" if result.debt_to_gross_profit is not None else "N/A",
                "Net Debt Ratio ((TD-Cash)/TA)": f"{result.net_debt_ratio:.4f}" if result.net_debt_ratio is not None else "N/A",
                "Debt Coverage (EBITDA/(IE+TD*0.1))": f"{result.debt_coverage_ratio:.4f}" if result.debt_coverage_ratio is not None else "N/A",
            }
            st.table(details)

        st.caption(result.summary)

    def _render_income_retention(self, df: pd.DataFrame):
        """Phase 196: Income Retention tab."""
        st.subheader("Income Retention Analysis")
        analyzer = CharlieAnalyzer()
        fin = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.income_retention_analysis(fin)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.ir_grade, "gray")
        st.markdown(f"**Grade:** :{color}[{result.ir_grade}] &emsp; **Score:** {result.ir_score:.1f}/10")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Net/Gross Ratio", f"{result.net_to_gross_ratio:.2f}" if result.net_to_gross_ratio is not None else "N/A")
        col2.metric("Net/Operating", f"{result.net_to_operating_ratio:.2f}" if result.net_to_operating_ratio is not None else "N/A")
        col3.metric("After-Tax Margin", f"{result.after_tax_margin:.2%}" if result.after_tax_margin is not None else "N/A")
        col4.metric("NI/EBITDA", f"{result.net_to_ebitda_ratio:.2f}" if result.net_to_ebitda_ratio is not None else "N/A")

        with st.expander("Details"):
            details = {
                "Net-to-Gross Ratio (NI/GP)": f"{result.net_to_gross_ratio:.4f}" if result.net_to_gross_ratio is not None else "N/A",
                "Net-to-Operating Ratio (NI/OI)": f"{result.net_to_operating_ratio:.4f}" if result.net_to_operating_ratio is not None else "N/A",
                "Net-to-EBITDA Ratio (NI/EBITDA)": f"{result.net_to_ebitda_ratio:.4f}" if result.net_to_ebitda_ratio is not None else "N/A",
                "Retention Rate (RE/NI)": f"{result.retention_rate:.4f}" if result.retention_rate is not None else "N/A",
                "Income-to-Asset Gen (NI/TA)": f"{result.income_to_asset_generation:.4f}" if result.income_to_asset_generation is not None else "N/A",
                "After-Tax Margin (NI/Rev)": f"{result.after_tax_margin:.4f}" if result.after_tax_margin is not None else "N/A",
            }
            st.table(details)

        st.caption(result.summary)

    def _render_operational_efficiency(self, df: pd.DataFrame):
        """Phase 195: Operational Efficiency tab."""
        st.subheader("Operational Efficiency Analysis")
        try:
            data = self.analyzer._dataframe_to_financial_data(df)
            result = self.analyzer.operational_efficiency_analysis(data)

            grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
            color = grade_colors.get(result.oe_grade, "gray")
            st.markdown(f"**Grade:** :{color}[{result.oe_grade}] | **Score:** {result.oe_score:.1f}/10")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("OI Margin", f"{result.oi_margin:.2f}" if result.oi_margin is not None else "N/A")
            col2.metric("Rev/Assets", f"{result.revenue_to_assets:.2f}" if result.revenue_to_assets is not None else "N/A")
            col3.metric("OpEx Effic", f"{result.opex_efficiency:.2f}" if result.opex_efficiency is not None else "N/A")
            col4.metric("OI/Liabilities", f"{result.income_per_liability:.2f}" if result.income_per_liability is not None else "N/A")

            with st.expander("Operational Efficiency Details"):
                details = {
                    "OI Margin (OI/Rev)": f"{result.oi_margin:.4f}" if result.oi_margin is not None else "N/A",
                    "Revenue/Assets": f"{result.revenue_to_assets:.4f}" if result.revenue_to_assets is not None else "N/A",
                    "GP/Assets": f"{result.gross_profit_per_asset:.4f}" if result.gross_profit_per_asset is not None else "N/A",
                    "OpEx Efficiency (Rev/OpEx)": f"{result.opex_efficiency:.4f}" if result.opex_efficiency is not None else "N/A",
                    "Asset Utilization (Rev/CA)": f"{result.asset_utilization:.4f}" if result.asset_utilization is not None else "N/A",
                    "OI/Liabilities": f"{result.income_per_liability:.4f}" if result.income_per_liability is not None else "N/A",
                }
                st.table(details)

            st.caption(result.summary)
        except Exception as e:
            st.error(f"Operational Efficiency analysis error: {e}")

    def _render_operating_momentum(self, df: pd.DataFrame):
        """Phase 191: Operating Momentum tab."""
        st.subheader("Operating Momentum Analysis")
        try:
            data = self.analyzer._dataframe_to_financial_data(df)
            result = self.analyzer.operating_momentum_analysis(data)

            grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
            color = grade_colors.get(result.om_grade, "gray")
            st.markdown(f"**Grade:** :{color}[{result.om_grade}] | **Score:** {result.om_score:.1f}/10")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("EBITDA Margin", f"{result.ebitda_margin:.2f}" if result.ebitda_margin is not None else "N/A")
            col2.metric("EBIT Margin", f"{result.ebit_margin:.2f}" if result.ebit_margin is not None else "N/A")
            col3.metric("OCF Margin", f"{result.ocf_margin:.2f}" if result.ocf_margin is not None else "N/A")
            col4.metric("GP→OI Conv", f"{result.gross_to_operating_conversion:.2f}" if result.gross_to_operating_conversion is not None else "N/A")

            with st.expander("Operating Momentum Details"):
                details = {
                    "EBITDA Margin (EBITDA/Rev)": f"{result.ebitda_margin:.4f}" if result.ebitda_margin is not None else "N/A",
                    "EBIT Margin (EBIT/Rev)": f"{result.ebit_margin:.4f}" if result.ebit_margin is not None else "N/A",
                    "OCF Margin (OCF/Rev)": f"{result.ocf_margin:.4f}" if result.ocf_margin is not None else "N/A",
                    "GP→OI Conversion (OI/GP)": f"{result.gross_to_operating_conversion:.4f}" if result.gross_to_operating_conversion is not None else "N/A",
                    "Operating Cash Conversion (OCF/OI)": f"{result.operating_cash_conversion:.4f}" if result.operating_cash_conversion is not None else "N/A",
                    "Overhead Absorption (OI/OpEx)": f"{result.overhead_absorption:.4f}" if result.overhead_absorption is not None else "N/A",
                }
                st.table(details)

            st.caption(result.summary)
        except Exception as e:
            st.error(f"Operating Momentum analysis error: {e}")

    def _render_payout_discipline(self, df: pd.DataFrame):
        """Phase 185: Payout Discipline tab."""
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = self.analyzer.payout_discipline_analysis(fd)

            grade = result.pd_grade
            if grade == "Excellent":
                color = "green"
            elif grade == "Good":
                color = "blue"
            elif grade == "Adequate":
                color = "orange"
            else:
                color = "red"

            st.markdown(f"**Payout Discipline Grade:** :{color}[{grade}] ({result.pd_score:.1f}/10)")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Cash Div Coverage", f"{result.cash_dividend_coverage:.2f}x" if result.cash_dividend_coverage is not None else "N/A")
            col2.metric("Payout Ratio", f"{result.payout_ratio:.2f}" if result.payout_ratio is not None else "N/A")
            col3.metric("CapEx Priority", f"{result.capex_priority:.2f}" if result.capex_priority is not None else "N/A")
            col4.metric("Retention", f"{result.retention_ratio:.2f}" if result.retention_ratio is not None else "N/A")

            with st.expander("Payout Discipline Details"):
                details = {"Metric": [], "Value": []}
                details["Metric"].append("Cash Div Coverage (OCF/Div)")
                details["Value"].append(f"{result.cash_dividend_coverage:.4f}" if result.cash_dividend_coverage is not None else "N/A")
                details["Metric"].append("Payout Ratio (Div/NI)")
                details["Value"].append(f"{result.payout_ratio:.4f}" if result.payout_ratio is not None else "N/A")
                details["Metric"].append("Retention Ratio ((NI-Div)/NI)")
                details["Value"].append(f"{result.retention_ratio:.4f}" if result.retention_ratio is not None else "N/A")
                details["Metric"].append("Dividend-to-OCF (Div/OCF)")
                details["Value"].append(f"{result.dividend_to_ocf:.4f}" if result.dividend_to_ocf is not None else "N/A")
                details["Metric"].append("CapEx Priority (CapEx/(CapEx+Div))")
                details["Value"].append(f"{result.capex_priority:.4f}" if result.capex_priority is not None else "N/A")
                details["Metric"].append("Free Cash After Div ((OCF-CapEx-Div)/Rev)")
                details["Value"].append(f"{result.free_cash_after_dividends:.4f}" if result.free_cash_after_dividends is not None else "N/A")
                details["Metric"].append("Score")
                details["Value"].append(f"{result.pd_score:.1f}/10")
                details["Metric"].append("Grade")
                details["Value"].append(result.pd_grade)
                st.table(details)

            st.caption(result.summary)

    def _render_income_resilience(self, df: pd.DataFrame):
        """Phase 184: Income Resilience tab."""
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = self.analyzer.income_resilience_analysis(fd)

            grade = result.ir_grade
            if grade == "Excellent":
                color = "green"
            elif grade == "Good":
                color = "blue"
            elif grade == "Adequate":
                color = "orange"
            else:
                color = "red"

            st.markdown(f"**Income Resilience Grade:** :{color}[{grade}] ({result.ir_score:.1f}/10)")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("OI Stability", f"{result.operating_income_stability:.2f}" if result.operating_income_stability is not None else "N/A")
            col2.metric("EBIT Coverage", f"{result.ebit_coverage:.2f}x" if result.ebit_coverage is not None else "N/A")
            col3.metric("NM Resilience", f"{result.net_margin_resilience:.2f}" if result.net_margin_resilience is not None else "N/A")
            col4.metric("EBITDA Cushion", f"{result.ebitda_cushion:.2f}x" if result.ebitda_cushion is not None else "N/A")

            with st.expander("Income Resilience Details"):
                details = {"Metric": [], "Value": []}
                details["Metric"].append("OI Stability (OI/Rev)")
                details["Value"].append(f"{result.operating_income_stability:.4f}" if result.operating_income_stability is not None else "N/A")
                details["Metric"].append("EBIT Coverage (EBIT/IE)")
                details["Value"].append(f"{result.ebit_coverage:.4f}" if result.ebit_coverage is not None else "N/A")
                details["Metric"].append("NM Resilience (NI/OI)")
                details["Value"].append(f"{result.net_margin_resilience:.4f}" if result.net_margin_resilience is not None else "N/A")
                details["Metric"].append("Depreciation Buffer (D&A/OI)")
                details["Value"].append(f"{result.depreciation_buffer:.4f}" if result.depreciation_buffer is not None else "N/A")
                details["Metric"].append("Tax & Interest Drag ((OI-NI)/OI)")
                details["Value"].append(f"{result.tax_interest_drag:.4f}" if result.tax_interest_drag is not None else "N/A")
                details["Metric"].append("EBITDA Cushion (EBITDA/IE)")
                details["Value"].append(f"{result.ebitda_cushion:.4f}" if result.ebitda_cushion is not None else "N/A")
                details["Metric"].append("Score")
                details["Value"].append(f"{result.ir_score:.1f}/10")
                details["Metric"].append("Grade")
                details["Value"].append(result.ir_grade)
                st.table(details)

            st.caption(result.summary)

    def _render_structural_strength(self, df: pd.DataFrame):
        """Phase 182: Structural Strength tab."""
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = self.analyzer.structural_strength_analysis(fd)

            grade = result.ss_grade
            if grade == "Excellent":
                color = "green"
            elif grade == "Good":
                color = "blue"
            elif grade == "Adequate":
                color = "orange"
            else:
                color = "red"

            st.markdown(f"**Structural Strength Grade:** :{color}[{grade}] ({result.ss_score:.1f}/10)")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Equity Multiplier", f"{result.equity_multiplier:.2f}x" if result.equity_multiplier is not None else "N/A")
            col2.metric("Debt-to-Equity", f"{result.debt_to_equity:.2f}x" if result.debt_to_equity is not None else "N/A")
            col3.metric("Equity Cushion", f"{result.equity_cushion:.2f}" if result.equity_cushion is not None else "N/A")
            col4.metric("Fin Leverage", f"{result.financial_leverage_ratio:.2f}x" if result.financial_leverage_ratio is not None else "N/A")

            with st.expander("Structural Strength Details"):
                details = {"Metric": [], "Value": []}
                details["Metric"].append("Equity Multiplier (TA/TE)")
                details["Value"].append(f"{result.equity_multiplier:.4f}" if result.equity_multiplier is not None else "N/A")
                details["Metric"].append("Debt-to-Equity (TD/TE)")
                details["Value"].append(f"{result.debt_to_equity:.4f}" if result.debt_to_equity is not None else "N/A")
                details["Metric"].append("Liability Composition (CL/TL)")
                details["Value"].append(f"{result.liability_composition:.4f}" if result.liability_composition is not None else "N/A")
                details["Metric"].append("Equity Cushion ((TE-TD)/TA)")
                details["Value"].append(f"{result.equity_cushion:.4f}" if result.equity_cushion is not None else "N/A")
                details["Metric"].append("Fixed Asset Coverage (TE/(TA-CA))")
                details["Value"].append(f"{result.fixed_asset_coverage:.4f}" if result.fixed_asset_coverage is not None else "N/A")
                details["Metric"].append("Financial Leverage (TL/TE)")
                details["Value"].append(f"{result.financial_leverage_ratio:.4f}" if result.financial_leverage_ratio is not None else "N/A")
                details["Metric"].append("Score")
                details["Value"].append(f"{result.ss_score:.1f}/10")
                details["Metric"].append("Grade")
                details["Value"].append(result.ss_grade)
                st.table(details)

            st.caption(result.summary)

    def _render_profit_conversion(self, df: pd.DataFrame):
        """Phase 179: Profit Conversion tab."""
        for _, row in df.iterrows():
            fd = self._row_to_financial_data(row)
            result = self.analyzer.profit_conversion_analysis(fd)
            period = row.get("Period", "N/A")

            grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
            color = grade_colors.get(result.pc_grade, "gray")
            st.markdown(f"### {period}: :{color}[{result.pc_grade}] (Score: {result.pc_score:.1f}/10)")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Gross Conv", f"{result.gross_conversion:.2%}" if result.gross_conversion is not None else "N/A")
            c2.metric("Operating Conv", f"{result.operating_conversion:.2%}" if result.operating_conversion is not None else "N/A")
            c3.metric("Net Conv", f"{result.net_conversion:.2%}" if result.net_conversion is not None else "N/A")
            c4.metric("Cash Conv", f"{result.cash_conversion:.2%}" if result.cash_conversion is not None else "N/A")

            with st.expander(f"Profit Conversion Details — {period}"):
                detail_data = {
                    "Metric": ["Gross Conversion", "Operating Conversion", "Net Conversion", "EBITDA Conversion", "Cash Conversion", "Profit-to-Cash Ratio"],
                    "Value": [
                        f"{result.gross_conversion:.4f}" if result.gross_conversion is not None else "N/A",
                        f"{result.operating_conversion:.4f}" if result.operating_conversion is not None else "N/A",
                        f"{result.net_conversion:.4f}" if result.net_conversion is not None else "N/A",
                        f"{result.ebitda_conversion:.4f}" if result.ebitda_conversion is not None else "N/A",
                        f"{result.cash_conversion:.4f}" if result.cash_conversion is not None else "N/A",
                        f"{result.profit_to_cash_ratio:.4f}" if result.profit_to_cash_ratio is not None else "N/A",
                    ],
                }
                st.table(detail_data)

            st.caption(result.summary)

    def _render_asset_deployment_efficiency(self, df: pd.DataFrame):
        """Phase 172: Asset Deployment Efficiency tab."""
        from financial_analyzer import AssetDeploymentEfficiencyResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result: AssetDeploymentEfficiencyResult = analyzer.asset_deployment_efficiency_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.ade_grade, "gray")
        st.markdown(f"### Asset Deployment Efficiency: :{color}[{result.ade_grade}] ({result.ade_score:.1f}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Asset Turnover", f"{result.asset_turnover:.2f}" if result.asset_turnover is not None else "N/A")
        c2.metric("Fixed Asset Lev", f"{result.fixed_asset_leverage:.2f}" if result.fixed_asset_leverage is not None else "N/A")
        c3.metric("Income Yield", f"{result.asset_income_yield:.1%}" if result.asset_income_yield is not None else "N/A")
        c4.metric("Cash Yield", f"{result.asset_cash_yield:.1%}" if result.asset_cash_yield is not None else "N/A")

        details = {
            "Metric": ["Asset Turnover", "Fixed Asset Leverage", "Income Yield", "Cash Yield", "Inventory Velocity", "Receivables Velocity"],
            "Value": [
                f"{result.asset_turnover:.4f}" if result.asset_turnover is not None else "N/A",
                f"{result.fixed_asset_leverage:.4f}" if result.fixed_asset_leverage is not None else "N/A",
                f"{result.asset_income_yield:.4f}" if result.asset_income_yield is not None else "N/A",
                f"{result.asset_cash_yield:.4f}" if result.asset_cash_yield is not None else "N/A",
                f"{result.inventory_velocity:.4f}" if result.inventory_velocity is not None else "N/A",
                f"{result.receivables_velocity:.4f}" if result.receivables_velocity is not None else "N/A",
            ],
        }
        import pandas as _pd
        with st.expander("Detail Breakdown"):
            st.table(_pd.DataFrame(details))

            st.caption(result.summary)

    def _render_profit_sustainability(self, df: pd.DataFrame):
        """Phase 171: Profit Sustainability tab."""
        from financial_analyzer import ProfitSustainabilityResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result: ProfitSustainabilityResult = analyzer.profit_sustainability_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.ps_grade, "gray")
        st.markdown(f"### Profit Sustainability: :{color}[{result.ps_grade}] ({result.ps_score:.1f}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cash Backing", f"{result.profit_cash_backing:.2f}" if result.profit_cash_backing is not None else "N/A")
        c2.metric("Margin Depth", f"{result.profit_margin_depth:.1%}" if result.profit_margin_depth is not None else "N/A")
        c3.metric("Reinvestment", f"{result.profit_reinvestment:.1%}" if result.profit_reinvestment is not None else "N/A")
        c4.metric("Profit/Asset", f"{result.profit_to_asset:.1%}" if result.profit_to_asset is not None else "N/A")

        details = {
            "Metric": ["Cash Backing", "Margin Depth", "Reinvestment", "Profit/Asset", "Stability Proxy", "Profit Leverage"],
            "Value": [
                f"{result.profit_cash_backing:.4f}" if result.profit_cash_backing is not None else "N/A",
                f"{result.profit_margin_depth:.4f}" if result.profit_margin_depth is not None else "N/A",
                f"{result.profit_reinvestment:.4f}" if result.profit_reinvestment is not None else "N/A",
                f"{result.profit_to_asset:.4f}" if result.profit_to_asset is not None else "N/A",
                f"{result.profit_stability_proxy:.4f}" if result.profit_stability_proxy is not None else "N/A",
                f"{result.profit_leverage:.4f}" if result.profit_leverage is not None else "N/A",
            ],
        }
        import pandas as _pd
        with st.expander("Detail Breakdown"):
            st.table(_pd.DataFrame(details))

            st.caption(result.summary)

    def _render_debt_discipline(self, df: pd.DataFrame):
        """Phase 170: Debt Discipline tab."""
        from financial_analyzer import DebtDisciplineResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result: DebtDisciplineResult = analyzer.debt_discipline_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.dd_grade, "gray")
        st.markdown(f"### Debt Discipline: :{color}[{result.dd_grade}] ({result.dd_score:.1f}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Debt Prudence", f"{result.debt_prudence_ratio:.1%}" if result.debt_prudence_ratio is not None else "N/A")
        c2.metric("Servicing Power", f"{result.debt_servicing_power:.2f}" if result.debt_servicing_power is not None else "N/A")
        c3.metric("Coverage Spread", f"{result.debt_coverage_spread:.2f}" if result.debt_coverage_spread is not None else "N/A")
        c4.metric("D/E Leverage", f"{result.debt_to_equity_leverage:.2f}" if result.debt_to_equity_leverage is not None else "N/A")

        details = {
            "Metric": ["Debt Prudence", "Servicing Power", "Coverage Spread", "D/E Leverage", "Interest Absorption", "Repayment Capacity"],
            "Value": [
                f"{result.debt_prudence_ratio:.4f}" if result.debt_prudence_ratio is not None else "N/A",
                f"{result.debt_servicing_power:.4f}" if result.debt_servicing_power is not None else "N/A",
                f"{result.debt_coverage_spread:.4f}" if result.debt_coverage_spread is not None else "N/A",
                f"{result.debt_to_equity_leverage:.4f}" if result.debt_to_equity_leverage is not None else "N/A",
                f"{result.interest_absorption:.4f}" if result.interest_absorption is not None else "N/A",
                f"{result.debt_repayment_capacity:.4f}" if result.debt_repayment_capacity is not None else "N/A",
            ],
        }
        import pandas as _pd
        with st.expander("Detail Breakdown"):
            st.table(_pd.DataFrame(details))

            st.caption(result.summary)

    def _render_capital_preservation(self, df: pd.DataFrame):
        """Phase 168: Capital Preservation tab."""
        from financial_analyzer import CapitalPreservationResult
        analyzer = CharlieAnalyzer()
        fin_data = self.analyzer._dataframe_to_financial_data(df)
        result: CapitalPreservationResult = analyzer.capital_preservation_analysis(fin_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.cp_grade, "gray")
        st.markdown(f"### Capital Preservation: :{color}[{result.cp_grade}] ({result.cp_score:.1f}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RE Power", f"{result.retained_earnings_power:.1%}" if result.retained_earnings_power is not None else "N/A")
        c2.metric("Capital Erosion", f"{result.capital_erosion_rate:.2f}" if result.capital_erosion_rate is not None else "N/A")
        c3.metric("Asset Integrity", f"{result.asset_integrity_ratio:.1%}" if result.asset_integrity_ratio is not None else "N/A")
        c4.metric("Op Capital Ratio", f"{result.operating_capital_ratio:.2f}" if result.operating_capital_ratio is not None else "N/A")

        details = {
            "Metric": ["RE Power", "Capital Erosion Rate", "Asset Integrity", "Op Capital Ratio", "NW Growth Proxy", "Capital Buffer"],
            "Value": [
                f"{result.retained_earnings_power:.4f}" if result.retained_earnings_power is not None else "N/A",
                f"{result.capital_erosion_rate:.4f}" if result.capital_erosion_rate is not None else "N/A",
                f"{result.asset_integrity_ratio:.4f}" if result.asset_integrity_ratio is not None else "N/A",
                f"{result.operating_capital_ratio:.4f}" if result.operating_capital_ratio is not None else "N/A",
                f"{result.net_worth_growth_proxy:.4f}" if result.net_worth_growth_proxy is not None else "N/A",
                f"{result.capital_buffer:.4f}" if result.capital_buffer is not None else "N/A",
            ],
        }
        import pandas as _pd
        with st.expander("Detail Breakdown"):
            st.table(_pd.DataFrame(details))

            st.caption(result.summary)

    def _render_obligation_coverage(self, df: pd.DataFrame):
        """Phase 160: Obligation Coverage tab."""
        from financial_analyzer import ObligationCoverageResult
        st.subheader("Obligation Coverage Analysis")
        fd = self.analyzer._dataframe_to_financial_data(df)
        analyzer = self.analyzer
        result = analyzer.obligation_coverage_analysis(fd)

        if result.oc_grade:
            color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.oc_grade, "gray")
            st.markdown(f"**Grade:** :{color}[{result.oc_grade}] &nbsp; **Score:** {result.oc_score:.1f}/10")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EBITDA Interest Coverage", f"{result.ebitda_interest_coverage:.2f}x" if result.ebitda_interest_coverage is not None else "N/A")
        c2.metric("Cash Interest Coverage", f"{result.cash_interest_coverage:.2f}x" if result.cash_interest_coverage is not None else "N/A")
        c3.metric("Fixed Charge Coverage", f"{result.fixed_charge_coverage:.2f}x" if result.fixed_charge_coverage is not None else "N/A")
        c4.metric("Debt Burden Ratio", f"{result.debt_burden_ratio:.2f}x" if result.debt_burden_ratio is not None else "N/A")

        with st.expander("Details", expanded=False):
            details = {"Metric": [], "Value": []}
            for label, val, fmt in [
                ("EBITDA Interest Coverage", result.ebitda_interest_coverage, ".2f"),
                ("Cash Interest Coverage", result.cash_interest_coverage, ".2f"),
                ("Debt Amortization Capacity", result.debt_amortization_capacity, ".2f"),
                ("Fixed Charge Coverage", result.fixed_charge_coverage, ".2f"),
                ("Debt Burden Ratio", result.debt_burden_ratio, ".2f"),
                ("Interest to Revenue", result.interest_to_revenue, ".4f"),
            ]:
                details["Metric"].append(label)
                details["Value"].append(f"{val:{fmt}}" if val is not None else "N/A")
            details["Metric"].append("Score")
            details["Value"].append(f"{result.oc_score:.1f}")
            details["Metric"].append("Grade")
            details["Value"].append(result.oc_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_internal_growth_capacity(self, df: pd.DataFrame):
        """Phase 159: Internal Growth Capacity tab."""
        from financial_analyzer import InternalGrowthCapacityResult
        st.subheader("Internal Growth Capacity Analysis")
        fd = self.analyzer._dataframe_to_financial_data(df)
        analyzer = self.analyzer
        result = analyzer.internal_growth_capacity_analysis(fd)

        if result.igc_grade:
            color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.igc_grade, "gray")
            st.markdown(f"**Grade:** :{color}[{result.igc_grade}] &nbsp; **Score:** {result.igc_score:.1f}/10")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sustainable Growth Rate", f"{result.sustainable_growth_rate:.2%}" if result.sustainable_growth_rate is not None else "N/A")
        c2.metric("Internal Growth Rate", f"{result.internal_growth_rate:.2%}" if result.internal_growth_rate is not None else "N/A")
        c3.metric("Growth Financing Ratio", f"{result.growth_financing_ratio:.2f}x" if result.growth_financing_ratio is not None else "N/A")
        c4.metric("Plowback Ratio", f"{result.plowback_ratio:.2%}" if result.plowback_ratio is not None else "N/A")

        with st.expander("Details", expanded=False):
            details = {"Metric": [], "Value": []}
            for label, val, fmt in [
                ("Sustainable Growth Rate", result.sustainable_growth_rate, ".2%"),
                ("Internal Growth Rate", result.internal_growth_rate, ".2%"),
                ("Plowback Ratio", result.plowback_ratio, ".2%"),
                ("Reinvestment Rate", result.reinvestment_rate, ".2f"),
                ("Growth Financing Ratio", result.growth_financing_ratio, ".2f"),
                ("Equity Growth Rate", result.equity_growth_rate, ".2%"),
            ]:
                details["Metric"].append(label)
                details["Value"].append(f"{val:{fmt}}" if val is not None else "N/A")
            details["Metric"].append("Score")
            details["Value"].append(f"{result.igc_score:.1f}")
            details["Metric"].append("Grade")
            details["Value"].append(result.igc_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_liability_management(self, df: pd.DataFrame):
        """Render Phase 146: Liability Management tab."""
        from financial_analyzer import CharlieAnalyzer, FinancialData
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            period = row.get("Period", "N/A")
            fd = self._row_to_financial_data(row)
            result = analyzer.liability_management_analysis(fd)
            color = "green" if result.lm_grade == "Excellent" else "blue" if result.lm_grade == "Good" else "orange" if result.lm_grade == "Adequate" else "red"
            st.markdown(f"### {period} — Liability Management: :{color}[{result.lm_grade}] ({result.lm_score}/10)")
            c1, c2, c3, c4 = st.columns(4)
            _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
            _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
            c1.metric("Liab/Assets", _pct(result.liability_to_assets))
            c2.metric("Liab/Equity", _r2(result.liability_to_equity))
            c3.metric("Liab Coverage", _r2(result.liability_coverage))
            c4.metric("Net Liability", _pct(result.net_liability))
            detail = {
                "Metric": ["Liab/Assets", "Liab/Equity", "Current Liab Ratio", "Liab Coverage", "Liab/Revenue", "Net Liability"],
                "Value": [_pct(result.liability_to_assets), _r2(result.liability_to_equity), _pct(result.current_liability_ratio), _r2(result.liability_coverage), _r2(result.liability_to_revenue), _pct(result.net_liability)],
            }
            st.table(detail)
            st.caption(result.summary)

    def _render_revenue_predictability(self, df: pd.DataFrame):
        """Render Phase 142: Revenue Predictability tab."""
        from financial_analyzer import CharlieAnalyzer, FinancialData
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            period = row.get("Period", "N/A")
            fd = self._row_to_financial_data(row)
            result = analyzer.revenue_predictability_analysis(fd)
            color = "green" if result.rp_grade == "Excellent" else "blue" if result.rp_grade == "Good" else "orange" if result.rp_grade == "Adequate" else "red"
            st.markdown(f"### {period} — Revenue Predictability: :{color}[{result.rp_grade}] ({result.rp_score}/10)")
            c1, c2, c3, c4 = st.columns(4)
            _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
            _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
            c1.metric("Op Margin", _pct(result.operating_margin))
            c2.metric("Gross Margin", _pct(result.gross_margin))
            c3.metric("Net Margin", _pct(result.net_margin))
            c4.metric("Rev/Assets", _r2(result.revenue_to_assets))
            detail = {
                "Metric": ["Revenue/Assets", "Revenue/Equity", "Revenue/Debt", "Gross Margin", "Operating Margin", "Net Margin"],
                "Value": [_r2(result.revenue_to_assets), _r2(result.revenue_to_equity), _r2(result.revenue_to_debt), _pct(result.gross_margin), _pct(result.operating_margin), _pct(result.net_margin)],
            }
            st.table(detail)
            st.caption(result.summary)

    def _render_equity_reinvestment(self, df: pd.DataFrame):
        """Render Phase 139: Equity Reinvestment tab."""
        from financial_analyzer import CharlieAnalyzer, FinancialData
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            period = row.get("Period", "N/A")
            fd = self._row_to_financial_data(row)
            result = analyzer.equity_reinvestment_analysis(fd)
            color = "green" if result.er_grade == "Excellent" else "blue" if result.er_grade == "Good" else "orange" if result.er_grade == "Adequate" else "red"
            st.markdown(f"### {period} — Equity Reinvestment: :{color}[{result.er_grade}] ({result.er_score}/10)")
            c1, c2, c3, c4 = st.columns(4)
            _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
            _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
            c1.metric("Retention Ratio", _pct(result.retention_ratio))
            c2.metric("Reinvestment Rate", _pct(result.reinvestment_rate))
            c3.metric("Equity Growth Proxy", _pct(result.equity_growth_proxy))
            c4.metric("Div Coverage", _r2(result.dividend_coverage))
            detail = {
                "Metric": ["Retention Ratio", "Reinvestment Rate", "Equity Growth Proxy", "Plowback/Assets", "Internal Growth Rate", "Dividend Coverage"],
                "Value": [_pct(result.retention_ratio), _pct(result.reinvestment_rate), _pct(result.equity_growth_proxy), _pct(result.plowback_to_assets), _pct(result.internal_growth_rate), _r2(result.dividend_coverage)],
            }
            st.table(detail)
            st.caption(result.summary)

    def _render_fixed_asset_efficiency(self, df: pd.DataFrame):
        """Render Phase 138: Fixed Asset Efficiency tab."""
        from financial_analyzer import CharlieAnalyzer, FinancialData
        analyzer = CharlieAnalyzer()
        for _, row in df.iterrows():
            period = row.get("Period", "N/A")
            fd = self._row_to_financial_data(row)
            result = analyzer.fixed_asset_efficiency_analysis(fd)
            color = "green" if result.fae_grade == "Excellent" else "blue" if result.fae_grade == "Good" else "orange" if result.fae_grade == "Adequate" else "red"
            st.markdown(f"### {period} — Fixed Asset Efficiency: :{color}[{result.fae_grade}] ({result.fae_score}/10)")
            c1, c2, c3, c4 = st.columns(4)
            _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
            _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
            c1.metric("FA Ratio", _pct(result.fixed_asset_ratio))
            c2.metric("FA Turnover", _r2(result.fixed_asset_turnover))
            c3.metric("FA Coverage", _r2(result.fixed_asset_coverage))
            c4.metric("CapEx/FA", _pct(result.capex_to_fixed))
            detail = {
                "Metric": ["Fixed Asset Ratio", "FA Turnover", "Fixed/Equity", "FA Coverage", "Depr/FA", "CapEx/FA"],
                "Value": [_pct(result.fixed_asset_ratio), _r2(result.fixed_asset_turnover), _r2(result.fixed_to_equity), _r2(result.fixed_asset_coverage), _pct(result.depreciation_to_fixed), _pct(result.capex_to_fixed)],
            }
            st.table(detail)
            st.caption(result.summary)

    def _render_funding_efficiency(self, df: pd.DataFrame):
        """Render Phase 131: Funding Efficiency Analysis."""
        from financial_analyzer import CharlieAnalyzer, FundingEfficiencyResult
        analyzer = CharlieAnalyzer()
        rows = df.to_dict("records")
        if not rows:
            st.warning("No data available for Funding Efficiency analysis.")
            return
        for row in rows:
            period = row.get("Period", "N/A")
            fd = self._row_to_financial_data(row)
            result = analyzer.funding_efficiency_analysis(fd)
            color = "green" if result.fe_grade == "Excellent" else "blue" if result.fe_grade == "Good" else "orange" if result.fe_grade == "Adequate" else "red"
            st.markdown(f"### {period} — Funding Efficiency: :{color}[{result.fe_grade}] ({result.fe_score}/10)")

            c1, c2, c3, c4 = st.columns(4)
            _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
            _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
            c1.metric("Debt/Cap", _pct(result.debt_to_capitalization))
            c2.metric("Equity Multiplier", f"{_r2(result.equity_multiplier)}x")
            c3.metric("EBITDA Coverage", f"{_r2(result.interest_coverage_ebitda)}x")
            c4.metric("Cost of Debt", _pct(result.cost_of_debt))

            details = {"Metric": [], "Value": []}
            details["Metric"].append("Weighted Funding Cost")
            details["Value"].append(_pct(result.weighted_funding_cost))
            details["Metric"].append("Funding Spread")
            details["Value"].append(_pct(result.funding_spread))
            details["Metric"].append("Score")
            details["Value"].append(f"{result.fe_score}/10")
            details["Metric"].append("Grade")
            details["Value"].append(result.fe_grade)
            st.table(details)

            st.caption(result.summary)

    def _render_cash_flow_stability(self, df: pd.DataFrame):
        """Render Phase 125: Cash Flow Stability Analysis."""
        analyzer = CharlieAnalyzer()
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.cash_flow_stability_analysis(data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.cfs_grade, "gray")
        st.markdown(f"**Cash Flow Stability Grade:** :{color}[{result.cfs_grade}] ({result.cfs_score}/10)")

        c1, c2, c3, c4 = st.columns(4)
        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        c1.metric("OCF Margin", _pct(result.ocf_margin))
        c2.metric("OCF/EBITDA", _r2(result.ocf_to_ebitda))
        c3.metric("Dividend Coverage", f"{_r2(result.dividend_coverage)}x")
        c4.metric("CF Sufficiency", _r2(result.cash_flow_sufficiency))

        detail = {
            "Metric": ["OCF Margin", "OCF/EBITDA", "OCF/Debt Service",
                        "CapEx/OCF", "Dividend Coverage", "Cash Flow Sufficiency"],
            "Value": [_pct(result.ocf_margin), _r2(result.ocf_to_ebitda),
                      f"{_r2(result.ocf_to_debt_service)}x", _pct(result.capex_to_ocf),
                      f"{_r2(result.dividend_coverage)}x", _r2(result.cash_flow_sufficiency)],
        }
        st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)

        st.caption(result.summary)

    def _render_income_quality(self, df: pd.DataFrame):
        """Render Phase 124: Income Quality Analysis."""
        analyzer = CharlieAnalyzer()
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.income_quality_analysis(data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.iq_grade, "gray")
        st.markdown(f"**Income Quality Grade:** :{color}[{result.iq_grade}] ({result.iq_score}/10)")

        c1, c2, c3, c4 = st.columns(4)
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        c1.metric("OCF/Net Income", _r2(result.ocf_to_net_income))
        c2.metric("Accruals Ratio", _pct(result.accruals_ratio))
        c3.metric("Cash Earnings", _r2(result.cash_earnings_ratio))
        c4.metric("Earnings Persistence", _pct(result.earnings_persistence))

        detail = {
            "Metric": ["OCF/Net Income", "Accruals Ratio", "Cash Earnings Ratio",
                        "Non-Cash Ratio", "Earnings Persistence", "Operating Income Ratio"],
            "Value": [_r2(result.ocf_to_net_income), _pct(result.accruals_ratio),
                      _r2(result.cash_earnings_ratio), _r2(result.non_cash_ratio),
                      _pct(result.earnings_persistence), _r2(result.operating_income_ratio)],
        }
        st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)

        st.caption(result.summary)

    def _render_dupont_analysis(self, df: pd.DataFrame):
        """Render Phase 119: DuPont Analysis."""
        analyzer = CharlieAnalyzer()
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.dupont_analysis(data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.da_grade, "gray")
        st.markdown(f"**DuPont Analysis Grade:** :{color}[{result.da_grade}] ({result.da_score}/10)")

        c1, c2, c3, c4 = st.columns(4)
        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        c1.metric("ROE (DuPont)", _pct(result.roe_dupont))
        c2.metric("Net Profit Margin", _pct(result.net_profit_margin))
        c3.metric("Asset Turnover", _r2(result.asset_turnover))
        c4.metric("Equity Multiplier", _r2(result.equity_multiplier))

        detail = {
            "Metric": ["ROE (DuPont)", "Net Profit Margin", "Asset Turnover",
                        "Equity Multiplier", "ROA", "Leverage Effect"],
            "Value": [_pct(result.roe_dupont), _pct(result.net_profit_margin),
                      _r2(result.asset_turnover), _r2(result.equity_multiplier),
                      _pct(result.roa), _pct(result.leverage_effect)],
        }
        st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)

        st.caption(result.summary)

    def _render_receivables_management(self, df: pd.DataFrame):
        """Render Phase 114: Receivables Management Analysis."""
        from financial_analyzer import ReceivablesManagementResult
        analyzer = CharlieAnalyzer()
        fd = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.receivables_management_analysis(fd)
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.rm_grade, "gray")
        st.markdown(f"### Receivables Management &mdash; :{color}[{result.rm_grade}] ({result.rm_score}/10)")
        c1, c2, c3, c4 = st.columns(4)
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        _d = lambda v: f"{v:.1f}" if v is not None else "N/A"
        c1.metric("DSO (days)", _d(result.dso))
        c2.metric("AR/Revenue", _r2(result.ar_to_revenue))
        c3.metric("Recv. Turnover", _r2(result.receivables_turnover))
        c4.metric("Collection Eff.", _r2(result.collection_effectiveness))
        detail = {
            "DSO (days)": _d(result.dso),
            "AR / Revenue": _r2(result.ar_to_revenue),
            "AR / Current Assets": _r2(result.ar_to_current_assets),
            "Receivables Turnover": _r2(result.receivables_turnover),
            "Collection Effectiveness": _r2(result.collection_effectiveness),
            "AR Concentration": _r2(result.ar_concentration),
        }
        st.table(detail)
        st.caption(result.summary)

    def _render_solvency_depth(self, df: pd.DataFrame):
        """Render Phase 109: Solvency Depth Analysis."""
        from financial_analyzer import SolvencyDepthResult
        analyzer = CharlieAnalyzer()
        fd = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.solvency_depth_analysis(fd)
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.sd_grade, "gray")
        st.markdown(f"### Solvency Depth &mdash; :{color}[{result.sd_grade}] ({result.sd_score}/10)")
        c1, c2, c3, c4 = st.columns(4)
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        c1.metric("D/EBITDA", _r2(result.debt_to_ebitda))
        c2.metric("D/E", _r2(result.debt_to_equity))
        c3.metric("Interest Coverage", _r2(result.interest_coverage_ratio))
        c4.metric("Financial Leverage", _r2(result.financial_leverage))
        detail = {
            "Debt-to-Equity": _r2(result.debt_to_equity),
            "Debt-to-Assets": _r2(result.debt_to_assets),
            "Equity-to-Assets": _r2(result.equity_to_assets),
            "Interest Coverage": _r2(result.interest_coverage_ratio),
            "Debt-to-EBITDA": _r2(result.debt_to_ebitda),
            "Financial Leverage": _r2(result.financial_leverage),
        }
        st.table(detail)
        st.caption(result.summary)

    def _render_operational_leverage_depth(self, df: pd.DataFrame):
        """Render Phase 105: Operational Leverage Depth tab."""
        from financial_analyzer import CharlieAnalyzer, OperationalLeverageDepthResult
        analyzer = CharlieAnalyzer()
        fd = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.operational_leverage_depth_analysis(fd)
        grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.old_grade, "gray")
        st.markdown(f"### Operational Leverage Depth: :{grade_color}[{result.old_grade}] ({result.old_score}/10)")
        c1, c2, c3, c4 = st.columns(4)
        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        c1.metric("Contribution Margin", _pct(result.contribution_margin))
        c2.metric("Breakeven Coverage", _r2(result.breakeven_coverage))
        c3.metric("Fixed Cost Ratio", _pct(result.fixed_cost_ratio))
        c4.metric("Cost Flexibility", _pct(result.cost_flexibility))
        detail = {
            "Fixed Cost Ratio": _pct(result.fixed_cost_ratio),
            "Variable Cost Ratio": _pct(result.variable_cost_ratio),
            "Contribution Margin": _pct(result.contribution_margin),
            "DOL Proxy": _r2(result.dol_proxy),
            "Breakeven Coverage": _r2(result.breakeven_coverage),
            "Cost Flexibility": _pct(result.cost_flexibility),
        }
        st.table(detail)
        st.caption(result.summary)

    def _render_profitability_depth(self, df: pd.DataFrame):
        """Phase 103: Profitability Depth tab."""
        from financial_analyzer import ProfitabilityDepthResult
        analyzer = self.analyzer
        fd = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.profitability_depth_analysis(fd)
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.pd_grade, "gray")
        st.markdown(f"**Profitability Depth Grade:** :{color}[{result.pd_grade}] ({result.pd_score}/10)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gross Margin", f"{result.gross_margin:.1%}" if result.gross_margin is not None else "N/A")
        c2.metric("Operating Margin", f"{result.operating_margin:.1%}" if result.operating_margin is not None else "N/A")
        c3.metric("Net Margin", f"{result.net_margin:.1%}" if result.net_margin is not None else "N/A")
        c4.metric("EBITDA Margin", f"{result.ebitda_margin:.1%}" if result.ebitda_margin is not None else "N/A")
        details = {
            "Margin Spread (GM-NM)": result.margin_spread,
            "Profit Retention Ratio": result.profit_retention_ratio,
        }
        rows = [[k, f"{v:.4f}" if v is not None else "N/A"] for k, v in details.items()]
        st.table(pd.DataFrame(rows, columns=["Metric", "Value"]))
        st.caption(result.summary)

    def _render_revenue_efficiency(self, df: pd.DataFrame):
        """Phase 102: Revenue Efficiency tab."""
        from financial_analyzer import RevenueEfficiencyResult
        analyzer = self.analyzer
        fd = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.revenue_efficiency_analysis(fd)
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.rev_eff_grade, "gray")
        st.markdown(f"**Revenue Efficiency Grade:** :{color}[{result.rev_eff_grade}] ({result.rev_eff_score}/10)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cash Conversion", f"{result.cash_conversion_efficiency:.1%}" if result.cash_conversion_efficiency is not None else "N/A")
        c2.metric("Gross Margin", f"{result.gross_margin_efficiency:.1%}" if result.gross_margin_efficiency is not None else "N/A")
        c3.metric("Op Leverage", f"{result.operating_leverage_ratio:.2f}" if result.operating_leverage_ratio is not None else "N/A")
        c4.metric("Rev/Equity", f"{result.revenue_to_equity:.2f}x" if result.revenue_to_equity is not None else "N/A")
        details = {
            "Revenue/Assets": result.revenue_per_asset,
            "Net Revenue Retention": result.net_revenue_retention,
        }
        rows = [[k, f"{v:.4f}" if v is not None else "N/A"] for k, v in details.items()]
        st.table(pd.DataFrame(rows, columns=["Metric", "Value"]))
        st.caption(result.summary)

    def _render_debt_composition(self, df: pd.DataFrame):
        """Phase 101: Debt Composition tab."""
        from financial_analyzer import DebtCompositionResult
        analyzer = self.analyzer
        fd = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.debt_composition_analysis(fd)
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.dco_grade, "gray")
        st.markdown(f"**Debt Composition Grade:** :{color}[{result.dco_grade}] ({result.dco_score}/10)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Debt/Equity", f"{result.debt_to_equity:.2f}x" if result.debt_to_equity is not None else "N/A")
        c2.metric("Debt/Assets", f"{result.debt_to_assets:.1%}" if result.debt_to_assets is not None else "N/A")
        c3.metric("Interest Burden", f"{result.interest_burden:.1%}" if result.interest_burden is not None else "N/A")
        c4.metric("Coverage Margin", f"{result.debt_coverage_margin:.2f}" if result.debt_coverage_margin is not None else "N/A")
        details = {
            "Long-term Debt Ratio": result.long_term_debt_ratio,
            "Debt Cost Ratio": result.debt_cost_ratio,
        }
        rows = [[k, f"{v:.4f}" if v is not None else "N/A"] for k, v in details.items()]
        st.table(pd.DataFrame(rows, columns=["Metric", "Value"]))
        st.caption(result.summary)

    def _render_operational_risk(self, df: pd.DataFrame):
        """Phase 92: Operational Risk tab."""
        from financial_analyzer import OperationalRiskResult
        analyzer = self.analyzer
        fd = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.operational_risk_analysis(fd)
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.or_grade, "gray")
        st.markdown(f"**Operational Risk Grade:** :{color}[{result.or_grade}] ({result.or_score}/10)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Margin of Safety", f"{result.margin_of_safety:.1%}" if result.margin_of_safety is not None else "N/A")
        c2.metric("Cost Rigidity", f"{result.cost_rigidity:.1%}" if result.cost_rigidity is not None else "N/A")
        c3.metric("Risk Buffer", f"{result.risk_buffer:.2f}x" if result.risk_buffer is not None else "N/A")
        c4.metric("Cash Burn Ratio", f"{result.cash_burn_ratio:.2f}" if result.cash_burn_ratio is not None else "N/A")
        details = {
            "Breakeven Ratio": result.breakeven_ratio,
            "Operating Leverage": result.operating_leverage,
        }
        rows = [[k, f"{v:.4f}" if v is not None else "N/A"] for k, v in details.items()]
        st.table(pd.DataFrame(rows, columns=["Metric", "Value"]))
        st.caption(result.summary)

    def _render_financial_health_score(self, df: pd.DataFrame):
        """Phase 90: Financial Health Score tab."""
        from financial_analyzer import FinancialHealthScoreResult
        analyzer = self.analyzer
        fd = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.financial_health_score_analysis(fd)
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.fh_grade, "gray")
        st.markdown(f"**Financial Health Grade:** :{color}[{result.fh_grade}] ({result.fh_score}/10)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Profitability", f"{result.profitability_score:.1f}" if result.profitability_score is not None else "N/A")
        c2.metric("Liquidity", f"{result.liquidity_score:.1f}" if result.liquidity_score is not None else "N/A")
        c3.metric("Solvency", f"{result.solvency_score:.1f}" if result.solvency_score is not None else "N/A")
        c4.metric("Composite", f"{result.composite_score:.1f}" if result.composite_score is not None else "N/A")
        details = {
            "Efficiency": result.efficiency_score,
            "Coverage": result.coverage_score,
        }
        rows = [[k, f"{v:.1f}" if v is not None else "N/A"] for k, v in details.items()]
        st.table(pd.DataFrame(rows, columns=["Pillar", "Score"]))
        st.caption(result.summary)

    def _render_asset_quality(self, df: pd.DataFrame):
        """Phase 86: Asset Quality Analysis tab."""
        from financial_analyzer import AssetQualityResult
        fd = self.analyzer._dataframe_to_financial_data(df)
        result: AssetQualityResult = self.analyzer.asset_quality_analysis(fd)

        grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.aq_grade, "gray")
        st.markdown(f"**Grade:** :{grade_color}[{result.aq_grade}] | **Score:** {result.aq_score:.1f}/10")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Asset Ratio", f"{result.current_asset_ratio:.1%}" if result.current_asset_ratio is not None else "N/A")
        c2.metric("Fixed Asset Ratio", f"{result.fixed_asset_ratio:.1%}" if result.fixed_asset_ratio is not None else "N/A")
        c3.metric("Cash/CA", f"{result.cash_to_current_assets:.1%}" if result.cash_to_current_assets is not None else "N/A")
        c4.metric("AR/TA", f"{result.receivables_to_assets:.1%}" if result.receivables_to_assets is not None else "N/A")

        details = {"Metric": [], "Value": []}
        for label, val, fmt in [
            ("Tangible Asset Ratio", result.tangible_asset_ratio, ".4f"),
            ("Fixed Asset Ratio", result.fixed_asset_ratio, ".4f"),
            ("Current Asset Ratio", result.current_asset_ratio, ".4f"),
            ("Cash / Current Assets", result.cash_to_current_assets, ".4f"),
            ("Receivables / Assets", result.receivables_to_assets, ".4f"),
            ("Inventory / Assets", result.inventory_to_assets, ".4f"),
            ("Score", result.aq_score, ".1f"),
            ("Grade", result.aq_grade, ""),
        ]:
            details["Metric"].append(label)
            details["Value"].append(f"{val:{fmt}}" if val is not None and fmt else (val if val is not None else "N/A"))
        st.table(details)

        st.caption(result.summary)

    def _render_financial_resilience(self, df: pd.DataFrame):
        """Phase 82: Financial Resilience Analysis tab."""
        from financial_analyzer import FinancialResilienceResult
        fd = self.analyzer._dataframe_to_financial_data(df)
        result: FinancialResilienceResult = self.analyzer.financial_resilience_analysis(fd)

        grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.fr_grade, "gray")
        st.markdown(f"**Grade:** :{grade_color}[{result.fr_grade}] | **Score:** {result.fr_score:.1f}/10")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cash/Assets", f"{result.cash_to_assets:.1%}" if result.cash_to_assets is not None else "N/A")
        c2.metric("Cash/Debt", f"{result.cash_to_debt:.2f}x" if result.cash_to_debt is not None else "N/A")
        c3.metric("OCF Coverage", f"{result.operating_cash_coverage:.2f}x" if result.operating_cash_coverage is not None else "N/A")
        c4.metric("Resilience Buffer", f"{result.resilience_buffer:.2f}x" if result.resilience_buffer is not None else "N/A")

        details = {"Metric": [], "Value": []}
        for label, val, fmt in [
            ("Cash / Assets", result.cash_to_assets, ".4f"),
            ("Cash / Debt", result.cash_to_debt, ".4f"),
            ("Operating Cash Coverage", result.operating_cash_coverage, ".4f"),
            ("Interest Coverage (Cash)", result.interest_coverage_cash, ".4f"),
            ("Free Cash Margin", result.free_cash_margin, ".4f"),
            ("Resilience Buffer", result.resilience_buffer, ".4f"),
            ("Score", result.fr_score, ".1f"),
            ("Grade", result.fr_grade, ""),
        ]:
            details["Metric"].append(label)
            details["Value"].append(f"{val:{fmt}}" if val is not None and fmt else (val if val is not None else "N/A"))
        st.table(details)

        st.caption(result.summary)

    def _render_equity_multiplier(self, df: pd.DataFrame):
        """Phase 81: Equity Multiplier Analysis tab."""
        from financial_analyzer import EquityMultiplierResult
        fd = self.analyzer._dataframe_to_financial_data(df)
        result: EquityMultiplierResult = self.analyzer.equity_multiplier_analysis(fd)

        grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.em_grade, "gray")
        st.markdown(f"**Grade:** :{grade_color}[{result.em_grade}] | **Score:** {result.em_score:.1f}/10")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equity Multiplier", f"{result.equity_multiplier:.2f}x" if result.equity_multiplier is not None else "N/A")
        c2.metric("Debt Ratio", f"{result.debt_ratio:.1%}" if result.debt_ratio is not None else "N/A")
        c3.metric("Equity Ratio", f"{result.equity_ratio:.1%}" if result.equity_ratio is not None else "N/A")
        c4.metric("DuPont ROE", f"{result.dupont_roe:.2%}" if result.dupont_roe is not None else "N/A")

        details = {"Metric": [], "Value": []}
        for label, val, fmt in [
            ("Equity Multiplier", result.equity_multiplier, ".2f"),
            ("Debt Ratio", result.debt_ratio, ".4f"),
            ("Equity Ratio", result.equity_ratio, ".4f"),
            ("Financial Leverage Index", result.financial_leverage_index, ".4f"),
            ("DuPont ROE", result.dupont_roe, ".4f"),
            ("Leverage Spread", result.leverage_spread, ".4f"),
            ("Score", result.em_score, ".1f"),
            ("Grade", result.em_grade, ""),
        ]:
            details["Metric"].append(label)
            details["Value"].append(f"{val:{fmt}}" if val is not None and fmt else (val if val is not None else "N/A"))
        st.table(details)

        st.caption(result.summary)

    def _render_defensive_interval(self, df: pd.DataFrame):
        """Phase 80: Defensive Interval Analysis tab."""
        from financial_analyzer import DefensiveIntervalResult
        fd = self.analyzer._dataframe_to_financial_data(df)
        result: DefensiveIntervalResult = self.analyzer.defensive_interval_analysis(fd)

        grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.di_grade, "gray")
        st.markdown(f"### Defensive Interval &mdash; :{grade_color}[{result.di_grade}] ({result.di_score}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Defensive Interval", f"{result.defensive_interval_days:.0f} days" if result.defensive_interval_days is not None else "N/A")
        c2.metric("Cash Interval", f"{result.cash_interval_days:.0f} days" if result.cash_interval_days is not None else "N/A")
        c3.metric("Liquid Reserve Adequacy", f"{result.liquid_reserve_adequacy:.2f}x" if result.liquid_reserve_adequacy is not None else "N/A")
        c4.metric("OpEx Coverage", f"{result.operating_expense_coverage:.2f}x" if result.operating_expense_coverage is not None else "N/A")

        details = {"Metric": [], "Value": []}
        for label, val in [
            ("Defensive Interval (days)", result.defensive_interval_days),
            ("Cash Interval (days)", result.cash_interval_days),
            ("Liquid Assets Ratio", result.liquid_assets_ratio),
            ("Days Cash on Hand", result.days_cash_on_hand),
            ("Liquid Reserve Adequacy", result.liquid_reserve_adequacy),
            ("OpEx Coverage", result.operating_expense_coverage),
            ("Score", result.di_score),
            ("Grade", result.di_grade),
        ]:
            details["Metric"].append(label)
            fmt = f"{val:.4f}" if isinstance(val, float) else (val if val is not None else "N/A")
            details["Value"].append(fmt)
        st.table(details)

        st.caption(result.summary)

    def _render_cash_burn(self, df: pd.DataFrame):
        """Phase 79: Cash Burn Analysis tab."""
        from financial_analyzer import CashBurnResult
        fd = self.analyzer._dataframe_to_financial_data(df)
        result: CashBurnResult = self.analyzer.cash_burn_analysis(fd)

        grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.cb_grade, "gray")
        st.markdown(f"### Cash Burn &mdash; :{grade_color}[{result.cb_grade}] ({result.cb_score}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OCF Margin", f"{result.ocf_margin:.1%}" if result.ocf_margin is not None else "N/A")
        c2.metric("FCF Margin", f"{result.fcf_margin:.1%}" if result.fcf_margin is not None else "N/A")
        c3.metric("CapEx Intensity", f"{result.capex_intensity:.1%}" if result.capex_intensity is not None else "N/A")
        c4.metric("Cash Self-Sufficiency", f"{result.cash_self_sufficiency:.2f}x" if result.cash_self_sufficiency is not None else "N/A")

        details = {"Metric": [], "Value": []}
        for label, val in [
            ("OCF Margin", result.ocf_margin),
            ("CapEx Intensity", result.capex_intensity),
            ("FCF Margin", result.fcf_margin),
            ("Cash Self-Sufficiency", result.cash_self_sufficiency),
            ("Cash Runway (months)", result.cash_runway_months),
            ("Net Cash Position", result.net_cash_position),
            ("Score", result.cb_score),
            ("Grade", result.cb_grade),
        ]:
            details["Metric"].append(label)
            fmt = f"{val:.4f}" if isinstance(val, float) else (val if val is not None else "N/A")
            details["Value"].append(fmt)
        st.table(details)

        st.caption(result.summary)

    def _render_profit_retention(self, df: pd.DataFrame):
        """Phase 78: Profit Retention Analysis tab."""
        from financial_analyzer import ProfitRetentionResult
        fd = self.analyzer._dataframe_to_financial_data(df)
        result: ProfitRetentionResult = self.analyzer.profit_retention_analysis(fd)

        grade_color = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}.get(result.pr_grade, "gray")
        st.markdown(f"### Profit Retention &mdash; :{grade_color}[{result.pr_grade}] ({result.pr_score}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retention Ratio", f"{result.retention_ratio:.1%}" if result.retention_ratio is not None else "N/A")
        c2.metric("Payout Ratio", f"{result.payout_ratio:.1%}" if result.payout_ratio is not None else "N/A")
        c3.metric("RE / Equity", f"{result.re_to_equity:.1%}" if result.re_to_equity is not None else "N/A")
        c4.metric("Sustainable Growth", f"{result.sustainable_growth_rate:.1%}" if result.sustainable_growth_rate is not None else "N/A")

        details = {"Metric": [], "Value": []}
        for label, val in [
            ("Retention Ratio", result.retention_ratio),
            ("Payout Ratio", result.payout_ratio),
            ("RE / Equity", result.re_to_equity),
            ("Sustainable Growth Rate", result.sustainable_growth_rate),
            ("Internal Growth Rate", result.internal_growth_rate),
            ("Plowback Amount", result.plowback_amount),
            ("Score", result.pr_score),
            ("Grade", result.pr_grade),
        ]:
            details["Metric"].append(label)
            fmt = f"{val:.4f}" if isinstance(val, float) else (val if val is not None else "N/A")
            details["Value"].append(fmt)
        st.table(details)

        st.caption(result.summary)

    def _render_debt_service_coverage(self, df: pd.DataFrame):
        """Render Debt Service Coverage tab."""
        from financial_analyzer import DebtServiceCoverageResult
        analyzer = CharlieAnalyzer()
        financial_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.debt_service_coverage_analysis(financial_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.dsc_grade, "gray")
        st.markdown(f"**Debt Service Coverage Grade:** :{color}[{result.dsc_grade}] (Score: {result.dsc_score:.1f}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DSCR", f"{result.dscr:.2f}x" if result.dscr is not None else "N/A")
        c2.metric("OCF/Debt Service", f"{result.ocf_to_debt_service:.2f}x" if result.ocf_to_debt_service is not None else "N/A")
        c3.metric("EBITDA/Interest", f"{result.ebitda_to_interest:.2f}x" if result.ebitda_to_interest is not None else "N/A")
        c4.metric("Coverage Cushion", f"{result.coverage_cushion:.2f}x" if result.coverage_cushion is not None else "N/A")

        detail_data = {
            "Metric": ["DSCR", "OCF/Debt Svc", "EBITDA/Interest", "FCF/Debt Svc", "Debt Svc/Revenue", "Coverage Cushion"],
            "Value": [
                f"{result.dscr:.4f}" if result.dscr is not None else "N/A",
                f"{result.ocf_to_debt_service:.4f}" if result.ocf_to_debt_service is not None else "N/A",
                f"{result.ebitda_to_interest:.4f}" if result.ebitda_to_interest is not None else "N/A",
                f"{result.fcf_to_debt_service:.4f}" if result.fcf_to_debt_service is not None else "N/A",
                f"{result.debt_service_to_revenue:.4f}" if result.debt_service_to_revenue is not None else "N/A",
                f"{result.coverage_cushion:.4f}" if result.coverage_cushion is not None else "N/A",
            ],
        }
        st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)
        st.caption(result.summary)

    def _render_capital_allocation(self, df: pd.DataFrame):
        """Render Capital Allocation tab."""
        from financial_analyzer import CapitalAllocationResult
        analyzer = CharlieAnalyzer()
        financial_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.capital_allocation_analysis(financial_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.ca_grade, "gray")
        st.markdown(f"**Capital Allocation Grade:** :{color}[{result.ca_grade}] (Score: {result.ca_score:.1f}/10)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CapEx/Revenue", f"{result.capex_to_revenue:.2%}" if result.capex_to_revenue is not None else "N/A")
        c2.metric("CapEx/OCF", f"{result.capex_to_ocf:.2%}" if result.capex_to_ocf is not None else "N/A")
        c3.metric("Reinvestment Rate", f"{result.reinvestment_rate:.2%}" if result.reinvestment_rate is not None else "N/A")
        c4.metric("FCF Yield", f"{result.fcf_yield:.2%}" if result.fcf_yield is not None else "N/A")

        detail_data = {
            "Metric": ["CapEx/Revenue", "CapEx/OCF", "Shareholder Return", "Reinvestment Rate", "FCF Yield", "Payout/FCF"],
            "Value": [
                f"{result.capex_to_revenue:.4f}" if result.capex_to_revenue is not None else "N/A",
                f"{result.capex_to_ocf:.4f}" if result.capex_to_ocf is not None else "N/A",
                f"{result.shareholder_return_ratio:.4f}" if result.shareholder_return_ratio is not None else "N/A",
                f"{result.reinvestment_rate:.4f}" if result.reinvestment_rate is not None else "N/A",
                f"{result.fcf_yield:.4f}" if result.fcf_yield is not None else "N/A",
                f"{result.total_payout_to_fcf:.4f}" if result.total_payout_to_fcf is not None else "N/A",
            ],
        }
        st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)
        st.caption(result.summary)

    def _render_tax_efficiency(self, df: pd.DataFrame):
        """Render Tax Efficiency tab."""
        from financial_analyzer import TaxEfficiencyResult
        analyzer = CharlieAnalyzer()
        financial_data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.tax_efficiency_analysis(financial_data)

        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.te_grade, "gray")
        st.markdown(f"**Tax Efficiency Grade:** :{color}[{result.te_grade}] — Score: {result.te_score}/10")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Effective Tax Rate", f"{result.effective_tax_rate:.1%}" if result.effective_tax_rate is not None else "N/A")
        c2.metric("Tax/Revenue", f"{result.tax_to_revenue:.1%}" if result.tax_to_revenue is not None else "N/A")
        c3.metric("After-Tax Margin", f"{result.after_tax_margin:.1%}" if result.after_tax_margin is not None else "N/A")
        c4.metric("Tax Shield Ratio", f"{result.tax_shield_ratio:.2f}" if result.tax_shield_ratio is not None else "N/A")

        details = {
            "Effective Tax Rate": f"{result.effective_tax_rate:.2%}" if result.effective_tax_rate is not None else "N/A",
            "Tax / Revenue": f"{result.tax_to_revenue:.2%}" if result.tax_to_revenue is not None else "N/A",
            "Tax / EBITDA": f"{result.tax_to_ebitda:.2%}" if result.tax_to_ebitda is not None else "N/A",
            "After-Tax Margin": f"{result.after_tax_margin:.2%}" if result.after_tax_margin is not None else "N/A",
            "Tax Shield Ratio": f"{result.tax_shield_ratio:.4f}" if result.tax_shield_ratio is not None else "N/A",
            "PreTax / EBIT": f"{result.pretax_to_ebit:.2f}" if result.pretax_to_ebit is not None else "N/A",
        }
        st.table(details)

        st.caption(result.summary)

    def _render_roic_analysis(self, df: pd.DataFrame):
        """Render ROIC Analysis tab."""
        st.subheader("Return on Invested Capital (ROIC)")
        from financial_analyzer import CharlieAnalyzer, ROICResult
        analyzer = CharlieAnalyzer()
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.roic_analysis(data)

        if result.roic_pct is None:
            st.warning("Insufficient data for ROIC analysis.")
            return

        # Grade badge
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.roic_grade, "gray")
        st.markdown(
            f"<span style='background-color:{color};color:white;padding:4px 12px;"
            f"border-radius:8px;font-weight:bold'>{result.roic_grade}</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        # 4-column metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROIC", f"{result.roic_pct:.1f}%")
        c2.metric("NOPAT", f"${result.nopat:,.0f}" if result.nopat is not None else "N/A")
        c3.metric("Invested Capital", f"${result.invested_capital:,.0f}" if result.invested_capital is not None else "N/A")
        c4.metric("Score", f"{result.roic_score:.1f} / 10")

        # Detail table
        details = {"Metric": [], "Value": []}
        for label, val, fmt in [
            ("ROIC %", result.roic_pct, ".1f"),
            ("NOPAT", result.nopat, ",.0f"),
            ("Invested Capital", result.invested_capital, ",.0f"),
            ("ROIC-ROA Spread", result.roic_roa_spread, ".1f"),
            ("Capital Efficiency", result.capital_efficiency, ".2f"),
            ("ROIC Score", result.roic_score, ".1f"),
        ]:
            details["Metric"].append(label)
            details["Value"].append(f"{val:{fmt}}" if val is not None else "N/A")
        details["Metric"].append("Grade")
        details["Value"].append(result.roic_grade)
        st.table(details)

        st.caption(result.summary)

    def _render_roa_quality(self, df: pd.DataFrame):
        """Render Return on Assets Quality tab."""
        st.subheader("Return on Assets (ROA) Quality")
        from financial_analyzer import CharlieAnalyzer, ROAQualityResult
        analyzer = CharlieAnalyzer()
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.roa_quality_analysis(data)

        if result.roa_pct is None:
            st.warning("Insufficient data for ROA Quality analysis.")
            return

        # Grade badge
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.roa_grade, "gray")
        st.markdown(
            f"<span style='background-color:{color};color:white;padding:4px 12px;"
            f"border-radius:8px;font-weight:bold'>{result.roa_grade}</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        # 4-column metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROA", f"{result.roa_pct:.1f}%")
        c2.metric("Asset Turnover", f"{result.asset_turnover:.2f}x" if result.asset_turnover is not None else "N/A")
        c3.metric("Capital Intensity", f"{result.capital_intensity:.2f}x" if result.capital_intensity is not None else "N/A")
        c4.metric("Score", f"{result.roa_score:.1f} / 10")

        # Detail table
        details = {"Metric": [], "Value": []}
        for label, val, fmt in [
            ("ROA %", result.roa_pct, ".1f"),
            ("Operating ROA %", result.operating_roa_pct, ".1f"),
            ("Cash ROA %", result.cash_roa_pct, ".1f"),
            ("Asset Turnover", result.asset_turnover, ".2f"),
            ("Fixed Asset Turnover", result.fixed_asset_turnover, ".2f"),
            ("Capital Intensity", result.capital_intensity, ".2f"),
            ("ROA Score", result.roa_score, ".1f"),
        ]:
            details["Metric"].append(label)
            details["Value"].append(f"{val:{fmt}}" if val is not None else "N/A")
        details["Metric"].append("Grade")
        details["Value"].append(result.roa_grade)
        st.table(details)

        # Bar chart: ROA vs Operating ROA vs Cash ROA
        import plotly.graph_objects as go
        labels = []
        vals = []
        colors = []
        if result.roa_pct is not None:
            labels.append("ROA")
            vals.append(result.roa_pct)
            colors.append("#4CAF50")
        if result.operating_roa_pct is not None:
            labels.append("Operating ROA")
            vals.append(result.operating_roa_pct)
            colors.append("#FF9800")
        if result.cash_roa_pct is not None:
            labels.append("Cash ROA")
            vals.append(result.cash_roa_pct)
            colors.append("#2196F3")

        if labels:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels, y=vals, marker_color=colors))
            fig.update_layout(
                title="ROA Comparison",
                yaxis_title="Return (%)",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_roe_analysis(self, df: pd.DataFrame):
        """Render Return on Equity Analysis tab."""
        st.subheader("Return on Equity (ROE) Analysis")
        from financial_analyzer import CharlieAnalyzer, ROEAnalysisResult
        analyzer = CharlieAnalyzer()
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.roe_analysis(data)

        if result.roe_pct is None:
            st.warning("Insufficient data for ROE analysis.")
            return

        # Grade badge
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.roe_grade, "gray")
        st.markdown(
            f"<span style='background-color:{color};color:white;padding:4px 12px;"
            f"border-radius:8px;font-weight:bold'>{result.roe_grade}</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        # 4-column metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROE", f"{result.roe_pct:.1f}%")
        c2.metric("Asset Turnover", f"{result.asset_turnover:.2f}x" if result.asset_turnover is not None else "N/A")
        c3.metric("Equity Multiplier", f"{result.equity_multiplier:.2f}x" if result.equity_multiplier is not None else "N/A")
        c4.metric("Score", f"{result.roe_score:.1f} / 10")

        # Detail table
        details = {"Metric": [], "Value": []}
        for label, val, fmt in [
            ("ROE %", result.roe_pct, ".1f"),
            ("Net Margin %", result.net_margin_pct, ".1f"),
            ("Asset Turnover", result.asset_turnover, ".2f"),
            ("Equity Multiplier", result.equity_multiplier, ".2f"),
            ("ROA %", result.roa_pct, ".1f"),
            ("Retention Ratio", result.retention_ratio, ".2f"),
            ("Sustainable Growth %", result.sustainable_growth_rate, ".1f"),
            ("ROE Score", result.roe_score, ".1f"),
        ]:
            details["Metric"].append(label)
            details["Value"].append(f"{val:{fmt}}" if val is not None else "N/A")
        details["Metric"].append("Grade")
        details["Value"].append(result.roe_grade)
        st.table(details)

        # DuPont decomposition bar chart
        import plotly.graph_objects as go
        components = []
        values = []
        colors = []
        if result.net_margin_pct is not None:
            components.append("Net Margin %")
            values.append(result.net_margin_pct)
            colors.append("#4CAF50")
        if result.asset_turnover is not None:
            components.append("Asset Turnover")
            values.append(result.asset_turnover)
            colors.append("#FF9800")
        if result.equity_multiplier is not None:
            components.append("Equity Multiplier")
            values.append(result.equity_multiplier)
            colors.append("#2196F3")

        if components:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=components, y=values, marker_color=colors,
            ))
            fig.update_layout(
                title="DuPont Decomposition",
                yaxis_title="Value",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_net_profit_margin(self, df: pd.DataFrame):
        """Render Net Profit Margin Analysis tab."""
        st.subheader("Net Profit Margin Analysis")
        from financial_analyzer import CharlieAnalyzer, NetProfitMarginResult
        analyzer = CharlieAnalyzer()
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.net_profit_margin_analysis(data)

        if result.net_margin_pct is None:
            st.warning("Insufficient data for Net Profit Margin analysis.")
            return

        # Grade badge
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.npm_grade, "gray")
        st.markdown(
            f"<span style='background-color:{color};color:white;padding:4px 12px;"
            f"border-radius:8px;font-weight:bold'>{result.npm_grade}</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        # 4-column metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net Margin", f"{result.net_margin_pct:.1f}%")
        c2.metric("Tax Burden", f"{result.tax_burden:.2f}" if result.tax_burden is not None else "N/A")
        c3.metric("Interest Burden", f"{result.interest_burden:.2f}" if result.interest_burden is not None else "N/A")
        c4.metric("Score", f"{result.npm_score:.1f} / 10")

        # Detail table
        details = {"Metric": [], "Value": []}
        for label, val, fmt in [
            ("Net Margin %", result.net_margin_pct, ".1f"),
            ("EBITDA Margin %", result.ebitda_margin_pct, ".1f"),
            ("EBIT Margin %", result.ebit_margin_pct, ".1f"),
            ("Tax Burden (NI/EBT)", result.tax_burden, ".2f"),
            ("Interest Burden (EBT/EBIT)", result.interest_burden, ".2f"),
            ("NI / EBITDA", result.net_to_ebitda, ".2f"),
            ("NPM Score", result.npm_score, ".1f"),
        ]:
            details["Metric"].append(label)
            details["Value"].append(f"{val:{fmt}}" if val is not None else "N/A")
        details["Metric"].append("Grade")
        details["Value"].append(result.npm_grade)
        st.table(details)

        # Waterfall: EBITDA -> EBIT -> EBT -> NI
        import plotly.graph_objects as go
        if (result.ebitda_margin_pct is not None and result.ebit_margin_pct is not None
                and result.net_margin_pct is not None):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["EBITDA Margin", "EBIT Margin", "Net Margin"],
                y=[result.ebitda_margin_pct, result.ebit_margin_pct, result.net_margin_pct],
                marker_color=["#4CAF50", "#FF9800", "#2196F3"],
            ))
            fig.update_layout(
                title="Margin Waterfall: EBITDA to Net",
                yaxis_title="Margin (%)",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_ebitda_margin_quality(self, df: pd.DataFrame):
        """Render EBITDA Margin Quality Analysis tab."""
        st.subheader("EBITDA Margin Quality Analysis")
        from financial_analyzer import CharlieAnalyzer, EbitdaMarginQualityResult
        analyzer = CharlieAnalyzer()
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.ebitda_margin_quality_analysis(data)

        if result.ebitda_margin_pct is None:
            st.warning("Insufficient data for EBITDA Margin Quality analysis.")
            return

        # Grade badge
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.ebitda_grade, "gray")
        st.markdown(
            f"<span style='background-color:{color};color:white;padding:4px 12px;"
            f"border-radius:8px;font-weight:bold'>{result.ebitda_grade}</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        # 4-column metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EBITDA Margin", f"{result.ebitda_margin_pct:.1f}%")
        c2.metric("D&A Intensity", f"{result.da_intensity:.1f}%" if result.da_intensity is not None else "N/A")
        c3.metric("EBITDA/GP", f"{result.ebitda_to_gp:.2f}" if result.ebitda_to_gp is not None else "N/A")
        c4.metric("Score", f"{result.ebitda_score:.1f} / 10")

        # Detail table
        details = {"Metric": [], "Value": []}
        for label, val, fmt in [
            ("EBITDA Margin %", result.ebitda_margin_pct, ".1f"),
            ("Operating Margin %", result.operating_margin_pct, ".1f"),
            ("D&A Intensity %", result.da_intensity, ".1f"),
            ("EBITDA-OI Spread %", result.ebitda_oi_spread, ".1f"),
            ("EBITDA / GP", result.ebitda_to_gp, ".2f"),
            ("EBITDA Score", result.ebitda_score, ".1f"),
        ]:
            details["Metric"].append(label)
            details["Value"].append(f"{val:{fmt}}" if val is not None else "N/A")
        details["Metric"].append("Grade")
        details["Value"].append(result.ebitda_grade)
        st.table(details)

        # Bar chart: EBITDA vs Operating margin
        import plotly.graph_objects as go
        if result.ebitda_margin_pct is not None and result.operating_margin_pct is not None:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["EBITDA Margin", "Operating Margin"],
                y=[result.ebitda_margin_pct, result.operating_margin_pct],
                marker_color=["#FF9800", "#2196F3"],
            ))
            fig.update_layout(
                title="EBITDA vs Operating Margin",
                yaxis_title="Margin (%)",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    def _render_gross_margin_stability(self, df: pd.DataFrame):
        """Render Gross Margin Stability Analysis tab."""
        st.subheader("Gross Margin Stability Analysis")
        from financial_analyzer import CharlieAnalyzer, GrossMarginStabilityResult
        analyzer = CharlieAnalyzer()
        data = self.analyzer._dataframe_to_financial_data(df)
        result = analyzer.gross_margin_stability_analysis(data)

        if result.gross_margin_pct is None:
            st.warning("Insufficient data for Gross Margin Stability analysis.")
            return

        # Grade badge
        grade_colors = {"Excellent": "green", "Good": "blue", "Adequate": "orange", "Weak": "red"}
        color = grade_colors.get(result.gm_stability_grade, "gray")
        st.markdown(
            f"<span style='background-color:{color};color:white;padding:4px 12px;"
            f"border-radius:8px;font-weight:bold'>{result.gm_stability_grade}</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        # 4-column metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gross Margin", f"{result.gross_margin_pct:.1f}%")
        c2.metric("COGS Ratio", f"{result.cogs_ratio:.1f}%" if result.cogs_ratio is not None else "N/A")
        c3.metric("OpEx Coverage", f"{result.opex_coverage:.2f}x" if result.opex_coverage is not None else "N/A")
        c4.metric("Score", f"{result.gm_stability_score:.1f} / 10")

        # Detail table
        details = {"Metric": [], "Value": []}
        for label, val, fmt in [
            ("Gross Margin %", result.gross_margin_pct, ".1f"),
            ("COGS Ratio %", result.cogs_ratio, ".1f"),
            ("Operating Margin %", result.operating_margin_pct, ".1f"),
            ("Margin Spread %", result.margin_spread, ".1f"),
            ("OpEx Coverage (x)", result.opex_coverage, ".2f"),
            ("Margin Buffer %", result.margin_buffer, ".1f"),
            ("Stability Score", result.gm_stability_score, ".1f"),
        ]:
            details["Metric"].append(label)
            details["Value"].append(f"{val:{fmt}}" if val is not None else "N/A")
        details["Metric"].append("Grade")
        details["Value"].append(result.gm_stability_grade)
        st.table(details)

        # Bar chart: Gross Margin vs Operating Margin
        import plotly.graph_objects as go
        if result.gross_margin_pct is not None and result.operating_margin_pct is not None:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Gross Margin", "Operating Margin"],
                y=[result.gross_margin_pct, result.operating_margin_pct],
                marker_color=["#4CAF50", "#2196F3"],
            ))
            fig.update_layout(
                title="Margin Comparison",
                yaxis_title="Margin (%)",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(result.summary)

    # ===== UNDERWRITING TABS =====

    def _render_credit_scorecard(self, df: pd.DataFrame):
        """Render credit scorecard from underwriting analysis."""
        from underwriting import UnderwritingAnalyzer

        data = self.analyzer._dataframe_to_financial_data(df)
        ua = UnderwritingAnalyzer(self.analyzer)
        scorecard = ua.credit_scorecard(data)

        col1, col2, col3 = st.columns(3)
        col1.metric("Credit Score", f"{scorecard.total_score}/100")
        col2.metric("Grade", scorecard.grade)
        col3.metric("Recommendation", scorecard.recommendation.title())

        st.subheader("Category Breakdown")
        for cat, score in scorecard.category_scores.items():
            st.progress(score / 20, text=f"{cat.title()}: {score}/20")

        if scorecard.strengths:
            st.success("**Strengths:** " + ", ".join(scorecard.strengths))
        if scorecard.weaknesses:
            st.warning("**Weaknesses:** " + ", ".join(scorecard.weaknesses))
        if scorecard.conditions:
            st.info("**Conditions:** " + ", ".join(scorecard.conditions))

    def _render_debt_capacity(self, df: pd.DataFrame):
        """Render debt capacity analysis."""
        from underwriting import UnderwritingAnalyzer, LoanStructure

        data = self.analyzer._dataframe_to_financial_data(df)
        ua = UnderwritingAnalyzer(self.analyzer)

        st.subheader("Proposed Loan Parameters")
        col1, col2, col3 = st.columns(3)
        principal = col1.number_input("Principal ($)", value=1_000_000, step=100_000, key="uw_principal")
        rate = col2.number_input("Annual Rate (%)", value=5.0, step=0.25, key="uw_rate") / 100
        term = col3.number_input("Term (years)", value=5, step=1, key="uw_term")

        loan = LoanStructure(principal=principal, annual_rate=rate, term_years=term)
        result = ua.debt_capacity(data, loan)

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Leverage", f"{result.current_leverage:.1f}x" if result.current_leverage else "N/A")
        col2.metric("Pro-forma Leverage", f"{result.pro_forma_leverage:.1f}x" if result.pro_forma_leverage else "N/A")
        col3.metric("Pro-forma DSCR", f"{result.pro_forma_dscr:.2f}x" if result.pro_forma_dscr else "N/A")

        if result.max_additional_debt is not None:
            st.info(f"Max additional debt capacity: **${result.max_additional_debt:,.0f}**")
        st.markdown(f"**Assessment:** {result.assessment}")

    def _render_covenant_package(self, df: pd.DataFrame):
        """Render recommended covenant package."""
        from underwriting import UnderwritingAnalyzer

        data = self.analyzer._dataframe_to_financial_data(df)
        ua = UnderwritingAnalyzer(self.analyzer)
        scorecard = ua.credit_scorecard(data)
        covenants = ua.recommend_covenants(data, scorecard)

        st.metric("Covenant Tier", covenants.covenant_tier.title())

        st.subheader("Financial Covenants")
        for name, details in covenants.financial_covenants.items():
            threshold = details.get("threshold", "N/A")
            freq = details.get("frequency", "quarterly")
            st.markdown(f"- **{name.replace('_', ' ').title()}**: {threshold} ({freq})")

        if covenants.reporting_requirements:
            st.subheader("Reporting Requirements")
            for req in covenants.reporting_requirements:
                st.markdown(f"- {req}")

        if covenants.events_of_default:
            st.subheader("Events of Default")
            for event in covenants.events_of_default:
                st.markdown(f"- {event}")

    # ===== STARTUP MODELING TABS =====

    def _render_saas_metrics(self, df: pd.DataFrame):
        """Render SaaS metrics dashboard."""
        from startup_model import StartupAnalyzer

        data = self.analyzer._dataframe_to_financial_data(df)
        sa = StartupAnalyzer()
        metrics = sa.saas_metrics(data)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MRR", f"${metrics.mrr:,.0f}" if metrics.mrr else "N/A")
        col2.metric("ARR", f"${metrics.arr:,.0f}" if metrics.arr else "N/A")
        col3.metric("ARPU", f"${metrics.arpu:,.0f}" if metrics.arpu else "N/A")
        col4.metric("Customers", f"{metrics.customers:,}" if metrics.customers else "N/A")

        col1, col2, col3 = st.columns(3)
        col1.metric("Gross Churn", f"{metrics.gross_churn_rate:.1%}" if metrics.gross_churn_rate is not None else "N/A")
        # P0-9: GRR (1 - gross churn) is what we can compute without expansion data;
        # NRR requires expansion-revenue tracking, hence often N/A.
        col2.metric("Gross Revenue Retention", f"{metrics.gross_revenue_retention:.1%}" if metrics.gross_revenue_retention is not None else "N/A")
        col3.metric("Net Revenue Retention", f"{metrics.net_revenue_retention:.1%}" if metrics.net_revenue_retention is not None else "N/A")

        if metrics.interpretation:
            st.info(metrics.interpretation)

    def _render_unit_economics(self, df: pd.DataFrame):
        """Render unit economics analysis."""
        from startup_model import StartupAnalyzer

        data = self.analyzer._dataframe_to_financial_data(df)
        sa = StartupAnalyzer()
        ue = sa.unit_economics(data)

        col1, col2, col3 = st.columns(3)
        col1.metric("CAC", f"${ue.cac:,.0f}" if ue.cac else "N/A")
        col2.metric("LTV", f"${ue.ltv:,.0f}" if ue.ltv else "N/A")
        col3.metric("LTV/CAC", f"{ue.ltv_to_cac:.1f}x" if ue.ltv_to_cac else "N/A")

        col1, col2 = st.columns(2)
        col1.metric("Payback", f"{ue.payback_months:.0f} months" if ue.payback_months else "N/A")
        col2.metric("GM-Adj LTV", f"${ue.gross_margin_adjusted_ltv:,.0f}" if ue.gross_margin_adjusted_ltv else "N/A")

        if ue.interpretation:
            if "Healthy" in ue.interpretation:
                st.success(ue.interpretation)
            elif "Unsustainable" in ue.interpretation:
                st.error(ue.interpretation)
            else:
                st.warning(ue.interpretation)

    def _render_burn_runway(self, df: pd.DataFrame):
        """Render burn rate and runway analysis."""
        from startup_model import StartupAnalyzer

        data = self.analyzer._dataframe_to_financial_data(df)
        sa = StartupAnalyzer()
        br = sa.burn_runway(data)

        col1, col2, col3 = st.columns(3)
        col1.metric("Net Burn", f"${br.net_burn:,.0f}/mo" if br.net_burn else "N/A")
        runway_label = "Infinite" if br.is_cash_flow_positive else (f"{br.runway_months:.0f} months" if br.runway_months else "N/A")
        col2.metric("Runway", runway_label)
        col3.metric("Category", br.category.title() if br.category else "N/A")

        if br.cash_on_hand:
            st.metric("Cash on Hand", f"${br.cash_on_hand:,.0f}")
        if br.breakeven_revenue_needed:
            st.info(f"Revenue needed to break even: **${br.breakeven_revenue_needed:,.0f}/mo**")

        if br.interpretation:
            color_map = {"critical": "error", "caution": "warning", "comfortable": "info", "strong": "success"}
            getattr(st, color_map.get(br.category, "info"))(br.interpretation)

    def _render_funding_scenarios(self, df: pd.DataFrame):
        """Render funding scenario analysis."""
        from startup_model import StartupAnalyzer

        data = self.analyzer._dataframe_to_financial_data(df)
        sa = StartupAnalyzer()

        st.subheader("Configure Funding Scenarios")
        n_scenarios = st.number_input("Number of scenarios", min_value=1, max_value=5, value=2, key="n_fund_scen")

        scenarios = []
        for i in range(n_scenarios):
            col1, col2 = st.columns(2)
            amount = col1.number_input(f"Raise Amount #{i+1} ($)", value=5_000_000, step=500_000, key=f"fund_amt_{i}")
            valuation = col2.number_input(f"Pre-Money Valuation #{i+1} ($)", value=20_000_000, step=1_000_000, key=f"fund_val_{i}")
            scenarios.append({"raise_amount": amount, "pre_money_valuation": valuation})

        if st.button("Analyze Funding Scenarios", key="btn_fund_analyze"):
            results = sa.funding_scenarios(data, scenarios)
            for fs in results:
                with st.expander(f"${fs.raise_amount:,.0f} raise @ ${fs.pre_money_valuation:,.0f} pre-money"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Post-Money", f"${fs.post_money_valuation:,.0f}")
                    col2.metric("Dilution", f"{fs.dilution_pct:.1%}")
                    col3.metric("New Runway", f"{fs.new_runway_months:.0f} mo" if fs.new_runway_months else "N/A")
                    if fs.implied_arr_multiple:
                        st.metric("Implied ARR Multiple", f"{fs.implied_arr_multiple:.1f}x")

    def _render_data_explorer(self, df: pd.DataFrame, workbook: WorkbookData):
        """Render data exploration section."""
        with st.expander("Data Explorer", expanded=False):
            st.subheader("Raw Data View")

            # Data filters
            col1, col2 = st.columns(2)

            with col1:
                show_rows = st.slider("Rows to display", 5, 100, 20)
            with col2:
                numeric_only = st.checkbox("Numeric columns only")

            display_df = df.select_dtypes(include=[np.number]) if numeric_only else df

            st.dataframe(display_df.head(show_rows), use_container_width=True)

            # Summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)

            # Column info
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Null': df.isnull().sum(),
                'Unique': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)

            # Export option
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Data as CSV",
                csv,
                "financial_data.csv",
                "text/csv"
            )

    # ===== PORTFOLIO ANALYSIS TABS =====

    def _sheets_to_companies(self, workbook) -> Dict[str, "FinancialData"]:
        """Convert workbook sheets to a dict of company name -> FinancialData."""
        companies = {}
        for sheet in workbook.sheets:
            data = self.analyzer._dataframe_to_financial_data(sheet.df)
            companies[sheet.name] = data
        return companies

    def _render_portfolio_overview(self, df: pd.DataFrame, workbook):
        """Render portfolio overview from multi-sheet workbook."""
        from portfolio_analyzer import PortfolioAnalyzer

        companies = self._sheets_to_companies(workbook)
        if len(companies) < 2:
            st.info("Upload a workbook with multiple sheets (one per company) for portfolio analysis.")
            return

        pa = PortfolioAnalyzer(self.analyzer)
        report = pa.full_portfolio_analysis(companies)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Companies", report.num_companies)
        col2.metric("Avg Health", f"{report.risk_summary.avg_health_score:.0f}/100")
        col3.metric("Risk Level", report.risk_summary.overall_risk_level.title())
        col4.metric("Diversification", f"{report.diversification.overall_score}/100")

        st.subheader("Company Health Scores")
        for snap in sorted(report.snapshots, key=lambda s: s.health_score, reverse=True):
            st.progress(snap.health_score / 100, text=f"{snap.name}: {snap.health_score}/100 (Grade {snap.health_grade})")

        if report.risk_summary.risk_flags:
            st.subheader("Risk Flags")
            for flag in report.risk_summary.risk_flags:
                st.warning(flag)

        st.markdown(f"**Summary:** {report.summary}")

    def _render_portfolio_correlation(self, df: pd.DataFrame, workbook):
        """Render correlation matrix for portfolio companies."""
        from portfolio_analyzer import PortfolioAnalyzer

        companies = self._sheets_to_companies(workbook)
        if len(companies) < 2:
            st.info("Need at least 2 sheets (companies) for correlation analysis.")
            return

        pa = PortfolioAnalyzer(self.analyzer)
        corr = pa.correlation_matrix(companies)

        st.metric("Avg Correlation", f"{corr.avg_correlation:.2f}")
        st.markdown(f"**Interpretation:** {corr.interpretation}")

        corr_df = pd.DataFrame(corr.matrix, index=corr.company_names, columns=corr.company_names)
        st.dataframe(corr_df.style.background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1), use_container_width=True)

    def _render_portfolio_diversification(self, df: pd.DataFrame, workbook):
        """Render diversification score for portfolio."""
        from portfolio_analyzer import PortfolioAnalyzer

        companies = self._sheets_to_companies(workbook)
        if len(companies) < 2:
            st.info("Need at least 2 sheets (companies) for diversification scoring.")
            return

        pa = PortfolioAnalyzer(self.analyzer)
        div = pa.diversification_score(companies)

        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Score", f"{div.overall_score}/100 ({div.grade})")
        col2.metric("Revenue HHI", f"{div.hhi_revenue:.3f}")
        col3.metric("Asset HHI", f"{div.hhi_assets:.3f}")

        st.markdown(f"- Revenue concentration: **{div.revenue_concentration}**")
        st.markdown(f"- Asset concentration: **{div.asset_concentration}**")
        st.markdown(f"- Avg correlation penalty: **{div.correlation_penalty:.2f}**")
        st.markdown(f"**{div.interpretation}**")

    # ===== REGULATORY & COMPLIANCE TABS =====

    def _render_sox_compliance(self, df: pd.DataFrame):
        """Render SOX compliance assessment."""
        from compliance_scorer import ComplianceScorer

        data = self.analyzer._dataframe_to_financial_data(df)
        cs = ComplianceScorer(self.analyzer)
        sox = cs.sox_compliance(data)

        col1, col2, col3 = st.columns(3)
        col1.metric("SOX Risk", sox.overall_risk.title())
        col2.metric("Risk Score", f"{sox.risk_score}/100")
        col3.metric("Checks Passed", f"{sox.checks_passed}/{sox.checks_performed}")

        if sox.material_weakness_indicators:
            st.error("**Material Weaknesses:** " + "; ".join(sox.material_weakness_indicators))
        if sox.significant_deficiency_indicators:
            st.warning("**Significant Deficiencies:** " + "; ".join(sox.significant_deficiency_indicators))
        if sox.flags:
            st.subheader("Detailed Flags")
            for flag in sox.flags:
                st.markdown(f"- {flag}")

    def _render_sec_filing_quality(self, df: pd.DataFrame):
        """Render SEC filing quality assessment."""
        from compliance_scorer import ComplianceScorer

        data = self.analyzer._dataframe_to_financial_data(df)
        cs = ComplianceScorer(self.analyzer)
        sec = cs.sec_filing_quality(data)

        col1, col2, col3 = st.columns(3)
        col1.metric("Disclosure Score", f"{sec.disclosure_score}/100")
        col2.metric("Grade", sec.grade)
        col3.metric("Completeness", f"{sec.data_completeness_pct:.0f}%")

        if sec.missing_critical_fields:
            st.warning("**Missing Critical Fields:** " + ", ".join(sec.missing_critical_fields))
        if sec.red_flags:
            st.error("**Red Flags:** " + "; ".join(sec.red_flags))

        st.markdown(f"Consistency checks: **{sec.consistency_checks_passed}/{sec.consistency_checks_total}** passed")

    def _render_regulatory_thresholds(self, df: pd.DataFrame):
        """Render regulatory threshold compliance check."""
        from compliance_scorer import ComplianceScorer

        data = self.analyzer._dataframe_to_financial_data(df)
        cs = ComplianceScorer(self.analyzer)
        reg = cs.regulatory_ratios(data)

        col1, col2, col3 = st.columns(3)
        col1.metric("Compliance", f"{reg.compliance_pct:.0f}%" if reg.compliance_pct is not None else "N/A")
        col2.metric("Passed", reg.pass_count)
        col3.metric("Failed", reg.fail_count)

        if reg.critical_failures:
            st.error("**Critical Failures:** " + "; ".join(reg.critical_failures))

        st.subheader("Threshold Details")
        for t in reg.thresholds_checked:
            val_str = f"{t.current_value:.4f}" if t.current_value is not None else "N/A"
            if t.passes is None:
                status = "N/A (insufficient data)"
            elif t.passes:
                status = "PASS"
            else:
                status = "FAIL"
            st.markdown(f"- **{t.rule_name}** ({t.framework}): {val_str} vs {t.threshold_value} -- {status}")

    def _render_audit_risk(self, df: pd.DataFrame):
        """Render audit risk assessment."""
        from compliance_scorer import ComplianceScorer

        data = self.analyzer._dataframe_to_financial_data(df)
        cs = ComplianceScorer(self.analyzer)
        report = cs.full_compliance_report(data)
        audit = report.audit_risk

        col1, col2, col3 = st.columns(3)
        col1.metric("Audit Risk", audit.risk_level.title())
        col2.metric("Score", f"{audit.score}/100")
        col3.metric("Grade", audit.grade)

        if audit.going_concern_risk:
            st.error("**GOING CONCERN RISK IDENTIFIED**")
        if audit.restatement_risk_indicators:
            st.warning("**Restatement Risk Indicators:**")
            for ind in audit.restatement_risk_indicators:
                st.markdown(f"- {ind}")
        if audit.recommendations:
            st.subheader("Recommendations")
            for rec in audit.recommendations:
                st.markdown(f"- {rec}")

    @staticmethod
    @st.cache_data(ttl=3600)
    def _extract_key_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Extract key financial metrics from DataFrame."""
        metrics = {}

        # Column name patterns to look for
        revenue_patterns = ['revenue', 'sales', 'income', 'turnover']
        income_patterns = ['net income', 'profit', 'earnings']

        for col in df.columns:
            col_lower = col.lower()

            # Revenue
            if any(p in col_lower for p in revenue_patterns) and 'net income' not in col_lower:
                if 'revenue' not in metrics:
                    try:
                        metrics['revenue'] = df[col].sum() if len(df) > 1 else df[col].iloc[0]
                    except (TypeError, ValueError, IndexError):
                        pass

            # Net Income
            if any(p in col_lower for p in income_patterns):
                if 'net_income' not in metrics:
                    try:
                        metrics['net_income'] = df[col].sum() if len(df) > 1 else df[col].iloc[0]
                    except (TypeError, ValueError, IndexError):
                        pass

        return metrics


def render_insights_page(docs_folder: str = "./documents"):
    """Convenience function to render the insights page."""
    page = FinancialInsightsPage(docs_folder)
    page.render()
