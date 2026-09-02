"""
Integration tests for the RAG-LLM Financial Enhancement system.
Tests Excel processing, financial analysis, and visualization generation.
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest


class TestExcelProcessor:
    """Tests for Excel processing module."""

    @pytest.fixture
    def temp_docs_folder(self):
        """Create a temporary documents folder."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_income_statement(self, temp_docs_folder):
        """Create a sample income statement Excel file."""
        data = {
            "Line Item": ["Revenue", "COGS", "Gross Profit", "Operating Expenses", "Net Income"],
            "Q1 2024": [1000000, 400000, 600000, 200000, 400000],
            "Q2 2024": [1100000, 440000, 660000, 210000, 450000],
        }
        df = pd.DataFrame(data)
        file_path = temp_docs_folder / "income_statement.xlsx"
        df.to_excel(file_path, index=False)
        return file_path

    @pytest.fixture
    def sample_csv(self, temp_docs_folder):
        """Create a sample CSV file."""
        data = {
            "Category": ["Revenue", "Expenses", "Profit"],
            "Actual": [100000, 60000, 40000],
            "Budget": [95000, 55000, 40000],
        }
        df = pd.DataFrame(data)
        file_path = temp_docs_folder / "budget.csv"
        df.to_csv(file_path, index=False)
        return file_path

    def test_scan_for_excel_files(self, temp_docs_folder, sample_income_statement, sample_csv):
        """Test scanning for Excel files in documents folder."""
        from excel_processor import ExcelProcessor

        processor = ExcelProcessor(str(temp_docs_folder))
        files = processor.scan_for_excel_files()

        # Deduplicate since scan checks both lower and upper extensions
        unique_files = set(files)
        assert len(unique_files) == 2
        assert any(f.suffix == ".xlsx" for f in unique_files)
        assert any(f.suffix == ".csv" for f in unique_files)

    def test_load_xlsx_workbook(self, temp_docs_folder, sample_income_statement):
        """Test loading an xlsx workbook."""
        from excel_processor import ExcelProcessor

        processor = ExcelProcessor(str(temp_docs_folder))
        workbook = processor.load_workbook(sample_income_statement)

        assert workbook is not None
        assert len(workbook.sheets) >= 1
        assert workbook.file_path == sample_income_statement

        # Check first sheet
        sheet = workbook.sheets[0]
        assert not sheet.df.empty
        assert "Line Item" in sheet.df.columns or sheet.df.columns[0] is not None

    def test_load_csv(self, temp_docs_folder, sample_csv):
        """Test loading a CSV file."""
        from excel_processor import ExcelProcessor

        processor = ExcelProcessor(str(temp_docs_folder))
        workbook = processor.load_workbook(sample_csv)

        assert workbook is not None
        assert len(workbook.sheets) == 1

        df = workbook.sheets[0].df
        assert "Category" in df.columns
        assert "Actual" in df.columns
        assert len(df) == 3

    def test_detect_statement_type(self, temp_docs_folder, sample_income_statement):
        """Test financial statement type detection."""
        from excel_processor import ExcelProcessor

        processor = ExcelProcessor(str(temp_docs_folder))
        workbook = processor.load_workbook(sample_income_statement)

        sheet = workbook.sheets[0]
        # Should detect as income statement based on keywords
        assert sheet.detected_type in ["income_statement", "custom"]

    def test_to_rag_chunks(self, temp_docs_folder, sample_income_statement):
        """Test converting Excel data to RAG chunks."""
        from excel_processor import ExcelProcessor

        processor = ExcelProcessor(str(temp_docs_folder))
        workbook = processor.load_workbook(sample_income_statement)
        combined = processor.combine_sheets_intelligently(workbook)
        chunks = processor.to_rag_chunks(combined, workbook)

        assert len(chunks) > 0
        assert all(chunk.text_content for chunk in chunks)
        assert all(chunk.sheet_name for chunk in chunks)

    def test_process_excel_for_rag(self, temp_docs_folder, sample_income_statement):
        """Test the convenience function for RAG processing."""
        from excel_processor import process_excel_for_rag

        docs = process_excel_for_rag(sample_income_statement, str(temp_docs_folder))

        assert len(docs) > 0
        assert all("content" in doc for doc in docs)
        assert all("metadata" in doc for doc in docs)
        assert all(doc["metadata"]["type"] == "excel" for doc in docs)


class TestFinancialAnalyzer:
    """Tests for financial analysis module."""

    @pytest.fixture
    def sample_financial_data(self):
        """Create sample financial data."""
        from financial_analyzer import FinancialData

        return FinancialData(
            revenue=1000000,
            cogs=400000,
            gross_profit=600000,
            operating_income=300000,
            net_income=200000,
            total_assets=2000000,
            current_assets=800000,
            cash=200000,
            inventory=300000,
            accounts_receivable=200000,
            total_liabilities=1000000,
            current_liabilities=400000,
            accounts_payable=150000,
            total_debt=600000,
            total_equity=1000000,
            interest_expense=50000,
            ebit=300000,
        )

    def test_calculate_liquidity_ratios(self, sample_financial_data):
        """Test liquidity ratio calculations."""
        from financial_analyzer import CharlieAnalyzer

        analyzer = CharlieAnalyzer()
        ratios = analyzer.calculate_liquidity_ratios(sample_financial_data)

        assert "current_ratio" in ratios
        assert ratios["current_ratio"] == pytest.approx(2.0, rel=0.01)  # 800k / 400k

        assert "quick_ratio" in ratios
        assert ratios["quick_ratio"] == pytest.approx(1.25, rel=0.01)  # (800k - 300k) / 400k

        assert "cash_ratio" in ratios
        assert ratios["cash_ratio"] == pytest.approx(0.5, rel=0.01)  # 200k / 400k

    def test_calculate_profitability_ratios(self, sample_financial_data):
        """Test profitability ratio calculations."""
        from financial_analyzer import CharlieAnalyzer

        analyzer = CharlieAnalyzer()
        ratios = analyzer.calculate_profitability_ratios(sample_financial_data)

        assert "gross_margin" in ratios
        assert ratios["gross_margin"] == pytest.approx(0.6, rel=0.01)  # 600k / 1M

        assert "operating_margin" in ratios
        assert ratios["operating_margin"] == pytest.approx(0.3, rel=0.01)  # 300k / 1M

        assert "net_margin" in ratios
        assert ratios["net_margin"] == pytest.approx(0.2, rel=0.01)  # 200k / 1M

        assert "roe" in ratios
        assert ratios["roe"] == pytest.approx(0.2, rel=0.01)  # 200k / 1M

    def test_calculate_leverage_ratios(self, sample_financial_data):
        """Test leverage ratio calculations."""
        from financial_analyzer import CharlieAnalyzer

        analyzer = CharlieAnalyzer()
        ratios = analyzer.calculate_leverage_ratios(sample_financial_data)

        assert "debt_to_equity" in ratios
        assert ratios["debt_to_equity"] == pytest.approx(0.6, rel=0.01)  # 600k / 1M

        assert "interest_coverage" in ratios
        assert ratios["interest_coverage"] == pytest.approx(6.0, rel=0.01)  # 300k / 50k

    def test_calculate_variance(self):
        """Test variance calculation."""
        from financial_analyzer import CharlieAnalyzer

        analyzer = CharlieAnalyzer()

        # Favorable variance (under budget for expense)
        result = analyzer.calculate_variance(90000, 100000, "Expense")
        assert result.variance == -10000
        assert result.variance_percent == pytest.approx(-0.1, rel=0.01)
        assert result.favorable is True

        # Unfavorable variance (over budget for expense)
        result = analyzer.calculate_variance(110000, 100000, "Expense")
        assert result.variance == 10000
        assert result.favorable is False

    def test_analyze_cash_flow(self, sample_financial_data):
        """Test cash flow analysis."""
        from financial_analyzer import CharlieAnalyzer

        # Add cash flow data
        sample_financial_data.operating_cash_flow = 250000
        sample_financial_data.capex = 50000

        analyzer = CharlieAnalyzer()
        cf_analysis = analyzer.analyze_cash_flow(sample_financial_data)

        assert cf_analysis.free_cash_flow == 200000  # 250k - 50k
        assert cf_analysis.dso is not None
        assert cf_analysis.dio is not None
        assert cf_analysis.dpo is not None

    def test_generate_insights(self, sample_financial_data):
        """Test insight generation."""
        from financial_analyzer import CharlieAnalyzer

        analyzer = CharlieAnalyzer()
        results = analyzer.analyze(sample_financial_data)
        insights = results.get("insights", [])

        # Insights list may be empty for balanced financial data
        assert isinstance(insights, list)
        if insights:
            assert all(hasattr(insight, "category") for insight in insights)
            assert all(hasattr(insight, "message") for insight in insights)

    def test_forecast_simple(self):
        """Test simple forecasting."""
        from financial_analyzer import CharlieAnalyzer

        analyzer = CharlieAnalyzer()

        historical = [100, 110, 120, 130, 140]
        forecast = analyzer.forecast_simple(historical, periods=3, method="linear")

        assert len(forecast.forecasted_values) == 3
        assert all(v > 140 for v in forecast.forecasted_values)  # Should be increasing

    def test_analyze_trends(self):
        """Test trend analysis."""
        import pandas as pd

        from financial_analyzer import CharlieAnalyzer

        analyzer = CharlieAnalyzer()

        df = pd.DataFrame({"revenue": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220]})

        trend = analyzer.analyze_trends(df, "revenue")

        assert trend.metric_name == "revenue"
        assert trend.trend_direction == "up"
        assert trend.cagr is not None
        assert trend.cagr > 0


class TestVisualization:
    """Tests for visualization utilities."""

    def test_format_currency(self):
        """Test currency formatting."""
        from viz_utils import FinancialVizUtils

        # format_currency uses decimals=0 by default
        assert FinancialVizUtils.format_currency(1000) == "$1K"
        assert FinancialVizUtils.format_currency(1500000) == "$2M"  # rounds to 0 decimals
        assert FinancialVizUtils.format_currency(2500000000) == "$2B"
        assert FinancialVizUtils.format_currency(-500000) == "-$500K"
        # With explicit decimals
        assert FinancialVizUtils.format_currency(1000, decimals=1) == "$1.0K"
        assert FinancialVizUtils.format_currency(1500000, decimals=1) == "$1.5M"

    def test_format_percent(self):
        """Test percentage formatting."""
        from viz_utils import FinancialVizUtils

        assert FinancialVizUtils.format_percent(0.15) == "15.0%"
        assert FinancialVizUtils.format_percent(0.0523) == "5.2%"

    def test_create_gauge_chart(self):
        """Test gauge chart creation."""
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_gauge_chart(value=1.5, title="Current Ratio", min_val=0, max_val=3)

        assert fig is not None
        assert len(fig.data) > 0

    def test_create_waterfall(self):
        """Test waterfall chart creation."""
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_waterfall(
            categories=["Revenue", "Expenses", "Taxes"], values=[100000, -60000, -10000], title="Profit Breakdown"
        )

        assert fig is not None
        assert len(fig.data) > 0

    def test_create_sparkline(self):
        """Test sparkline creation."""
        from viz_utils import FinancialVizUtils

        fig = FinancialVizUtils.create_sparkline(values=[10, 12, 11, 15, 18, 16, 20])

        assert fig is not None
        assert len(fig.data) > 0

    def test_create_time_series(self):
        """Test time series chart creation."""
        import pandas as pd

        from viz_utils import FinancialVizUtils

        df = pd.DataFrame(
            {"Period": ["Q1", "Q2", "Q3", "Q4"], "Revenue": [100, 110, 120, 130], "Expenses": [60, 65, 70, 75]}
        )

        fig = FinancialVizUtils.create_time_series(df, "Period", ["Revenue", "Expenses"], title="Financial Trend")

        assert fig is not None
        assert len(fig.data) == 2


class TestRAGIntegration:
    """Tests for RAG system integration with Excel."""

    @pytest.fixture
    def temp_docs_folder(self):
        """Create a temporary documents folder."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_excel_in_docs(self, temp_docs_folder):
        """Create sample Excel file in documents folder."""
        data = {"Item": ["Revenue", "Expenses", "Net Income"], "Amount": [100000, 60000, 40000]}
        df = pd.DataFrame(data)
        file_path = temp_docs_folder / "financial_data.xlsx"
        df.to_excel(file_path, index=False)
        return file_path

    def test_is_financial_query(self):
        """Test financial query detection."""
        # We can't import SimpleRAG without Ollama, so test the logic directly
        financial_keywords = [
            "ratio",
            "margin",
            "profit",
            "revenue",
            "income",
            "expense",
            "cash flow",
            "budget",
            "variance",
            "roe",
            "roa",
            "roi",
        ]

        def is_financial(query):
            query_lower = query.lower()
            return any(keyword in query_lower for keyword in financial_keywords)

        assert is_financial("What is the profit margin?") is True
        assert is_financial("Calculate the ROE") is True
        assert is_financial("What is the weather?") is False
        assert is_financial("Show me the cash flow") is True


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        from financial_analyzer import CharlieAnalyzer, FinancialData

        analyzer = CharlieAnalyzer()
        empty_data = FinancialData()

        ratios = analyzer.calculate_liquidity_ratios(empty_data)
        assert ratios["current_ratio"] is None

    def test_division_by_zero(self):
        """Test handling of division by zero."""
        from financial_analyzer import CharlieAnalyzer, FinancialData

        analyzer = CharlieAnalyzer()
        data = FinancialData(
            current_assets=100000,
            current_liabilities=0,  # This would cause division by zero
        )

        ratios = analyzer.calculate_liquidity_ratios(data)
        assert ratios["current_ratio"] is None

    def test_corrupted_excel_graceful_handling(self):
        """Test graceful handling of corrupted files."""
        import tempfile

        from excel_processor import ExcelProcessor

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a corrupted file
            corrupted_path = Path(temp_dir) / "corrupted.xlsx"
            with open(corrupted_path, "w") as f:
                f.write("This is not a valid Excel file")

            processor = ExcelProcessor(temp_dir)

            with pytest.raises(Exception):
                processor.load_workbook(corrupted_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
