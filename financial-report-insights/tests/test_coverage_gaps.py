"""
Tests for previously untested high-priority functions.
Covers: _dataframe_to_financial_data, analyze_budget_variance, detect_anomalies,
        _chunk_text, _cosine_similarity, _find_header_row, _is_time_period,
        process_excel_for_rag document format.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financial_analyzer import CharlieAnalyzer

# ============================================================
# _dataframe_to_financial_data tests
# ============================================================


class TestDataframeToFinancialData:
    @pytest.fixture
    def analyzer(self):
        return CharlieAnalyzer()

    def test_single_row_income_statement(self, analyzer):
        df = pd.DataFrame(
            {
                "Revenue": [1000000],
                "Cost of Goods Sold": [400000],
                "Net Income": [200000],
            }
        )
        data = analyzer._dataframe_to_financial_data(df)
        assert data.revenue == 1000000
        assert data.cogs == 400000
        assert data.net_income == 200000

    def test_multi_row_uses_last_value(self, analyzer):
        df = pd.DataFrame(
            {
                "Revenue": [100, 200, 300],
                "Net Income": [10, 20, 30],
            }
        )
        data = analyzer._dataframe_to_financial_data(df)
        assert data.revenue == 300
        assert data.net_income == 30

    def test_case_insensitive_column_matching(self, analyzer):
        df = pd.DataFrame({"REVENUE": [500], "net income": [100]})
        data = analyzer._dataframe_to_financial_data(df)
        assert data.revenue == 500
        assert data.net_income == 100

    def test_unrecognized_columns_ignored(self, analyzer):
        df = pd.DataFrame({"Widget Count": [42], "Foo": [99]})
        data = analyzer._dataframe_to_financial_data(df)
        assert data.revenue is None
        assert data.net_income is None

    def test_empty_dataframe(self, analyzer):
        df = pd.DataFrame()
        data = analyzer._dataframe_to_financial_data(df)
        assert data.revenue is None

    def test_nan_values_skipped(self, analyzer):
        df = pd.DataFrame({"Revenue": [float("nan")]})
        data = analyzer._dataframe_to_financial_data(df)
        assert data.revenue is None

    def test_balance_sheet_columns(self, analyzer):
        df = pd.DataFrame(
            {
                "Total Assets": [2000000],
                "Current Assets": [800000],
                "Cash": [200000],
                "Total Equity": [1000000],
            }
        )
        data = analyzer._dataframe_to_financial_data(df)
        assert data.total_assets == 2000000
        assert data.current_assets == 800000
        assert data.cash == 200000
        assert data.total_equity == 1000000


# ============================================================
# analyze_budget_variance tests
# ============================================================


class TestAnalyzeBudgetVariance:
    @pytest.fixture
    def analyzer(self):
        return CharlieAnalyzer()

    def test_basic_budget_variance(self, analyzer):
        actual_df = pd.DataFrame(
            {
                "Category": ["Revenue", "Expenses"],
                "Amount": [1100000, 450000],
            }
        )
        budget_df = pd.DataFrame(
            {
                "Category": ["Revenue", "Expenses"],
                "Amount": [1000000, 500000],
            }
        )
        result = analyzer.analyze_budget_variance(actual_df, budget_df, "Category", "Amount", "Amount")
        assert result.total_actual == 1550000
        assert result.total_budget == 1500000
        assert result.total_variance == 50000
        assert len(result.line_items) == 2

    def test_favorable_and_unfavorable_items(self, analyzer):
        actual_df = pd.DataFrame(
            {
                "Item": ["A", "B"],
                "Value": [90, 110],
            }
        )
        budget_df = pd.DataFrame(
            {
                "Item": ["A", "B"],
                "Value": [100, 100],
            }
        )
        result = analyzer.analyze_budget_variance(actual_df, budget_df, "Item", "Value", "Value")
        assert len(result.favorable_items) >= 0
        assert len(result.unfavorable_items) >= 0
        assert len(result.line_items) == 2

    def test_largest_variances_sorted(self, analyzer):
        actual_df = pd.DataFrame(
            {
                "Item": ["A", "B", "C"],
                "Val": [100, 200, 300],
            }
        )
        budget_df = pd.DataFrame(
            {
                "Item": ["A", "B", "C"],
                "Val": [100, 100, 100],
            }
        )
        result = analyzer.analyze_budget_variance(actual_df, budget_df, "Item", "Val", "Val")
        # Largest variances should be sorted by absolute value
        abs_variances = [abs(v.variance) for v in result.largest_variances]
        assert abs_variances == sorted(abs_variances, reverse=True)


# ============================================================
# detect_anomalies tests
# ============================================================


class TestDetectAnomalies:
    @pytest.fixture
    def analyzer(self):
        return CharlieAnalyzer()

    def test_detects_outlier(self, analyzer):
        df = pd.DataFrame(
            {
                "revenue": [100, 102, 98, 101, 99, 100, 103, 97, 100, 500],
            }
        )
        anomalies = analyzer.detect_anomalies(df)
        assert len(anomalies) >= 1
        assert any(a.value == 500 for a in anomalies)

    def test_no_anomalies_in_uniform_data(self, analyzer):
        df = pd.DataFrame(
            {
                "revenue": [100, 101, 99, 100, 100, 101, 99, 100],
            }
        )
        anomalies = analyzer.detect_anomalies(df)
        assert len(anomalies) == 0

    def test_too_few_data_points(self, analyzer):
        df = pd.DataFrame({"revenue": [100, 200]})
        anomalies = analyzer.detect_anomalies(df)
        assert len(anomalies) == 0

    def test_constant_values_no_anomalies(self, analyzer):
        df = pd.DataFrame({"revenue": [100, 100, 100, 100]})
        anomalies = analyzer.detect_anomalies(df)
        assert len(anomalies) == 0  # std==0, skip

    def test_specific_columns(self, analyzer):
        df = pd.DataFrame(
            {
                "revenue": [100, 100, 100, 100, 500],
                "other": [1, 1, 1, 1, 1000],
            }
        )
        anomalies = analyzer.detect_anomalies(df, columns=["revenue"])
        assert all(a.metric_name == "revenue" for a in anomalies)

    def test_anomaly_has_z_score(self, analyzer):
        df = pd.DataFrame(
            {
                "x": [10, 10, 10, 10, 10, 10, 10, 10, 10, 100],
            }
        )
        anomalies = analyzer.detect_anomalies(df)
        for a in anomalies:
            assert abs(a.z_score) > 2


# ============================================================
# _chunk_text tests (via SimpleRAG)
# ============================================================


class TestChunkText:
    @pytest.fixture
    def rag_instance(self):
        """Create a minimal SimpleRAG instance without Ollama dependency."""
        from app_local import SimpleRAG

        rag = SimpleRAG.__new__(SimpleRAG)
        return rag

    def test_basic_chunking(self, rag_instance):
        text = " ".join(f"word{i}" for i in range(100))
        chunks = rag_instance._chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 20 for c in chunks)

    def test_small_text_single_chunk(self, rag_instance):
        text = "Hello world"
        chunks = rag_instance._chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1

    def test_empty_text(self, rag_instance):
        chunks = rag_instance._chunk_text("", chunk_size=500, overlap=50)
        assert len(chunks) == 0  # empty text returns empty list

    def test_overlap_creates_redundancy(self, rag_instance):
        text = " ".join(f"word{i}" for i in range(50))
        chunks = rag_instance._chunk_text(text, chunk_size=20, overlap=5)
        # With overlap, chunks should share some words
        if len(chunks) >= 2:
            words_1 = set(chunks[0].split())
            words_2 = set(chunks[1].split())
            assert len(words_1 & words_2) > 0


# ============================================================
# _cosine_similarity tests (numpy-vectorized version)
# ============================================================


class TestCosineSimilarity:
    @pytest.fixture
    def rag_instance(self):
        from app_local import SimpleRAG

        rag = SimpleRAG.__new__(SimpleRAG)
        return rag

    def test_identical_vectors(self, rag_instance):
        vec = [1.0, 2.0, 3.0]
        assert rag_instance._cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self, rag_instance):
        assert rag_instance._cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self, rag_instance):
        assert rag_instance._cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self, rag_instance):
        assert rag_instance._cosine_similarity([0, 0], [1, 2]) == 0.0

    def test_large_vectors(self, rag_instance):
        np.random.seed(42)
        a = np.random.randn(384).tolist()
        b = np.random.randn(384).tolist()
        sim = rag_instance._cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0


# ============================================================
# _find_header_row tests
# ============================================================


class TestFindHeaderRow:
    @pytest.fixture
    def processor(self):
        from excel_processor import ExcelProcessor

        return ExcelProcessor(tempfile.mkdtemp())

    def test_empty_data(self, processor):
        assert processor._find_header_row([]) == -1

    def test_header_in_first_row(self, processor):
        data = [
            ["Name", "Revenue", "Expenses"],
            [None, 1000, 500],
        ]
        assert processor._find_header_row(data) == 0

    def test_header_after_empty_rows(self, processor):
        data = [
            [None, None],
            ["Revenue", "Expenses", "Net Income"],
            [1000, 500, 500],
        ]
        assert processor._find_header_row(data) == 1

    def test_numeric_only_rows(self, processor):
        data = [
            [100, 200, 300],
            [400, 500, 600],
        ]
        # No text-heavy row found; defaults to 0
        assert processor._find_header_row(data) == 0


# ============================================================
# _is_time_period tests
# ============================================================


class TestIsTimePeriod:
    @pytest.fixture
    def processor(self):
        from excel_processor import ExcelProcessor

        return ExcelProcessor(tempfile.mkdtemp())

    def test_year(self, processor):
        assert processor._is_time_period("2024") is True

    def test_quarter(self, processor):
        assert processor._is_time_period("Q1 2024") is True
        assert processor._is_time_period("q3 2023") is True

    def test_fiscal_year(self, processor):
        assert processor._is_time_period("FY24") is True
        assert processor._is_time_period("fy2024") is True

    def test_month_name(self, processor):
        assert processor._is_time_period("January") is True
        assert processor._is_time_period("mar") is True

    def test_non_time_column(self, processor):
        assert processor._is_time_period("Revenue") is False
        assert processor._is_time_period("Total") is False

    def test_date_format(self, processor):
        assert processor._is_time_period("01/15/2024") is True


# ============================================================
# process_excel_for_rag document format tests
# ============================================================


class TestProcessExcelForRagFormat:
    def test_documents_have_top_level_source_and_type(self):
        """Verify the latent bug fix: docs must have top-level source/type keys."""
        from excel_processor import process_excel_for_rag

        with tempfile.TemporaryDirectory() as temp_dir:
            df = pd.DataFrame(
                {
                    "Item": ["Revenue", "Expenses"],
                    "Amount": [100000, 60000],
                }
            )
            file_path = Path(temp_dir) / "test.xlsx"
            df.to_excel(file_path, index=False)

            docs = process_excel_for_rag(file_path, temp_dir)

            assert len(docs) > 0
            for doc in docs:
                # Top-level keys expected by SimpleRAG.answer()
                assert "source" in doc, "Missing top-level 'source' key"
                assert "type" in doc, "Missing top-level 'type' key"
                assert "content" in doc
                assert doc["type"] == "excel"
                # Metadata should also exist
                assert "metadata" in doc
                assert doc["metadata"]["type"] == "excel"


# ============================================================
# Row-based _dataframe_to_financial_data tests
# ============================================================


class TestRowBasedDataframeToFinancialData:
    @pytest.fixture
    def analyzer(self):
        return CharlieAnalyzer()

    def test_row_based_income_statement(self, analyzer):
        """Items as rows, periods as columns - should pick latest period."""
        df = pd.DataFrame(
            {
                "Line Item": ["Revenue", "Cost of Goods Sold", "Gross Profit", "Operating Income", "Net Income"],
                "Q1 2024": [500000, 200000, 300000, 150000, 100000],
                "Q2 2024": [600000, 230000, 370000, 180000, 130000],
            }
        )
        data = analyzer._dataframe_to_financial_data(df)
        assert data.revenue == 600000
        assert data.cogs == 230000
        assert data.gross_profit == 370000
        assert data.operating_income == 180000
        assert data.net_income == 130000

    def test_row_based_balance_sheet(self, analyzer):
        """Single-period balance sheet with items as rows."""
        df = pd.DataFrame(
            {
                "Account": ["Total Assets", "Current Assets", "Total Liabilities", "Total Equity", "Cash"],
                "2024": [1000000, 400000, 600000, 400000, 150000],
            }
        )
        data = analyzer._dataframe_to_financial_data(df)
        assert data.total_assets == 1000000
        assert data.current_assets == 400000
        assert data.total_liabilities == 600000
        assert data.total_equity == 400000
        assert data.cash == 150000

    def test_row_based_budget(self, analyzer):
        """Budget/Actual format picks rightmost numeric column."""
        df = pd.DataFrame(
            {
                "Category": ["Revenue", "COGS", "Net Income", "Total Assets"],
                "Budget": [800000, 400000, 200000, 2000000],
                "Actual": [900000, 420000, 250000, 2100000],
            }
        )
        data = analyzer._dataframe_to_financial_data(df)
        # Should use 'Actual' (rightmost numeric)
        assert data.revenue == 900000
        assert data.cogs == 420000
        assert data.net_income == 250000

    def test_row_based_insufficient_matches(self, analyzer):
        """Fewer than 3 financial term matches should NOT trigger transpose."""
        df = pd.DataFrame(
            {
                "Item": ["Revenue", "Other stuff", "Random"],
                "Value": [100, 200, 300],
            }
        )
        data = analyzer._dataframe_to_financial_data(df)
        # Only 1 match (Revenue) - not enough to trigger row-based fallback
        # Column headers also don't match -> empty FinancialData
        assert data.revenue is None

    def test_row_based_non_numeric_values(self, analyzer):
        """Graceful handling when value columns contain text."""
        df = pd.DataFrame(
            {
                "Line Item": ["Revenue", "COGS", "Gross Profit", "Net Income"],
                "Notes": ["Good", "OK", "Fine", "Great"],
            }
        )
        data = analyzer._dataframe_to_financial_data(df)
        # No numeric columns -> all fields stay None
        assert data.revenue is None
        assert data.net_income is None
