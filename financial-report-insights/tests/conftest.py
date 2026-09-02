"""
Pytest configuration and fixtures for financial insights tests.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def sample_financial_data_dict():
    """Sample financial data as a dictionary."""
    return {
        "revenue": 1000000,
        "cogs": 400000,
        "gross_profit": 600000,
        "operating_income": 300000,
        "net_income": 200000,
        "total_assets": 2000000,
        "current_assets": 800000,
        "cash": 200000,
        "inventory": 300000,
        "accounts_receivable": 200000,
        "total_liabilities": 1000000,
        "current_liabilities": 400000,
        "accounts_payable": 150000,
        "total_debt": 600000,
        "total_equity": 1000000,
    }


@pytest.fixture
def sample_income_statement_df():
    """Sample income statement DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        {
            "Line Item": [
                "Revenue",
                "Cost of Goods Sold",
                "Gross Profit",
                "Operating Expenses",
                "Operating Income",
                "Net Income",
            ],
            "Q1 2024": [1000000, 400000, 600000, 300000, 300000, 200000],
            "Q2 2024": [1100000, 440000, 660000, 310000, 350000, 250000],
            "Q3 2024": [1200000, 480000, 720000, 320000, 400000, 300000],
            "Q4 2024": [1300000, 520000, 780000, 330000, 450000, 350000],
        }
    )


@pytest.fixture
def sample_balance_sheet_df():
    """Sample balance sheet DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        {
            "Account": [
                "Cash",
                "Accounts Receivable",
                "Inventory",
                "Current Assets",
                "Property & Equipment",
                "Total Assets",
                "Accounts Payable",
                "Current Liabilities",
                "Long-term Debt",
                "Total Liabilities",
                "Shareholders Equity",
                "Total Equity",
            ],
            "2024": [
                200000,
                200000,
                300000,
                800000,
                1200000,
                2000000,
                150000,
                400000,
                600000,
                1000000,
                1000000,
                1000000,
            ],
        }
    )


@pytest.fixture
def sample_budget_df():
    """Sample budget vs actual DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        {
            "Category": ["Revenue", "Marketing", "Salaries", "Technology", "Other"],
            "Budget": [1000000, 100000, 400000, 80000, 50000],
            "Actual": [950000, 120000, 380000, 85000, 55000],
        }
    )
