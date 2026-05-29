"""Golden Q&A dataset for RAG evaluation.

Contains representative question-answer pairs spanning five query types
(ratio_lookup, trend_analysis, comparison, explanation, general) at three
difficulty levels (easy, medium, hard).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class GoldenQA:
    """A single golden question-answer pair for evaluation.

    Attributes:
        question: The user query.
        expected_answer: Reference answer text.
        expected_sources: Filenames that should be retrieved.
        query_type: One of ratio_lookup, trend_analysis, comparison,
                    explanation, general.
        difficulty: One of easy, medium, hard.
    """

    question: str
    expected_answer: str
    expected_sources: List[str] = field(default_factory=list)
    query_type: str = "general"
    difficulty: str = "easy"


GOLDEN_QA_PAIRS: List[GoldenQA] = [
    # --- ratio_lookup (4) ---
    GoldenQA(
        question="What is the current ratio for FY2025?",
        expected_answer=(
            "The current ratio for FY2025 is 1.85, calculated as current assets "
            "of $4.2B divided by current liabilities of $2.27B."
        ),
        expected_sources=["balance_sheet_2025.xlsx"],
        query_type="ratio_lookup",
        difficulty="easy",
    ),
    GoldenQA(
        question="What is the debt-to-equity ratio?",
        expected_answer=(
            "The debt-to-equity ratio is 0.62, with total debt of $3.1B and shareholders' equity of $5.0B."
        ),
        expected_sources=["balance_sheet_2025.xlsx"],
        query_type="ratio_lookup",
        difficulty="easy",
    ),
    GoldenQA(
        question="Calculate the return on equity (ROE) using DuPont decomposition.",
        expected_answer=(
            "ROE is 18.5% via DuPont: net profit margin 12.3% x asset turnover 0.85 x equity multiplier 1.77 = 18.5%."
        ),
        expected_sources=["income_statement_2025.xlsx", "balance_sheet_2025.xlsx"],
        query_type="ratio_lookup",
        difficulty="hard",
    ),
    GoldenQA(
        question="What is the gross profit margin for Q4 2025?",
        expected_answer=("Gross profit margin for Q4 2025 is 42.7%, with gross profit of $1.07B on revenue of $2.5B."),
        expected_sources=["income_statement_2025.xlsx"],
        query_type="ratio_lookup",
        difficulty="medium",
    ),
    # --- trend_analysis (4) ---
    GoldenQA(
        question="How has revenue changed from FY2023 to FY2025?",
        expected_answer=(
            "Revenue grew from $8.1B in FY2023 to $9.8B in FY2025, a compound "
            "annual growth rate (CAGR) of approximately 10%."
        ),
        expected_sources=[
            "income_statement_2023.xlsx",
            "income_statement_2025.xlsx",
        ],
        query_type="trend_analysis",
        difficulty="medium",
    ),
    GoldenQA(
        question="What is the year-over-year change in operating expenses?",
        expected_answer=("Operating expenses increased 7.2% year-over-year, from $5.6B in FY2024 to $6.0B in FY2025."),
        expected_sources=[
            "income_statement_2024.xlsx",
            "income_statement_2025.xlsx",
        ],
        query_type="trend_analysis",
        difficulty="medium",
    ),
    GoldenQA(
        question="Describe the trend in free cash flow over the past three years.",
        expected_answer=(
            "Free cash flow improved steadily: $1.1B (FY2023), $1.35B (FY2024), "
            "$1.6B (FY2025), reflecting better working capital management."
        ),
        expected_sources=["cash_flow_2023.xlsx", "cash_flow_2025.xlsx"],
        query_type="trend_analysis",
        difficulty="hard",
    ),
    GoldenQA(
        question="Has the net profit margin been improving or declining?",
        expected_answer=(
            "Net profit margin improved from 10.5% in FY2023 to 12.3% in "
            "FY2025, driven by operating leverage and cost controls."
        ),
        expected_sources=["income_statement_2023.xlsx", "income_statement_2025.xlsx"],
        query_type="trend_analysis",
        difficulty="easy",
    ),
    # --- comparison (4) ---
    GoldenQA(
        question="Compare the current ratio to the quick ratio for FY2025.",
        expected_answer=(
            "The current ratio is 1.85 while the quick ratio is 1.22, indicating "
            "that $1.43B in inventory accounts for the difference."
        ),
        expected_sources=["balance_sheet_2025.xlsx"],
        query_type="comparison",
        difficulty="medium",
    ),
    GoldenQA(
        question="How does our ROA compare to the industry average?",
        expected_answer=(
            "Our ROA of 10.5% exceeds the industry average of 7.8% by 2.7 "
            "percentage points, indicating superior asset utilization."
        ),
        expected_sources=["balance_sheet_2025.xlsx", "industry_benchmarks.pdf"],
        query_type="comparison",
        difficulty="hard",
    ),
    GoldenQA(
        question=("Compare operating cash flow to net income for FY2025."),
        expected_answer=(
            "Operating cash flow of $2.1B exceeds net income of $1.2B by $900M, "
            "indicating strong cash conversion and quality of earnings."
        ),
        expected_sources=[
            "cash_flow_2025.xlsx",
            "income_statement_2025.xlsx",
        ],
        query_type="comparison",
        difficulty="medium",
    ),
    GoldenQA(
        question="Which segment has the highest profit margin?",
        expected_answer=(
            "The Software segment has the highest profit margin at 34.2%, "
            "followed by Services at 18.5% and Hardware at 11.3%."
        ),
        expected_sources=["segment_report_2025.xlsx"],
        query_type="comparison",
        difficulty="hard",
    ),
    # --- explanation (4) ---
    GoldenQA(
        question="Why did working capital decrease in FY2025?",
        expected_answer=(
            "Working capital decreased primarily due to a $400M increase in "
            "short-term borrowings to fund the Q3 acquisition, partially "
            "offset by higher receivables collection."
        ),
        expected_sources=["balance_sheet_2025.xlsx", "notes_2025.pdf"],
        query_type="explanation",
        difficulty="hard",
    ),
    GoldenQA(
        question="What factors drove the improvement in EBITDA margin?",
        expected_answer=(
            "EBITDA margin improved from 22.1% to 24.8% due to economies of "
            "scale in manufacturing, renegotiated supplier contracts, and a "
            "shift toward higher-margin software revenue."
        ),
        expected_sources=["income_statement_2025.xlsx", "management_discussion.pdf"],
        query_type="explanation",
        difficulty="hard",
    ),
    GoldenQA(
        question="Explain how depreciation affects cash flow from operations.",
        expected_answer=(
            "Depreciation is a non-cash expense that reduces net income but is "
            "added back in the cash flow statement. The $350M depreciation charge "
            "in FY2025 increased reported operating cash flow relative to net income."
        ),
        expected_sources=["cash_flow_2025.xlsx"],
        query_type="explanation",
        difficulty="easy",
    ),
    GoldenQA(
        question="Why is the Altman Z-Score important for credit analysis?",
        expected_answer=(
            "The Altman Z-Score combines five financial ratios to predict "
            "bankruptcy risk. A score above 2.99 indicates safety, 1.81-2.99 is "
            "a grey zone, and below 1.81 signals distress."
        ),
        expected_sources=["balance_sheet_2025.xlsx", "income_statement_2025.xlsx"],
        query_type="explanation",
        difficulty="medium",
    ),
    # --- general (4) ---
    GoldenQA(
        question="Summarize the key findings from the FY2025 annual report.",
        expected_answer=(
            "FY2025 highlights: revenue grew 10% to $9.8B, net income rose to "
            "$1.2B, free cash flow reached $1.6B, and the balance sheet remains "
            "healthy with a debt-to-equity ratio of 0.62."
        ),
        expected_sources=["income_statement_2025.xlsx", "balance_sheet_2025.xlsx"],
        query_type="general",
        difficulty="medium",
    ),
    GoldenQA(
        question="What are the main risk factors mentioned in the report?",
        expected_answer=(
            "Key risk factors include foreign exchange exposure (32% international "
            "revenue), supply chain concentration, regulatory changes in the EU "
            "market, and rising interest rates on variable-rate debt."
        ),
        expected_sources=["risk_factors_2025.pdf"],
        query_type="general",
        difficulty="medium",
    ),
    GoldenQA(
        question="List the top three financial strengths of the company.",
        expected_answer=(
            "Top strengths: (1) strong free cash flow generation of $1.6B, "
            "(2) improving profit margins at 12.3% net margin, and (3) low "
            "leverage with debt-to-equity of 0.62."
        ),
        expected_sources=["income_statement_2025.xlsx", "balance_sheet_2025.xlsx"],
        query_type="general",
        difficulty="easy",
    ),
    GoldenQA(
        question="What is the company's total revenue for FY2025?",
        expected_answer="Total revenue for FY2025 is $9.8 billion.",
        expected_sources=["income_statement_2025.xlsx"],
        query_type="general",
        difficulty="easy",
    ),
]
