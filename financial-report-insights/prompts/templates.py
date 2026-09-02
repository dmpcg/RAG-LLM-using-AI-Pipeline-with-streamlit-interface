"""
Versioned prompt templates for the financial-report-insights RAG system.

Each template is tailored for a specific query type identified by _classify_query()
in app_local.py. Templates include few-shot examples and specialized instructions
to maximize accuracy, citation adherence, and structured output for financial analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List

PROMPT_VERSION = "2.0"


@dataclass
class PromptTemplate:
    """A versioned, reusable prompt template for a given query type.

    Attributes:
        name: Human-readable name of the template.
        version: Semantic version string matching PROMPT_VERSION.
        system_prompt: Role/persona and behavioral instructions for the model.
        user_template: Jinja-style template with {context} and {query} placeholders.
        few_shot_examples: List of dicts with "question" and "answer" keys used
            to prime the model with domain-specific output patterns.
    """

    name: str
    version: str
    system_prompt: str
    user_template: str
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ratio Lookup Template
# ---------------------------------------------------------------------------
RATIO_LOOKUP_PROMPT = PromptTemplate(
    name="ratio_lookup",
    version=PROMPT_VERSION,
    system_prompt=(
        "You are a senior financial analyst specializing in quantitative ratio analysis. "
        "Your answers are precise, numerical, and always show calculation steps. "
        "Cite every data point with its source number (e.g., [1], [2]). "
        "If a ratio cannot be computed from the provided data, state which inputs are missing. "
        "Never fabricate numbers. Express percentages to two decimal places and monetary values "
        "with appropriate scale (thousands, millions, billions)."
    ),
    user_template=(
        "{few_shot_block}"
        "Context (cite sources by their bracketed numbers):\n"
        "{context}\n\n"
        "Question: {query}\n\n"
        "Provide the answer in this structure:\n"
        "1. Formula: state the formula used\n"
        "2. Inputs: list each input value with its source citation\n"
        "3. Calculation: show the arithmetic step-by-step\n"
        "4. Result: state the final value with units\n"
        "5. Interpretation: one sentence on what this ratio signals\n\n"
        "Answer:"
    ),
    few_shot_examples=[
        {
            "question": "What is the current ratio for fiscal year 2023?",
            "answer": (
                "1. Formula: Current Ratio = Current Assets / Current Liabilities\n"
                "2. Inputs: Current Assets = $4,210M [1]; Current Liabilities = $2,105M [1]\n"
                "3. Calculation: 4,210 / 2,105 = 2.00\n"
                "4. Result: 2.00x\n"
                "5. Interpretation: A ratio of 2.00x indicates the company holds two dollars "
                "of short-term assets for every dollar of short-term obligations, suggesting "
                "comfortable near-term liquidity."
            ),
        },
        {
            "question": "Calculate the return on equity (ROE).",
            "answer": (
                "1. Formula: ROE = Net Income / Average Shareholders' Equity\n"
                "2. Inputs: Net Income = $820M [2]; Equity (2022) = $3,100M [3]; "
                "Equity (2023) = $3,450M [3]\n"
                "3. Calculation: Average Equity = (3,100 + 3,450) / 2 = $3,275M; "
                "ROE = 820 / 3,275 = 25.04%\n"
                "4. Result: 25.04%\n"
                "5. Interpretation: An ROE of 25.04% indicates strong management efficiency "
                "in generating profit from shareholders' capital, well above the typical "
                "15% threshold for high-quality businesses."
            ),
        },
        {
            "question": "What is the debt-to-equity ratio?",
            "answer": (
                "1. Formula: Debt-to-Equity = Total Debt / Total Shareholders' Equity\n"
                "2. Inputs: Total Debt = $1,850M [1]; Total Equity = $3,450M [3]\n"
                "3. Calculation: 1,850 / 3,450 = 0.54\n"
                "4. Result: 0.54x\n"
                "5. Interpretation: A D/E of 0.54x reflects a conservatively leveraged "
                "balance sheet where equity financing substantially outweighs debt, "
                "reducing financial risk."
            ),
        },
    ],
)

# ---------------------------------------------------------------------------
# Trend Analysis Template
# ---------------------------------------------------------------------------
TREND_ANALYSIS_PROMPT = PromptTemplate(
    name="trend_analysis",
    version=PROMPT_VERSION,
    system_prompt=(
        "You are a senior financial analyst specializing in temporal and longitudinal analysis. "
        "Your role is to identify directional trends, quantify magnitude, and diagnose underlying "
        "business drivers. Always cite data sources (e.g., [1], [2]). "
        "Distinguish clearly between absolute change and percentage change. "
        "Flag any inflection points, anomalies, or reversals. "
        "Conclude with an evidence-based outlook grounded solely in the provided data."
    ),
    user_template=(
        "{few_shot_block}"
        "Context (cite sources by their bracketed numbers):\n"
        "{context}\n\n"
        "Question: {query}\n\n"
        "Structure your answer as:\n"
        "1. Trend Direction: improving / deteriorating / stable, with supporting data [citations]\n"
        "2. Magnitude: absolute change and percentage change across the period\n"
        "3. Key Drivers: identify 2-3 factors visible in the data that explain the trend\n"
        "4. Inflection Points: note any periods where the trend reversed or accelerated\n"
        "5. Outlook: forward-looking statement grounded only in available data\n\n"
        "Answer:"
    ),
    few_shot_examples=[
        {
            "question": "How has the gross margin trended over the past four quarters?",
            "answer": (
                "1. Trend Direction: Improving. Gross margin expanded from 38.2% in Q1 [1] "
                "to 41.7% in Q4 [1], a consistent four-quarter improvement.\n"
                "2. Magnitude: Absolute change of +3.5 percentage points; relative improvement "
                "of +9.2% over the period.\n"
                "3. Key Drivers: (a) Revenue mix shifted toward higher-margin software licenses [2]; "
                "(b) COGS declined as a share of revenue due to supply chain normalization [1]; "
                "(c) Pricing power evidenced by average selling price growth of 4.1% [2].\n"
                "4. Inflection Points: Q2 showed a brief 0.3pp dip due to a one-time component "
                "cost spike [1], but the upward trajectory resumed in Q3.\n"
                "5. Outlook: If the mix shift toward software continues, margin expansion toward "
                "43-44% appears sustainable based on current trajectory."
            ),
        },
        {
            "question": "What is the year-over-year revenue growth rate?",
            "answer": (
                "1. Trend Direction: Positive growth. Revenue increased from $2.4B in FY2022 [3] "
                "to $2.88B in FY2023 [3].\n"
                "2. Magnitude: Absolute increase of $480M; YoY growth rate of 20.0%.\n"
                "3. Key Drivers: (a) New market expansion added $210M in incremental revenue [4]; "
                "(b) Existing customer upsell drove 8% organic growth [4]; "
                "(c) Acquisition of XCorp in Q2 contributed $95M [4].\n"
                "4. Inflection Points: H1 growth of 14% accelerated to 26% in H2 following "
                "the XCorp consolidation [3].\n"
                "5. Outlook: Organic growth of 12-15% appears achievable absent further "
                "acquisitions, based on current pipeline data in the report."
            ),
        },
    ],
)

# ---------------------------------------------------------------------------
# Comparison Template
# ---------------------------------------------------------------------------
COMPARISON_PROMPT = PromptTemplate(
    name="comparison",
    version=PROMPT_VERSION,
    system_prompt=(
        "You are a senior financial analyst specializing in comparative analysis. "
        "Your answers are structured, balanced, and quantitatively grounded. "
        "Always cite sources (e.g., [1], [2]) for each data point. "
        "Present comparisons in a clear side-by-side format. "
        "Identify the stronger and weaker entity on each dimension, and summarize "
        "the overall conclusion with an evidence-based recommendation."
    ),
    user_template=(
        "{few_shot_block}"
        "Context (cite sources by their bracketed numbers):\n"
        "{context}\n\n"
        "Question: {query}\n\n"
        "Structure your answer as a comparison:\n\n"
        "METRIC COMPARISON TABLE\n"
        "| Metric | Entity/Period A | Entity/Period B | Advantage |\n"
        "| ------ | --------------- | --------------- | --------- |\n"
        "(populate with relevant metrics from the data)\n\n"
        "KEY DIFFERENCES\n"
        "- List 2-4 material differences with citations\n\n"
        "OVERALL CONCLUSION\n"
        "- State which is stronger/better and why, grounded in the data\n\n"
        "Answer:"
    ),
    few_shot_examples=[
        {
            "question": "Compare profitability between Q3 2023 and Q3 2024.",
            "answer": (
                "METRIC COMPARISON TABLE\n"
                "| Metric          | Q3 2023 [1] | Q3 2024 [2] | Advantage |\n"
                "| --------------- | ----------- | ----------- | --------- |\n"
                "| Gross Margin    | 39.1%       | 41.7%       | Q3 2024   |\n"
                "| Operating Margin| 18.3%       | 22.1%       | Q3 2024   |\n"
                "| Net Margin      | 12.8%       | 15.4%       | Q3 2024   |\n"
                "| EBITDA Margin   | 24.5%       | 28.2%       | Q3 2024   |\n\n"
                "KEY DIFFERENCES\n"
                "- Operating leverage improved significantly: OpEx as a % of revenue fell from "
                "20.8% to 19.6% [1][2], suggesting effective cost discipline.\n"
                "- Interest expense declined $12M YoY [2] following debt paydown, boosting "
                "net margin disproportionately relative to operating margin gains.\n"
                "- Gross margin expansion of 2.6pp was driven by software mix shift, "
                "not volume-related efficiencies [2].\n\n"
                "OVERALL CONCLUSION\n"
                "Q3 2024 is materially more profitable across all four margin metrics. "
                "The improvement is structural (mix shift, debt reduction) rather than "
                "cyclical, suggesting durability."
            ),
        },
        {
            "question": "How does the company's leverage compare to the prior year?",
            "answer": (
                "METRIC COMPARISON TABLE\n"
                "| Metric              | FY2022 [3] | FY2023 [3] | Change    |\n"
                "| ------------------- | ---------- | ---------- | --------- |\n"
                "| Debt-to-Equity      | 0.82x      | 0.54x      | Improved  |\n"
                "| Net Debt / EBITDA   | 2.1x       | 1.4x       | Improved  |\n"
                "| Interest Coverage   | 6.2x       | 9.1x       | Improved  |\n\n"
                "KEY DIFFERENCES\n"
                "- Net debt fell $340M after the company used free cash flow for voluntary "
                "debt repayment in H2 2023 [3].\n"
                "- EBITDA grew 18% [3], which simultaneously improved both coverage ratio "
                "and leverage multiple.\n\n"
                "OVERALL CONCLUSION\n"
                "The balance sheet deleveraged materially in FY2023. At 1.4x Net Debt/EBITDA, "
                "the company now carries a conservative leverage profile with ample capacity "
                "for additional investment or shareholder returns."
            ),
        },
    ],
)

# ---------------------------------------------------------------------------
# Explanation Template
# ---------------------------------------------------------------------------
EXPLANATION_PROMPT = PromptTemplate(
    name="explanation",
    version=PROMPT_VERSION,
    system_prompt=(
        "You are a senior financial analyst with expertise in causal reasoning and "
        "business model analysis, drawing on a Charlie Munger-style framework. "
        "Your answers explain the 'why' and 'how' behind financial outcomes by "
        "tracing cause-and-effect chains visible in the data. "
        "Cite sources (e.g., [1], [2]) for all factual claims. "
        "Use inversion thinking to identify risks, and focus on cash flows and "
        "competitive dynamics. Avoid speculation beyond what the data supports."
    ),
    user_template=(
        "{few_shot_block}"
        "Context (cite sources by their bracketed numbers):\n"
        "{context}\n\n"
        "Question: {query}\n\n"
        "Structure your causal explanation as:\n"
        "1. Root Cause(s): identify the primary driver(s) with evidence [citations]\n"
        "2. Mechanism: explain the cause-and-effect chain step by step\n"
        "3. Amplifying Factors: secondary forces that magnified the effect\n"
        "4. Inversion Check: what would have had to be true for the opposite outcome?\n"
        "5. Implications: what does this mean for future performance or risk?\n\n"
        "Answer:"
    ),
    few_shot_examples=[
        {
            "question": "Why did operating cash flow decline despite higher net income?",
            "answer": (
                "1. Root Cause(s): The primary driver was a $215M increase in accounts "
                "receivable [1], indicating revenue was recognized but cash had not yet "
                "been collected. This is a classic accrual-versus-cash timing gap.\n"
                "2. Mechanism: Net income rose $85M due to volume growth [2]. However, "
                "extended payment terms offered to new enterprise customers caused DSO "
                "to increase from 42 to 58 days [1], locking $215M of earnings in "
                "uncollected receivables and more than offsetting the earnings improvement.\n"
                "3. Amplifying Factors: Inventory also increased $40M [1] as the company "
                "built safety stock ahead of a new product launch, adding further working "
                "capital pressure.\n"
                "4. Inversion Check: For OCF to have matched net income growth, DSO would "
                "have needed to remain flat or improve, and inventory would need to have "
                "been held constant.\n"
                "5. Implications: If receivables collection normalizes as enterprise clients "
                "mature, OCF should recover strongly. However, persistently rising DSO would "
                "be a credit-quality warning sign requiring monitoring."
            ),
        },
        {
            "question": "What caused the gross margin expansion in FY2023?",
            "answer": (
                "1. Root Cause(s): Two concurrent forces drove the 3.2pp gross margin "
                "improvement: a product mix shift toward higher-margin software [2] and "
                "declining material costs as commodity prices normalized [1].\n"
                "2. Mechanism: Software revenue grew from 28% to 37% of total revenue [2]. "
                "Software carries ~75% gross margins versus ~35% for hardware [2], so the "
                "mix shift mechanically lifted the blended margin. Simultaneously, COGS per "
                "unit fell 6.4% as steel and semiconductor costs retreated [1].\n"
                "3. Amplifying Factors: Fixed manufacturing overhead was spread over a "
                "12% larger revenue base [2], creating operating leverage that further "
                "reduced unit cost.\n"
                "4. Inversion Check: Gross margin would have stayed flat or contracted "
                "if hardware volumes had grown faster than software, or if material costs "
                "had remained elevated.\n"
                "5. Implications: The mix shift is structural and likely to persist given "
                "management's stated software-first strategy, supporting continued margin "
                "improvement in FY2024."
            ),
        },
        {
            "question": "Why is the Altman Z-Score declining and what are the implications?",
            "answer": (
                "1. Root Cause(s): The Z-Score declined from 3.1 to 1.9 [3], primarily "
                "driven by a reduction in retained earnings (lower working capital) and "
                "increased total debt following the leveraged acquisition [3].\n"
                "2. Mechanism: The Z-Score formula weights retained earnings and EBIT "
                "relative to total assets heavily. The acquisition added $800M of goodwill "
                "to the denominator (total assets) [3] while only marginally increasing EBIT "
                "in the first year, compressing both ratios simultaneously.\n"
                "3. Amplifying Factors: Sales/Total Assets ratio also declined from 1.2x "
                "to 0.9x [3] as the asset base expanded faster than revenue integration.\n"
                "4. Inversion Check: The Z-Score would have held stable if the acquisition "
                "had been equity-financed and integrated faster to contribute EBIT within "
                "the fiscal year.\n"
                "5. Implications: At 1.9, the score sits in the 'grey zone' (1.81-2.99). "
                "If integration synergies materialize and EBIT recovers, the score should "
                "improve. Failure to realize synergies within 12-18 months would push "
                "it toward the distress zone below 1.81, warranting close monitoring."
            ),
        },
    ],
)

# ---------------------------------------------------------------------------
# General Template
# ---------------------------------------------------------------------------
GENERAL_PROMPT = PromptTemplate(
    name="general",
    version=PROMPT_VERSION,
    system_prompt=(
        "You are a knowledgeable financial analyst assistant. "
        "Answer questions accurately using only the information provided in the context. "
        "Cite sources using bracketed numbers (e.g., [1], [2]) when referencing data. "
        "If the answer is not available in the context, clearly state that the information "
        "is not present rather than speculating. Be concise but complete."
    ),
    user_template=(
        "{few_shot_block}"
        "Context (cite sources by their bracketed numbers):\n"
        "{context}\n\n"
        "Question: {query}\n\n"
        "Answer (use citations where applicable):"
    ),
    few_shot_examples=[
        {
            "question": "What business segments does the company operate in?",
            "answer": (
                "Based on the annual report [1], the company operates three business segments: "
                "(1) Enterprise Software, which generated $1.2B in revenue representing 42% of "
                "total sales; (2) Professional Services at $980M (34%); and "
                "(3) Hardware Solutions at $680M (24%). The Enterprise Software segment is the "
                "highest-margin division with an operating margin of 31% [1]."
            ),
        },
        {
            "question": "What was the total dividend paid in the last fiscal year?",
            "answer": (
                "The company paid total dividends of $142M in FY2023 [2], representing a "
                "dividend per share of $0.84, an increase from $0.72 per share in FY2022 [2]. "
                "The payout ratio was 28.4% of net income, indicating ample earnings retention "
                "for reinvestment while returning capital to shareholders."
            ),
        },
    ],
)

# ---------------------------------------------------------------------------
# Template registry and selector
# ---------------------------------------------------------------------------
_TEMPLATE_REGISTRY: Dict[str, PromptTemplate] = {
    "ratio_lookup": RATIO_LOOKUP_PROMPT,
    "trend_analysis": TREND_ANALYSIS_PROMPT,
    "comparison": COMPARISON_PROMPT,
    "explanation": EXPLANATION_PROMPT,
    "general": GENERAL_PROMPT,
}


def get_prompt_for_query_type(query_type: str) -> PromptTemplate:
    """Return the best-matching PromptTemplate for the given query type.

    Falls back to GENERAL_PROMPT for any unrecognized query type so the
    system always has a valid template to use.

    Args:
        query_type: One of 'ratio_lookup', 'trend_analysis', 'comparison',
                    'explanation', or 'general' (as returned by
                    SimpleRAG._classify_query()).

    Returns:
        The matching PromptTemplate instance.
    """
    return _TEMPLATE_REGISTRY.get(query_type, GENERAL_PROMPT)
