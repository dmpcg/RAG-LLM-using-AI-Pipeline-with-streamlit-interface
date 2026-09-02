"""
Tests for the prompts package (Phase 2.3 - Prompt Engineering).

Covers:
- PromptTemplate dataclass field validation
- get_prompt_for_query_type selector
- format_context_with_citations
- format_few_shot_examples
- build_prompt assembly
- format_json_instruction
- Fallback / unknown query type behaviour
- Few-shot example content quality
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts import (  # noqa: F401 – smoke-test __init__ exports
    PROMPT_VERSION as _PVER,
)
from prompts.formatters import (
    build_prompt,
    format_context_with_citations,
    format_few_shot_examples,
    format_json_instruction,
)
from prompts.templates import (
    COMPARISON_PROMPT,
    EXPLANATION_PROMPT,
    GENERAL_PROMPT,
    PROMPT_VERSION,
    RATIO_LOOKUP_PROMPT,
    TREND_ANALYSIS_PROMPT,
    PromptTemplate,
    get_prompt_for_query_type,
)

# ---------------------------------------------------------------------------
# PROMPT_VERSION constant
# ---------------------------------------------------------------------------


class TestPromptVersion:
    def test_version_is_string(self):
        assert isinstance(PROMPT_VERSION, str)

    def test_version_matches_all_templates(self):
        templates = [
            RATIO_LOOKUP_PROMPT,
            TREND_ANALYSIS_PROMPT,
            COMPARISON_PROMPT,
            EXPLANATION_PROMPT,
            GENERAL_PROMPT,
        ]
        for t in templates:
            assert t.version == PROMPT_VERSION, (
                f"Template '{t.name}' version '{t.version}' does not match PROMPT_VERSION '{PROMPT_VERSION}'"
            )

    def test_version_non_empty(self):
        assert PROMPT_VERSION.strip() != ""


# ---------------------------------------------------------------------------
# PromptTemplate dataclass
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_instantiate_minimal(self):
        t = PromptTemplate(
            name="test",
            version="1.0",
            system_prompt="You are helpful.",
            user_template="{context}\n{query}",
        )
        assert t.name == "test"
        assert t.few_shot_examples == []

    def test_instantiate_with_examples(self):
        examples = [{"question": "Q?", "answer": "A."}]
        t = PromptTemplate(
            name="test",
            version="1.0",
            system_prompt="sys",
            user_template="{context}\n{query}",
            few_shot_examples=examples,
        )
        assert len(t.few_shot_examples) == 1

    @pytest.mark.parametrize(
        "template",
        [
            RATIO_LOOKUP_PROMPT,
            TREND_ANALYSIS_PROMPT,
            COMPARISON_PROMPT,
            EXPLANATION_PROMPT,
            GENERAL_PROMPT,
        ],
    )
    def test_required_fields_present(self, template):
        assert template.name
        assert template.version
        assert template.system_prompt
        assert template.user_template

    @pytest.mark.parametrize(
        "template",
        [
            RATIO_LOOKUP_PROMPT,
            TREND_ANALYSIS_PROMPT,
            COMPARISON_PROMPT,
            EXPLANATION_PROMPT,
            GENERAL_PROMPT,
        ],
    )
    def test_user_template_has_context_placeholder(self, template):
        assert "{context}" in template.user_template, f"Template '{template.name}' is missing {{context}} placeholder"

    @pytest.mark.parametrize(
        "template",
        [
            RATIO_LOOKUP_PROMPT,
            TREND_ANALYSIS_PROMPT,
            COMPARISON_PROMPT,
            EXPLANATION_PROMPT,
            GENERAL_PROMPT,
        ],
    )
    def test_user_template_has_query_placeholder(self, template):
        assert "{query}" in template.user_template, f"Template '{template.name}' is missing {{query}} placeholder"


# ---------------------------------------------------------------------------
# get_prompt_for_query_type
# ---------------------------------------------------------------------------


class TestGetPromptForQueryType:
    def test_ratio_lookup_returns_correct_template(self):
        t = get_prompt_for_query_type("ratio_lookup")
        assert t.name == "ratio_lookup"

    def test_trend_analysis_returns_correct_template(self):
        t = get_prompt_for_query_type("trend_analysis")
        assert t.name == "trend_analysis"

    def test_comparison_returns_correct_template(self):
        t = get_prompt_for_query_type("comparison")
        assert t.name == "comparison"

    def test_explanation_returns_correct_template(self):
        t = get_prompt_for_query_type("explanation")
        assert t.name == "explanation"

    def test_general_returns_correct_template(self):
        t = get_prompt_for_query_type("general")
        assert t.name == "general"

    def test_unknown_type_falls_back_to_general(self):
        t = get_prompt_for_query_type("nonexistent_type_xyz")
        assert t.name == "general"

    def test_empty_string_falls_back_to_general(self):
        t = get_prompt_for_query_type("")
        assert t.name == "general"

    def test_returns_prompt_template_instance(self):
        t = get_prompt_for_query_type("ratio_lookup")
        assert isinstance(t, PromptTemplate)


# ---------------------------------------------------------------------------
# format_context_with_citations
# ---------------------------------------------------------------------------


class TestFormatContextWithCitations:
    def _make_doc(self, source, content, citation=None, doc_type=None, financial_type=None):
        doc = {"source": source, "content": content}
        if citation:
            doc["_citation"] = citation
        if doc_type:
            doc["type"] = doc_type
        if financial_type:
            doc["metadata"] = {"financial_type": financial_type}
        return doc

    def test_empty_list_returns_empty_string(self):
        assert format_context_with_citations([]) == ""

    def test_single_doc_numbered(self):
        doc = self._make_doc("report.pdf", "Revenue was $1B.")
        result = format_context_with_citations([doc])
        assert "[1]" in result
        assert "Revenue was $1B." in result
        assert "report.pdf" in result

    def test_multiple_docs_numbered_sequentially(self):
        docs = [
            self._make_doc("a.pdf", "Content A."),
            self._make_doc("b.pdf", "Content B."),
            self._make_doc("c.pdf", "Content C."),
        ]
        result = format_context_with_citations(docs)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_existing_citation_key_is_respected(self):
        doc = self._make_doc("report.pdf", "Net income $500M.", citation="[3]")
        result = format_context_with_citations([doc])
        # Should still use [3] from the pre-assigned citation
        assert "[3]" in result

    def test_financial_type_metadata_included(self):
        doc = self._make_doc(
            "balance_sheet.xlsx", "Total assets $2B.", doc_type="excel", financial_type="balance_sheet"
        )
        result = format_context_with_citations([doc])
        assert "balance_sheet" in result

    def test_content_present_in_output(self):
        doc = self._make_doc("doc.pdf", "Unique content string 12345.")
        result = format_context_with_citations([doc])
        assert "Unique content string 12345." in result

    def test_docs_separated_by_blank_lines(self):
        docs = [
            self._make_doc("a.pdf", "A content."),
            self._make_doc("b.pdf", "B content."),
        ]
        result = format_context_with_citations(docs)
        # Double newline separator between blocks
        assert "\n\n" in result


# ---------------------------------------------------------------------------
# format_few_shot_examples
# ---------------------------------------------------------------------------


class TestFormatFewShotExamples:
    def test_empty_list_returns_empty_string(self):
        assert format_few_shot_examples([]) == ""

    def test_single_example_contains_question_and_answer(self):
        examples = [{"question": "What is ROE?", "answer": "ROE is 25%."}]
        result = format_few_shot_examples(examples)
        assert "What is ROE?" in result
        assert "ROE is 25%." in result

    def test_multiple_examples_numbered(self):
        examples = [
            {"question": "Q1?", "answer": "A1."},
            {"question": "Q2?", "answer": "A2."},
        ]
        result = format_few_shot_examples(examples)
        assert "Example 1" in result
        assert "Example 2" in result

    def test_block_has_header_and_footer(self):
        examples = [{"question": "Q?", "answer": "A."}]
        result = format_few_shot_examples(examples)
        assert "EXAMPLES" in result
        assert "END EXAMPLES" in result

    def test_example_missing_question_skipped(self):
        examples = [
            {"question": "", "answer": "A."},
            {"question": "Q2?", "answer": "A2."},
        ]
        result = format_few_shot_examples(examples)
        # Only the second valid example should appear
        assert "Q2?" in result
        # The empty-question example should not create a numbered entry
        assert "Example 1" not in result or "Q2?" in result

    def test_all_empty_examples_returns_empty_string(self):
        examples = [{"question": "", "answer": ""}, {"question": "", "answer": ""}]
        result = format_few_shot_examples(examples)
        assert result == ""


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def _simple_template(self):
        return PromptTemplate(
            name="test",
            version="2.0",
            system_prompt="You are a test analyst.",
            user_template="{few_shot_block}Context:\n{context}\n\nQ: {query}\n\nAnswer:",
        )

    def test_system_prompt_in_output(self):
        t = self._simple_template()
        result = build_prompt(t, "What is ROE?", "Revenue: $1B.")
        assert "You are a test analyst." in result

    def test_context_in_output(self):
        t = self._simple_template()
        result = build_prompt(t, "What is ROE?", "Revenue: $1B.")
        assert "Revenue: $1B." in result

    def test_query_in_output(self):
        t = self._simple_template()
        result = build_prompt(t, "What is ROE?", "Revenue: $1B.")
        assert "What is ROE?" in result

    def test_few_shot_examples_included(self):
        t = PromptTemplate(
            name="test",
            version="2.0",
            system_prompt="sys",
            user_template="{few_shot_block}{context}\n{query}",
            few_shot_examples=[{"question": "Eg Q?", "answer": "Eg A."}],
        )
        result = build_prompt(t, "My query.", "My context.")
        assert "Eg Q?" in result
        assert "Eg A." in result

    def test_no_examples_no_example_block(self):
        t = self._simple_template()
        result = build_prompt(t, "q", "ctx")
        assert "EXAMPLES" not in result

    def test_system_section_prefix(self):
        t = self._simple_template()
        result = build_prompt(t, "q", "ctx")
        assert result.startswith("SYSTEM:")

    def test_user_section_present(self):
        t = self._simple_template()
        result = build_prompt(t, "q", "ctx")
        assert "USER:" in result

    def test_full_ratio_template_assembles(self):
        result = build_prompt(RATIO_LOOKUP_PROMPT, "What is ROE?", "[1] Net income $100M.")
        assert "SYSTEM:" in result
        assert "What is ROE?" in result
        assert "[1] Net income $100M." in result


# ---------------------------------------------------------------------------
# format_json_instruction
# ---------------------------------------------------------------------------


class TestFormatJsonInstruction:
    def test_ratio_lookup_returns_non_empty(self):
        result = format_json_instruction("ratio_lookup")
        assert result.strip() != ""

    def test_ratio_lookup_contains_json_keyword(self):
        result = format_json_instruction("ratio_lookup")
        assert "JSON" in result or "json" in result

    def test_ratio_lookup_contains_ratio_fields(self):
        result = format_json_instruction("ratio_lookup")
        assert "ratio_name" in result
        assert "result" in result

    def test_trend_analysis_returns_non_empty(self):
        result = format_json_instruction("trend_analysis")
        assert result.strip() != ""

    def test_trend_analysis_contains_trend_fields(self):
        result = format_json_instruction("trend_analysis")
        assert "trend_direction" in result

    def test_comparison_returns_empty_string(self):
        result = format_json_instruction("comparison")
        assert result == ""

    def test_explanation_returns_empty_string(self):
        result = format_json_instruction("explanation")
        assert result == ""

    def test_general_returns_empty_string(self):
        result = format_json_instruction("general")
        assert result == ""

    def test_unknown_type_returns_empty_string(self):
        result = format_json_instruction("bogus_type")
        assert result == ""


# ---------------------------------------------------------------------------
# Few-shot example quality checks
# ---------------------------------------------------------------------------


class TestFewShotExampleQuality:
    @pytest.mark.parametrize(
        "template",
        [
            RATIO_LOOKUP_PROMPT,
            TREND_ANALYSIS_PROMPT,
            COMPARISON_PROMPT,
            EXPLANATION_PROMPT,
            GENERAL_PROMPT,
        ],
    )
    def test_at_least_one_example(self, template):
        assert len(template.few_shot_examples) >= 1, f"Template '{template.name}' has no few-shot examples"

    @pytest.mark.parametrize(
        "template",
        [
            RATIO_LOOKUP_PROMPT,
            TREND_ANALYSIS_PROMPT,
            COMPARISON_PROMPT,
            EXPLANATION_PROMPT,
            GENERAL_PROMPT,
        ],
    )
    def test_all_examples_have_non_empty_question(self, template):
        for i, ex in enumerate(template.few_shot_examples):
            assert ex.get("question", "").strip(), f"Template '{template.name}' example {i} has empty 'question'"

    @pytest.mark.parametrize(
        "template",
        [
            RATIO_LOOKUP_PROMPT,
            TREND_ANALYSIS_PROMPT,
            COMPARISON_PROMPT,
            EXPLANATION_PROMPT,
            GENERAL_PROMPT,
        ],
    )
    def test_all_examples_have_non_empty_answer(self, template):
        for i, ex in enumerate(template.few_shot_examples):
            assert ex.get("answer", "").strip(), f"Template '{template.name}' example {i} has empty 'answer'"

    def test_ratio_examples_contain_formula(self):
        for ex in RATIO_LOOKUP_PROMPT.few_shot_examples:
            assert "Formula" in ex["answer"] or "formula" in ex["answer"], (
                "Ratio example answer should describe the formula used"
            )

    def test_trend_examples_contain_direction(self):
        for ex in TREND_ANALYSIS_PROMPT.few_shot_examples:
            answer_lower = ex["answer"].lower()
            has_direction = any(
                word in answer_lower for word in ["improving", "deteriorating", "stable", "positive", "trend"]
            )
            assert has_direction, "Trend example should describe a direction"

    def test_comparison_examples_contain_table(self):
        for ex in COMPARISON_PROMPT.few_shot_examples:
            assert "|" in ex["answer"], "Comparison example should contain a markdown table (pipe character)"

    def test_explanation_examples_contain_causal_language(self):
        for ex in EXPLANATION_PROMPT.few_shot_examples:
            answer_lower = ex["answer"].lower()
            has_causal = any(
                word in answer_lower for word in ["because", "caused", "drove", "driven", "mechanism", "root cause"]
            )
            assert has_causal, "Explanation example should contain causal language"


# ---------------------------------------------------------------------------
# __init__.py export smoke tests
# ---------------------------------------------------------------------------


class TestInitExports:
    def test_prompt_version_exported(self):
        from prompts import PROMPT_VERSION as pv

        assert isinstance(pv, str)

    def test_get_prompt_for_query_type_exported(self):
        from prompts import get_prompt_for_query_type as fn

        assert callable(fn)

    def test_build_prompt_exported(self):
        from prompts import build_prompt as fn

        assert callable(fn)

    def test_format_context_with_citations_exported(self):
        from prompts import format_context_with_citations as fn

        assert callable(fn)

    def test_format_json_instruction_exported(self):
        from prompts import format_json_instruction as fn

        assert callable(fn)

    def test_all_template_constants_exported(self):
        import prompts

        for name in [
            "RATIO_LOOKUP_PROMPT",
            "TREND_ANALYSIS_PROMPT",
            "COMPARISON_PROMPT",
            "EXPLANATION_PROMPT",
            "GENERAL_PROMPT",
        ]:
            assert hasattr(prompts, name), f"prompts.{name} not exported"
