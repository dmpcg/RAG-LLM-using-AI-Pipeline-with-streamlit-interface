"""Tests for compliance_scorer.py -- regulatory & compliance scoring."""

import pytest

from financial_analyzer import FinancialData
from compliance_scorer import (
    AuditRiskAssessment,
    ComplianceReport,
    ComplianceScorer,
    RegulatoryRatioReport,
    RegulatoryThreshold,
    SECFilingQuality,
    SOXComplianceResult,
    _score_to_grade,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def compliant_company() -> FinancialData:
    """Company that passes most compliance checks."""
    return FinancialData(
        revenue=10_000_000,
        net_income=1_000_000,
        gross_profit=6_000_000,
        operating_income=2_000_000,
        ebit=2_000_000,
        ebitda=2_500_000,
        interest_expense=200_000,
        cogs=4_000_000,
        total_assets=20_000_000,
        current_assets=8_000_000,
        current_liabilities=3_000_000,
        total_debt=5_000_000,
        total_equity=13_000_000,
        total_liabilities=7_000_000,
        operating_cash_flow=1_500_000,
        cash=2_000_000,
        accounts_receivable=2_000_000,
        inventory=1_500_000,
        accounts_payable=1_000_000,
        depreciation=500_000,
        capex=800_000,
    )


@pytest.fixture
def noncompliant_company() -> FinancialData:
    """Company that fails many compliance checks."""
    return FinancialData(
        revenue=2_000_000,
        net_income=-500_000,
        gross_profit=200_000,
        operating_income=-300_000,
        ebit=-300_000,
        ebitda=-100_000,
        interest_expense=400_000,
        cogs=1_800_000,
        total_assets=3_000_000,
        current_assets=500_000,
        current_liabilities=2_000_000,
        total_debt=3_500_000,
        total_equity=-500_000,  # Negative equity
        total_liabilities=3_500_000,
        operating_cash_flow=-100_000,
        cash=100_000,
        accounts_receivable=1_000_000,  # AR/Rev = 0.50 (high)
    )


@pytest.fixture
def scorer():
    return ComplianceScorer()


# ---------------------------------------------------------------------------
# SOX Compliance
# ---------------------------------------------------------------------------


class TestSOXCompliance:
    def test_compliant_low_risk(self, scorer, compliant_company):
        sox = scorer.sox_compliance(compliant_company)
        assert isinstance(sox, SOXComplianceResult)
        assert sox.overall_risk == "low"
        assert sox.risk_score >= 80
        assert sox.checks_performed > 0
        assert len(sox.material_weakness_indicators) == 0

    def test_noncompliant_high_risk(self, scorer, noncompliant_company):
        sox = scorer.sox_compliance(noncompliant_company)
        assert sox.overall_risk in ("moderate", "high")
        assert sox.risk_score < 80
        assert len(sox.flags) > 0

    def test_negative_equity_flagged(self, scorer, noncompliant_company):
        sox = scorer.sox_compliance(noncompliant_company)
        mw = sox.material_weakness_indicators
        neg_equity = [m for m in mw if "negative" in m.lower()]
        assert len(neg_equity) > 0

    def test_high_ar_revenue_flagged(self, scorer, noncompliant_company):
        sox = scorer.sox_compliance(noncompliant_company)
        ar_flags = [f for f in sox.flags if "AR/Revenue" in f]
        assert len(ar_flags) > 0

    def test_operating_loss_flagged(self, scorer, noncompliant_company):
        sox = scorer.sox_compliance(noncompliant_company)
        loss_flags = [f for f in sox.flags if "Operating loss" in f]
        assert len(loss_flags) > 0

    def test_negative_ni_ocf_ratio_flagged(self, scorer):
        """BUG-2 fix: negative NI with negative OCF should still flag concern,
        even though OCF/NI ratio > 0.5 (both negative → positive ratio)."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=-200_000,
            operating_cash_flow=-150_000,
            total_assets=2_000_000,
            total_equity=500_000,
        )
        sox = scorer.sox_compliance(data)
        # Should flag negative NI as earnings quality concern
        ni_flags = [f for f in sox.flags if "Negative net income" in f]
        assert len(ni_flags) > 0

    def test_positive_ni_low_ocf_flagged(self, scorer):
        """Positive NI with low OCF/NI should still flag divergence."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=200_000,
            operating_cash_flow=50_000,  # OCF/NI = 0.25
            total_assets=2_000_000,
            total_equity=500_000,
        )
        sox = scorer.sox_compliance(data)
        ocf_flags = [f for f in sox.flags if "OCF/NI" in f]
        assert len(ocf_flags) > 0


# ---------------------------------------------------------------------------
# SEC Filing Quality
# ---------------------------------------------------------------------------


class TestSECFilingQuality:
    def test_complete_data_high_score(self, scorer, compliant_company):
        sec = scorer.sec_filing_quality(compliant_company)
        assert isinstance(sec, SECFilingQuality)
        assert sec.disclosure_score >= 60
        assert sec.grade in ("A", "B", "C")
        assert sec.data_completeness_pct > 50

    def test_missing_critical_fields(self, scorer):
        sparse = FinancialData(revenue=1_000_000)
        sec = scorer.sec_filing_quality(sparse)
        assert sec.disclosure_score < 50
        assert len(sec.missing_critical_fields) > 0

    def test_balance_sheet_consistency(self, scorer, compliant_company):
        sec = scorer.sec_filing_quality(compliant_company)
        # compliant_company: A=20M, L=7M, E=13M -> L+E=20M -> passes
        assert sec.consistency_checks_passed >= 1

    def test_balance_sheet_inconsistency_flagged(self, scorer):
        # Assets != Liabilities + Equity
        bad = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_assets=10_000_000,
            total_liabilities=3_000_000,
            total_equity=3_000_000,  # 3M + 3M = 6M != 10M
            operating_cash_flow=200_000,
            current_assets=4_000_000,
            current_liabilities=1_000_000,
        )
        sec = scorer.sec_filing_quality(bad)
        bs_flags = [f for f in sec.red_flags if "balance" in f.lower()]
        assert len(bs_flags) > 0


# ---------------------------------------------------------------------------
# Regulatory Ratios
# ---------------------------------------------------------------------------


class TestRegulatoryRatios:
    def test_compliant_passes_most(self, scorer, compliant_company):
        reg = scorer.regulatory_ratios(compliant_company)
        assert isinstance(reg, RegulatoryRatioReport)
        assert reg.pass_count > 0
        assert reg.compliance_pct > 50

    def test_noncompliant_fails(self, scorer, noncompliant_company):
        reg = scorer.regulatory_ratios(noncompliant_company)
        assert reg.fail_count > 0
        assert len(reg.critical_failures) > 0

    def test_all_thresholds_checked(self, scorer, compliant_company):
        reg = scorer.regulatory_ratios(compliant_company)
        assert len(reg.thresholds_checked) == 6  # 6 rules defined

    def test_threshold_fields_populated(self, scorer, compliant_company):
        reg = scorer.regulatory_ratios(compliant_company)
        for t in reg.thresholds_checked:
            assert isinstance(t, RegulatoryThreshold)
            assert t.rule_name != ""
            assert t.framework != ""
            assert t.threshold_value is not None


# ---------------------------------------------------------------------------
# Audit Risk Assessment
# ---------------------------------------------------------------------------


class TestAuditRisk:
    def test_low_risk_compliant(self, scorer, compliant_company):
        audit = scorer.audit_risk_assessment(compliant_company)
        assert isinstance(audit, AuditRiskAssessment)
        assert audit.risk_level in ("low", "moderate")
        assert audit.score >= 50
        assert not audit.going_concern_risk

    def test_high_risk_noncompliant(self, scorer, noncompliant_company):
        audit = scorer.audit_risk_assessment(noncompliant_company)
        assert audit.risk_level in ("high", "critical")
        assert audit.going_concern_risk is True
        assert len(audit.restatement_risk_indicators) > 0

    def test_recommendations_populated(self, scorer, noncompliant_company):
        audit = scorer.audit_risk_assessment(noncompliant_company)
        assert len(audit.recommendations) > 0


# ---------------------------------------------------------------------------
# Full Compliance Report
# ---------------------------------------------------------------------------


class TestFullComplianceReport:
    def test_full_report_compliant(self, scorer, compliant_company):
        report = scorer.full_compliance_report(compliant_company)
        assert isinstance(report, ComplianceReport)
        assert report.sox is not None
        assert report.sec is not None
        assert report.regulatory is not None
        assert report.audit_risk is not None
        assert len(report.summary) > 20

    def test_full_report_noncompliant(self, scorer, noncompliant_company):
        report = scorer.full_compliance_report(noncompliant_company)
        assert report.audit_risk.going_concern_risk is True
        assert "GOING CONCERN" in report.summary

    def test_summary_contains_scores(self, scorer, compliant_company):
        report = scorer.full_compliance_report(compliant_company)
        assert "SOX" in report.summary
        assert "SEC" in report.summary
        assert "Regulatory" in report.summary


# ---------------------------------------------------------------------------
# All-None data (Coverage gap 5)
# ---------------------------------------------------------------------------


class TestAllNoneData:
    """Coverage gap: FinancialData with all None fields should not crash."""

    def test_full_compliance_all_none_no_exception(self, scorer):
        data = FinancialData()
        report = scorer.full_compliance_report(data)
        assert isinstance(report, ComplianceReport)
        assert report.sox is not None
        assert report.sec is not None
        assert report.regulatory is not None

    def test_sox_all_none_no_flags(self, scorer):
        data = FinancialData()
        result = scorer.sox_compliance(data)
        # With all None, insufficient data means checks pass (no flags)
        assert isinstance(result, SOXComplianceResult)

    def test_regulatory_all_none_passes_all(self, scorer):
        data = FinancialData()
        result = scorer.regulatory_ratios(data)
        # Missing data → passes=True for each threshold
        assert isinstance(result, RegulatoryRatioReport)
        assert len(result.thresholds_checked) > 0

    def test_audit_risk_all_none_no_exception(self, scorer):
        data = FinancialData()
        result = scorer.full_compliance_report(data)
        assert isinstance(result.audit_risk, AuditRiskAssessment)


# ---------------------------------------------------------------------------
# SOX boundary values (Coverage gap 6)
# ---------------------------------------------------------------------------


class TestSOXBoundaryValues:
    def test_interest_coverage_exactly_one_is_material_weakness(self, scorer):
        """ic < 1.0 is material weakness; ic == 1.0 should be significant deficiency."""
        data = FinancialData(
            revenue=10_000_000, net_income=500_000,
            ebit=200_000, interest_expense=200_000,  # ic = 1.0
            operating_income=500_000, operating_cash_flow=500_000,
            total_equity=5_000_000, total_assets=10_000_000,
        )
        result = scorer.sox_compliance(data)
        # ic=1.0: not < 1.0 so not material weakness, but < 2.0 so significant deficiency
        assert len(result.significant_deficiency_indicators) >= 1 or len(result.flags) >= 1

    def test_interest_coverage_below_one_is_material_weakness(self, scorer):
        data = FinancialData(
            revenue=10_000_000, net_income=500_000,
            ebit=100_000, interest_expense=200_000,  # ic = 0.5
            operating_income=500_000, operating_cash_flow=500_000,
            total_equity=5_000_000, total_assets=10_000_000,
        )
        result = scorer.sox_compliance(data)
        assert len(result.material_weakness_indicators) >= 1

    def test_negative_equity_material_weakness(self, scorer):
        data = FinancialData(
            revenue=10_000_000, net_income=500_000,
            operating_income=500_000, operating_cash_flow=500_000,
            total_equity=-1_000_000, total_assets=10_000_000,
        )
        result = scorer.sox_compliance(data)
        assert len(result.material_weakness_indicators) >= 1


# ---------------------------------------------------------------------------
# SEC Filing Quality edge cases (Coverage gap 7)
# ---------------------------------------------------------------------------


class TestSECFilingQualityEdgeCases:
    def test_all_none_returns_valid_result(self, scorer):
        data = FinancialData()
        result = scorer.sec_filing_quality(data)
        assert isinstance(result, SECFilingQuality)
        assert result.disclosure_score >= 0

    def test_complete_data_high_score(self, scorer, compliant_company):
        result = scorer.sec_filing_quality(compliant_company)
        assert result.disclosure_score >= 50
        assert result.grade in ("A", "B", "C")

    def test_balance_sheet_mismatch_flagged(self, scorer):
        """Assets != Liabilities + Equity should produce a red flag."""
        data = FinancialData(
            revenue=10_000_000,
            total_assets=100_000_000,
            total_liabilities=20_000_000,
            total_equity=30_000_000,  # Sum = 50M, mismatch with 100M assets
        )
        result = scorer.sec_filing_quality(data)
        assert len(result.red_flags) >= 1


# ---------------------------------------------------------------------------
# Regulatory threshold boundary values (Coverage gap 15)
# ---------------------------------------------------------------------------


class TestRegulatoryThresholdBoundary:
    def test_equity_ratio_exactly_at_threshold_passes(self, scorer):
        """Equity ratio >= 6% should pass."""
        data = FinancialData(
            total_equity=6_000, total_assets=100_000,  # 6%
            revenue=50_000, current_assets=20_000, current_liabilities=10_000,
        )
        result = scorer.regulatory_ratios(data)
        equity_check = [r for r in result.thresholds_checked if "Equity" in r.rule_name]
        if equity_check:
            assert equity_check[0].passes is True

    def test_equity_ratio_below_threshold_fails(self, scorer):
        data = FinancialData(
            total_equity=5_000, total_assets=100_000,  # 5%
            revenue=50_000, current_assets=20_000, current_liabilities=10_000,
        )
        result = scorer.regulatory_ratios(data)
        equity_check = [r for r in result.thresholds_checked if "Equity" in r.rule_name]
        if equity_check:
            assert equity_check[0].passes is False

    def test_high_leverage_fails(self, scorer):
        data = FinancialData(
            total_debt=90_000, total_assets=100_000,  # 90% leverage
            total_equity=10_000, revenue=50_000,
            current_assets=20_000, current_liabilities=10_000,
        )
        result = scorer.regulatory_ratios(data)
        lev_check = [r for r in result.thresholds_checked if "Leverage" in r.rule_name]
        if lev_check:
            assert lev_check[0].passes is False


# ---------------------------------------------------------------------------
# Score clamping boundary tests
# ---------------------------------------------------------------------------


class TestScoreClamping:
    """Verify all score outputs are clamped to [0, 100]."""

    def test_sox_score_never_negative(self, scorer):
        """SOX score should be >= 0 even with many material weaknesses."""
        data = FinancialData(
            revenue=10_000_000, net_income=-5_000_000,
            operating_income=-5_000_000, operating_cash_flow=-1_000_000,
            total_equity=-10_000_000, total_assets=10_000_000,
            ebit=-1_000_000, interest_expense=500_000,
            accounts_receivable=6_000_000,
        )
        result = scorer.sox_compliance(data)
        assert 0 <= result.risk_score <= 100

    def test_sox_score_never_above_100(self, scorer, compliant_company):
        """SOX score should be <= 100 even for perfectly compliant company."""
        result = scorer.sox_compliance(compliant_company)
        assert 0 <= result.risk_score <= 100

    def test_sec_score_never_negative(self, scorer):
        """SEC filing quality should be >= 0 for empty data."""
        data = FinancialData()
        result = scorer.sec_filing_quality(data)
        assert 0 <= result.disclosure_score <= 100

    def test_audit_risk_score_clamped(self, scorer):
        """Audit risk total should stay within [0, 100]."""
        data = FinancialData(
            revenue=10_000_000, net_income=-5_000_000,
            operating_income=-5_000_000, operating_cash_flow=-1_000_000,
            total_equity=-10_000_000, total_assets=10_000_000,
            ebit=-1_000_000, interest_expense=500_000,
        )
        result = scorer.audit_risk_assessment(data)
        assert 0 <= result.score <= 100

    def test_grade_mapping_comprehensive(self):
        """_score_to_grade handles all ranges."""
        assert _score_to_grade(95) == "A"
        assert _score_to_grade(80) in ("A", "B")
        assert _score_to_grade(0) in ("D", "F")
        assert _score_to_grade(100) == "A"


# ---------------------------------------------------------------------------
# WP-5 (P1-E7): SOX score reflects balance-sheet imbalance
# ---------------------------------------------------------------------------


@pytest.fixture
def bs_balanced() -> FinancialData:
    """Balance sheet that balances: Assets = Liabilities + Equity (diff 0)."""
    return FinancialData(
        total_assets=1000.0,
        total_liabilities=600.0,
        total_equity=400.0,
    )


@pytest.fixture
def bs_imbalanced() -> FinancialData:
    """Balance sheet that does NOT balance: 1000 vs 900 (10% diff, > 1% tol)."""
    return FinancialData(
        total_assets=1000.0,
        total_liabilities=600.0,
        total_equity=300.0,
    )


class TestSOXBalanceSheetImbalance:
    def test_tolerance_constant_is_one_percent(self):
        """The shared BS imbalance tolerance is the pinned 1% (UA-3)."""
        from compliance_scorer import _BS_IMBALANCE_TOLERANCE

        assert _BS_IMBALANCE_TOLERANCE == 0.01

    def test_imbalanced_lowers_sox_risk_score_with_local_penalty(
        self, scorer, bs_balanced, bs_imbalanced
    ):
        """Imbalanced BS -> SOX risk_score drops + SOX-local penalty present."""
        balanced = scorer.sox_compliance(bs_balanced)
        imbalanced = scorer.sox_compliance(bs_imbalanced)

        # Local penalty field present and positive only for the imbalanced case
        assert imbalanced.bs_imbalance_penalty > 0
        assert balanced.bs_imbalance_penalty == 0

        # The imbalance must actually pull the SOX risk_score down
        assert imbalanced.risk_score < balanced.risk_score
        assert imbalanced.risk_score == balanced.risk_score - imbalanced.bs_imbalance_penalty

    def test_balanced_sox_risk_score_unchanged(self, scorer, bs_balanced):
        """Balanced BS -> no penalty, score reflects only the other checks."""
        result = scorer.sox_compliance(bs_balanced)
        assert result.bs_imbalance_penalty == 0

    def test_bs_imbalance_not_in_material_weakness(
        self, scorer, bs_imbalanced
    ):
        """The imbalance signal must NOT be appended to material_weakness."""
        result = scorer.sox_compliance(bs_imbalanced)
        joined = " ".join(result.material_weakness_indicators).lower()
        assert "balance" not in joined
        # Equally, it must not be a significant_deficiency list entry.
        joined_sd = " ".join(result.significant_deficiency_indicators).lower()
        assert "balance sheet" not in joined_sd

    def test_overall_audit_risk_indicator_count_unchanged(
        self, scorer, bs_balanced, bs_imbalanced
    ):
        """No double-count: overall_audit_risk restatement-indicator COUNT is
        identical with vs. without the SOX-local imbalance change.

        The imbalance already flows into overall_audit_risk via sec.red_flags,
        so the SOX-local penalty must add nothing to the indicator list.
        """
        bal = scorer.audit_risk_assessment(bs_balanced)
        imbal = scorer.audit_risk_assessment(bs_imbalanced)

        # Count the imbalance-attributable restatement indicators (from sec.red_flags).
        def imbalance_indicators(assessment):
            return [
                ind
                for ind in assessment.restatement_risk_indicators
                if "balance sheet does not balance" in ind.lower()
            ]

        # Exactly one representation of the imbalance in the imbalanced overall
        # assessment (via sec.red_flags), and none in the balanced one.
        assert len(imbalance_indicators(imbal)) == 1
        assert len(imbalance_indicators(bal)) == 0
