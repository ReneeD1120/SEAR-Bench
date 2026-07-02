from __future__ import annotations

from .data import EquityFrame
from .evidence import FactorEvidence, extract_factor_evidence, split_time_evidence
from .factors import add_basic_factors, make_candidate_factor_sets
from .judge import LinearEvidenceJudge, evidence_to_features, fit_judge_from_rows, rule_based_judge


def summarize_factor(df, factor, horizon: int = 5) -> FactorEvidence:
    return extract_factor_evidence(df, factor, symbol="", factor_name="", horizon=horizon)


__all__ = [
    "EquityFrame",
    "FactorEvidence",
    "LinearEvidenceJudge",
    "add_basic_factors",
    "evidence_to_features",
    "extract_factor_evidence",
    "fit_judge_from_rows",
    "make_candidate_factor_sets",
    "rule_based_judge",
    "split_time_evidence",
    "summarize_factor",
]
