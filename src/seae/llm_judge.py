from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

import pandas as pd


SYSTEM_PROMPT = """You are the reasoning layer in SEAR-Bench.
You are not allowed to inspect raw prices, raw returns, hidden labels, or external market data.
You must only use the structured evidence JSON provided by the benchmark.
Your job is to make factor validity and regime-dependence decisions, then explain them.
Return strict JSON only."""


USER_PROMPT_TEMPLATE = """We are testing structured-evidence agentic reasoning for factor validity.

Agentic reasoning protocol:
- Use only the provided structured evidence.
- Each candidate includes factor_name, formula, train_factor_sample, and train evidence.
- First form a candidate-specific hypothesis from formula and factor_name.
- Use train_factor_sample only as in-sample factor behavior, not as future performance.
- A factor can be useful even if IC is negative, if the oriented strategy proxy is strong.
- Some ablation views may include evidence_tags. If present, treat them as derived train-only summaries, not labels or benchmark answers.
- Some ablation views may include family or family_summary. If present, treat them only as coarse metadata, not as a decision rule.
- Prefer factors with consistent train evidence, stronger train strategy Sharpe, acceptable drawdown, and sufficient train_n_obs.
- Explicitly check counter-evidence before deciding.
- Decide active_regime from train_regime_ic_high_vol, train_regime_ic_low_vol, and train_regime_contrast when evidence supports it.
- Use uncertain only when the regime evidence is weak or contradictory.
- Be conservative when evidence conflicts, but do not use a fixed keep/drop template.
- Do not invent data or mention raw price patterns.
- Do not compare against risk-free rates, transaction costs, sectors, or any baseline that is not explicitly present in the structured evidence.
- If a field is missing, say it is unavailable instead of inferring it.
- Make one decision for each provided top_factors candidate.
- The number of decisions must exactly equal candidate_selection.candidate_count.
- Copy candidate_id exactly from the provided top_factors list.
- Also copy symbol and factor_name exactly from the same candidate.
- If family is present in a candidate, copy it exactly; if family is absent, omit it from that decision.
- If you are unsure about a candidate, choose drop rather than inventing a new factor identity.
- Set global_assessment to exactly one of: mostly_positive, mixed, mostly_negative.
- Set confidence as your calibrated certainty in [0, 1]. It should vary across candidates when the evidence differs.
- Do not copy example values. Repeated identical confidence, regime, or rationale across all candidates is considered failed reasoning.
- Use concise natural language in evidence_audit. Each audit field must be at most 12 words.
- Do not output hidden chain-of-thought; output only the final audit summary.
- Minimize whitespace in the returned JSON.
- Return compact JSON only; do not include markdown fences, comments, or trailing text.

Allowed regimes:
- high_vol
- low_vol
- none
- uncertain

Return one JSON object with these top-level keys:
- model_role: exactly "structured_evidence_reasoner"
- reasoning_protocol: exactly "agentic_evidence_audit_v1"
- global_assessment: one of mostly_positive, mixed, mostly_negative
- decisions: list with exactly one object per candidate

Each decision object must contain:
- candidate_id
- symbol
- factor_name
- decision: keep or drop
- active_regime: high_vol, low_vol, none, or uncertain
- confidence: number in [0, 1]
- evidence_audit: object with exactly these string fields:
  - formula_hypothesis: what the formula is trying to capture
  - support_summary: strongest train-only evidence supporting usefulness
  - counter_evidence: strongest train-only evidence against usefulness
  - regime_summary: why the active_regime was selected
  - decision_logic: why keep/drop follows from the evidence balance

Structured evidence:
{evidence_json}
"""


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    api_key: str | None = None
    temperature: float = 0.0
    timeout: int = 120
    max_new_tokens: int = 2048


def _finite_json_value(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _finite_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_json_value(v) for v in value]
    return value


def build_llm_messages(reasoning_view: dict[str, object]) -> list[dict[str, str]]:
    clean_view = _finite_json_value(reasoning_view)
    evidence_json = json.dumps(clean_view, ensure_ascii=False, indent=2, sort_keys=True)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(evidence_json=evidence_json)},
    ]


def write_prompt(messages: list[dict[str, str]], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


def call_openai_compatible_chat(messages: list[dict[str, str]], config: LLMConfig) -> str:
    endpoint = config.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM endpoint returned HTTP {exc.code}: {detail}") from exc
    data = json.loads(raw)
    return str(data["choices"][0]["message"]["content"])


def call_huggingface_local_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
    device_map: str | None = None,
    torch_dtype: str = "auto",
    trust_remote_code: bool = False,
) -> str:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face local backend requires transformers. "
            "Install optional dependencies, for example: pip install transformers torch accelerate"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    model_kwargs: dict[str, object] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    hf_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages) + "\n\nASSISTANT:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        target_device = next(hf_model.parameters()).device
        inputs = inputs.to(target_device)
    except (AttributeError, StopIteration):
        pass
    generate_kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0.0:
        generate_kwargs["temperature"] = temperature
    output = hf_model.generate(**inputs, **generate_kwargs)
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return str(tokenizer.decode(generated, skip_special_tokens=True))


def parse_llm_json(content: str) -> dict[str, object]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_llm_decisions(response: dict[str, object]) -> pd.DataFrame:
    decisions = response.get("decisions", [])
    if not isinstance(decisions, list):
        raise ValueError("LLM response must contain a list field named 'decisions'")
    rows = []
    for item in decisions:
        if not isinstance(item, dict):
            continue
        decision = str(item.get("decision", "")).lower().strip()
        if decision not in {"keep", "drop"}:
            decision = "drop"
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        active_regime = str(item.get("active_regime", "uncertain")).lower().strip()
        if active_regime not in {"high_vol", "low_vol", "none", "uncertain"}:
            active_regime = "uncertain"
        evidence_audit = item.get("evidence_audit", {})
        if not isinstance(evidence_audit, dict):
            evidence_audit = {}
        audit_json = json.dumps(evidence_audit, ensure_ascii=False, sort_keys=True) if evidence_audit else ""
        rows.append(
            {
                "candidate_id": str(item.get("candidate_id", "")),
                "symbol": str(item.get("symbol", "")),
                "factor_name": str(item.get("factor_name", "")),
                "family": str(item.get("family", "")),
                "llm_decision": decision,
                "llm_active_regime": active_regime,
                "llm_confidence": max(0.0, min(1.0, confidence)),
                "llm_rationale": str(item.get("rationale", "")),
                "llm_formula_hypothesis": str(evidence_audit.get("formula_hypothesis", "")),
                "llm_support_summary": str(evidence_audit.get("support_summary", "")),
                "llm_counter_evidence": str(evidence_audit.get("counter_evidence", "")),
                "llm_regime_summary": str(evidence_audit.get("regime_summary", "")),
                "llm_decision_logic": str(evidence_audit.get("decision_logic", "")),
                "llm_evidence_audit": audit_json,
            }
        )
    return pd.DataFrame(rows)


def _nonempty_text_rate(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.fillna("").astype(str).str.strip().ne("").mean())


def _unique_text_rate(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    normalized = series.fillna("").astype(str).str.strip()
    nonempty = normalized[normalized.ne("")]
    if nonempty.empty:
        return 0.0
    return float(nonempty.nunique() / len(nonempty))


POSITIVE_EVIDENCE_TERMS = {
    "positive",
    "strong",
    "stable",
    "stability",
    "high",
    "higher",
    "useful",
    "utility",
    "support",
    "supports",
    "keep",
    "good",
}

NEGATIVE_EVIDENCE_TERMS = {
    "negative",
    "weak",
    "low",
    "lower",
    "drawdown",
    "risk",
    "conflict",
    "caution",
    "poor",
    "lack",
    "large",
    "severe",
    "drop",
}


def _count_terms(text: object, terms: set[str]) -> int:
    tokens = re.findall(r"[a-z_]+", str(text).lower())
    return sum(1 for token in tokens if token in terms)


def _expected_regime(row: pd.Series, *, min_contrast: float = 0.02) -> str:
    high = row.get("train_regime_ic_high_vol", float("nan"))
    low = row.get("train_regime_ic_low_vol", float("nan"))
    try:
        high_value = float(high)
        low_value = float(low)
    except (TypeError, ValueError):
        return "uncertain"
    if not math.isfinite(high_value) or not math.isfinite(low_value):
        return "uncertain"
    if abs(high_value - low_value) < min_contrast:
        return "uncertain"
    return "high_vol" if abs(high_value) > abs(low_value) else "low_vol"


def add_explanation_faithfulness(scored: pd.DataFrame) -> pd.DataFrame:
    """Attach heuristic faithfulness checks for structured LLM evidence audits."""
    if scored.empty:
        return scored
    required = {
        "llm_support_summary",
        "llm_counter_evidence",
        "llm_active_regime",
        "llm_decision",
        "llm_decision_logic",
    }
    if not required.issubset(scored.columns):
        return scored
    out = scored.copy()
    support_positive = out["llm_support_summary"].map(lambda text: _count_terms(text, POSITIVE_EVIDENCE_TERMS))
    support_negative = out["llm_support_summary"].map(lambda text: _count_terms(text, NEGATIVE_EVIDENCE_TERMS))
    counter_positive = out["llm_counter_evidence"].map(lambda text: _count_terms(text, POSITIVE_EVIDENCE_TERMS))
    counter_negative = out["llm_counter_evidence"].map(lambda text: _count_terms(text, NEGATIVE_EVIDENCE_TERMS))
    out["faith_support_polarity_ok"] = support_positive >= support_negative
    out["faith_counter_polarity_ok"] = counter_negative >= counter_positive
    out["faith_expected_regime"] = out.apply(_expected_regime, axis=1)
    out["faith_regime_ok"] = (out["faith_expected_regime"] == "uncertain") | (
        out["llm_active_regime"] == out["faith_expected_regime"]
    )
    out["faith_decision_conflict"] = (
        (out["llm_decision"] == "keep")
        & out["llm_decision_logic"].fillna("").astype(str).str.lower().str.contains("lack|not|caution|drop|negative")
    ) | (
        (out["llm_decision"] == "drop")
        & out["llm_decision_logic"].fillna("").astype(str).str.lower().str.contains("useful|utility|keep|strong")
    )
    out["faithfulness_ok"] = (
        out["faith_support_polarity_ok"]
        & out["faith_counter_polarity_ok"]
        & out["faith_regime_ok"]
        & ~out["faith_decision_conflict"]
    )
    return out


def candidate_frame_from_view(reasoning_view: dict[str, object]) -> pd.DataFrame:
    top_factors = reasoning_view.get("top_factors", [])
    if not isinstance(top_factors, list):
        return pd.DataFrame()
    return pd.DataFrame(top_factors)


def evaluate_llm_decisions(
    table: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    reasoning_view: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if decisions.empty:
        return decisions, {
            "n_decisions": 0.0,
            "n_matched": 0.0,
            "match_rate": float("nan"),
            "n_valid_test_strategy_sharpe": 0.0,
            "valid_test_strategy_sharpe_rate": float("nan"),
            "n_valid_test_strategy_cum_return": 0.0,
            "valid_test_strategy_cum_return_rate": float("nan"),
            "keep_rate": float("nan"),
            "mean_test_ic_kept": float("nan"),
            "mean_test_ic_dropped": float("nan"),
            "mean_test_strategy_sharpe_kept": float("nan"),
            "mean_test_strategy_sharpe_dropped": float("nan"),
        }
    table_metrics = table.copy()
    if "evidence" in table_metrics.columns:
        table_metrics = table_metrics.assign(
            train_ic_ir=table_metrics["evidence"].map(lambda ev: getattr(ev, "ic_ir", float("nan"))),
            train_win_rate=table_metrics["evidence"].map(lambda ev: getattr(ev, "win_rate", float("nan"))),
            train_stability=table_metrics["evidence"].map(lambda ev: getattr(ev, "stability", float("nan"))),
            train_n_obs=table_metrics["evidence"].map(lambda ev: getattr(ev, "n_obs", float("nan"))),
            train_regime_ic_high_vol=table_metrics["evidence"].map(
                lambda ev: getattr(ev, "regime_ic_high_vol", float("nan"))
            ),
            train_regime_ic_low_vol=table_metrics["evidence"].map(
                lambda ev: getattr(ev, "regime_ic_low_vol", float("nan"))
            ),
            train_regime_contrast=table_metrics["evidence"].map(
                lambda ev: getattr(ev, "regime_contrast", float("nan"))
            ),
        )
    metric_cols = [
        "symbol",
        "factor_name",
        "train_ic",
        "train_ic_ir",
        "train_win_rate",
        "train_stability",
        "train_n_obs",
        "train_regime_ic_high_vol",
        "train_regime_ic_low_vol",
        "train_regime_contrast",
        "train_strategy_mean_return",
        "train_strategy_sharpe",
        "train_strategy_cum_return",
        "train_strategy_max_drawdown",
        "test_ic",
        "test_strategy_mean_return",
        "test_strategy_sharpe",
        "test_strategy_cum_return",
        "test_strategy_max_drawdown",
    ]
    metric_cols = [c for c in metric_cols if c in table_metrics.columns]
    optional_cols = [c for c in ["decision", "active_regime", "label_keep"] if c in table_metrics.columns]
    candidate_frame = candidate_frame_from_view(reasoning_view or {})
    working = decisions.copy()
    if "candidate_id" in working.columns and not candidate_frame.empty and "candidate_id" in candidate_frame.columns:
        candidate_cols = [c for c in ["candidate_id", "symbol", "factor_name", "family"] if c in candidate_frame.columns]
        candidate_map = candidate_frame.loc[:, candidate_cols].drop_duplicates("candidate_id")
        working = working.drop(columns=[c for c in ["symbol", "factor_name", "family"] if c in working.columns])
        working = working.merge(candidate_map, on="candidate_id", how="left")
    merge_keys = ["symbol", "factor_name"]
    if "family" in working.columns:
        merge_keys.append("family")
    merged = working.merge(table_metrics[metric_cols + ["family"] + optional_cols], on=merge_keys, how="left")
    merged = add_explanation_faithfulness(merged)
    keep = merged["llm_decision"] == "keep"
    valid_strategy_sharpe = merged["test_strategy_sharpe"].notna()
    valid_cum_return = merged["test_strategy_cum_return"].notna()
    summary = {
        "n_decisions": float(len(merged)),
        "n_matched": float(merged["test_ic"].notna().sum()),
        "match_rate": float(merged["test_ic"].notna().mean()),
        "n_valid_test_strategy_sharpe": float(valid_strategy_sharpe.sum()),
        "valid_test_strategy_sharpe_rate": float(valid_strategy_sharpe.mean()),
        "n_valid_test_strategy_cum_return": float(valid_cum_return.sum()),
        "valid_test_strategy_cum_return_rate": float(valid_cum_return.mean()),
        "keep_rate": float(keep.mean()),
        "mean_test_ic_kept": float(merged.loc[keep, "test_ic"].mean()),
        "mean_test_ic_dropped": float(merged.loc[~keep, "test_ic"].mean()),
        "mean_test_strategy_sharpe_kept": float(merged.loc[keep, "test_strategy_sharpe"].mean()),
        "mean_test_strategy_sharpe_dropped": float(merged.loc[~keep, "test_strategy_sharpe"].mean()),
    }
    if "llm_confidence" in merged.columns:
        summary["confidence_unique_count"] = float(merged["llm_confidence"].nunique(dropna=True))
        summary["confidence_std"] = float(merged["llm_confidence"].std(ddof=0))
    if "llm_active_regime" in merged.columns:
        summary["non_uncertain_regime_rate"] = float((merged["llm_active_regime"] != "uncertain").mean())
        summary["regime_unique_count"] = float(merged["llm_active_regime"].nunique(dropna=True))
    if "llm_evidence_audit" in merged.columns:
        summary["evidence_audit_nonempty_rate"] = _nonempty_text_rate(merged["llm_evidence_audit"])
        summary["evidence_audit_unique_rate"] = _unique_text_rate(merged["llm_evidence_audit"])
    for column in [
        "faith_support_polarity_ok",
        "faith_counter_polarity_ok",
        "faith_regime_ok",
        "faith_decision_conflict",
        "faithfulness_ok",
    ]:
        if column in merged.columns:
            if column == "faith_decision_conflict":
                summary[f"{column}_rate"] = float(merged[column].mean())
            else:
                summary[f"{column}_rate"] = float(merged[column].mean())
    for column in [
        "llm_formula_hypothesis",
        "llm_support_summary",
        "llm_counter_evidence",
        "llm_regime_summary",
        "llm_decision_logic",
    ]:
        if column in merged.columns:
            summary[f"{column}_nonempty_rate"] = _nonempty_text_rate(merged[column])
            summary[f"{column}_unique_rate"] = _unique_text_rate(merged[column])
    if "decision" in merged.columns:
        summary["rule_agreement_rate"] = float((merged["llm_decision"] == merged["decision"]).mean())
    if "label_keep" in merged.columns:
        summary["label_keep_accuracy"] = float((keep.astype(int) == merged["label_keep"].astype(int)).mean())
    return merged, summary


def config_from_env(
    *,
    model: str,
    base_url: str | None = None,
    api_key_env: str = "QWEN_API_KEY",
    temperature: float = 0.0,
    timeout: int = 120,
    max_new_tokens: int = 2048,
) -> LLMConfig:
    resolved_base_url = base_url or os.environ.get("QWEN_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1"
    api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
    return LLMConfig(
        model=model,
        base_url=resolved_base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        max_new_tokens=max_new_tokens,
    )
