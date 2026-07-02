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

Decision protocol:
- Use only the provided structured evidence.
- A factor can be useful even if IC is negative, if the oriented strategy proxy is strong.
- Prefer factors with consistent train evidence, stronger test strategy Sharpe, acceptable drawdown, and meaningful family-level support.
- Be conservative when evidence conflicts.
- Do not invent data or mention raw price patterns.

Allowed regimes:
- high_vol
- low_vol
- none
- uncertain

Return JSON with this exact schema:
{{
  "model_role": "structured_evidence_reasoner",
  "global_assessment": "...",
  "decisions": [
    {{
      "symbol": "...",
      "factor_name": "...",
      "family": "...",
      "decision": "keep/drop",
      "active_regime": "high_vol/low_vol/none/uncertain",
      "confidence": 0.0,
      "rationale": "..."
    }}
  ]
}}

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
    if not device_map:
        try:
            inputs = inputs.to(hf_model.device)
        except AttributeError:
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
        rows.append(
            {
                "symbol": str(item.get("symbol", "")),
                "factor_name": str(item.get("factor_name", "")),
                "family": str(item.get("family", "")),
                "llm_decision": decision,
                "llm_active_regime": str(item.get("active_regime", "uncertain")),
                "llm_confidence": max(0.0, min(1.0, confidence)),
                "llm_rationale": str(item.get("rationale", "")),
            }
        )
    return pd.DataFrame(rows)


def evaluate_llm_decisions(table: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    if decisions.empty:
        return decisions, {
            "n_decisions": 0.0,
            "keep_rate": float("nan"),
            "mean_test_ic_kept": float("nan"),
            "mean_test_ic_dropped": float("nan"),
            "mean_test_strategy_sharpe_kept": float("nan"),
            "mean_test_strategy_sharpe_dropped": float("nan"),
        }
    metric_cols = [
        "symbol",
        "factor_name",
        "train_ic",
        "test_ic",
        "test_strategy_mean_return",
        "test_strategy_sharpe",
        "test_strategy_cum_return",
        "test_strategy_max_drawdown",
    ]
    optional_cols = [c for c in ["decision", "active_regime", "label_keep"] if c in table.columns]
    merged = decisions.merge(table[metric_cols + optional_cols], on=["symbol", "factor_name"], how="left")
    keep = merged["llm_decision"] == "keep"
    summary = {
        "n_decisions": float(len(merged)),
        "n_matched": float(merged["test_ic"].notna().sum()),
        "keep_rate": float(keep.mean()),
        "mean_test_ic_kept": float(merged.loc[keep, "test_ic"].mean()),
        "mean_test_ic_dropped": float(merged.loc[~keep, "test_ic"].mean()),
        "mean_test_strategy_sharpe_kept": float(merged.loc[keep, "test_strategy_sharpe"].mean()),
        "mean_test_strategy_sharpe_dropped": float(merged.loc[~keep, "test_strategy_sharpe"].mean()),
    }
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
