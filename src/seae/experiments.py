from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .data import load_zip_archive
from .evidence import FactorEvidence
from .evidence import split_time_evidence
from .factors import factor_series_map
from .judge import LinearEvidenceJudge, fit_judge_from_rows, rule_based_judge
from .synthetic import SyntheticConfig, generate_synthetic_benchmark


def _labels_for_synthetic_factor(family: str, factor_name: str) -> dict[str, object]:
    truth = {
        "momentum": {"label_keep": 1, "active_regime": "low_vol"},
        "return": {"label_keep": 1, "active_regime": "low_vol"},
        "mean_reversion": {"label_keep": 1, "active_regime": "low_vol"},
        "volume": {"label_keep": 1, "active_regime": "high_vol"},
        "volatility": {"label_keep": 0, "active_regime": "none"},
        "range": {"label_keep": 0, "active_regime": "none"},
        "gap": {"label_keep": 0, "active_regime": "none"},
        "intraday": {"label_keep": 0, "active_regime": "none"},
        "normalization": {"label_keep": 1, "active_regime": "low_vol"},
        "rank": {"label_keep": 0, "active_regime": "none"},
        "momentum_zscore": {"label_keep": 1, "active_regime": "low_vol"},
        "momentum_lag": {"label_keep": 1, "active_regime": "low_vol"},
        "return_zscore": {"label_keep": 1, "active_regime": "low_vol"},
        "return_lag": {"label_keep": 1, "active_regime": "low_vol"},
        "volume_zscore": {"label_keep": 1, "active_regime": "high_vol"},
        "volume_lag": {"label_keep": 1, "active_regime": "high_vol"},
        "volatility_zscore": {"label_keep": 0, "active_regime": "none"},
        "volatility_lag": {"label_keep": 0, "active_regime": "none"},
        "mean_reversion_zscore": {"label_keep": 1, "active_regime": "low_vol"},
        "mean_reversion_lag": {"label_keep": 1, "active_regime": "low_vol"},
        "range_zscore": {"label_keep": 0, "active_regime": "none"},
        "range_lag": {"label_keep": 0, "active_regime": "none"},
        "gap_zscore": {"label_keep": 0, "active_regime": "none"},
        "gap_lag": {"label_keep": 0, "active_regime": "none"},
        "intraday_zscore": {"label_keep": 0, "active_regime": "none"},
        "intraday_lag": {"label_keep": 0, "active_regime": "none"},
        "spread": {"label_keep": 1, "active_regime": "low_vol"},
        "interaction": {"label_keep": 1, "active_regime": "high_vol"},
        "volume_volatility": {"label_keep": 1, "active_regime": "high_vol"},
        "volume_volatility_zscore": {"label_keep": 1, "active_regime": "high_vol"},
        "volume_volatility_lag": {"label_keep": 1, "active_regime": "high_vol"},
        "volume_volatility_rank": {"label_keep": 1, "active_regime": "high_vol"},
        "volume_volatility_minmax": {"label_keep": 1, "active_regime": "high_vol"},
        "candlestick": {"label_keep": 0, "active_regime": "none"},
        "candlestick_zscore": {"label_keep": 0, "active_regime": "none"},
        "candlestick_lag": {"label_keep": 0, "active_regime": "none"},
        "candlestick_rank": {"label_keep": 0, "active_regime": "none"},
        "candlestick_minmax": {"label_keep": 0, "active_regime": "none"},
        "candlestick_spread": {"label_keep": 0, "active_regime": "none"},
        "price_level": {"label_keep": 0, "active_regime": "none"},
    }
    if family in truth:
        return truth[family]
    if "volume" in family:
        return {"label_keep": 1, "active_regime": "high_vol"}
    if "momentum" in family or "return" in family or "mean_reversion" in family:
        return {"label_keep": 1, "active_regime": "low_vol"}
    return {"label_keep": 0, "active_regime": "none"}


def _factor_library(df: pd.DataFrame, *, synthetic: bool) -> dict[str, tuple[str, pd.Series]]:
    factors = factor_series_map(df, bank="alpha360")
    if synthetic:
        factors["noise_factor"] = ("noise", df["noise_factor"])
    return factors


def _factor_rows(
    df: pd.DataFrame,
    *,
    symbol: str,
    synthetic: bool,
    horizon: int = 5,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    factors = _factor_library(df, synthetic=synthetic)
    for factor_name, value in factors.items():
        family, series = value
        ev = split_time_evidence(df, series, symbol=symbol, factor_name=factor_name, horizon=horizon)
        row = {
            "symbol": symbol,
            "factor_name": factor_name,
            "family": family,
            "evidence": ev,
            "train_ic": ev.train_ic,
            "test_ic": ev.test_ic,
            "train_strategy_mean_return": ev.strategy_mean_return,
            "train_strategy_sharpe": ev.strategy_sharpe,
            "train_strategy_cum_return": ev.strategy_cum_return,
            "train_strategy_max_drawdown": ev.strategy_max_drawdown,
            "test_strategy_mean_return": ev.test_strategy_mean_return,
            "test_strategy_sharpe": ev.test_strategy_sharpe,
            "test_strategy_cum_return": ev.test_strategy_cum_return,
            "test_strategy_max_drawdown": ev.test_strategy_max_drawdown,
        }
        if synthetic:
            row.update(_labels_for_synthetic_factor(family, factor_name))
        rows.append(row)
    return rows


def summarize_family_scores(table: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "mean_train_ic": ("train_ic", "mean"),
        "mean_test_ic": ("test_ic", "mean"),
        "mean_test_strategy_return": ("test_strategy_mean_return", "mean"),
        "mean_test_strategy_sharpe": ("test_strategy_sharpe", "mean"),
        "mean_test_strategy_cum_return": ("test_strategy_cum_return", "mean"),
        "mean_test_strategy_max_drawdown": ("test_strategy_max_drawdown", "mean"),
        "n_obs": ("factor_name", "count"),
    }
    if "label_keep" in table.columns:
        agg_dict["keep_rate"] = ("label_keep", "mean")
    elif "decision" in table.columns:
        keep_flag = (table["decision"] == "keep").astype(float)
        table = table.assign(_keep_flag=keep_flag)
        agg_dict["keep_rate"] = ("_keep_flag", "mean")
    agg = (
        table.groupby(["family", "factor_name"], as_index=False)
        .agg(**agg_dict)
        .sort_values(["family", "mean_test_ic"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return agg


def summarize_family_ablation(table: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "factor_rows": ("factor_name", "count"),
        "unique_factors": ("factor_name", "nunique"),
        "mean_train_ic": ("train_ic", "mean"),
        "mean_test_ic": ("test_ic", "mean"),
        "mean_test_strategy_return": ("test_strategy_mean_return", "mean"),
        "mean_test_strategy_sharpe": ("test_strategy_sharpe", "mean"),
        "mean_test_strategy_cum_return": ("test_strategy_cum_return", "mean"),
        "mean_test_strategy_max_drawdown": ("test_strategy_max_drawdown", "mean"),
    }
    if "label_keep" in table.columns:
        agg_dict["truth_keep_rate"] = ("label_keep", "mean")
    if "decision" in table.columns:
        table = table.assign(_keep_flag=(table["decision"] == "keep").astype(float))
        agg_dict["judge_keep_rate"] = ("_keep_flag", "mean")
    return (
        table.groupby("family", as_index=False)
        .agg(**agg_dict)
        .sort_values(["mean_test_strategy_sharpe", "mean_test_ic"], ascending=False)
        .reset_index(drop=True)
    )


def build_reasoning_view(table: pd.DataFrame, top_k: int = 5) -> dict[str, object]:
    family_agg = {
        "mean_train_ic": ("train_ic", "mean"),
        "mean_test_ic": ("test_ic", "mean"),
        "mean_test_strategy_return": ("test_strategy_mean_return", "mean"),
        "mean_test_strategy_sharpe": ("test_strategy_sharpe", "mean"),
        "mean_test_strategy_cum_return": ("test_strategy_cum_return", "mean"),
        "factor_count": ("factor_name", "count"),
    }
    if "label_keep" in table.columns:
        family_agg["keep_rate"] = ("label_keep", "mean")
    elif "decision" in table.columns:
        keep_flag = (table["decision"] == "keep").astype(float)
        table = table.assign(_keep_flag=keep_flag)
        family_agg["keep_rate"] = ("_keep_flag", "mean")

    family_summary = (
        table.groupby("family", as_index=False)
        .agg(**family_agg)
        .sort_values(["mean_test_strategy_sharpe", "mean_test_ic"], ascending=False)
        .head(top_k)
    )
    top_cols = [
        "symbol",
        "family",
        "factor_name",
        "train_ic",
        "test_ic",
        "train_strategy_mean_return",
        "train_strategy_sharpe",
        "train_strategy_cum_return",
        "test_strategy_mean_return",
        "test_strategy_sharpe",
        "test_strategy_cum_return",
        "test_strategy_max_drawdown",
    ]
    if "label_keep" in table.columns:
        top_cols.append("label_keep")
    if "decision" in table.columns:
        top_cols.append("decision")
    top_factors = table.sort_values(["test_strategy_sharpe", "test_ic"], ascending=False).loc[:, top_cols].head(top_k * 3).copy()
    top_factors.insert(0, "candidate_id", [f"C{i:03d}" for i in range(len(top_factors))])
    return {
        "decision_boundary": "LLM/agent should only read these structured fields; benchmark labels and returns are computed outside the LLM.",
        "family_summary": family_summary.to_dict(orient="records"),
        "top_factors": top_factors.to_dict(orient="records"),
    }


def _diversified_top_candidates(
    table: pd.DataFrame,
    *,
    candidate_count: int,
    min_train_n_obs: int = 252,
    max_per_symbol: int = 2,
    max_per_family: int | None = None,
) -> pd.DataFrame:
    """Rank by train evidence while preventing one symbol/family from dominating."""
    if table.empty:
        return table.copy()

    ranked = table.copy()
    ranked["_abs_train_ic"] = ranked["train_ic"].abs()
    ranked["_train_sharpe_rank"] = ranked["train_strategy_sharpe"].fillna(-999.0)
    ranked["_train_n_obs"] = ranked["evidence"].map(lambda ev: getattr(ev, "n_obs", 0))
    valid = ranked[
        ranked["train_ic"].notna()
        & ranked["train_strategy_sharpe"].notna()
        & (ranked["_train_n_obs"] >= min_train_n_obs)
    ].copy()
    if valid.empty:
        valid = ranked[
            ranked["train_ic"].notna()
            & ranked["train_strategy_sharpe"].notna()
            & (ranked["_train_n_obs"] > 0)
        ].copy()
    if valid.empty:
        valid = ranked
    ranked = valid
    ranked = ranked.sort_values(
        ["_train_sharpe_rank", "_abs_train_ic", "_train_n_obs"],
        ascending=False,
    )

    selected: list[int] = []
    symbol_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    effective_symbol_cap = max(
        max_per_symbol,
        int(np.ceil(candidate_count / max(1, ranked["symbol"].nunique()))),
    )
    family_cap = max_per_family or max(2, int(np.ceil(candidate_count / max(1, ranked["family"].nunique()))))

    for idx, row in ranked.iterrows():
        symbol = str(row["symbol"])
        family = str(row["family"])
        if symbol_counts.get(symbol, 0) >= effective_symbol_cap:
            continue
        if family_counts.get(family, 0) >= family_cap:
            continue
        selected.append(idx)
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= candidate_count:
            break

    if len(selected) < candidate_count:
        for idx, row in ranked.iterrows():
            if idx in selected:
                continue
            symbol = str(row["symbol"])
            if symbol_counts.get(symbol, 0) >= effective_symbol_cap:
                continue
            selected.append(idx)
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            if len(selected) >= candidate_count:
                break

    out = ranked.loc[selected].reset_index(drop=True)
    out.attrs["effective_symbol_cap"] = effective_symbol_cap
    out.attrs["candidate_count_requested"] = candidate_count
    out.attrs["candidate_count_after_filter"] = len(ranked)
    out.attrs["min_train_n_obs"] = min_train_n_obs
    return out


def _evidence_tags(row: pd.Series, ev: FactorEvidence) -> dict[str, str]:
    """Discrete train-only evidence tags to make LLM reasoning less numerically brittle."""
    sharpe = float(row["train_strategy_sharpe"]) if pd.notna(row["train_strategy_sharpe"]) else float("nan")
    cum_return = float(row["train_strategy_cum_return"]) if pd.notna(row["train_strategy_cum_return"]) else float("nan")
    drawdown = float(row["train_strategy_max_drawdown"]) if pd.notna(row["train_strategy_max_drawdown"]) else float("nan")
    train_ic = float(row["train_ic"]) if pd.notna(row["train_ic"]) else float("nan")

    if np.isfinite(sharpe) and np.isfinite(cum_return) and sharpe >= 2.0 and cum_return > 0:
        strategy_strength = "strong"
    elif np.isfinite(sharpe) and np.isfinite(cum_return) and sharpe >= 1.0 and cum_return > 0:
        strategy_strength = "moderate"
    else:
        strategy_strength = "weak"

    if np.isfinite(train_ic) and train_ic >= 0.02:
        ic_signal = "supportive"
    elif np.isfinite(train_ic) and train_ic <= -0.02:
        ic_signal = "conflicting"
    else:
        ic_signal = "neutral"

    if ev.n_obs >= 756:
        history_quality = "long"
    elif ev.n_obs >= 252:
        history_quality = "sufficient"
    else:
        history_quality = "short"

    if np.isfinite(drawdown) and drawdown <= -0.5:
        drawdown_risk = "high"
    elif np.isfinite(drawdown) and drawdown <= -0.25:
        drawdown_risk = "moderate"
    else:
        drawdown_risk = "acceptable"

    if np.isfinite(ev.regime_contrast) and ev.regime_contrast >= 0.15:
        regime_signal = "strong"
    elif np.isfinite(ev.regime_contrast) and ev.regime_contrast >= 0.05:
        regime_signal = "moderate"
    else:
        regime_signal = "weak"

    return {
        "strategy_strength": strategy_strength,
        "ic_signal": ic_signal,
        "history_quality": history_quality,
        "drawdown_risk": drawdown_risk,
        "regime_signal": regime_signal,
    }


def build_llm_reasoning_view(table: pd.DataFrame, top_k: int = 5, *, include_tags: bool = False) -> dict[str, object]:
    """Build a leakage-free view for LLM decisions.

    The LLM sees only train/in-sample structured evidence. Held-out test metrics
    are joined later by `evaluate_llm_decisions`.
    """
    work = table.copy()
    work["_abs_train_ic"] = work["train_ic"].abs()
    work["_train_n_obs"] = work["evidence"].map(lambda ev: getattr(ev, "n_obs", 0))
    family_summary = (
        work.groupby("family", as_index=False)
        .agg(
            mean_train_ic=("train_ic", "mean"),
            mean_abs_train_ic=("_abs_train_ic", "mean"),
            mean_train_n_obs=("_train_n_obs", "mean"),
            mean_train_strategy_return=("train_strategy_mean_return", "mean"),
            mean_train_strategy_sharpe=("train_strategy_sharpe", "mean"),
            mean_train_strategy_cum_return=("train_strategy_cum_return", "mean"),
            mean_train_strategy_max_drawdown=("train_strategy_max_drawdown", "mean"),
            factor_count=("factor_name", "count"),
        )
        .sort_values(["mean_train_strategy_sharpe", "mean_abs_train_ic"], ascending=False)
        .head(top_k)
    )

    top = _diversified_top_candidates(
        work,
        candidate_count=top_k * 3,
        min_train_n_obs=252,
        max_per_symbol=2,
    )
    rows: list[dict[str, object]] = []
    for idx, row in top.iterrows():
        ev = row["evidence"]
        candidate = {
            "candidate_id": f"C{idx:03d}",
            "symbol": row["symbol"],
            "family": row["family"],
            "factor_name": row["factor_name"],
            "train_ic": row["train_ic"],
            "train_ic_ir": ev.ic_ir,
            "train_win_rate": ev.win_rate,
            "train_stability": ev.stability,
            "train_n_obs": ev.n_obs,
            "train_regime_ic_high_vol": ev.regime_ic_high_vol,
            "train_regime_ic_low_vol": ev.regime_ic_low_vol,
            "train_regime_contrast": ev.regime_contrast,
            "train_strategy_mean_return": row["train_strategy_mean_return"],
            "train_strategy_sharpe": row["train_strategy_sharpe"],
            "train_strategy_cum_return": row["train_strategy_cum_return"],
            "train_strategy_max_drawdown": row["train_strategy_max_drawdown"],
        }
        if include_tags:
            candidate["evidence_tags"] = _evidence_tags(row, ev)
        rows.append(candidate)
    return {
        "decision_boundary": "LLM/agent may only read train/in-sample structured evidence. Held-out test metrics are hidden until benchmark evaluation.",
        "family_summary": family_summary.drop(columns=["mean_abs_train_ic"]).to_dict(orient="records"),
        "candidate_selection": {
            "rank_basis": "train_strategy_sharpe, abs(train_ic), train_n_obs",
            "candidate_count": len(rows),
            "candidate_count_requested": top.attrs.get("candidate_count_requested", top_k * 3),
            "candidate_count_after_train_filter": top.attrs.get("candidate_count_after_filter", len(top)),
            "min_train_n_obs": top.attrs.get("min_train_n_obs", 252),
            "base_max_per_symbol": 2,
            "effective_max_per_symbol": top.attrs.get("effective_symbol_cap", 2),
            "held_out_metrics_visible_to_llm": False,
            "include_evidence_tags": include_tags,
        },
        "top_factors": rows,
    }


def audit_llm_reasoning_view(view: dict[str, object]) -> list[str]:
    """Return forbidden leakage-like keys found in an LLM reasoning view."""
    forbidden_exact = {
        "active_regime",
        "decision",
        "label_keep",
        "label_regime",
        "pred_keep",
        "pred_regime",
        "rule_keep",
        "rule_regime",
        "score",
    }
    findings: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                next_path = f"{path}.{key}" if path else str(key)
                if key in forbidden_exact or key.startswith("test_"):
                    findings.append(next_path)
                visit(child, next_path)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                visit(child, f"{path}[{idx}]")

    visit(view, "")
    return findings


def build_synthetic_dataset(config: SyntheticConfig | None = None) -> pd.DataFrame:
    df, _ = generate_synthetic_benchmark(config or SyntheticConfig())
    return df


def run_synthetic_experiment(
    config: SyntheticConfig | None = None,
    *,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float], LinearEvidenceJudge]:
    df, truth = generate_synthetic_benchmark(config or SyntheticConfig())
    rows: list[dict[str, object]] = []
    for symbol, group in df.groupby("asset", sort=True):
        rows.extend(_factor_rows(group.reset_index(drop=True), symbol=symbol, synthetic=True))

    table = pd.DataFrame(rows)
    train_asset_cutoff = int(len(df["asset"].unique()) * 0.7)
    train = table[table["symbol"].str.extract(r"(\d+)").astype(int)[0] < train_asset_cutoff]
    test = table[~table.index.isin(train.index)]
    judge = fit_judge_from_rows(train)

    pred_rows = []
    for _, row in test.iterrows():
        ev = row["evidence"]
        pred = judge.predict(ev)
        rule_pred = rule_based_judge(ev)
        pred_rows.append(
            {
                "symbol": row["symbol"],
                "factor_name": row["factor_name"],
                "family": row["family"],
                "label_keep": int(row["label_keep"]),
                "label_regime": row["active_regime"],
                "pred_keep": 1 if pred["decision"] == "keep" else 0,
                "pred_regime": pred["active_regime"],
                "confidence": pred["confidence"],
                "rule_keep": 1 if rule_pred["decision"] == "keep" else 0,
                "rule_regime": rule_pred["active_regime"],
                "rule_confidence": rule_pred["confidence"],
                "test_ic": row["test_ic"],
                "test_strategy_mean_return": row["test_strategy_mean_return"],
                "test_strategy_sharpe": row["test_strategy_sharpe"],
                "test_strategy_cum_return": row["test_strategy_cum_return"],
                "test_strategy_max_drawdown": row["test_strategy_max_drawdown"],
            }
        )
    pred = pd.DataFrame(pred_rows)
    keep_acc = float((pred["label_keep"] == pred["pred_keep"]).mean())
    regime_acc = float((pred["label_regime"] == pred["pred_regime"]).mean())
    rule_keep_acc = float((pred["label_keep"] == pred["rule_keep"]).mean())
    rule_regime_acc = float((pred["label_regime"] == pred["rule_regime"]).mean())
    kept_test_ic = float(pred.loc[pred["pred_keep"] == 1, "test_ic"].mean())
    dropped_test_ic = float(pred.loc[pred["pred_keep"] == 0, "test_ic"].mean())
    rule_kept_test_ic = float(pred.loc[pred["rule_keep"] == 1, "test_ic"].mean())
    rule_dropped_test_ic = float(pred.loc[pred["rule_keep"] == 0, "test_ic"].mean())
    kept_strategy_sharpe = float(pred.loc[pred["pred_keep"] == 1, "test_strategy_sharpe"].mean())
    dropped_strategy_sharpe = float(pred.loc[pred["pred_keep"] == 0, "test_strategy_sharpe"].mean())
    rule_kept_strategy_sharpe = float(pred.loc[pred["rule_keep"] == 1, "test_strategy_sharpe"].mean())
    rule_dropped_strategy_sharpe = float(pred.loc[pred["rule_keep"] == 0, "test_strategy_sharpe"].mean())
    summary = {
        "keep_accuracy": keep_acc,
        "regime_accuracy": regime_acc,
        "rule_keep_accuracy": rule_keep_acc,
        "rule_regime_accuracy": rule_regime_acc,
        "mean_test_ic_kept": kept_test_ic,
        "mean_test_ic_dropped": dropped_test_ic,
        "rule_mean_test_ic_kept": rule_kept_test_ic,
        "rule_mean_test_ic_dropped": rule_dropped_test_ic,
        "mean_test_strategy_sharpe_kept": kept_strategy_sharpe,
        "mean_test_strategy_sharpe_dropped": dropped_strategy_sharpe,
        "rule_mean_test_strategy_sharpe_kept": rule_kept_strategy_sharpe,
        "rule_mean_test_strategy_sharpe_dropped": rule_dropped_strategy_sharpe,
        "n_test_samples": float(len(pred)),
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        table_with_evidence = table.assign(
            evidence_json=table["evidence"].map(lambda e: e.to_json()),
        ).drop(columns=["evidence"])
        table_with_evidence.to_csv(out / "synthetic_factor_table.csv", index=False)
        pred.to_csv(out / "synthetic_predictions.csv", index=False)
        (out / "synthetic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        comparison = pred[[
            "symbol",
            "factor_name",
            "family",
            "label_keep",
            "label_regime",
            "pred_keep",
            "pred_regime",
            "rule_keep",
            "rule_regime",
            "confidence",
            "rule_confidence",
            "test_ic",
            "test_strategy_mean_return",
            "test_strategy_sharpe",
            "test_strategy_cum_return",
            "test_strategy_max_drawdown",
        ]].copy()
        comparison.to_csv(out / "synthetic_comparison.csv", index=False)
        family_summary = summarize_family_scores(table)
        family_summary.to_csv(out / "synthetic_family_summary.csv", index=False)
        summarize_family_ablation(table).to_csv(out / "synthetic_family_ablation.csv", index=False)
        (out / "synthetic_reasoning_view.json").write_text(
            json.dumps(build_reasoning_view(table), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return pred, summary, judge


def run_real_market_experiment(
    zip_path: str | Path,
    *,
    limit: int | None = 10,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    equities = load_zip_archive(zip_path, limit=limit)
    rows: list[dict[str, object]] = []
    for eq in equities:
        factors = _factor_library(eq.frame, synthetic=False)
        for factor_name, value in factors.items():
            family, series = value
            ev = split_time_evidence(eq.frame, series, symbol=eq.symbol, factor_name=factor_name)
            pred = rule_based_judge(ev)
            rows.append(
                {
                    "symbol": eq.symbol,
                    "factor_name": factor_name,
                    "family": family,
                    "train_ic": ev.train_ic,
                    "test_ic": ev.test_ic,
                    "train_strategy_mean_return": ev.strategy_mean_return,
                    "train_strategy_sharpe": ev.strategy_sharpe,
                    "train_strategy_cum_return": ev.strategy_cum_return,
                    "train_strategy_max_drawdown": ev.strategy_max_drawdown,
                    "test_strategy_mean_return": ev.test_strategy_mean_return,
                    "test_strategy_sharpe": ev.test_strategy_sharpe,
                    "test_strategy_cum_return": ev.test_strategy_cum_return,
                    "test_strategy_max_drawdown": ev.test_strategy_max_drawdown,
                    "score": pred["confidence"],
                    "decision": pred["decision"],
                    "active_regime": pred["active_regime"],
                    "evidence": ev,
                }
            )
    table = pd.DataFrame(rows)
    summary = {
        "n_symbols": float(len(equities)),
        "n_factor_rows": float(len(table)),
        "avg_train_ic": float(table["train_ic"].mean()),
        "avg_test_ic": float(table["test_ic"].mean()),
        "avg_test_ic_kept": float(table.loc[table["decision"] == "keep", "test_ic"].mean()),
        "avg_test_strategy_return": float(table["test_strategy_mean_return"].mean()),
        "avg_test_strategy_sharpe": float(table["test_strategy_sharpe"].mean()),
        "avg_test_strategy_sharpe_kept": float(table.loc[table["decision"] == "keep", "test_strategy_sharpe"].mean()),
        "avg_test_strategy_sharpe_dropped": float(table.loc[table["decision"] == "drop", "test_strategy_sharpe"].mean()),
        "keep_rate": float((table["decision"] == "keep").mean()),
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.assign(evidence_json=table["evidence"].map(lambda e: e.to_json())).drop(columns=["evidence"]).to_csv(out / "real_market_factor_table.csv", index=False)
        summarize_family_scores(table).to_csv(out / "real_market_family_summary.csv", index=False)
        summarize_family_ablation(table).to_csv(out / "real_market_family_ablation.csv", index=False)
        (out / "real_market_diagnostic_reasoning_view.json").write_text(
            json.dumps(build_reasoning_view(table), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (out / "real_market_llm_reasoning_view.json").write_text(
            json.dumps(build_llm_reasoning_view(table), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (out / "real_market_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return table, summary, pd.DataFrame(equities)
