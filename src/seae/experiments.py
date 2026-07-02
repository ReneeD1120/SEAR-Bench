from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from .benchmark import make_candidate_factor_sets
from .data import load_zip_archive
from .evidence import split_time_evidence
from .factors import add_basic_factors
from .judge import LinearEvidenceJudge, fit_judge_from_rows, rule_based_judge
from .synthetic import SyntheticConfig, generate_synthetic_benchmark


def _labels_for_synthetic_factor(factor_name: str) -> dict[str, object]:
    low_vol_keep = {
        "momentum_20d",
        "momentum_60d",
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "ret_60d",
        "price_to_ma_20",
        "price_to_ma_60",
    }
    high_vol_keep = {
        "reversal_5d",
        "reversal_10d",
        "volume_surge",
        "volume_ma_ratio_20",
        "volume_ma_ratio_60",
    }
    neutral_drop = {
        "range_pct",
        "hl_spread",
        "close_to_open",
        "gap_return",
        "vol_20d",
        "vol_60d",
        "intraday_reversal",
        "noise_factor",
    }
    if factor_name in low_vol_keep:
        return {"label_keep": 1, "active_regime": "low_vol"}
    if factor_name in high_vol_keep:
        return {"label_keep": 1, "active_regime": "high_vol"}
    if factor_name in neutral_drop:
        return {"label_keep": 0, "active_regime": "none"}
    return {"label_keep": 0, "active_regime": "none"}


def _factor_library(df: pd.DataFrame, *, synthetic: bool) -> dict[str, pd.Series]:
    df = add_basic_factors(df)
    factors = make_candidate_factor_sets(df)
    if synthetic:
        factors["noise_factor"] = df["noise_factor"]
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
    for factor_name, series in factors.items():
        ev = split_time_evidence(df, series, symbol=symbol, factor_name=factor_name, horizon=horizon)
        row = {
            "symbol": symbol,
            "factor_name": factor_name,
            "evidence": ev,
            "train_ic": ev.train_ic,
            "test_ic": ev.test_ic,
        }
        if synthetic:
            row.update(_labels_for_synthetic_factor(factor_name))
        rows.append(row)
    return rows


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
                "label_keep": int(row["label_keep"]),
                "label_regime": row["active_regime"],
                "pred_keep": 1 if pred["decision"] == "keep" else 0,
                "pred_regime": pred["active_regime"],
                "confidence": pred["confidence"],
                "rule_keep": 1 if rule_pred["decision"] == "keep" else 0,
                "rule_regime": rule_pred["active_regime"],
                "rule_confidence": rule_pred["confidence"],
                "test_ic": row["test_ic"],
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
    summary = {
        "keep_accuracy": keep_acc,
        "regime_accuracy": regime_acc,
        "rule_keep_accuracy": rule_keep_acc,
        "rule_regime_accuracy": rule_regime_acc,
        "mean_test_ic_kept": kept_test_ic,
        "mean_test_ic_dropped": dropped_test_ic,
        "rule_mean_test_ic_kept": rule_kept_test_ic,
        "rule_mean_test_ic_dropped": rule_dropped_test_ic,
        "n_test_samples": float(len(pred)),
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.assign(
            evidence_json=table["evidence"].map(lambda e: e.to_json()),
        ).drop(columns=["evidence"]).to_csv(out / "synthetic_factor_table.csv", index=False)
        pred.to_csv(out / "synthetic_predictions.csv", index=False)
        (out / "synthetic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        comparison = pred[[
            "symbol",
            "factor_name",
            "label_keep",
            "label_regime",
            "pred_keep",
            "pred_regime",
            "rule_keep",
            "rule_regime",
            "confidence",
            "rule_confidence",
            "test_ic",
        ]].copy()
        comparison.to_csv(out / "synthetic_comparison.csv", index=False)
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
        for factor_name, series in factors.items():
            ev = split_time_evidence(eq.frame, series, symbol=eq.symbol, factor_name=factor_name)
            pred = rule_based_judge(ev)
            rows.append(
                {
                    "symbol": eq.symbol,
                    "factor_name": factor_name,
                    "train_ic": ev.train_ic,
                    "test_ic": ev.test_ic,
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
        "keep_rate": float((table["decision"] == "keep").mean()),
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.assign(evidence_json=table["evidence"].map(lambda e: e.to_json())).drop(columns=["evidence"]).to_csv(out / "real_market_factor_table.csv", index=False)
        (out / "real_market_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return table, summary, pd.DataFrame(equities)
