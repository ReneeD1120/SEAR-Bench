from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .benchmark import rule_based_judge, summarize_factor
from .data import load_zip_archive
from .factors import add_basic_factors, future_return, make_candidate_factor_sets
from .synthetic import generate_synthetic_market


def cmd_summarize(zip_path: str, limit: int | None = None) -> None:
    equities = load_zip_archive(zip_path, limit=limit)
    print(f"loaded_symbols={len(equities)}")
    for eq in equities[:5]:
        df = add_basic_factors(eq.frame)
        factors = make_candidate_factor_sets(df)
        for name, series in factors.items():
            ev = summarize_factor(df, series, horizon=5)
            ev = ev.__class__(
                symbol=eq.symbol,
                factor_name=name,
                horizon=ev.horizon,
                ic=ev.ic,
                ic_ir=ev.ic_ir,
                win_rate=ev.win_rate,
                stability=ev.stability,
                n_obs=ev.n_obs,
                regime_summary=ev.regime_summary,
            )
            decision = rule_based_judge(ev)
            print(eq.symbol, name, ev.ic, ev.win_rate, decision["decision"], decision["active_regime"])


def cmd_synthetic() -> None:
    df = generate_synthetic_market()
    print(df.head().to_string())
    print(df.groupby("regime")["future_return"].mean().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(prog="seae")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--zip-path", required=True)
    p_sum.add_argument("--limit", type=int, default=5)
    sub.add_parser("synthetic")
    args = parser.parse_args()
    if args.cmd == "summarize":
        cmd_summarize(args.zip_path, limit=args.limit)
    elif args.cmd == "synthetic":
        cmd_synthetic()


if __name__ == "__main__":
    main()
