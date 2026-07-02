from __future__ import annotations

import argparse

from .experiments import run_real_market_experiment, run_synthetic_experiment
from .synthetic import SyntheticConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog="sear")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_syn = sub.add_parser("synthetic")
    p_syn.add_argument("--output-dir", default="outputs")
    p_syn.add_argument("--n-assets", type=int, default=120)
    p_syn.add_argument("--n-days", type=int, default=720)
    p_syn.add_argument("--seed", type=int, default=7)
    p_real = sub.add_parser("real")
    p_real.add_argument("--zip-path", required=True)
    p_real.add_argument("--limit", type=int, default=10)
    p_real.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    if args.cmd == "synthetic":
        config = SyntheticConfig(n_assets=args.n_assets, n_days=args.n_days, seed=args.seed)
        pred, summary, judge = run_synthetic_experiment(config, output_dir=args.output_dir)
        cols = [
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
        ]
        print(pred[cols].head().to_string(index=False))
        print(summary)
        print("judge_weights=", judge.weights.tolist(), "bias=", judge.bias)
    elif args.cmd == "real":
        table, summary, _ = run_real_market_experiment(args.zip_path, limit=args.limit, output_dir=args.output_dir)
        print(table[["symbol", "factor_name", "train_ic", "test_ic", "score", "decision", "active_regime"]].head().to_string(index=False))
        print(summary)


if __name__ == "__main__":
    main()
