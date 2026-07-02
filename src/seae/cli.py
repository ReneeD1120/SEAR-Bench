from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments import build_reasoning_view, run_real_market_experiment, run_synthetic_experiment
from .llm_judge import (
    build_llm_messages,
    call_openai_compatible_chat,
    config_from_env,
    evaluate_llm_decisions,
    normalize_llm_decisions,
    parse_llm_json,
    write_prompt,
)
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
    p_reason = sub.add_parser("reason")
    p_reason.add_argument("--zip-path", required=True)
    p_reason.add_argument("--limit", type=int, default=10)
    p_reason.add_argument("--top-k", type=int, default=5)
    p_llm = sub.add_parser("llm")
    p_llm.add_argument("--zip-path", required=True)
    p_llm.add_argument("--limit", type=int, default=10)
    p_llm.add_argument("--top-k", type=int, default=5)
    p_llm.add_argument("--model", default="Qwen/Qwen3-8B")
    p_llm.add_argument("--base-url", default=None)
    p_llm.add_argument("--temperature", type=float, default=0.0)
    p_llm.add_argument("--timeout", type=int, default=120)
    p_llm.add_argument("--prompt-out", default="outputs/llm_prompt.json")
    p_llm.add_argument("--response-out", default="outputs/llm_response.json")
    p_llm.add_argument("--decisions-out", default="outputs/llm_decisions.csv")
    p_llm.add_argument("--dry-run", action="store_true")
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
            "test_strategy_sharpe",
            "test_strategy_cum_return",
        ]
        print(pred[cols].head().to_string(index=False))
        print(summary)
        print("judge_weights=", judge.weights.tolist(), "bias=", judge.bias)
    elif args.cmd == "real":
        table, summary, _ = run_real_market_experiment(args.zip_path, limit=args.limit, output_dir=args.output_dir)
        print(table[["symbol", "factor_name", "family", "train_ic", "test_ic", "test_strategy_sharpe", "score", "decision", "active_regime"]].head().to_string(index=False))
        print(summary)
    elif args.cmd == "reason":
        table, _, _ = run_real_market_experiment(args.zip_path, limit=args.limit, output_dir=None)
        view = build_reasoning_view(table, top_k=args.top_k)
        print(json.dumps(view, indent=2, sort_keys=True))
    elif args.cmd == "llm":
        table, _, _ = run_real_market_experiment(args.zip_path, limit=args.limit, output_dir=None)
        view = build_reasoning_view(table, top_k=args.top_k)
        messages = build_llm_messages(view)
        prompt_out = Path(args.prompt_out)
        prompt_out.parent.mkdir(parents=True, exist_ok=True)
        write_prompt(messages, prompt_out)
        if args.dry_run:
            print(f"prompt_written={prompt_out}")
            print(json.dumps(view, indent=2, sort_keys=True))
            return
        config = config_from_env(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        raw = call_openai_compatible_chat(messages, config)
        response = parse_llm_json(raw)
        response_out = Path(args.response_out)
        response_out.parent.mkdir(parents=True, exist_ok=True)
        response_out.write_text(json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        decisions = normalize_llm_decisions(response)
        scored, summary = evaluate_llm_decisions(table, decisions)
        decisions_out = Path(args.decisions_out)
        decisions_out.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(decisions_out, index=False)
        print(f"prompt_written={prompt_out}")
        print(f"response_written={response_out}")
        print(f"decisions_written={decisions_out}")
        print(summary)


if __name__ == "__main__":
    main()
