from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import PortfolioBacktestConfig, run_portfolio_backtest
from .experiments import (
    audit_llm_reasoning_view,
    build_llm_reasoning_view,
    build_reasoning_view,
    run_real_market_experiment,
    run_synthetic_experiment,
)
from .llm_judge import (
    build_revision_messages,
    build_llm_messages,
    call_huggingface_local_chat,
    call_openai_compatible_chat,
    config_from_env,
    critique_response_against_view,
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
    p_reason.add_argument("--candidate-count", type=int, default=None)
    p_reason.add_argument("--diagnostic-leaky", action="store_true")
    p_reason.add_argument("--include-evidence-tags", action="store_true")
    p_reason.add_argument("--factor-sample-size", type=int, default=6)
    p_reason.add_argument("--include-family", action="store_true")
    p_reason.add_argument("--include-family-summary", action="store_true")
    p_llm = sub.add_parser("llm")
    p_llm.add_argument("--zip-path", required=True)
    p_llm.add_argument("--limit", type=int, default=10)
    p_llm.add_argument("--top-k", type=int, default=5)
    p_llm.add_argument("--candidate-count", type=int, default=None)
    p_llm.add_argument("--backend", choices=["openai-compatible", "hf-local"], default="openai-compatible")
    p_llm.add_argument("--model", default="Qwen/Qwen3-8B")
    p_llm.add_argument("--base-url", default=None)
    p_llm.add_argument("--temperature", type=float, default=0.0)
    p_llm.add_argument("--timeout", type=int, default=120)
    p_llm.add_argument("--max-new-tokens", type=int, default=2048)
    p_llm.add_argument("--hf-device-map", default=None)
    p_llm.add_argument("--hf-torch-dtype", default="auto")
    p_llm.add_argument("--trust-remote-code", action="store_true")
    p_llm.add_argument("--prompt-out", default="outputs/llm_prompt.json")
    p_llm.add_argument("--response-out", default="outputs/llm_response.json")
    p_llm.add_argument("--decisions-out", default="outputs/llm_decisions.csv")
    p_llm.add_argument("--summary-out", default="outputs/llm_summary.json")
    p_llm.add_argument("--dry-run", action="store_true")
    p_llm.add_argument("--include-evidence-tags", action="store_true")
    p_llm.add_argument("--factor-sample-size", type=int, default=6)
    p_llm.add_argument("--include-family", action="store_true")
    p_llm.add_argument("--include-family-summary", action="store_true")
    p_llm.add_argument("--revision-rounds", type=int, default=0)
    p_llm.add_argument("--initial-response-out", default=None)
    p_llm.add_argument("--critic-out", default=None)
    p_score = sub.add_parser("score-response")
    p_score.add_argument("--zip-path", required=True)
    p_score.add_argument("--response-in", required=True)
    p_score.add_argument("--limit", type=int, default=10)
    p_score.add_argument("--top-k", type=int, default=5)
    p_score.add_argument("--candidate-count", type=int, default=None)
    p_score.add_argument("--include-evidence-tags", action="store_true")
    p_score.add_argument("--factor-sample-size", type=int, default=6)
    p_score.add_argument("--include-family", action="store_true")
    p_score.add_argument("--include-family-summary", action="store_true")
    p_score.add_argument("--decisions-out", default="outputs/llm_scored_decisions.csv")
    p_score.add_argument("--summary-out", default="outputs/llm_scored_summary.json")
    p_port = sub.add_parser("portfolio")
    p_port.add_argument("--zip-path", required=True)
    p_port.add_argument("--decisions-path", required=True)
    p_port.add_argument("--limit", type=int, default=10)
    p_port.add_argument("--decision-col", default="llm_decision")
    p_port.add_argument("--keep-value", default="keep")
    p_port.add_argument("--cost-bps", type=float, default=5.0)
    p_port.add_argument("--long-quantile", type=float, default=0.3)
    p_port.add_argument("--signal-window", type=int, default=20)
    p_port.add_argument("--factor-weighting", choices=["equal", "ic", "both"], default="both")
    p_port.add_argument("--factor-bank", default="alpha1000")
    p_port.add_argument("--output-dir", default="outputs/portfolio_backtest")
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
        view = (
            build_reasoning_view(table, top_k=args.top_k)
            if args.diagnostic_leaky
            else build_llm_reasoning_view(
                table,
                top_k=args.top_k,
                candidate_count=args.candidate_count,
                include_tags=args.include_evidence_tags,
                factor_sample_size=args.factor_sample_size,
                include_family=args.include_family,
                include_family_summary=args.include_family_summary,
            )
        )
        print(json.dumps(view, indent=2, sort_keys=True))
    elif args.cmd == "llm":
        table, _, _ = run_real_market_experiment(args.zip_path, limit=args.limit, output_dir=None)
        view = build_llm_reasoning_view(
            table,
            top_k=args.top_k,
            candidate_count=args.candidate_count,
            include_tags=args.include_evidence_tags,
            factor_sample_size=args.factor_sample_size,
            include_family=args.include_family,
            include_family_summary=args.include_family_summary,
        )
        leakage_findings = audit_llm_reasoning_view(view)
        if leakage_findings:
            raise RuntimeError(f"LLM reasoning view contains forbidden leakage keys: {leakage_findings}")
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
            max_new_tokens=args.max_new_tokens,
        )

        def call_model(current_messages: list[dict[str, str]]) -> str:
            if args.backend == "openai-compatible":
                return call_openai_compatible_chat(current_messages, config)
            return call_huggingface_local_chat(
                current_messages,
                model=args.model,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                device_map=args.hf_device_map,
                torch_dtype=args.hf_torch_dtype,
                trust_remote_code=args.trust_remote_code,
            )

        raw = call_model(messages)
        response_out = Path(args.response_out)
        response_out.parent.mkdir(parents=True, exist_ok=True)
        raw_out = response_out.with_suffix(response_out.suffix + ".raw.txt")
        try:
            response = parse_llm_json(raw)
            parse_success = True
        except json.JSONDecodeError as exc:
            response = {
                "model_role": "structured_evidence_reasoner",
                "global_assessment": "parse_failed",
                "parse_error": str(exc),
                "decisions": [],
            }
            parse_success = False
        if args.initial_response_out:
            initial_response_out = Path(args.initial_response_out)
            initial_response_out.parent.mkdir(parents=True, exist_ok=True)
            initial_response_out.write_text(json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

        critic_feedback = critique_response_against_view(view, response) if parse_success else {}
        revision_rounds_completed = 0
        for _ in range(max(0, args.revision_rounds)):
            critic_summary = critic_feedback.get("summary", {}) if isinstance(critic_feedback, dict) else {}
            if not parse_success or float(critic_summary.get("n_failed", 1.0)) <= 0.0:
                break
            revision_messages = build_revision_messages(view, response, critic_feedback)
            raw = call_model(revision_messages)
            try:
                response = parse_llm_json(raw)
                parse_success = True
            except json.JSONDecodeError as exc:
                response = {
                    "model_role": "structured_evidence_reasoner",
                    "global_assessment": "parse_failed",
                    "parse_error": str(exc),
                    "decisions": [],
                }
                parse_success = False
                break
            revision_rounds_completed += 1
            critic_feedback = critique_response_against_view(view, response)

        if args.critic_out:
            critic_out = Path(args.critic_out)
            critic_out.parent.mkdir(parents=True, exist_ok=True)
            critic_out.write_text(json.dumps(critic_feedback, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        raw_out.write_text(raw, encoding="utf-8")
        response_out.write_text(json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        decisions = normalize_llm_decisions(response)
        scored, summary = evaluate_llm_decisions(table, decisions, reasoning_view=view)
        summary["parse_success"] = float(parse_success)
        summary["backend"] = args.backend
        summary["model"] = args.model
        summary["limit"] = float(args.limit)
        summary["top_k"] = float(args.top_k)
        summary["candidate_count_requested"] = float(
            args.candidate_count if args.candidate_count is not None else args.top_k * 3
        )
        summary["candidate_count_actual"] = float(view.get("candidate_selection", {}).get("candidate_count", float("nan")))
        summary["include_evidence_tags"] = float(args.include_evidence_tags)
        summary["factor_sample_size"] = float(args.factor_sample_size)
        summary["include_family"] = float(args.include_family)
        summary["include_family_summary"] = float(args.include_family_summary)
        summary["revision_rounds_requested"] = float(args.revision_rounds)
        summary["revision_rounds_completed"] = float(revision_rounds_completed)
        if isinstance(critic_feedback, dict) and "summary" in critic_feedback:
            critic_summary = critic_feedback["summary"]
            summary["critic_pass_rate"] = float(critic_summary.get("pass_rate", float("nan")))
            summary["critic_n_failed"] = float(critic_summary.get("n_failed", float("nan")))
        decisions_out = Path(args.decisions_out)
        decisions_out.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(decisions_out, index=False)
        summary_out = Path(args.summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"prompt_written={prompt_out}")
        print(f"response_written={response_out}")
        print(f"raw_response_written={raw_out}")
        print(f"decisions_written={decisions_out}")
        print(f"summary_written={summary_out}")
        print(summary)
    elif args.cmd == "score-response":
        table, _, _ = run_real_market_experiment(args.zip_path, limit=args.limit, output_dir=None)
        view = build_llm_reasoning_view(
            table,
            top_k=args.top_k,
            candidate_count=args.candidate_count,
            include_tags=args.include_evidence_tags,
            factor_sample_size=args.factor_sample_size,
            include_family=args.include_family,
            include_family_summary=args.include_family_summary,
        )
        leakage_findings = audit_llm_reasoning_view(view)
        if leakage_findings:
            raise RuntimeError(f"LLM reasoning view contains forbidden leakage keys: {leakage_findings}")
        content = Path(args.response_in).read_text(encoding="utf-8")
        try:
            response = json.loads(content)
        except json.JSONDecodeError:
            response = parse_llm_json(content)
        decisions = normalize_llm_decisions(response)
        scored, summary = evaluate_llm_decisions(table, decisions, reasoning_view=view)
        summary["parse_success"] = 1.0
        summary["response_in"] = str(args.response_in)
        summary["limit"] = float(args.limit)
        summary["top_k"] = float(args.top_k)
        summary["candidate_count_requested"] = float(
            args.candidate_count if args.candidate_count is not None else args.top_k * 3
        )
        summary["candidate_count_actual"] = float(view.get("candidate_selection", {}).get("candidate_count", float("nan")))
        summary["include_evidence_tags"] = float(args.include_evidence_tags)
        summary["factor_sample_size"] = float(args.factor_sample_size)
        summary["include_family"] = float(args.include_family)
        summary["include_family_summary"] = float(args.include_family_summary)
        decisions_out = Path(args.decisions_out)
        decisions_out.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(decisions_out, index=False)
        summary_out = Path(args.summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"decisions_written={decisions_out}")
        print(f"summary_written={summary_out}")
        print(summary)
    elif args.cmd == "portfolio":
        config = PortfolioBacktestConfig(
            zip_path=args.zip_path,
            decisions_path=args.decisions_path,
            limit=args.limit,
            decision_col=args.decision_col,
            keep_value=args.keep_value,
            signal_window=args.signal_window,
            long_quantile=args.long_quantile,
            cost_bps=args.cost_bps,
            factor_weighting=args.factor_weighting,
            factor_bank=args.factor_bank,
            output_dir=args.output_dir,
        )
        _, summary = run_portfolio_backtest(config)
        print(summary)


if __name__ == "__main__":
    main()
