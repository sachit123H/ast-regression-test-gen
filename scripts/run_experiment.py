"""Main CLI runner executing Control (`raw_diff`) and Treatment (`ast_edit`) trials."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.llm_client import LLMClient
from src.agent.loop import run_agentic_trial
from src.config import ExperimentConfig
from src.mcp.sandbox import PrefixSandbox


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Control (raw_diff) and Treatment (ast_edit) BRT generation trials."
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["raw_diff", "ast_edit"],
        choices=["raw_diff", "ast_edit", "hyrep"],
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of bugs (e.g., pilot run)")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip already completed trials")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    bugs = json.loads(cfg.paths.curated_manifest.read_text(encoding="utf-8"))
    if args.limit is not None:
        bugs = bugs[: args.limit]

    trials_dir = cfg.paths.results_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMClient(
        provider=cfg.agent.provider,
        model=cfg.agent.model,
        temperature=cfg.agent.temperature,
        max_tokens=cfg.agent.max_tokens,
        api_base_url=cfg.agent.api_base_url,
    )

    for idx, bug in enumerate(bugs, start=1):
        bug_key = bug["bug_key"]
        raw_pre_dir = cfg.paths.prefix_root / f"{bug_key}_raw"
        if not raw_pre_dir.exists():
            print(f"[SKIP] Missing pre-fix checkout for {bug_key}: {raw_pre_dir}")
            continue

        for cond in args.conditions:
            out_path = trials_dir / f"{bug_key}__{cond}.json"
            if args.resume and out_path.exists():
                print(f"[RESUME] Skipping existing {out_path.name}")
                continue

            agent_work = cfg.paths.prefix_root / f"{bug_key}_agent_{cond}"
            sandbox = PrefixSandbox.materialize_clean(
                raw_prefix_checkout=raw_pre_dir,
                agent_workspace_dir=agent_work,
                dir_src_tests=bug["dir_src_tests"],
                target_class_fqn=bug["modified_classes"][0],
                enclosing_method_start=int(bug["enclosing_method_start"]),
                enclosing_method_end=int(bug["enclosing_method_end"]),
                triggering_tests=bug["triggering_tests"],
                strip_triggers=cfg.dataset.strip_triggering_tests,
                d4j_bin=cfg.tools.defects4j_bin,
                compile_timeout=cfg.tools.compile_timeout_sec,
                test_timeout=cfg.tools.test_timeout_sec,
                coverage_timeout=cfg.tools.coverage_timeout_sec,
            )

            try:
                print(f"[{idx}/{len(bugs)}] Running {bug_key} ({cond})...")
                res = run_agentic_trial(
                    bug=bug,
                    condition=cond,
                    sandbox=sandbox,
                    llm=llm,
                    max_retries=cfg.agent.max_retries,
                    max_tool_calls_per_retry=cfg.agent.max_tool_calls_per_retry,
                    inline_enclosing_method=cfg.agent.inline_enclosing_method,
                )
                out_path.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
                print(
                    f"  -> {res.terminated_reason} | attempts={res.num_run_test_attempts} "
                    f"| fail_type={res.final_prefix_failure_type} | tokens={res.prompt_tokens + res.completion_tokens}"
                )
            except Exception as exc:
                print(f"  [ERROR] Trial failed for {bug_key} ({cond}): {exc}")
            finally:
                if agent_work.exists():
                    shutil.rmtree(agent_work, ignore_errors=True)


if __name__ == "__main__":
    main()
