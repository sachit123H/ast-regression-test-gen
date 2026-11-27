"""Main CLI runner executing Control (`raw_diff`) and Treatment (`ast_edit`) trials."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shutil
import subprocess
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


def _current_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


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
    parser.add_argument("--repeats", type=int, default=None, help="Number of repeat trials per (bug, condition)")
    parser.add_argument("--no-resume", action="store_true", help="Re-run even if trial JSON already exists")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    repeats = args.repeats if args.repeats is not None else max(1, int(cfg.agent.repeats))
    bugs = json.loads(cfg.paths.curated_manifest.read_text(encoding="utf-8"))
    if args.limit is not None:
        bugs = bugs[: args.limit]

    trials_dir = cfg.paths.results_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _current_git_commit(),
        "config_path": str(args.config),
        "conditions": list(args.conditions),
        "repeats": repeats,
        "num_bugs": len(bugs),
        "agent": {
            "provider": cfg.agent.provider,
            "model": cfg.agent.model,
            "temperature": cfg.agent.temperature,
            "max_retries": cfg.agent.max_retries,
            "inline_enclosing_method": cfg.agent.inline_enclosing_method,
        },
        "dataset": {
            "filter_ast_header_noise": cfg.dataset.filter_ast_header_noise,
            "strip_triggering_tests": cfg.dataset.strip_triggering_tests,
        },
    }
    (cfg.paths.results_dir / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )

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
            for rep in range(1, repeats + 1):
                suffix = f"__r{rep}" if repeats > 1 else ""
                out_path = trials_dir / f"{bug_key}__{cond}{suffix}.json"
                if (not args.no_resume) and out_path.exists():
                    print(f"[RESUME] Skipping existing {out_path.name}")
                    continue

                agent_work = cfg.paths.prefix_root / f"{bug_key}_agent_{cond}{suffix}"
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
                    print(f"[{idx}/{len(bugs)}] Running {bug_key} ({cond}, repeat {rep}/{repeats})...")
                    res = run_agentic_trial(
                        bug=bug,
                        condition=cond,
                        sandbox=sandbox,
                        llm=llm,
                        max_retries=cfg.agent.max_retries,
                        max_tool_calls_per_retry=cfg.agent.max_tool_calls_per_retry,
                        inline_enclosing_method=cfg.agent.inline_enclosing_method,
                    )
                    payload = res.to_dict()
                    payload["repeat_index"] = rep
                    payload["config_snapshot"] = run_meta["agent"]
                    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    print(
                        f"  -> {res.terminated_reason} | attempts={res.num_run_test_attempts} "
                        f"| fail_type={res.final_prefix_failure_type} | tokens={res.prompt_tokens + res.completion_tokens}"
                    )
                except Exception as exc:
                    print(f"  [ERROR] Trial failed for {bug_key} ({cond} r{rep}): {exc}")
                finally:
                    if agent_work.exists():
                        shutil.rmtree(agent_work, ignore_errors=True)


if __name__ == "__main__":
    main()
