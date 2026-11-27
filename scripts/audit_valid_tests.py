"""Manual & programmatic quality audit of all generated tests labeled valid."""
from __future__ import annotations

import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def audit_all_trials() -> dict:
    eval_summary = json.loads(Path("data/results/evaluation_summary.json").read_text(encoding="utf-8"))
    curated = {b["bug_key"]: b for b in json.loads(Path("data/curated_bugs.json").read_text(encoding="utf-8"))}

    valid_reviews = []
    degenerate_flags = []

    for rec in eval_summary:
        bk = rec["bug_key"]
        bug = curated[bk]
        target_cls_simple = bug["modified_classes"][0].split(".")[-1]
        enclosing_method = bug["enclosing_method"]

        for cond in ("raw_diff", "ast_edit"):
            v = rec["verdicts"].get(cond, {})
            trial_path = Path(f"data/results/trials/{bk}__{cond}.json")
            trial = json.loads(trial_path.read_text(encoding="utf-8"))
            src = trial.get("final_java_source", "")
            last_att = trial["attempts"][-1] if trial.get("attempts") else {}

            # Check for method-copying / helper-reimplementation patterns
            has_copy_helper = bool(
                re.search(rf"\b{re.escape(enclosing_method)}ForTest\b", src)
                or (f" {enclosing_method}(" in src and "new " + target_cls_simple not in src and target_cls_simple + "." not in src)
            )
            calls_target = target_cls_simple in src

            if has_copy_helper:
                degenerate_flags.append({
                    "bug_key": bk,
                    "condition": cond,
                    "classification": v.get("classification"),
                    "reason": "Reimplements or defines helper with target method name in test class",
                })

            if v.get("is_valid_libro"):
                has_assert = bool(
                    re.search(r"\bassert\w*\s*\(|expected\s*=|\bfail\s*\(", src)
                )
                uses_reflection = "getDeclaredMethod" in src or "setAccessible" in src
                hits_enclosing = bool(last_att.get("enclosing_method_hit"))

                notes = []
                if not v.get("is_valid_strict_assertion"):
                    notes.append(
                        f"Exception-only validity ({last_att.get('exception_type')}); excluded under strict assertion criterion"
                    )
                if not hits_enclosing:
                    notes.append(
                        "enclosing_method_hit=False on recorded span (e.g., overloaded constructor in Math-35)"
                    )
                if not has_assert:
                    notes.append("No explicit JUnit assertion/fail/expected= token")
                if uses_reflection:
                    notes.append("Uses reflection to access non-public member")

                valid_reviews.append({
                    "bug_key": bk,
                    "condition": cond,
                    "classification": v["classification"],
                    "is_valid_strict_assertion": v["is_valid_strict_assertion"],
                    "calls_target_class": calls_target,
                    "enclosing_method_hit": hits_enclosing,
                    "has_assertion_construct": has_assert,
                    "uses_reflection": uses_reflection,
                    "prefix_exception": last_att.get("exception_type", ""),
                    "attempts_to_stop": trial.get("num_run_test_attempts", 0),
                    "notes": notes,
                    "manual_status": "VALID_CONFIRMED" if calls_target else "NEEDS_REVIEW",
                })

    strict_valid = [r for r in valid_reviews if r["is_valid_strict_assertion"]]
    exception_only = [r for r in valid_reviews if not r["is_valid_strict_assertion"]]
    out = {
        "total_libro_valid_tests": len(valid_reviews),
        "total_strict_assertion_valid": len(strict_valid),
        "total_exception_only_valid": len(exception_only),
        "all_valid_call_target_class": all(r["calls_target_class"] for r in valid_reviews),
        "valid_with_enclosing_hit_false": [
            {"bug_key": r["bug_key"], "condition": r["condition"], "notes": r["notes"]}
            for r in valid_reviews if not r["enclosing_method_hit"]
        ],
        "degenerate_test_flags_across_all_trials": degenerate_flags,
        "valid_tests": valid_reviews,
    }
    Path("data/results/valid_tests_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"Audit complete: {len(valid_reviews)} LIBRO-valid ({len(strict_valid)} strict, "
        f"{len(exception_only)} exception-only), all_call_target={out['all_valid_call_target_class']}, "
        f"degenerate_flags={len(degenerate_flags)}"
    )
    return out


if __name__ == "__main__":
    audit_all_trials()
