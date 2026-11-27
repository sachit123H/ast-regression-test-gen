"""Tests for Multi-SWE-bench exclusion logging and FDG relevant passing test discovery."""

import json
from pathlib import Path
from unittest.mock import patch

from src.dataset.multi_swe_bench import build_multi_swe_manifest
from src.evaluation.pipeline import extract_relevant_passing_test_fqns


def test_multi_swe_manifest_writes_exclusion_audit(tmp_path: Path):
    rows = [
        {
            "instance_id": "demo__1",
            "repo": "org/demo",
            "base_commit": "abc123",
            "problem_statement": "Fix crash when @Test assertEquals fails",
            "patch": "diff --git a/src/A.java b/src/A.java\n",
        },
        {
            "instance_id": "demo__2",
            "repo": "org/demo",
            "base_commit": "abc123",
            "problem_statement": "NullPointerException on empty input",
            "patch": "diff --git a/pom.xml b/pom.xml\n+++ b/pom.xml\n",
        },
        {
            "instance_id": "demo__3",
            "repo": "org/demo",
            "base_commit": "abc123",
            "problem_statement": "Handle negative offset cleanly",
            "target_rel_file": "src/main/java/Parser.java",
            "pre_code": "class Parser {\n  int f(int x) {\n    return x;\n  }\n}\n",
            "post_code": "class Parser {\n  int f(int x) {\n    return x < 0 ? 0 : x;\n  }\n}\n",
            "patch": "@@ -2,3 +2,3 @@\n-    return x;\n+    return x < 0 ? 0 : x;\n",
        },
    ]
    out_manifest = tmp_path / "mswe.json"
    out_log = tmp_path / "mswe_exclusions.json"
    items = build_multi_swe_manifest(
        rows=rows,
        repos_cache_dir=tmp_path / "cache",
        output_path=out_manifest,
        exclusion_log_path=out_log,
    )
    assert len(items) == 1
    assert items[0].instance_id == "demo__3"

    audit = json.loads(out_log.read_text(encoding="utf-8"))
    assert audit["total_scanned"] == 3
    assert audit["total_curated"] == 1
    assert audit["exclusion_counts"]["issue_leaked_brt"] == 1
    assert audit["exclusion_counts"]["non_single_java_file"] == 1


def test_extract_relevant_passing_test_fqns(tmp_path: Path):
    (tmp_path / "src/test/java/org/demo").mkdir(parents=True)
    (tmp_path / "src/test/java/org/demo/FooTest.java").write_text(
        "package org.demo;\n"
        "import org.junit.Test;\n"
        "public class FooTest {\n"
        "  @Test public void testTrig() {}\n"
        "  @Test public void testPassOne() {}\n"
        "  public void testPassLegacy() {}\n"
        "}\n",
        encoding="utf-8",
    )

    class _FakeProc:
        returncode = 0
        stdout = "org.demo.FooTest\n"

    with patch("src.evaluation.pipeline.subprocess.run", return_value=_FakeProc()):
        found = extract_relevant_passing_test_fqns(
            prefix_work_dir=tmp_path,
            dir_src_tests="src/test/java",
            triggering_tests=["org.demo.FooTest::testTrig"],
            max_tests=10,
        )
    assert found == [
        "org.demo.FooTest::testPassOne",
        "org.demo.FooTest::testPassLegacy",
    ]
