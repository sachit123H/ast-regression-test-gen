"""Unit tests for Cobertura XML parsing and JUnit failure classification."""

from pathlib import Path

from src.evaluation.cobertura_parser import build_coverage_matrix, parse_cobertura_xml
from src.evaluation.pipeline import _find_faulty_methods
from src.mcp.sandbox import classify_junit_failure, parse_enclosing_method_coverage


SAMPLE_COBERTURA_XML = """<?xml version="1.0"?>
<coverage>
  <packages>
    <package name="org.demo">
      <classes>
        <class name="org.demo.MathUtil" filename="org/demo/MathUtil.java">
          <methods>
            <method name="clamp" signature="(III)I">
              <lines>
                <line number="10" hits="1"/>
                <line number="11" hits="0"/>
                <line number="12" hits="1"/>
              </lines>
            </method>
          </methods>
          <lines>
            <line number="10" hits="1"/>
            <line number="11" hits="0"/>
            <line number="12" hits="1"/>
          </lines>
        </class>
        <class name="org.demo.Irrelevant" filename="org/demo/Irrelevant.java">
          <lines>
            <line number="5" hits="1"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""


def test_parse_cobertura_filters_relevant_classes(tmp_path: Path):
    xml_path = tmp_path / "coverage.xml"
    xml_path.write_text(SAMPLE_COBERTURA_XML, encoding="utf-8")

    cov = parse_cobertura_xml(xml_path, relevant_classes=["org.demo.MathUtil"])
    assert "org.demo.MathUtil:10" in cov.covered_lines
    assert "org.demo.MathUtil:11" not in cov.covered_lines
    assert "org.demo.Irrelevant:5" not in cov.all_lines
    assert cov.line_to_method["org.demo.MathUtil:10"] == "org.demo.MathUtil#clamp(III)I"

    elements, mat, _ = build_coverage_matrix([cov])
    assert elements == ["org.demo.MathUtil:10", "org.demo.MathUtil:11", "org.demo.MathUtil:12"]
    assert mat == [[1, 0, 1]]

    covered, missed = parse_enclosing_method_coverage(xml_path, "org.demo.MathUtil", 10, 12)
    assert covered == [10, 12]
    assert missed == [11]


def test_classify_junit_failure():
    assert_log = "--- org.demo.FooTest::testBug\njunit.framework.AssertionFailedError: expected:<5> but was:<4>\n\tat org.demo.FooTest.testBug(FooTest.java:15)"
    ftype, etype, msg, _ = classify_junit_failure(assert_log, "FooTest")
    assert ftype == "ASSERTION_FAILURE"
    assert "AssertionFailedError" in etype
    assert "expected:<5>" in msg

    npe_log = "--- org.demo.FooTest::testBug\njava.lang.NullPointerException: null input\n\tat org.demo.Foo.bar(Foo.java:42)"
    ftype2, etype2, _, _ = classify_junit_failure(npe_log, "FooTest")
    assert ftype2 == "RUNTIME_ERROR"
    assert "NullPointerException" in etype2


def test_classify_junit_failure_targets_test_class_among_multiple():
    multi_log = """--- org.demo.OtherTest::testOther
junit.framework.AssertionFailedError: other failure
\tat org.demo.OtherTest.testOther(OtherTest.java:10)
--- org.demo.MyBRT_Test::testReproduceBug
java.lang.IllegalArgumentException: bad value
\tat org.demo.MyBRT_Test.testReproduceBug(MyBRT_Test.java:20)
"""
    ftype, etype, msg, _ = classify_junit_failure(multi_log, "MyBRT_Test")
    assert ftype == "RUNTIME_ERROR"
    assert "IllegalArgumentException" in etype
    assert "bad value" in msg


def test_find_faulty_methods_constructor_and_fallback():
    line_to_method = {
        "org.demo.Fraction:10": "org.demo.Fraction#<init>(D)V",
        "org.demo.Fraction:20": "org.demo.Fraction#add(Lorg/demo/Fraction;)Lorg/demo/Fraction;",
    }
    # Constructor matching (Fraction -> <init>)
    matched = _find_faulty_methods("org.demo.Fraction", "Fraction", line_to_method, 10, 15)
    assert "org.demo.Fraction#<init>(D)V" in matched

    # Method matching (add -> add)
    matched2 = _find_faulty_methods("org.demo.Fraction", "add", line_to_method, 20, 25)
    assert "org.demo.Fraction#add(Lorg/demo/Fraction;)Lorg/demo/Fraction;" in matched2

    # Line-range fallback matching
    matched3 = _find_faulty_methods("org.demo.Fraction", "unknownName", line_to_method, 10, 12)
    assert "org.demo.Fraction#<init>(D)V" in matched3
