"""Unit tests for GumTree JSON/text normalization and positional metadata stripping."""

from src.ast_diff.gumtree_normalizer import (
    format_normalized_script,
    parse_gumtree_json,
    parse_gumtree_text,
    strip_positional_metadata,
)


def test_strip_positional_metadata():
    raw = "InfixExpression: > [1240,1255] @(1240,15) #42"
    assert strip_positional_metadata(raw) == "InfixExpression: >"


def test_parse_gumtree_json_and_format():
    payload = {
        "actions": [
            {
                "action": "update-node",
                "tree": "INFIX_EXPRESSION_OPERATOR: > [412,413]",
                "label": ">=",
            },
            {
                "action": "insert-tree",
                "tree": "IfStatement [500,560]\n  InfixExpression: == [504,515]\n    SimpleName: x [504,505]\n    NullLiteral [509,513]",
                "parent": "Block [480,620]",
                "at": 0,
            },
            {
                "action": "delete-node",
                "tree": "SimpleName: oldVar [700,706]",
                "parent": "MethodInvocation: foo [690,710]",
            },
        ]
    }
    actions = parse_gumtree_json(payload)
    assert len(actions) == 3
    assert actions[0].action == "UPDATE"
    assert actions[0].label == ">"
    assert actions[0].new_label == ">="
    assert actions[1].action == "INSERT"
    assert actions[1].parent_type == "Block"
    assert "[500,560]" not in actions[1].subtree_summary

    text = format_normalized_script("src/main/java/Foo.java", "createNumber", actions)
    assert "UPDATE INFIX_EXPRESSION_OPERATOR: `>` -> `>=`" in text
    assert "INSERT IfStatement into Block" in text
    assert "[412,413]" not in text


def test_parse_gumtree_text_fallback():
    raw_text = """===
match
SimpleName: foo [10,13]
===
update-node
INFIX_EXPRESSION_OPERATOR: < [20,21]
replace < by <=
===
insert-node
ReturnStatement [30,45]
to Block [15,60] at 2
"""
    actions = parse_gumtree_text(raw_text)
    assert len(actions) == 2
    assert actions[0].action == "UPDATE"
    assert actions[0].new_label == "<="
    assert actions[1].action == "INSERT"
    assert actions[1].parent_type == "Block"
