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


def test_filter_phantom_header_noise_preserves_genuine_edits():
    payload = {
        "actions": [
            {"action": "insert-node", "tree": "Modifier: public [0,6]", "parent": "TypeDeclaration [0,100]"},
            {"action": "insert-node", "tree": "TYPE_DECLARATION_KIND: class [7,12]", "parent": "TypeDeclaration [0,100]"},
            {"action": "insert-node", "tree": "SimpleName: NumberUtils [13,24]", "parent": "TypeDeclaration [0,100]"},
            {"action": "insert-tree", "tree": "MethodInvocation [50,70]", "parent": "InfixExpression [30,70]"},
            {"action": "delete-node", "tree": "SimpleName: oldHelper [40,49]"},
            {"action": "delete-node", "tree": "Modifier: public [0,6]"},
            {"action": "delete-node", "tree": "TYPE_DECLARATION_KIND: class [7,12]"},
            {"action": "delete-node", "tree": "SimpleName: NumberUtils [13,24]"},
        ]
    }
    dst_src = "0" * 50 + 'str.startsWith("0X")' + "0" * 40
    pre_src = "0" * 40 + "oldHelper" + "0" * 40
    actions = parse_gumtree_json(payload, src_text=pre_src, dst_text=dst_src)
    # Phantom TypeDeclaration header churn must be dropped; real method edit & method-body delete kept
    assert len(actions) == 2
    assert actions[0].action == "INSERT"
    assert actions[0].node_type == "MethodInvocation"
    assert '"0X"' in actions[0].subtree_summary
    assert actions[1].action == "DELETE"
    assert actions[1].label == "oldHelper"


def test_parse_gumtree_v3_text_with_dash_separators():
    raw_text = """===
insert-tree
---
IfStatement [100,160]
    InfixExpression: >= [104,135]
        SimpleName: lRow [104,108]
        SimpleName: Integer.MAX_VALUE [120,135]
---
to
Block [80,200]
at 1
"""
    actions = parse_gumtree_text(raw_text)
    assert len(actions) == 1
    assert actions[0].action == "INSERT"
    assert actions[0].parent_type == "Block"
    assert "Integer.MAX_VALUE" in actions[0].subtree_summary


def test_move_action_preserves_subtree_summary_and_method_header_noise_cancelled():
    payload = {
        "actions": [
            {"action": "insert-node", "tree": "Modifier: private [0,7]", "parent": "MethodDeclaration [0,80]"},
            {"action": "insert-node", "tree": "SimpleName: squareRoot [15,25]", "parent": "MethodDeclaration [0,80]"},
            {"action": "move-tree", "tree": "InfixExpression [30,44]", "parent": "InfixExpression [28,60]"},
            {"action": "delete-node", "tree": "Modifier: private [0,7]"},
            {"action": "delete-node", "tree": "SimpleName: squareRoot [15,25]"},
        ]
    }
    src = "0" * 30 + "hexDigits > 16" + "0" * 50
    actions = parse_gumtree_json(payload, src_text=src, dst_text=src)
    assert len(actions) == 1
    assert actions[0].action == "MOVE"
    line = actions[0].format_line()
    assert "`hexDigits > 16`" in line
