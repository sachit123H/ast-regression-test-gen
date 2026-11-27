"""Unit tests for HyRep-BRT (Hybrid Change Representation Engine)."""

from src.ast_diff.gumtree_normalizer import NormalizedASTAction
from src.ast_diff.hyrep import generate_hyrep_diff, parse_ast_action_summary_from_text


def test_hyrep_preserves_lexical_literals():
    raw_diff = """--- a/NumberUtils.java
+++ b/NumberUtils.java
@@ -464,3 +464,5 @@
-            return Long.parseLong(str);
+            if (str.startsWith("0x") || str.startsWith("-0x") || str.startsWith("0X") || str.startsWith("-0X")) {
+                return createInteger(str);
+            }
"""
    ast_actions = [
        NormalizedASTAction(
            action="UPDATE",
            node_type="IfStatement",
            label="startsWith",
            new_label="0X prefix checks",
            parent_type="MethodDeclaration",
        )
    ]

    hyrep_diff = generate_hyrep_diff(raw_diff, ast_actions)

    # 1. Exact string literals must be preserved verbatim (solves Lang-16 failure mode)
    assert '"0x"' in hyrep_diff
    assert '"-0x"' in hyrep_diff
    assert '"0X"' in hyrep_diff
    assert '"-0X"' in hyrep_diff
    assert "createInteger(str)" in hyrep_diff

    # 2. AST syntactic context is embedded
    assert "[AST: UPDATE IfStatement" in hyrep_diff
    assert "[AST-Node: IfStatement -> InfixExpression (OR)]" in hyrep_diff
    assert "// --- Structural AST Operation Trace (GumTree) ---" in hyrep_diff


def test_hyrep_with_text_script():
    raw_diff = """--- a/UTF8StreamJsonParser.java
+++ b/UTF8StreamJsonParser.java
@@ -204,3 +204,3 @@
-        if (ptr >= inputEnd) {
+        if (ptr >= inputEnd || ptr < 0) {
             return false;
"""
    ast_text = """File: src/UTF8StreamJsonParser.java
Enclosing Method: checkBounds
AST Edit Operations (1):
  1. UPDATE InfixExpression: `raw` -> `guarded`
"""
    hyrep_diff = generate_hyrep_diff(raw_diff, ast_text)

    # Verifies literal tokens preserved
    assert "ptr < 0" in hyrep_diff
    assert "inputEnd" in hyrep_diff

    # Verifies AST node annotation
    assert "[AST: UPDATE InfixExpression" in hyrep_diff
    assert "[AST-Node: IfStatement -> InfixExpression (OR)]" in hyrep_diff


def test_hyrep_filters_header_noise_and_aligns_multihunk():
    raw_diff = """--- a/Demo.java
+++ b/Demo.java
@@ -10,3 +10,4 @@
+        if (x > 100) {
+            throw new IllegalArgumentException();
@@ -40,2 +41,3 @@
+        return flag ? 1 : 0;
"""
    noisy_script = """File: src/Demo.java
Enclosing Method: check
AST Edit Operations (8):
  1. INSERT Modifier(public) into TypeDeclaration
  2. INSERT TYPE_DECLARATION_KIND(class) into TypeDeclaration
  3. INSERT SimpleName(Demo) into TypeDeclaration
  4. INSERT ThrowStatement into Block :: `throw new IllegalArgumentException()`
  5. INSERT ReturnStatement into Block :: `return flag ? 1 : 0;`
  6. DELETE Modifier(public)
  7. DELETE TYPE_DECLARATION_KIND(class)
  8. DELETE SimpleName(Demo)
"""
    hyrep = generate_hyrep_diff(raw_diff, noisy_script)
    assert "Modifier(public)" not in hyrep
    assert "TYPE_DECLARATION_KIND" not in hyrep
    assert "@@ -10,3 +10,4 @@ [AST: INSERT ThrowStatement" in hyrep
    assert "@@ -40,2 +41,3 @@ [AST: INSERT ReturnStatement" in hyrep
    assert "[AST-Node: ReturnStatement -> ConditionalExpression]" in hyrep
