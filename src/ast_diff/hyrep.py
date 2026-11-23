"""HyRep-BRT: Hybrid Change Representation Engine.

Fuses unified git diffs (lexical literal preservation) with GumTree AST edit
scripts (structural syntax node semantics).

Addresses the primary failure mode of pure AST edit scripts (RQ1):
- Normalized AST edit scripts abstract away vital lexical literals (e.g., `"0x"`, `"0X"`,
  numeric comparison bounds, exception messages), depriving the LLM of the concrete
  values needed to synthesize non-trivial trigger inputs (leading to `PREFIX_PASS`).
- Raw unified diffs preserve verbatim lexical literals but lack method-boundary
  and syntax-tree context, leading to occasional scope confusion and `FAIL_FAIL`.
- HyRep-BRT embeds concise AST syntactic tags directly into unified diff hunks,
  providing the model with BOTH lexical literals for input synthesis AND explicit
  syntax node typing for semantic precision.
"""

from __future__ import annotations

import re
from typing import List, Optional, Union

from src.ast_diff.gumtree_normalizer import NormalizedASTAction


def parse_ast_action_summary_from_text(ast_edit_script: str) -> List[str]:
    """Extract concise action descriptors from normalized GumTree script text."""
    descriptors: List[str] = []
    for line in ast_edit_script.splitlines():
        line = line.strip()
        if not line:
            continue
        # Lines starting with numbered actions e.g. "1. UPDATE InfixExpression: ..."
        m = re.match(r"^\d+\.\s*(.+)$", line)
        if m:
            descriptors.append(m.group(1).strip())
        elif line.startswith(("INSERT", "DELETE", "UPDATE", "MOVE")):
            descriptors.append(line)
    return descriptors


def generate_hyrep_diff(
    raw_diff: str,
    ast_actions_or_script: Union[List[NormalizedASTAction], str],
) -> str:
    """Generate a HyRep-BRT hybrid representation combining git diff with AST annotations.

    Args:
        raw_diff: Verbatim unified `git diff` output.
        ast_actions_or_script: Either a list of `NormalizedASTAction` objects or
                               the normalized GumTree text script.

    Returns:
        A diff decorated with syntactic AST node classifications while preserving
        exact literal constants and code formatting.
    """
    if isinstance(ast_actions_or_script, str):
        descriptors = parse_ast_action_summary_from_text(ast_actions_or_script)
    else:
        descriptors = [a.format_line() for a in ast_actions_or_script]

    lines = raw_diff.splitlines()
    out_lines: List[str] = []

    hunk_index = 0
    desc_idx = 0
    num_descs = len(descriptors)

    for line in lines:
        if line.startswith("@@"):
            # Decorate hunk header with high-level AST operation if available
            annotation = ""
            if desc_idx < num_descs:
                main_desc = descriptors[desc_idx]
                # Shorten descriptor if long
                if len(main_desc) > 60:
                    main_desc = main_desc[:57] + "..."
                annotation = f" [AST: {main_desc}]"
                desc_idx += 1
            out_lines.append(f"{line}{annotation}")
            hunk_index += 1
            continue

        if line.startswith("+") and not line.startswith("+++"):
            # Check for specific syntax constructs in added line to annotate
            code = line[1:].strip()
            tag = ""
            if "throw new " in code:
                exc_m = re.search(r"throw new (\w+)", code)
                if exc_m:
                    tag = f"  // [AST-Node: ThrowStatement -> {exc_m.group(1)}]"
            elif "catch (" in code or "catch(" in code:
                catch_m = re.search(r"catch\s*\(([^)]+)\)", code)
                if catch_m:
                    tag = f"  // [AST-Node: CatchClause -> {catch_m.group(1).strip()}]"
            elif "if (" in code or "if(" in code:
                if "||" in code:
                    tag = "  // [AST-Node: IfStatement -> InfixExpression (OR)]"
                elif "&&" in code:
                    tag = "  // [AST-Node: IfStatement -> InfixExpression (AND)]"
                elif "<=" in code or ">=" in code or "<" in code or ">" in code:
                    tag = "  // [AST-Node: IfStatement -> RelationalExpression]"
                elif "==" in code or "!=" in code:
                    tag = "  // [AST-Node: IfStatement -> EqualityExpression]"
            elif "return " in code:
                if "?" in code and ":" in code:
                    tag = "  // [AST-Node: ReturnStatement -> ConditionalExpression]"
                else:
                    tag = "  // [AST-Node: ReturnStatement]"
            elif "? " in code and ":" in code:
                tag = "  // [AST-Node: ConditionalExpression (Ternary)]"

            out_lines.append(f"{line}{tag}")
            continue

        if line.startswith("-") and not line.startswith("---"):
            out_lines.append(line)
            continue

        out_lines.append(line)

    # Append remaining structural AST actions as a structured metadata summary footer
    if descriptors:
        out_lines.append("\n// --- Structural AST Operation Trace (GumTree) ---")
        for i, d in enumerate(descriptors, start=1):
            out_lines.append(f"// AST-{i:02d}: {d}")

    return "\n".join(out_lines)
