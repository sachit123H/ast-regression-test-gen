"""HyRep-BRT: Hybrid Change Representation Engine (Conceptual Prototype).

Fuses unified git diffs (lexical literal preservation) with GumTree AST edit
scripts (structural syntax node semantics).

Addresses the primary failure mode of lossy AST edit scripts (RQ1):
- Over-normalized AST edit scripts stripped lexical literals (e.g., `"0X"`,
  `Integer.MAX_VALUE`) and carried class-header noise, forcing the LLM to guess
  post-fix behavior (`FAIL_FAIL`).
- Raw unified diffs preserve verbatim lexical literals but lack structural node
  boundaries for multi-line refactorings (`Lang-17`).
- HyRep-BRT filters phantom `TypeDeclaration` header noise, aligns AST actions to
  matching diff hunks via token/construct correspondence, and annotates modified
  lines with structural node tags.

Note: HyRep-BRT is a prototype architecture motivating future evaluation; it has
not yet been benchmarked in end-to-end GPT-4o agentic trials.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple, Union

from src.ast_diff.gumtree_normalizer import (
    NormalizedASTAction,
    filter_phantom_header_noise,
)


_HEADER_NOISE_LINE_RE = re.compile(
    r"^(INSERT|DELETE)\s+(Modifier|TYPE_DECLARATION_KIND|SimpleName)\(([^)]*)\)(?:\s+into\s+TypeDeclaration)?$"
)


def _filter_noise_descriptors(descriptors: List[str]) -> List[str]:
    """Cancel balanced phantom TypeDeclaration header lines from serialized AST scripts."""
    ins_counts: Counter[Tuple[str, str]] = Counter()
    del_counts: Counter[Tuple[str, str]] = Counter()
    parsed: List[Tuple[str, str, str, bool]] = []

    for desc in descriptors:
        m = _HEADER_NOISE_LINE_RE.match(desc.strip())
        if m:
            kind, ntype, label = m.group(1), m.group(2), m.group(3)
            is_td_ins = kind == "INSERT" and "into TypeDeclaration" in desc
            if is_td_ins:
                ins_counts[(ntype, label)] += 1
                parsed.append((desc, "INS", f"{ntype}:{label}", True))
                continue
            if kind == "DELETE":
                del_counts[(ntype, label)] += 1
                parsed.append((desc, "DEL", f"{ntype}:{label}", True))
                continue
        parsed.append((desc, "", "", False))

    drop_ins: Counter[str] = Counter()
    drop_del: Counter[str] = Counter()
    for (ntype, label), cnt in ins_counts.items():
        paired = min(cnt, del_counts.get((ntype, label), 0))
        if paired > 0:
            key = f"{ntype}:{label}"
            drop_ins[key] = paired
            drop_del[key] = paired

    if not drop_ins:
        return descriptors

    out: List[str] = []
    for desc, tag, key, _ in parsed:
        if tag == "INS" and drop_ins[key] > 0:
            drop_ins[key] -= 1
            continue
        if tag == "DEL" and drop_del[key] > 0:
            drop_del[key] -= 1
            continue
        out.append(desc)
    return out


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
    return _filter_noise_descriptors(descriptors)


def _split_diff_hunks(lines: List[str]) -> List[Tuple[int, int]]:
    """Return `[start_idx, end_idx)` index spans in `lines` for each `@@` diff hunk."""
    starts = [i for i, ln in enumerate(lines) if ln.startswith("@@")]
    spans: List[Tuple[int, int]] = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        spans.append((s, e))
    return spans


def _match_descriptors_to_hunk(
    hunk_lines: List[str],
    descriptors: List[str],
    used_indices: set[int],
    single_hunk: bool,
) -> str:
    """Select the most relevant AST operation descriptor(s) for a diff hunk."""
    if not descriptors:
        return ""
    hunk_code = "\n".join(
        ln[1:] for ln in hunk_lines if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))
    )
    best_idx = -1
    best_score = -1

    for idx, desc in enumerate(descriptors):
        if idx in used_indices and not single_hunk:
            continue
        score = 0
        # Match quoted or parenthesized identifiers/literals in the descriptor against changed hunk lines
        for tok in re.findall(r"`([^`]+)`|\(([^)]+)\)", desc):
            val = tok[0] or tok[1]
            if val and val in hunk_code:
                score += 3
        # Match Java AST construct types against changed code patterns
        if "IfStatement" in desc and "if" in hunk_code:
            score += 1
        if "ThrowStatement" in desc and "throw" in hunk_code:
            score += 1
        if "CatchClause" in desc and "catch" in hunk_code:
            score += 1
        if "ReturnStatement" in desc and "return" in hunk_code:
            score += 1
        if "ForStatement" in desc and "for" in hunk_code:
            score += 1
        if "WhileStatement" in desc and "while" in hunk_code:
            score += 1
        if "MethodInvocation" in desc and "(" in hunk_code:
            score += 1
        if "VariableDeclarationStatement" in desc and ("=" in hunk_code or ";" in hunk_code):
            score += 1
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx == -1:
        for idx in range(len(descriptors)):
            if idx not in used_indices:
                best_idx = idx
                break
    if best_idx == -1:
        return ""

    used_indices.add(best_idx)
    main_desc = descriptors[best_idx]
    if len(main_desc) > 68:
        main_desc = main_desc[:65] + "..."
    return f" [AST: {main_desc}]"


def _classify_added_line(code: str) -> str:
    """Annotate an added Java diff line with its primary syntactic construct."""
    if "throw new " in code:
        exc_m = re.search(r"throw new (\w+)", code)
        if exc_m:
            return f"  // [AST-Node: ThrowStatement -> {exc_m.group(1)}]"
    if "catch (" in code or "catch(" in code:
        catch_m = re.search(r"catch\s*\(([^)]+)\)", code)
        if catch_m:
            return f"  // [AST-Node: CatchClause -> {catch_m.group(1).strip()}]"
    if re.match(r"^(?:else\s+)?if\s*\(", code):
        if "||" in code:
            return "  // [AST-Node: IfStatement -> InfixExpression (OR)]"
        if "&&" in code:
            return "  // [AST-Node: IfStatement -> InfixExpression (AND)]"
        if any(op in code for op in ("<=", ">=", "<", ">")):
            return "  // [AST-Node: IfStatement -> RelationalExpression]"
        if "==" in code or "!=" in code:
            return "  // [AST-Node: IfStatement -> EqualityExpression]"
        return "  // [AST-Node: IfStatement -> Condition]"
    if re.match(r"^(?:for|while)\s*\(", code):
        loop_kw = "ForStatement" if code.startswith("for") else "WhileStatement"
        return f"  // [AST-Node: {loop_kw}]"
    if code.startswith("return "):
        if "?" in code and ":" in code:
            return "  // [AST-Node: ReturnStatement -> ConditionalExpression]"
        return "  // [AST-Node: ReturnStatement]"
    if "? " in code and ":" in code:
        return "  // [AST-Node: ConditionalExpression (Ternary)]"
    return ""


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
        A diff decorated with hunk-aligned syntactic AST node classifications while
        preserving exact literal constants and code formatting.
    """
    if isinstance(ast_actions_or_script, str):
        descriptors = parse_ast_action_summary_from_text(ast_actions_or_script)
    else:
        clean_actions = filter_phantom_header_noise(ast_actions_or_script)
        descriptors = [a.format_line() for a in clean_actions]

    lines = raw_diff.splitlines()
    hunk_spans = _split_diff_hunks(lines)
    hunk_annotations: dict[int, str] = {}
    used_indices: set[int] = set()
    single_hunk = len(hunk_spans) <= 1

    for start_idx, end_idx in hunk_spans:
        hunk_annotations[start_idx] = _match_descriptors_to_hunk(
            lines[start_idx:end_idx], descriptors, used_indices, single_hunk
        )

    out_lines: List[str] = []
    for idx, line in enumerate(lines):
        if line.startswith("@@"):
            out_lines.append(f"{line}{hunk_annotations.get(idx, '')}")
            continue

        if line.startswith("+") and not line.startswith("+++"):
            tag = _classify_added_line(line[1:].strip())
            out_lines.append(f"{line}{tag}")
            continue

        out_lines.append(line)

    # Append remaining structural AST actions as a structured metadata summary footer
    if descriptors:
        out_lines.append("\n// --- Structural AST Operation Trace (GumTree) ---")
        for i, d in enumerate(descriptors, start=1):
            out_lines.append(f"// AST-{i:02d}: {d}")

    return "\n".join(out_lines)
