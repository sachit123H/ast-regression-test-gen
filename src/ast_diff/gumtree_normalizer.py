"""GumTree AST edit script extraction, normalization, and raw git diff baseline.

Implements Proposal Section 4.4:
- Runs GumTree to generate AST edit scripts (Insert, Delete, Update, Move).
- Normalizes output by stripping internal node IDs and positional metadata
  (character offsets/lengths like `[123,456]`), producing concise, deterministic
  structural summaries for the Treatment condition.
- Produces minimal unified `git diff` output for the Control condition so both
  conditions differ in structure rather than content.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Regexes for stripping GumTree positional metadata e.g. [1240,1265] or @(1240,25)
_POS_BRACKET_RE = re.compile(r"\s*\[(\d+)\s*,\s*(\d+)\]")
_POS_AT_RE = re.compile(r"\s*@\((\d+)\s*,\s*(\d+)\)")
_NODE_ID_RE = re.compile(r"#\d+\b")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# GumTree JDT frequently emits balanced phantom TypeDeclaration / MethodDeclaration header churn
_HEADER_NOISE_TYPES = {"Modifier", "TYPE_DECLARATION_KIND", "SimpleName", "PrimitiveType"}
_HEADER_NOISE_PARENTS = {"TypeDeclaration", "MethodDeclaration"}


@dataclass
class NormalizedASTAction:
    """Single normalized AST edit operation without positional/internal IDs."""

    action: str          # INSERT, DELETE, UPDATE, MOVE
    node_type: str       # e.g., InfixExpression, IfStatement, MethodInvocation
    label: str = ""      # Original token/label if present
    new_label: str = ""  # Replacement label for UPDATE actions
    parent_type: str = ""
    parent_label: str = ""
    subtree_summary: str = ""

    def format_line(self) -> str:
        """Render a concise, LLM-readable structural edit line."""
        node_desc = f"{self.node_type}({self.label})" if self.label else self.node_type
        parent_desc = (
            f"{self.parent_type}({self.parent_label})"
            if self.parent_label
            else self.parent_type
        )
        if self.action == "UPDATE":
            return f"UPDATE {self.node_type}: `{self.label}` -> `{self.new_label}`"
        if self.action == "INSERT":
            target = f" into {parent_desc}" if parent_desc else ""
            body = f" :: {self.subtree_summary}" if self.subtree_summary else ""
            return f"INSERT {node_desc}{target}{body}"
        if self.action == "DELETE":
            target = f" from {parent_desc}" if parent_desc else ""
            body = f" :: {self.subtree_summary}" if self.subtree_summary else ""
            return f"DELETE {node_desc}{target}{body}"
        if self.action == "MOVE":
            target = f" into {parent_desc}" if parent_desc else ""
            body = f" :: {self.subtree_summary}" if self.subtree_summary else ""
            return f"MOVE {node_desc}{target}{body}"
        return f"{self.action} {node_desc}"


@dataclass
class ASTDiffResult:
    """Result of running GumTree and normalizing the edit script."""

    success: bool
    file_path: str
    enclosing_method: str
    actions: List[NormalizedASTAction]
    normalized_text: str
    raw_action_count: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "file_path": self.file_path,
            "enclosing_method": self.enclosing_method,
            "actions": [asdict(a) for a in self.actions],
            "normalized_text": self.normalized_text,
            "raw_action_count": self.raw_action_count,
            "error": self.error,
        }


def strip_positional_metadata(text: str) -> str:
    """Remove GumTree character offsets `[start,end]`, `@(pos,len)`, and internal IDs."""
    cleaned = _POS_BRACKET_RE.sub("", text)
    cleaned = _POS_AT_RE.sub("", cleaned)
    cleaned = _NODE_ID_RE.sub("", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _split_type_and_label(tree_str: str) -> tuple[str, str]:
    """Parse GumTree node descriptor like 'SimpleName: foo [10,13]' -> ('SimpleName', 'foo')."""
    cleaned = strip_positional_metadata(tree_str)
    first_line = cleaned.splitlines()[0].strip() if cleaned else ""
    if ": " in first_line:
        node_type, label = first_line.split(": ", 1)
        return node_type.strip(), label.strip()
    return first_line.strip(), ""


def _summarize_subtree(tree_str: str, max_lines: int = 12) -> str:
    """Compact multi-line GumTree subtree into a clean indented structural snippet."""
    lines = [strip_positional_metadata(ln) for ln in tree_str.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return ""
    body_lines = lines[1 : max_lines + 1]
    suffix = " ..." if len(lines) > max_lines + 1 else ""
    return " ; ".join(body_lines) + suffix


def _extract_char_span(tree_str: str) -> Optional[Tuple[int, int]]:
    """Extract the first `[start, end]` or `@(pos, len)` character span from GumTree output."""
    first_line = tree_str.splitlines()[0] if tree_str else ""
    m_bracket = _POS_BRACKET_RE.search(first_line)
    if m_bracket:
        start, end = int(m_bracket.group(1)), int(m_bracket.group(2))
        if 0 <= start < end:
            return start, end
    m_at = _POS_AT_RE.search(first_line)
    if m_at:
        pos, length = int(m_at.group(1)), int(m_at.group(2))
        if pos >= 0 and length > 0:
            return pos, pos + length
    return None


def _extract_source_snippet(
    source_text: Optional[str],
    span: Optional[Tuple[int, int]],
    existing_label: str = "",
    max_chars: int = 160,
) -> str:
    """Recover exact lexical tokens/literals from source text using GumTree offsets."""
    if not source_text or span is None:
        return ""
    start, end = span
    if start < 0 or end > len(source_text) or start >= end:
        return ""
    raw_slice = re.sub(r"\s+", " ", source_text[start:end]).strip()
    if not raw_slice or raw_slice == existing_label:
        return ""
    if len(raw_slice) > max_chars:
        return raw_slice[: max_chars - 3].rstrip() + "..."
    return raw_slice


def filter_phantom_header_noise(
    actions: List[NormalizedASTAction],
) -> List[NormalizedASTAction]:
    """Remove balanced phantom class-header churn emitted on TypeDeclaration nodes.

    GumTree v3.0.0 JDT frequently pairs `INSERT Modifier(public) into TypeDeclaration`,
    `INSERT TYPE_DECLARATION_KIND(class) into TypeDeclaration`, and `INSERT SimpleName(...)
    into TypeDeclaration` with matching `DELETE` actions whenever enclosing body trees shift.
    Cancelling only balanced `(node_type, label)` pairs preserves genuine header edits
    (e.g., adding `abstract`) and genuine method-body `SimpleName`/`Modifier` deletions.
    """
    ins_counts: Counter[Tuple[str, str]] = Counter()
    del_counts: Counter[Tuple[str, str]] = Counter()

    for act in actions:
        if (
            act.action == "INSERT"
            and act.parent_type in _HEADER_NOISE_PARENTS
            and act.node_type in _HEADER_NOISE_TYPES
        ):
            ins_counts[(act.node_type, act.label)] += 1
        elif (
            act.action == "DELETE"
            and act.parent_type in ("", *_HEADER_NOISE_PARENTS)
            and act.node_type in _HEADER_NOISE_TYPES
        ):
            del_counts[(act.node_type, act.label)] += 1

    # Only cancel pairs that appear in both INSERT (into Type/MethodDeclaration) and DELETE
    to_drop_ins: Counter[Tuple[str, str]] = Counter()
    to_drop_del: Counter[Tuple[str, str]] = Counter()
    for key, cnt in ins_counts.items():
        paired = min(cnt, del_counts.get(key, 0))
        if paired > 0:
            to_drop_ins[key] = paired
            to_drop_del[key] = paired

    if not to_drop_ins:
        return actions

    filtered: List[NormalizedASTAction] = []
    for act in actions:
        key = (act.node_type, act.label)
        if (
            act.action == "INSERT"
            and act.parent_type in _HEADER_NOISE_PARENTS
            and to_drop_ins[key] > 0
        ):
            to_drop_ins[key] -= 1
            continue
        if (
            act.action == "DELETE"
            and act.parent_type in ("", *_HEADER_NOISE_PARENTS)
            and to_drop_del[key] > 0
        ):
            to_drop_del[key] -= 1
            continue
        filtered.append(act)
    return filtered


def _normalize_action_kind(raw_action: str) -> str:
    raw = raw_action.lower().strip()
    if raw.startswith("insert"):
        return "INSERT"
    if raw.startswith("delete"):
        return "DELETE"
    if raw.startswith("update"):
        return "UPDATE"
    if raw.startswith("move"):
        return "MOVE"
    return raw.upper()


def parse_gumtree_json(
    payload: Dict[str, Any],
    src_text: Optional[str] = None,
    dst_text: Optional[str] = None,
    filter_header_noise: bool = True,
) -> List[NormalizedASTAction]:
    """Normalize GumTree `textdiff -f JSON` output into `NormalizedASTAction` items."""
    actions_out: List[NormalizedASTAction] = []
    for item in payload.get("actions", []):
        action_kind = _normalize_action_kind(str(item.get("action", "")))
        tree_raw = str(item.get("tree", item.get("node", "")))
        node_type, label = _split_type_and_label(tree_raw)
        subtree_summary = _summarize_subtree(tree_raw)

        # GumTree v3.0 JSON omits child subtrees in `tree`; recover exact literals/tokens via offsets
        if not subtree_summary and action_kind in {"INSERT", "DELETE", "MOVE"}:
            span = _extract_char_span(tree_raw)
            ref_text = dst_text if action_kind == "INSERT" else src_text
            snippet = _extract_source_snippet(ref_text, span, existing_label=label)
            if snippet:
                subtree_summary = f"`{snippet}`"

        parent_raw = str(item.get("parent", item.get("to", ""))) if action_kind != "UPDATE" else str(item.get("parent", ""))
        parent_type, parent_label = _split_type_and_label(parent_raw) if parent_raw else ("", "")

        new_label = strip_positional_metadata(str(item.get("label", "")))
        if action_kind == "UPDATE" and not new_label:
            # GumTree v3/v4 occasionally nests replacement under 'value' or 'to'
            new_label = strip_positional_metadata(
                str(item.get("value", item.get("to", "")))
            )

        actions_out.append(
            NormalizedASTAction(
                action=action_kind,
                node_type=node_type,
                label=label,
                new_label=new_label,
                parent_type=parent_type,
                parent_label=parent_label,
                subtree_summary=subtree_summary,
            )
        )
    if filter_header_noise:
        return filter_phantom_header_noise(actions_out)
    return actions_out


def parse_gumtree_text(
    raw_text: str,
    src_text: Optional[str] = None,
    dst_text: Optional[str] = None,
    filter_header_noise: bool = True,
) -> List[NormalizedASTAction]:
    """Fallback parser for GumTree `textdiff` plain-text output (v3.x and v4.x)."""
    actions_out: List[NormalizedASTAction] = []
    blocks = re.split(r"\n===+\n|^===+\s*\n", raw_text.strip())
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip() and ln.strip() != "---"]
        if not lines:
            continue
        header = lines[0].lower()
        if header.startswith("match"):
            continue
        action_kind = _normalize_action_kind(header)
        if action_kind not in {"INSERT", "DELETE", "UPDATE", "MOVE"}:
            continue

        body = "\n".join(lines[1:])
        if action_kind == "UPDATE":
            # Pattern: "SimpleName: a [10,11]\nreplace a by b" or "to <new_label>"
            node_type, label = _split_type_and_label(lines[1]) if len(lines) > 1 else ("Node", "")
            new_label = ""
            for ln in lines[2:]:
                m = re.search(r"replace\s+.+?\s+by\s+(.+)$", ln)
                if m:
                    new_label = strip_positional_metadata(m.group(1))
                    break
                if ln.lower().startswith("to "):
                    new_label = strip_positional_metadata(ln[3:])
                    break
            actions_out.append(
                NormalizedASTAction(
                    action="UPDATE",
                    node_type=node_type,
                    label=label,
                    new_label=new_label,
                )
            )
        else:
            # Handle both GumTree v4 ("to <parent> at <idx>") and v3 ("to\n<parent>\nat <idx>")
            parent_type, parent_label = "", ""
            tree_lines: List[str] = []
            idx = 1
            while idx < len(lines):
                ln = lines[idx]
                low = ln.lower()
                if low == "to" and not parent_type and idx + 1 < len(lines):
                    parent_part = re.sub(r"\s+at\s+\d+\s*$", "", lines[idx + 1], flags=re.IGNORECASE)
                    parent_type, parent_label = _split_type_and_label(parent_part)
                    idx += 2
                    continue
                if low.startswith("to ") and not parent_type:
                    parent_part = re.sub(r"\s+at\s+\d+\s*$", "", ln[3:], flags=re.IGNORECASE)
                    parent_type, parent_label = _split_type_and_label(parent_part)
                    idx += 1
                    continue
                if re.match(r"^at\s+\d+$", low) and parent_type:
                    idx += 1
                    continue
                tree_lines.append(ln)
                idx += 1

            tree_str = "\n".join(tree_lines) if tree_lines else body
            node_type, label = _split_type_and_label(tree_str)
            subtree_summary = _summarize_subtree(tree_str)
            if not subtree_summary:
                span = _extract_char_span(tree_str)
                ref_text = dst_text if action_kind == "INSERT" else src_text
                snippet = _extract_source_snippet(ref_text, span, existing_label=label)
                if snippet:
                    subtree_summary = f"`{snippet}`"
            actions_out.append(
                NormalizedASTAction(
                    action=action_kind,
                    node_type=node_type,
                    label=label,
                    parent_type=parent_type,
                    parent_label=parent_label,
                    subtree_summary=subtree_summary,
                )
            )
    if filter_header_noise:
        return filter_phantom_header_noise(actions_out)
    return actions_out


def format_normalized_script(
    file_path: str,
    enclosing_method: str,
    actions: List[NormalizedASTAction],
) -> str:
    """Format normalized AST actions into a deterministic structural summary."""
    header = [
        f"File: {file_path}",
        f"Enclosing Method: {enclosing_method or 'Unknown'}",
        f"AST Edit Operations ({len(actions)}):",
    ]
    body = [f"  {idx}. {act.format_line()}" for idx, act in enumerate(actions, start=1)]
    return "\n".join(header + body)


def run_gumtree_diff(
    src_file: Path,
    dst_file: Path,
    rel_path: str,
    enclosing_method: str = "",
    gumtree_bin: str = "gumtree",
    java_home: str = "",
    timeout_sec: int = 60,
    filter_header_noise: bool = True,
) -> ASTDiffResult:
    """Invoke GumTree CLI on `(src_file, dst_file)` and return normalized AST edit script."""
    env = os.environ.copy()
    if java_home:
        env["JAVA_HOME"] = java_home
        env["PATH"] = f"{java_home}/bin:{env.get('PATH', '')}"

    src_text = src_file.read_text(encoding="utf-8", errors="replace") if src_file.exists() else ""
    dst_text = dst_file.read_text(encoding="utf-8", errors="replace") if dst_file.exists() else ""

    # Try JSON format first (supported in GumTree 3.x / 4.x)
    cmd_json = [gumtree_bin, "textdiff", "-f", "JSON", str(src_file), str(dst_file)]
    try:
        proc = subprocess.run(
            cmd_json,
            capture_output=True,
            text=True,
            errors="replace",
            env=env,
            timeout=timeout_sec,
        )
        if proc.returncode == 0 and proc.stdout.strip().startswith("{"):
            payload = json.loads(proc.stdout)
            actions = parse_gumtree_json(
                payload,
                src_text=src_text,
                dst_text=dst_text,
                filter_header_noise=filter_header_noise,
            )
            return ASTDiffResult(
                success=True,
                file_path=rel_path,
                enclosing_method=enclosing_method,
                actions=actions,
                normalized_text=format_normalized_script(rel_path, enclosing_method, actions),
                raw_action_count=len(actions),
            )
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as exc:
        # Fall through to plain-text mode only on timeout or non-JSON output
        if isinstance(exc, OSError):
            return ASTDiffResult(
                success=False,
                file_path=rel_path,
                enclosing_method=enclosing_method,
                actions=[],
                normalized_text="",
                error=f"Failed to invoke GumTree binary ({gumtree_bin}): {exc}",
            )

    # Fallback: plain-text textdiff
    cmd_text = [gumtree_bin, "textdiff", str(src_file), str(dst_file)]
    try:
        proc = subprocess.run(
            cmd_text,
            capture_output=True,
            text=True,
            errors="replace",
            env=env,
            timeout=timeout_sec,
        )
        if proc.returncode != 0:
            return ASTDiffResult(
                success=False,
                file_path=rel_path,
                enclosing_method=enclosing_method,
                actions=[],
                normalized_text="",
                error=proc.stderr.strip() or f"GumTree exited with code {proc.returncode}",
            )
        actions = parse_gumtree_text(
            proc.stdout,
            src_text=src_text,
            dst_text=dst_text,
            filter_header_noise=filter_header_noise,
        )
        return ASTDiffResult(
            success=True,
            file_path=rel_path,
            enclosing_method=enclosing_method,
            actions=actions,
            normalized_text=format_normalized_script(rel_path, enclosing_method, actions),
            raw_action_count=len(actions),
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        msg = f"GumTree timed out after {timeout_sec}s" if isinstance(exc, subprocess.TimeoutExpired) else f"Failed to invoke GumTree binary ({gumtree_bin}): {exc}"
        return ASTDiffResult(
            success=False,
            file_path=rel_path,
            enclosing_method=enclosing_method,
            actions=[],
            normalized_text="",
            error=msg,
        )


def extract_raw_git_diff(
    src_file: Path,
    dst_file: Path,
    rel_path: str,
    context_lines: int = 3,
) -> str:
    """Compute minimal unified git diff (Control condition) for the exact same file pair."""
    proc = subprocess.run(
        [
            "git",
            "diff",
            "--no-index",
            f"-U{context_lines}",
            "--",
            str(src_file),
            str(dst_file),
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    # git diff --no-index returns 1 when files differ, 0 when identical
    diff_text = proc.stdout
    # Normalize absolute paths in diff header to clean relative path (both OS and POSIX forms)
    for p in (str(src_file), src_file.as_posix()):
        diff_text = diff_text.replace(p, f"a/{rel_path}")
    for p in (str(dst_file), dst_file.as_posix()):
        diff_text = diff_text.replace(p, f"b/{rel_path}")
    return diff_text.strip()
