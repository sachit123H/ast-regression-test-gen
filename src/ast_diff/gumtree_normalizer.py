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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# Regexes for stripping GumTree positional metadata e.g. [1240,1265] or @(1240,25)
_POS_BRACKET_RE = re.compile(r"\s*\[\d+\s*,\s*\d+\]")
_POS_AT_RE = re.compile(r"\s*@\(\d+\s*,\s*\d+\)")
_NODE_ID_RE = re.compile(r"#\d+\b")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


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
            return f"MOVE {node_desc}{target}"
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


def parse_gumtree_json(payload: Dict[str, Any]) -> List[NormalizedASTAction]:
    """Normalize GumTree `textdiff -f JSON` output into `NormalizedASTAction` items."""
    actions_out: List[NormalizedASTAction] = []
    for item in payload.get("actions", []):
        action_kind = _normalize_action_kind(str(item.get("action", "")))
        tree_raw = str(item.get("tree", item.get("node", "")))
        node_type, label = _split_type_and_label(tree_raw)
        subtree_summary = _summarize_subtree(tree_raw)

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
    return actions_out


def parse_gumtree_text(raw_text: str) -> List[NormalizedASTAction]:
    """Fallback parser for GumTree `textdiff` plain-text output (v3.x and v4.x)."""
    actions_out: List[NormalizedASTAction] = []
    blocks = re.split(r"\n===+\n", raw_text.strip())
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
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
            # Split around "to <parent> at <idx>" if present
            parent_type, parent_label = "", ""
            tree_lines: List[str] = []
            for ln in lines[1:]:
                if ln.lower().startswith("to ") and not parent_type:
                    parent_part = re.sub(r"\s+at\s+\d+\s*$", "", ln[3:], flags=re.IGNORECASE)
                    parent_type, parent_label = _split_type_and_label(parent_part)
                else:
                    tree_lines.append(ln)
            tree_str = "\n".join(tree_lines) if tree_lines else body
            node_type, label = _split_type_and_label(tree_str)
            actions_out.append(
                NormalizedASTAction(
                    action=action_kind,
                    node_type=node_type,
                    label=label,
                    parent_type=parent_type,
                    parent_label=parent_label,
                    subtree_summary=_summarize_subtree(tree_str),
                )
            )
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
) -> ASTDiffResult:
    """Invoke GumTree CLI on `(src_file, dst_file)` and return normalized AST edit script."""
    env = os.environ.copy()
    if java_home:
        env["JAVA_HOME"] = java_home
        env["PATH"] = f"{java_home}/bin:{env.get('PATH', '')}"

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
            actions = parse_gumtree_json(payload)
            return ASTDiffResult(
                success=True,
                file_path=rel_path,
                enclosing_method=enclosing_method,
                actions=actions,
                normalized_text=format_normalized_script(rel_path, enclosing_method, actions),
                raw_action_count=len(actions),
            )
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as exc:
        # Fall through to plain-text mode unless binary is missing
        if isinstance(exc, FileNotFoundError):
            return ASTDiffResult(
                success=False,
                file_path=rel_path,
                enclosing_method=enclosing_method,
                actions=[],
                normalized_text="",
                error=f"GumTree binary not found: {gumtree_bin}",
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
        actions = parse_gumtree_text(proc.stdout)
        return ASTDiffResult(
            success=True,
            file_path=rel_path,
            enclosing_method=enclosing_method,
            actions=actions,
            normalized_text=format_normalized_script(rel_path, enclosing_method, actions),
            raw_action_count=len(actions),
        )
    except subprocess.TimeoutExpired:
        return ASTDiffResult(
            success=False,
            file_path=rel_path,
            enclosing_method=enclosing_method,
            actions=[],
            normalized_text="",
            error=f"GumTree timed out after {timeout_sec}s",
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
