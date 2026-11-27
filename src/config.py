"""Typed configuration loader for the AST vs. Raw Diff BRT generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    # Fallback for environments where PyYAML won't build (e.g., Python 3.15)
    from ruamel.yaml import YAML as _YAML
    class _YamlCompat:
        @staticmethod
        def safe_load(stream):
            y = _YAML(typ='safe')
            return y.load(stream)
    yaml = _YamlCompat()


@dataclass
class PathsConfig:
    data_dir: Path = Path("data")
    curated_manifest: Path = Path("data/curated_bugs.json")
    exclusion_log: Path = Path("data/exclusion_log.json")
    prefix_root: Path = Path("tmp/prefix_workspaces")
    postfix_root: Path = Path("tmp/postfix_workspaces")
    results_dir: Path = Path("data/results")
    figures_dir: Path = Path("docs/figures")


@dataclass
class ToolsConfig:
    defects4j_bin: str = "defects4j"
    gumtree_bin: str = "gumtree"
    gumtree_java_home: str = ""
    compile_timeout_sec: int = 120
    test_timeout_sec: int = 120
    coverage_timeout_sec: int = 180


@dataclass
class DatasetConfig:
    projects: List[str] = field(
        default_factory=lambda: ["Lang", "Math", "Chart", "Time", "Closure"]
    )
    require_single_method_fix: bool = True
    min_viable_bugs: int = 50
    max_bugs: Optional[int] = None
    strip_triggering_tests: bool = True
    filter_ast_header_noise: bool = True


@dataclass
class AgentConfig:
    provider: str = "openai"
    model: str = "gpt-4o-2024-08-06"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_retries: int = 5
    max_tool_calls_per_retry: int = 6
    api_base_url: Optional[str] = None
    inline_enclosing_method: bool = True
    repeats: int = 1


@dataclass
class EvaluationConfig:
    fdg_alpha: float = 0.5
    dstar_star: int = 2
    acc_n_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    strict_assertion_only: bool = False
    include_passing_baseline_tests: bool = True
    max_baseline_passing_tests: int = 25


@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        paths_raw = raw.get("paths", {})
        paths = PathsConfig(
            **{k: Path(v) for k, v in paths_raw.items() if hasattr(PathsConfig, k)}
        )
        tools = ToolsConfig(**raw.get("tools", {}))
        dataset = DatasetConfig(**raw.get("dataset", {}))
        agent = AgentConfig(**raw.get("agent", {}))
        evaluation = EvaluationConfig(**raw.get("evaluation", {}))
        return cls(
            paths=paths,
            tools=tools,
            dataset=dataset,
            agent=agent,
            evaluation=evaluation,
        )
