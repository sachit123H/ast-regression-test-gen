# Post-Fix Bug-Reproducing Test Generation: AST Edit Scripts vs. Raw Diffs

> **CS.91000 Graduation Research**  
> **Author:** Sachit Varshney (20220859)  
> **Advisor:** Prof. Dr. Shin Yoo (COINSE Lab, School of Computing, KAIST)  
> **Date:** Fall 2026

*A graduation research project investigating whether providing an LLM-based test generation agent with structural Abstract Syntax Tree (AST) edit scripts (generated via GumTree) improves the quality of generated bug-reproducing tests compared to standard raw line diffs.*

---

## Overview

When a developer fixes a bug without writing a corresponding bug-reproducing test (BRT), the fix goes into production unguarded — a BRT must still be backfilled into the regression suite. This project asks: **does the structural representation of a bug-fixing commit affect the quality of an LLM-generated BRT?**

Three realistic post-fix use cases motivate this work:
1. **Time-pressured hotfixes** deployed without a BRT under tight deadlines.
2. **External contributor PRs** where the author lacks test-framework knowledge.
3. **Edge-case expansion** beyond a minimal symptom test.

We construct two conditions differing **only** in the change representation:
- **Control (`raw_diff`):** Standard unified `git diff` output.
- **Treatment (`ast_edit`):** Normalized GumTree AST edit script (Insert, Delete, Update, Move operations on syntax-tree nodes).

All other experimental factors — the agent prompt, tool schemas, feedback signals, retry budget, and stopping criterion — are held **byte-for-byte identical** across both conditions.

---

## Research Questions

- **RQ1 (Validity Rate):** Does providing a normalized AST edit script instead of a raw diff improve the agent's ability to generate a valid bug-reproducing test (satisfying the differential execution oracle)?
- **RQ2 (Diagnostic Quality):** To what extent does AST-level change context improve the Fault Diagnosability Gain (FDG) of the generated test compared to the raw-diff baseline over jointly valid cases?
- **RQ3 (Efficiency & Dynamics):** How do textual versus structural change representations affect prompt token overhead and retry convergence dynamics?

---

## Repository Structure

```text
ast-regression-test-gen/
├── PROPOSAL.md              # Full research proposal text
├── README.md                # Project overview and documentation
├── Dockerfile               # Reproducible Docker environment (Ubuntu 22.04 + Java 8/17 + Defects4J + GumTree + Python 3.11)
├── Makefile                 # Convenience targets: test, curate, verify-isolation, pilot, run-all, evaluate, analyze
├── pyproject.toml           # Python project metadata
├── requirements.txt         # Python dependencies
├── configs/
│   ├── pilot_65.yaml        # Initial 65-bug pilot config + passing-test FDG evaluation
│   └── experiment.yaml      # Full 80-bug rerun config (patched AST + inline method + 3 repeats)
├── data/                    # Dataset artifacts (curated_bugs*.json, exclusion_log*.json, multi_swe_java.json)
├── docs/
│   ├── figures/             # LaTeX tables (table_rq1.tex, table_rq2.tex, table_summary_metrics.tex) and 300-DPI plots
│   └── thesis/              # Complete graduation research thesis (thesis.tex, references.bib, thesis.pdf)
├── scripts/
│   ├── audit_normalizer_80.py   # Verifies patched GumTree normalizer across all 80 curated bugs
│   ├── audit_valid_tests.py     # Audits all generated valid tests and flags degenerate helpers
│   ├── build_curated_80.py      # Builds 80-bug rerun manifest with patched AST + inlined methods
│   ├── run_experiment.py        # Main CLI runner for Control + Treatment trials (supports --repeats)
│   ├── update_readme_stats.py   # Auto-generates README empirical summary from statistical_summary.json
│   └── verify_isolation.py      # Automated isolation & leakage audit
├── src/
│   ├── config.py            # Typed configuration loader
│   ├── agent/
│   │   ├── llm_client.py    # Provider-agnostic LLM client (OpenAI, Anthropic, OpenAI-compatible)
│   │   ├── loop.py          # Closed 5-retry agentic BRT generation loop
│   │   └── prompts.py       # Identical prompt templates with change representation switch
│   ├── ast_diff/
│   │   ├── gumtree_normalizer.py  # GumTree AST extraction, normalization, and raw diff baseline
│   │   └── hyrep.py               # HyRep-BRT hybrid representation engine (lexical diff + AST node tags)
│   ├── dataset/
│   │   ├── defects4j_curate.py    # 3-filter Defects4J curation with exclusion audit
│   │   └── multi_swe_bench.py     # Multi-SWE-bench exploratory adapter prototype
│   ├── evaluation/
│   │   ├── oracle.py              # Isolated differential execution oracle (Sec. 4.5)
│   │   ├── cobertura_parser.py    # Cobertura XML line/method coverage parser
│   │   ├── sbfl.py                # Ochiai + DStar(*=2) SBFL, ranking, acc@n, mAP
│   │   ├── fdg.py                 # Exact FDG (An & Yoo ISSTA 2022) implementation
│   │   ├── pipeline.py            # End-to-end: Oracle → Cobertura Spectra → FDG/SBFL
│   │   └── stats_and_plots.py     # McNemar, Wilcoxon, Vargha-Delaney A12, LaTeX + plots
│   └── mcp/
│       ├── sandbox.py             # Hardened pre-fix sandbox with double isolation
│       └── server.py              # MCP stdio server (FastMCP) for get_context + run_test
└── tests/                         # Unit & integration tests (all pure Python, no Docker needed)
    ├── test_gumtree_normalizer.py
    ├── test_hyrep.py
    ├── test_oracle_and_cobertura.py
    ├── test_sandbox_isolation.py
    ├── test_sbfl_and_fdg.py
    └── test_stats_and_plots.py
```

---

## Quick Start

### 1. Build & Enter Docker Environment
```bash
docker build -t ast-brt-gen .
docker run -it --rm -e OPENAI_API_KEY="$OPENAI_API_KEY" -v "$(pwd):/workspace" ast-brt-gen
```

### 2. Verify Installation & Run Tests
```bash
defects4j pids && gumtree --version
make test                    # All unit tests (pure Python)
make verify-isolation        # Double isolation audit
```

### 3. Curate Defects4J Dataset
```bash
make curate                  # 3-filter curation → data/curated_bugs.json + data/exclusion_log.json
```

### 4. Run Pilot (10 bugs) → Full Experiment
```bash
make pilot                   # 10-bug pilot across both conditions
make run-all                 # Full paired benchmark (supports --resume)
```

### 5. Evaluate & Analyze
```bash
make evaluate-pilot          # Re-evaluate initial 65-bug pilot with passing+failing FDG baseline
make evaluate                # Full rerun evaluation (strict assertion primary + passing+failing FDG)
make analyze                 # Statistical tests, LaTeX tables, 300-DPI plots, and README sync
```

---

<!-- BEGIN_AUTOGEN_RESULTS -->
## Empirical Results Summary (Initial 65-Bug Pilot)

Evaluated on $N=65$ Defects4J bugs across Apache Commons-Lang (40 bugs) and Commons-Math (25 bugs) with GPT-4o (`gpt-4o-2024-08-06`, temperature 0.0, 130 total trials) using the initial lossy AST normalizer and on-demand `get_context` (`configs/pilot_65.yaml`), after 15 late-stage Math bugs from the 80-bug curated manifest were halted by OpenAI API billing/quota exhaustion (`data/exclusion_log.json`):

| Metric | Raw Diff (Control) | AST Edit Script (Treatment) | Comparison / Statistical Test |
|:-------|:-------------------|:----------------------------|:------------------------------|
| **Strict Assertion Validity (Primary, RQ1)** | **32.3%** (21/65) | 18.5% (12/65) | McNemar $p = 0.0225$ |
| **LIBRO Fail/Pass Validity (Secondary, RQ1)** | **41.5%** (27/65) | 24.6% (16/65) | McNemar $p = 0.0127$ |
| **Mean FDG (Ochiai, Strict Paired, $N=10$)** | 0.5047 | 0.5039 | Wilcoxon $p = 0.6250$, $\hat{A}_{12} = 0.500$ (1 AST wins / 6 ties / 3 raw wins; sign $p = 0.6250$) |
| **Mean FDG (DStar $*=2$, Strict Paired, $N=10$)** | 0.4988 | 0.4982 | Wilcoxon $p = 0.6250$, $\hat{A}_{12} = 0.490$ |
| **Realised Mean $\Delta\text{AP}$ (Ochiai, $N=10$)** | +0.0127 | +0.0627 | Wilcoxon $p = 1.0000$, $\hat{A}_{12} = 0.550$ |
| **Method acc@3 (Strict Paired, $N=10$)** | 8 ($\Delta = -1$) | 8 ($\Delta = -1$) | Baseline = 9 |
| **Expected Diagnosability $E[\text{FDG}]$ (Strict, $N=65$)** | **0.1654** | 0.0933 | Wilcoxon $p = 0.0148$, $\hat{A}_{12} = 0.428$ |
| **Mean Token Overhead (RQ3)** | 26,027 tokens | **22,505 tokens** | **13.5% reduction** with AST (2.94 vs. 2.97 attempts) |

**Pilot Key Finding:** In the initial 65-bug pilot, raw unified diffs outperformed the initial over-normalized AST edit scripts (32.3% vs. 18.5% strict assertion validity, McNemar $p = 0.0225$; 41.5% vs. 24.6% LIBRO fail/pass; 32.5% vs. 17.5% strict on Lang, 32.0% vs. 20.0% strict on Math; LIBRO by-project: 45.0% vs. 25.0% Lang, 36.0% vs. 24.0% Math) because GumTree v3.0.0 JSON omitted nested subtrees (stripping operand literals) and retained spurious `TypeDeclaration` header noise in 49/65 bugs. Over strict jointly valid pairs ($N_{\text{paired}}=10$), paired FDG counts were 1 AST wins / 6 ties / 3 raw wins (Wilcoxon $p = 0.6250$). Diagnostic FDG/SBFL spectra incorporate both `tests.trigger` failing tests and up to 12 verified passing tests sampled across `tests.relevant`. For the full rerun (`configs/experiment.yaml`), we patched `gumtree_normalizer.py` (verified across all 80 curated bugs with 0 header-noise actions and full subtree/MOVE literal recovery) and enabled `inline_enclosing_method: true`.
<!-- END_AUTOGEN_RESULTS -->

---

## Key Design Decisions & Advisor Guidance Alignment

| Decision | Source / Resolution |
|:---------|:--------------------|
| **Task Framing** | Post-fix Bug-Reproducing Test (BRT) generation (reframed per Prof. Shin Yoo's Aug 7–8 guidance) |
| **Enclosing-Method Context** | `inline_enclosing_method: true` in `configs/experiment.yaml` (Prof. Yoo's Nov 25 guidance; initial 65-bug pilot used on-demand `get_context` in `configs/pilot_65.yaml`) |
| **Dual Exclusion Audits** | Explicit exclusion criteria and logs for both Defects4J (`data/exclusion_log.json`) and Multi-SWE-bench (`data/swe_bench_exclusion_log.json`, Prof. Yoo's Nov 25 guidance) |
| **Primary Validity Criterion** | Strict assertion failure on pre-fix and pass on post-fix (Proposal Sec. 4.5); LIBRO fail/pass reported secondarily |
| **Diagnostic Metric** | An & Yoo (ISSTA 2022) FDG ("Start with FDG", Nov 25 guidance) with passing (`tests.relevant`) + failing (`tests.trigger`) baseline tests |
| **Triggering-Test Isolation** | Double isolation: post-fix root confinement + physical `tests.trigger` redaction (Proposal Sec. 4.3 + Aug 12 isolation check) |
| **Contamination Scope** | Anticipated risk in Proposal Sec. 5.1; Multi-SWE-bench (Java) curation audited with written exclusion rules; end-to-end execution deferred to future work |
| **Future Work: HyRep-BRT** | Conceptual hybrid representation (`src/ast_diff/hyrep.py`); un-evaluated in LLM trials and reserved for future work |
| **Repository Name** | Historical identifier `ast-regression-test-gen` preserved for git remote/workspace stability |

---

## References

- Gabin An and Shin Yoo. *FDG: A Precise Measurement of Fault Diagnosability Gain of Test Cases.* ISSTA 2022.
- René Just et al. *Defects4J: A Database of Existing Faults.* ISSTA 2014.
- Jean-Rémy Falleri et al. *Fine-Grained and Accurate Source Code Differencing.* ASE 2014.
- W. Eric Wong et al. *The DStar Method for Effective Software Fault Localization.* IEEE TR 2014.
