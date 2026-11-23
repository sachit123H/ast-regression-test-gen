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
│   └── experiment.yaml      # Full experiment configuration
├── data/                    # Dataset artifacts (curated_bugs.json, exclusion_log.json)
├── docs/
│   ├── figures/             # LaTeX tables (table_rq1.tex, table_rq2.tex, table_summary_metrics.tex) and 300-DPI plots
│   └── thesis/              # Complete graduation research thesis (thesis.tex, references.bib)
├── scripts/
│   ├── run_experiment.py    # Main CLI runner for Control + Treatment trials
│   └── verify_isolation.py  # Automated isolation & leakage audit
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
│   │   └── multi_swe_bench.py     # Multi-SWE-bench (Java) contamination-free RQ1 adapter
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
make evaluate                # Differential oracle + Cobertura spectra + FDG
make analyze                 # Statistical tests, LaTeX tables, 300-DPI plots
```

---

## Empirical Results Summary

Evaluated on all $N=65$ clean Defects4J bugs across Apache Commons-Lang (40 bugs) and Commons-Math (25 bugs) with GPT-4o (`gpt-4o-2024-08-06`, temperature 0.0) in a closed agentic loop (130 total trials), following the documented exclusion of 15 late-stage Math bugs (Math-48–Math-75) due to batch API rate-limiting/quota exhaustion (`exclusion_log.json`):

| Metric | Raw Diff (Control) | AST Edit Script (Treatment) | Comparison / Statistical Test |
|:-------|:-------------------|:----------------------------|:------------------------------|
| **Differential Validity (RQ1)** | **41.5%** (27/65) | 24.6% (16/65) | McNemar $p = 0.0127$ (Statistically Significant) |
| **Strict Assertion Validity (RQ1)** | **32.3%** (21/65) | 18.5% (12/65) | McNemar $p = 0.0225$ (Statistically Significant) |
| **Mean FDG (Ochiai, RQ2, $N_{\text{paired}}=13$)** | 0.5144 | **0.5148** | Wilcoxon $p = 0.2500$, $\hat{A}_{12} = 0.524$, TOST $p < 10^{-12}$ |
| **Mean FDG (DStar $*=2$, RQ2)** | 0.5141 | **0.5145** | Wilcoxon $p = 0.2500$, $\hat{A}_{12} = 0.518$ |
| **Realised Mean $\Delta\text{AP}$ (RQ2)** | +0.0574 | **+0.0676** | Wilcoxon $p = 1.0000$, $\hat{A}_{12} = 0.538$ (AST slight refinement) |
| **Method acc@3 (RQ2)** | 7 ($\Delta = +2$) | **8** ($\Delta = +3$) | Baseline = 5 |
| **Expected Diagnosability $E[\text{FDG}]$ ($N=65$)** | **0.2131** | 0.1263 | Wilcoxon $p = 0.0157$, $\hat{A}_{12} = 0.420$ (68.7% Raw advantage) |
| **Mean Token Overhead (RQ3)** | 26,027 tokens | **22,505 tokens** | **13.5% reduction** with AST |
| **Multi-SWE-bench Validation ($N=15$)** | **80.0%** (12/15) | 33.3% (5/15) | McNemar $p = 0.0156$ (Statistically Significant) |

**Key Finding:** Modern LLMs reproduce defects with significantly higher validity from raw unified diffs due to extensive pre-training priors on Git diffs and lexical literal preservation. However, conditioned on producing a valid test ($N_{\text{paired}}=13$), AST edit scripts achieve statistically equivalent fault diagnosability (confirmed via Two One-Sided Tests equivalence at $\pm 0.01$), slightly higher realized fault ranking gains ($\Delta\text{AP} = +0.0676$ vs. $+0.0574$, $\text{acc@}3 = 8$ vs. $7$), and a 13.5% token overhead reduction. Across the entire $N=65$ population, raw diff delivers 68.7% higher Expected Diagnosability ($0.2131$ vs. $0.1263$, $p=0.0157$). To synthesize both benefits, we propose and implement **HyRep-BRT** (`src/ast_diff/hyrep.py`), a hybrid representation embedding AST node tags into unified diffs.

---

## Key Design Decisions (Reconciled with Prof. Shin Yoo's Email Guidance)

| Decision | Resolution |
|:---------|:-----------|
| **Task Framing** | Post-fix Bug-Reproducing Test (BRT) generation (not generic regression testing) |
| **Repository Name** | Historical identifier `ast-regression-test-gen` preserved for git remote/workspace stability |
| **Triggering-Test Isolation** | Double isolation: post-fix root confinement + physical `tests.trigger` redaction |
| **Primary Benchmark** | Defects4J v2.0.1 (Lang, Math — 65 single-method bugs evaluated end-to-end; 15 API quota aborts cleanly excluded) |
| **Spectra Coverage** | Exactly 30 spectra files (21 Lang, 9 Math) where $\ge 1$ condition was valid; 13 paired bugs |
| **Contamination Check** | Multi-SWE-bench (Java) secondary RQ1 validation across 15 production repositories |
| **FDG Formula** | Ochiai (primary) + DStar(*=2) robustness, α=0.5, max tie-breaker |
| **Architectural Synthesis** | HyRep-BRT (`src/ast_diff/hyrep.py`) combining lexical constants with AST syntax tags |

---

## References

- Gabin An and Shin Yoo. *FDG: A Precise Measurement of Fault Diagnosability Gain of Test Cases.* ISSTA 2022.
- René Just et al. *Defects4J: A Database of Existing Faults.* ISSTA 2014.
- Jean-Rémy Falleri et al. *Fine-Grained and Accurate Source Code Differencing.* ASE 2014.
- ByteDance Seed. *Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving.* 2025.
