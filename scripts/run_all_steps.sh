#!/usr/bin/env bash
# =============================================================================
# CS.91000 Graduation Research — ASAP Full Execution Script
# Sachit Varshney (20220859) — Advisor: Prof. Dr. Shin Yoo (KAIST)
#
# Run this INSIDE the Docker container:
#   docker build -t ast-brt-gen .
#   docker run -it --rm -e OPENAI_API_KEY="$OPENAI_API_KEY" -v "$(pwd):/workspace" ast-brt-gen
#   bash scripts/run_all_steps.sh
#
# This script runs all pipeline steps from start to finish.
# Estimated time: 4-8 hours depending on API latency and bug count.
# Estimated cost: ~$5-$15 for 80-150 paired bugs on gpt-4o-2024-08-06.
# =============================================================================
set -euo pipefail

PYTHON="${PYTHON:-python3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "CS.91000 GRADUATION RESEARCH — FULL PIPELINE EXECUTION"
echo "Started at: $(date)"
echo "Log directory: ${LOG_DIR}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────
# STEP 0: Verify environment
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[STEP 0] Verifying environment..."
defects4j pids | head -5
gumtree --help | head -5
${PYTHON} -c "import numpy, scipy, matplotlib, pandas; print('Python deps OK')"
echo "  ✓ Environment verified."

# ─────────────────────────────────────────────────────────────────
# STEP 1: Run unit tests & isolation audit
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[STEP 1] Running unit tests and isolation audit..."
${PYTHON} -m pytest -v 2>&1 | tee "${LOG_DIR}/step1_pytest.log"
${PYTHON} scripts/verify_isolation.py --config configs/experiment.yaml 2>&1 | tee "${LOG_DIR}/step1_isolation.log"
echo "  ✓ All tests pass. Isolation audit PASSED."

# ─────────────────────────────────────────────────────────────────
# STEP 2: Curate Defects4J dataset (3-filter pipeline)
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[STEP 2] Curating Defects4J dataset..."
${PYTHON} -m src.dataset.defects4j_curate --config configs/experiment.yaml 2>&1 | tee "${LOG_DIR}/step2_curate.log"
echo ""
echo "  Curated manifest: data/curated_bugs.json"
echo "  Exclusion log:    data/exclusion_log.json"
N_BUGS=$(${PYTHON} -c "import json; print(len(json.load(open('data/curated_bugs.json'))))")
echo "  ✓ Curated ${N_BUGS} bugs."

# ─────────────────────────────────────────────────────────────────
# STEP 3: 10-bug pilot run
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[STEP 3] Running 10-bug pilot (Control + Treatment)..."
${PYTHON} scripts/run_experiment.py --config configs/experiment.yaml --limit 10 --conditions raw_diff ast_edit 2>&1 | tee "${LOG_DIR}/step3_pilot.log"
echo "  ✓ Pilot complete. Check data/results/trials/ for outputs."

# Quick sanity: count pilot output files
PILOT_COUNT=$(find data/results/trials/ -name "*.json" | wc -l)
echo "  Trial JSONs generated: ${PILOT_COUNT}"

# ─────────────────────────────────────────────────────────────────
# STEP 4: Full paired experiment (Control + Treatment)
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[STEP 4] Running full paired experiment on all ${N_BUGS} curated bugs..."
echo "  This will take several hours. Supports safe --resume if interrupted."
${PYTHON} scripts/run_experiment.py --config configs/experiment.yaml --conditions raw_diff ast_edit 2>&1 | tee "${LOG_DIR}/step4_full_run.log"
echo "  ✓ Full experiment complete."

# ─────────────────────────────────────────────────────────────────
# STEP 5: Differential oracle + Cobertura spectra + FDG evaluation
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[STEP 5] Running evaluation pipeline (Oracle → Cobertura → FDG)..."
${PYTHON} -m src.evaluation.pipeline --config configs/experiment.yaml 2>&1 | tee "${LOG_DIR}/step5_evaluate.log"
echo "  ✓ Evaluation complete."
echo "  Summary: data/results/evaluation_summary.json"
echo "  CSV:     data/results/evaluation_paired.csv"

# ─────────────────────────────────────────────────────────────────
# STEP 6: Statistical analysis, LaTeX tables, publication plots
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[STEP 6] Generating statistical tests, LaTeX tables, and 300-DPI plots..."
${PYTHON} -m src.evaluation.stats_and_plots --config configs/experiment.yaml 2>&1 | tee "${LOG_DIR}/step6_analyze.log"
echo "  ✓ Analysis complete."
echo "  LaTeX tables: docs/figures/table_rq1.tex, docs/figures/table_rq2.tex"
echo "  Plots:        docs/figures/rq1_validity_comparison.png, docs/figures/rq2_fdg_paired.png"
echo "  Stats:        docs/figures/statistical_summary.json"

# ─────────────────────────────────────────────────────────────────
# STEP 7: Multi-SWE-bench (Java) secondary validation (optional, needs network)
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[STEP 7] (Optional) Running Multi-SWE-bench (Java) RQ1 validation..."
echo "  Note: This requires network access to clone GitHub repos."
echo "  Skipping if network is unavailable or if this is an offline run."
if ${PYTHON} -c "import datasets; print('OK')" 2>/dev/null; then
    ${PYTHON} -m src.dataset.multi_swe_bench --limit 30 --output data/multi_swe_java.json 2>&1 | tee "${LOG_DIR}/step7_multi_swe.log" || echo "  ⚠ Multi-SWE-bench step failed (network or data issue). Continuing."
else
    echo "  ⚠ datasets library not available. Install with: pip install datasets"
fi

# ─────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "CS.91000 GRADUATION RESEARCH — PIPELINE COMPLETE"
echo "Finished at: $(date)"
echo ""
echo "KEY DELIVERABLES:"
echo "  data/curated_bugs.json        — Curated bug manifest"
echo "  data/exclusion_log.json       — Filter exclusion audit"
echo "  data/results/trials/          — Raw trial JSONs (per bug × condition)"
echo "  data/results/spectra/         — Stored Cobertura spectra"
echo "  data/results/evaluation_summary.json  — Full evaluation results"
echo "  data/results/evaluation_paired.csv    — Flat CSV for analysis"
echo "  docs/figures/table_rq1.tex    — RQ1 LaTeX table"
echo "  docs/figures/table_rq2.tex    — RQ2 LaTeX table"
echo "  docs/figures/rq1_validity_comparison.png  — RQ1 bar chart (300 DPI)"
echo "  docs/figures/rq2_fdg_paired.png           — RQ2 boxplot (300 DPI)"
echo "  docs/figures/statistical_summary.json     — All p-values, A12, etc."
echo ""
echo "NEXT STEPS:"
echo "  1. Copy docs/figures/*.tex and docs/figures/*.png into docs/thesis/"
echo "  2. Fill in the PLACEHOLDER sections in docs/thesis/thesis.tex"
echo "  3. Compile: cd docs/thesis && pdflatex thesis && bibtex thesis && pdflatex thesis && pdflatex thesis"
echo "  4. Push to GitHub: git add -A && git commit -m 'Final results' && git push"
echo "  5. Email thesis PDF + replication link to Prof. Shin Yoo"
echo "============================================================"

