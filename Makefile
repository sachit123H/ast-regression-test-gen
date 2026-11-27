.PHONY: test curate verify-isolation pilot run-all evaluate evaluate-pilot analyze sync-readme clean

PYTHON ?= python3

test:
	$(PYTHON) -m pytest -v

curate:
	$(PYTHON) -m src.dataset.defects4j_curate --config configs/experiment.yaml

verify-isolation:
	$(PYTHON) scripts/verify_isolation.py --config configs/experiment.yaml

pilot:
	$(PYTHON) scripts/run_experiment.py --config configs/pilot_65.yaml --limit 10 --conditions raw_diff ast_edit

run-all:
	$(PYTHON) scripts/run_experiment.py --config configs/experiment.yaml --conditions raw_diff ast_edit

evaluate:
	$(PYTHON) -m src.evaluation.pipeline --config configs/experiment.yaml --workers 3

evaluate-pilot:
	$(PYTHON) -m src.evaluation.pipeline --config configs/pilot_65.yaml --workers 3

analyze:
	$(PYTHON) -m src.evaluation.stats_and_plots --config configs/pilot_65.yaml
	$(PYTHON) -c "import shutil, pathlib; [shutil.copy2(p, pathlib.Path('docs/thesis') / p.name) for p in pathlib.Path('docs/figures').iterdir() if p.is_file()]"
	$(PYTHON) scripts/update_readme_stats.py

sync-readme:
	$(PYTHON) scripts/update_readme_stats.py

clean:
	rm -rf tmp/ .pytest_cache/ __pycache__/
