.PHONY: test curate verify-isolation pilot run-all evaluate analyze clean

PYTHON ?= python3

test:
	$(PYTHON) -m pytest -v

curate:
	$(PYTHON) -m src.dataset.defects4j_curate --config configs/experiment.yaml

verify-isolation:
	$(PYTHON) scripts/verify_isolation.py --config configs/experiment.yaml

pilot:
	$(PYTHON) scripts/run_experiment.py --config configs/experiment.yaml --limit 10 --conditions raw_diff ast_edit

run-all:
	$(PYTHON) scripts/run_experiment.py --config configs/experiment.yaml --conditions raw_diff ast_edit

evaluate:
	$(PYTHON) -m src.evaluation.pipeline --config configs/experiment.yaml

analyze:
	$(PYTHON) -m src.evaluation.stats_and_plots --config configs/experiment.yaml

clean:
	rm -rf tmp/ .pytest_cache/ __pycache__/
