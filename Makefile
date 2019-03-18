run_tests:
	PYTHONPATH=. pytest -l tests/

cov_tests:
	PYTHONPATH=. pytest --cov-report html:.covreport --cov=tmtoolkit tests/; rm .coverage*

