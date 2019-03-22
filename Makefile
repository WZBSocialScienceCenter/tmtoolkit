run_tests:
	PYTHONPATH=. pytest -l tests/

cov_tests:
	PYTHONPATH=. pytest --cov-report html:.covreport --cov=tmtoolkit tests/; rm .coverage*

example_preproc_gen_dtm_de:
	PYTHONPATH=. python examples/preproc_gen_dtm_de.py | tee tests/examples_output/preproc_gen_dtm_de.txt

example_preproc_gen_dtm_en:
	PYTHONPATH=. python examples/preproc_gen_dtm_en.py | tee tests/examples_output/preproc_gen_dtm_en.txt

