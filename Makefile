run_tests:
	PYTHONPATH=. pytest -l tests/

cov_tests:
	PYTHONPATH=. pytest --cov-report html:.covreport --cov=tmtoolkit tests/; rm .coverage*

example_preproc_gen_dtm_de:
	PYTHONPATH=. python examples/preproc_gen_dtm_de.py | tee tests/examples_output/preproc_gen_dtm_de.txt

example_preproc_gen_dtm_en:
	PYTHONPATH=. python examples/preproc_gen_dtm_en.py | tee tests/examples_output/preproc_gen_dtm_en.txt

example_read_corpus_de:
	cd examples; PYTHONPATH=.. python read_corpus_de.py | tee ../tests/examples_output/read_corpus_de.txt; cd ..

sdist:
	python setup.py sdist

wheel:
	python setup.py bdist_wheel

readme:
	cat doc/source/intro.rst doc/source/install.rst doc/source/license_note.rst > README.rst

