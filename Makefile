run_tests:
	PYTHONPATH=. pytest -l tests/

cov_tests:
	PYTHONPATH=. pytest --cov-report html:.covreport --cov=tmtoolkit tests/; rm .coverage*

sdist:
	python setup.py sdist

wheel:
	python setup.py bdist_wheel

readme:
	cat doc/source/intro.rst doc/source/install.rst > README.rst
	echo >> README.rst
	echo >> README.rst
	cat doc/source/license_note.rst >> README.rst

