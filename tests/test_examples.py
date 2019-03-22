"""
pytest tests for Python scripts in "examples" directory.
Each script comes with an expected output text file located at "tests/examples_output".
"""


def _run_script_against_expected_output(script_runner, script):
    script_file = 'examples/%s.py' % script
    output_file = 'tests/examples_output/%s.txt' % script

    with open(output_file) as f:
        expected_output = f.read()

    ret = script_runner.run(script_file)
    assert ret.success
    assert ret.stdout == expected_output


def test_preproc_gen_dtm_de(script_runner):
    _run_script_against_expected_output(script_runner, 'preproc_gen_dtm_de')


def test_preproc_gen_dtm_en(script_runner):
    _run_script_against_expected_output(script_runner, 'preproc_gen_dtm_en')
