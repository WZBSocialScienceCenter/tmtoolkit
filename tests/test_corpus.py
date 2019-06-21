import string
from random import sample

import pytest
import hypothesis.strategies as st
from hypothesis import given

from tmtoolkit.corpus import path_recursive_split, paragraphs_from_lines, read_text_file, Corpus
from tmtoolkit.preprocess import TMPreproc


TEMP_PICKLE_FILE = '/tmp/tmtoolkit_corpus_test.pickle'


def test_path_recursive_split():
    assert path_recursive_split('') == []
    assert path_recursive_split('/') == []
    assert path_recursive_split('a') == ['a']
    assert path_recursive_split('/a') == ['a']
    assert path_recursive_split('/a/') == ['a']
    assert path_recursive_split('a/') == ['a']
    assert path_recursive_split('a/b') == ['a', 'b']
    assert path_recursive_split('a/b/c') == ['a', 'b', 'c']
    assert path_recursive_split('/a/b/c') == ['a', 'b', 'c']
    assert path_recursive_split('/a/b/c/') == ['a', 'b', 'c']
    assert path_recursive_split('/a/../b/c/') == ['a', '..', 'b', 'c']
    assert path_recursive_split('/a/b/c/d.txt') == ['a', 'b', 'c', 'd.txt']


def test_paragraphs_from_lines():
    with pytest.raises(ValueError):
        paragraphs_from_lines("foo", splitchar=None)

    assert len(paragraphs_from_lines('')) == 0
    assert len(paragraphs_from_lines(' ')) == 0
    assert len(paragraphs_from_lines('\n')) == 0
    assert len(paragraphs_from_lines('\n\n')) == 0
    assert len(paragraphs_from_lines('\n\n\n')) == 0

    pars = paragraphs_from_lines("foo")
    assert len(pars) == 1
    assert pars[0] == "foo"

    testlines1 = u"""

par1 lorem
ipsum

par2 lorem


par3 ipsum
lorem
dorem


par4

"""

    pars = paragraphs_from_lines(testlines1)  # with default break_on_num_newlines=2

    assert len(pars) == 4
    assert pars[0] == 'par1 lorem ipsum'
    assert pars[1] == 'par2 lorem'
    assert pars[2] == 'par3 ipsum lorem dorem'
    assert pars[3] == 'par4'

    assert paragraphs_from_lines(testlines1.split('\n'), splitchar=None) == pars

    pars = paragraphs_from_lines(testlines1, break_on_num_newlines=1)
    assert len(pars) == 7
    assert pars[0] == 'par1 lorem'
    assert pars[1] == 'ipsum'
    assert pars[6] == 'par4'

    pars = paragraphs_from_lines(testlines1, break_on_num_newlines=3)
    assert len(pars) == 3
    assert pars[0] == 'par1 lorem ipsum par2 lorem'
    assert pars[1] == 'par3 ipsum lorem dorem'
    assert pars[2] == 'par4'


@given(st.text(string.printable))
def test_paragraphs_from_lines_hypothesis(lines):
    pars = paragraphs_from_lines(lines)
    assert len(pars) <= len(lines)
    assert all(len(p) > 0 for p in pars)


@given(st.lists(st.text(string.printable)))
def test_paragraphs_from_lines_already_split_hypothesis(lines):
    pars = paragraphs_from_lines(lines, splitchar=None)
    assert len(pars) <= len(lines)
    assert all(len(p) > 0 for p in pars)


def test_read_text_file():
    contents = read_text_file('examples/data/gutenberg/kafka_verwandlung.txt', encoding='utf-8')
    assert len(contents) > 0
    contents = read_text_file('examples/data/gutenberg/kafka_verwandlung.txt', encoding='utf-8', read_size=100)
    assert 0 < len(contents) <= 100


def test_empty_corpora():
    c1 = Corpus()
    c2 = Corpus.from_files([])
    c3 = Corpus.from_files([]).add_files([])
    assert c1.docs == c2.docs == c3.docs == {}


def test_corpus_dict_methods():
    c = Corpus()
    assert len(c) == 0
    with pytest.raises(KeyError):
        x = c['x']

    with pytest.raises(KeyError):
        c[1] = 'abc'

    with pytest.raises(KeyError):
        c[''] = 'abc'

    with pytest.raises(ValueError):
        c['d1'] = None

    c['d1'] = 'd1 text'
    assert len(c) == 1
    assert 'd1' in c
    assert set(c.keys()) == {'d1'}
    assert c['d1'] == 'd1 text'

    c['d2'] = 'd2 text'
    assert len(c) == 2
    for dl in c:
        assert dl in {'d1', 'd2'}
    assert set(c.keys()) == {'d1', 'd2'}

    for dl, dt in c.items():
        assert dl in {'d1', 'd2'}
        assert c[dl] == dt

    with pytest.raises(KeyError):
        del c['d3']

    del c['d1']
    assert len(c) == 1
    assert set(c.keys()) == {'d2'}

    del c['d2']
    assert len(c) == 0
    assert set(c.keys()) == set()


@given(texts=st.lists(st.text()))
def test_corpus_copy(texts):
    c1 = Corpus({str(i): t for i, t in enumerate(texts)})
    c2 = c1.copy()

    assert c1.docs is not c2.docs
    assert c1.docs == c2.docs

    assert c1.doc_paths is not c2.doc_paths
    assert c1.doc_paths == c2.doc_paths

    assert c1.doc_labels == c2.doc_labels
    assert c1.doc_lengths == c2.doc_lengths
    assert c1.unique_characters == c2.unique_characters


def test_corpus_n_docs():
    c = Corpus({'a': 'Doc A text', 'b': 'Doc B text', 'c': 'Doc C text'})
    assert c.n_docs == len(c) == 3


def test_corpus_doc_labels():
    c = Corpus({'a': 'Doc A text', 'b': 'Doc B text', 'c': 'Doc C text'})
    assert isinstance(c.doc_labels, list)
    assert list(c.doc_labels) == c.get_doc_labels(sort=True) == list('abc')


def test_corpus_doc_lengths():
    c = Corpus({'a': '1', 'b': '22', 'c': '333'})
    assert isinstance(c.doc_lengths, dict)
    assert c.doc_lengths == {'a': 1, 'b': 2, 'c': 3}


@given(texts=st.lists(st.text()))
def test_corpus_unique_characters(texts):
    all_chars = set(''.join(texts))

    c = Corpus({str(i): t for i, t in enumerate(texts)})
    res_chars = c.unique_characters
    assert isinstance(res_chars, set)
    assert res_chars == all_chars


def test_corpus_add_doc():
    c = Corpus()
    with pytest.raises(ValueError):
        c.add_doc('', 'x')
    with pytest.raises(ValueError):
        c.add_doc(123, 'x')
    with pytest.raises(ValueError):
        c.add_doc('d1', None)

    c.add_doc('d1', 'd1 text')
    with pytest.raises(ValueError):
        c.add_doc('d1', 'd1 text')

    c.add_doc('d2', '')

    assert set(c.keys()) == {'d1', 'd2'}


def test_corpus_from_files():
    doc_path = 'examples/data/gutenberg/kafka_verwandlung.txt'
    c1 = Corpus.from_files([doc_path])
    c2 = Corpus().add_files([doc_path])

    assert len(c1.docs) == len(c1.doc_paths) == 1
    assert len(c2.docs) == len(c2.doc_paths) == 1
    assert c1.docs.keys() == c2.docs.keys() == c1.doc_paths.keys() == c2.doc_paths.keys()

    only_doc_label = next(iter(c1.docs.keys()))
    assert only_doc_label.endswith('kafka_verwandlung')

    only_doc = c1.docs[only_doc_label]
    assert len(only_doc) > 0

    assert c1.doc_paths[only_doc_label] == doc_path


def test_corpus_from_files2():
    c = Corpus.from_files(['examples/data/gutenberg/werther/goethe_werther1.txt',
                           'examples/data/gutenberg/werther/goethe_werther2.txt'])
    assert len(c.docs) == len(c.doc_paths) == 2

    for k, d in c.docs.items():
        assert k[:-1].endswith('goethe_werther')
        assert len(d) > 0


def test_corpus_from_files_nonlist_arg():
    with pytest.raises(ValueError):
        Corpus.from_files('wrong')


def test_corpus_from_files_not_existent():
    with pytest.raises(IOError):
        Corpus.from_files(['examples/data/gutenberg/werther/goethe_werther1.txt',
                           'not_existent'])


def test_corpus_from_folder():
    c1 = Corpus({'a': '1', 'b': '22', 'c': '333'})
    c1.to_pickle(TEMP_PICKLE_FILE)

    c2 = Corpus.from_pickle(TEMP_PICKLE_FILE)

    assert c1.docs == c2.docs


def test_corpus_from_folder_valid_ext():
    assert len(Corpus.from_folder('examples/data/gutenberg', valid_extensions='txt').docs) == 3
    assert len(Corpus.from_folder('examples/data/gutenberg', valid_extensions='foo').docs) == 0
    assert len(Corpus.from_folder('examples/data/gutenberg', valid_extensions=('foo', 'txt')).docs) == 3


def test_corpus_from_folder_not_existent():
    with pytest.raises(IOError):
        Corpus.from_folder('not_existent')


def test_corpus_pickle():
    c = Corpus.from_folder('examples/data/gutenberg')
    assert len(c.docs) == 3


def test_corpus_get_doc_labels():
    c = Corpus.from_folder('examples/data/gutenberg')
    assert set(c.docs.keys()) == set(c.get_doc_labels())


def test_corpus_sample():
    c = Corpus.from_folder('examples/data/gutenberg')
    n_docs_orig = c.n_docs

    sampled_docs = c.sample(2)
    assert isinstance(sampled_docs, dict)
    assert len(sampled_docs) == 2
    assert c.n_docs == n_docs_orig

    assert isinstance(c.sample(2, inplace=True), Corpus)
    assert c.n_docs == 2


def test_corpus_filter_by_min_length():
    c = Corpus.from_folder('examples/data/gutenberg', force_unix_linebreaks=False)
    assert len(c.filter_by_min_length(1).docs) == 3
    assert len(c.filter_by_min_length(142694).docs) == 1
    assert len(c.filter_by_min_length(142695).docs) == 0
    assert len(c.filter_by_min_length(1).docs) == 0


def test_corpus_filter_by_max_length():
    c = Corpus.from_folder('examples/data/gutenberg', force_unix_linebreaks=False)
    assert len(c.filter_by_max_length(999999).docs) == 3
    assert len(c.filter_by_max_length(142694).docs) == 3
    assert len(c.filter_by_max_length(142693).docs) == 2
    assert len(c.filter_by_max_length(0).docs) == 0
    assert len(c.filter_by_max_length(999999).docs) == 0


def test_corpus_split_by_paragraphs():
    c = Corpus.from_folder('examples/data/gutenberg', doc_label_fmt='{basename}')

    orig_docs = c.docs
    orig_doc_paths = c.doc_paths
    c.split_by_paragraphs()
    par_docs = c.docs

    assert len(par_docs) >= len(orig_docs)
    assert len(set(orig_doc_paths.values())) == len(set(c.doc_paths.values()))

    for k, d in orig_docs.items():
        assert k in ('goethe_werther1', 'goethe_werther2', 'kafka_verwandlung')
        pars = [par_docs[par_k] for par_k in sorted(par_docs.keys()) if par_k.startswith(k)]
        assert len(pars) > 0

        pars_ = paragraphs_from_lines(d)
        assert len(pars_) == len(pars)
        assert set(pars_) == set(pars)


def test_corpus_split_by_paragraphs_rejoin():
    # TODO: better tests here
    c = Corpus.from_folder('examples/data/gutenberg', doc_label_fmt='{basename}')
    c2 = Corpus.from_folder('examples/data/gutenberg', doc_label_fmt='{basename}')

    orig_docs = c.docs
    #par_docs = c.split_by_paragraphs().docs
    par_docs_joined = c2.split_by_paragraphs(join_paragraphs=5).docs

    assert len(par_docs_joined) >= len(orig_docs)

    for k, d in orig_docs.items():
        assert k in ('goethe_werther1', 'goethe_werther2', 'kafka_verwandlung')
        pars = [par_docs_joined[par_k] for par_k in sorted(par_docs_joined.keys()) if par_k.startswith(k)]
        assert len(pars) > 0


@given(texts=st.lists(st.text()))
def test_corpus_apply(texts):
    c = Corpus({str(i): t for i, t in enumerate(texts)})
    c_orig = c.copy()
    orig_doc_labels = c.doc_labels
    orig_doc_lengths = c.doc_lengths

    assert isinstance(c.apply(str.upper), Corpus)

    assert c.doc_labels == orig_doc_labels
    assert c.doc_lengths == orig_doc_lengths

    for dl, dt in c.items():
        assert c_orig[dl].upper() == dt


@given(texts=st.lists(st.text()))
def test_corpus_filter_characters(texts):
    c = Corpus({str(i): t for i, t in enumerate(texts)})
    c_orig = c.copy()

    orig_doc_labels = c.doc_labels
    orig_doc_lengths = c.doc_lengths
    orig_uniq_chars = c.unique_characters

    assert isinstance(c.filter_characters(orig_uniq_chars), Corpus)
    assert c.doc_labels == orig_doc_labels
    assert c.doc_lengths == orig_doc_lengths
    assert c.unique_characters == orig_uniq_chars

    not_in_corpus_chars = set(string.printable) - orig_uniq_chars
    if len(not_in_corpus_chars) > 0:
        c.filter_characters(not_in_corpus_chars)
        assert c.doc_labels == orig_doc_labels
        assert c.doc_lengths == {dl: 0 for dl in c.doc_labels}
        assert c.unique_characters == set()

    c = c_orig.copy()
    c.filter_characters(set())
    assert c.doc_labels == orig_doc_labels
    assert c.doc_lengths == {dl: 0 for dl in c.doc_labels}
    assert c.unique_characters == set()

    if len(orig_uniq_chars) > 3:
        c = c_orig.copy()
        only_chars = set(sample(list(orig_uniq_chars), 3))
        c.filter_characters(only_chars)
        assert c.doc_labels == orig_doc_labels
        assert c.doc_lengths != orig_doc_lengths
        assert c.unique_characters == only_chars

        c = c_orig.copy()
        only_chars = set(sample(list(orig_uniq_chars), 3))
        c.filter_characters(''.join(only_chars))  # as char sequence
        assert c.doc_labels == orig_doc_labels
        assert c.doc_lengths != orig_doc_lengths
        assert c.unique_characters == only_chars


def test_corpus_replace_characters_simple():
    c = Corpus({'doc1': 'ABC', 'doc2': 'abcDeF'})
    c.replace_characters({'a': None, 'C': 'c', 'e': ord('X')})

    assert c.docs == {
        'doc1': 'ABc',
        'doc2': 'bcDXF',
    }

    c.replace_characters({ord('A'): None})

    assert c.docs == {
        'doc1': 'Bc',
        'doc2': 'bcDXF',
    }

    c.replace_characters(str.maketrans('DXFY', '1234'))

    assert c.docs == {
        'doc1': 'Bc',
        'doc2': 'bc123',
    }

    c.replace_characters({})

    assert c.docs == {
        'doc1': 'Bc',
        'doc2': 'bc123',
    }


def test_corpus_pass_tmpreproc():
    c = Corpus()
    c['doc1'] = 'A simple example in simple English.'
    c['doc2'] = 'It contains only three very simple documents.'
    c['doc3'] = 'Simply written documents are very brief.'

    preproc = TMPreproc(c)
    tok = preproc.tokens
    assert set(tok.keys()) == set(c.keys())
    assert len(tok['doc1']) == 7
