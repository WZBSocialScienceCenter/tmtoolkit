import string

import pytest
import hypothesis.strategies as st
from hypothesis import given

from tmtoolkit.corpus import path_recursive_split, paragraphs_from_lines, read_full_file, Corpus


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
        paragraphs_from_lines(u"foo", splitchar=None)

    assert len(paragraphs_from_lines('')) == 0
    assert len(paragraphs_from_lines(' ')) == 0
    assert len(paragraphs_from_lines('\n')) == 0
    assert len(paragraphs_from_lines('\n\n')) == 0
    assert len(paragraphs_from_lines('\n\n\n')) == 0

    pars = paragraphs_from_lines(u"foo")
    assert len(pars) == 1
    assert pars[0] == u"foo"

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
    assert pars[0] == u'par1 lorem ipsum'
    assert pars[1] == u'par2 lorem'
    assert pars[2] == u'par3 ipsum lorem dorem'
    assert pars[3] == u'par4'

    assert paragraphs_from_lines(testlines1.split('\n'), splitchar=None) == pars

    pars = paragraphs_from_lines(testlines1, break_on_num_newlines=1)
    assert len(pars) == 7
    assert pars[0] == u'par1 lorem'
    assert pars[1] == u'ipsum'
    assert pars[6] == u'par4'

    pars = paragraphs_from_lines(testlines1, break_on_num_newlines=3)
    assert len(pars) == 3
    assert pars[0] == u'par1 lorem ipsum par2 lorem'
    assert pars[1] == u'par3 ipsum lorem dorem'
    assert pars[2] == u'par4'


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


def test_read_full_file():
    contents = read_full_file('examples/data/gutenberg/kafka_verwandlung.txt', encoding='utf-8')
    assert len(contents) > 0
    contents = read_full_file('examples/data/gutenberg/kafka_verwandlung.txt', encoding='utf-8', read_size=100)
    assert 0 < len(contents) <= 100


def test_empty_corpora():
    c1 = Corpus()
    c2 = Corpus.from_files([])
    c3 = Corpus.from_files([]).add_files([])
    assert c1.docs == c2.docs == c3.docs == {}


def test_corpus_from_files():
    c1 = Corpus.from_files(['examples/data/gutenberg/kafka_verwandlung.txt'])
    c2 = Corpus().add_files(['examples/data/gutenberg/kafka_verwandlung.txt'])

    assert len(c1.docs) == 1
    assert len(c2.docs) == 1
    assert c1.docs.keys() == c2.docs.keys()

    only_doc_label = next(iter(c1.docs.keys()))
    assert only_doc_label.endswith('kafka_verwandlung')

    only_doc = c1.docs[only_doc_label]
    assert len(only_doc) > 0


def test_corpus_from_files2():
    c = Corpus.from_files(['examples/data/gutenberg/werther/goethe_werther1.txt',
                           'examples/data/gutenberg/werther/goethe_werther2.txt'])
    assert len(c.docs) == 2

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
    c = Corpus.from_folder('examples/data/gutenberg')
    assert len(c.docs) == 3


def test_corpus_from_folder_valid_ext():
    assert len(Corpus.from_folder('examples/data/gutenberg', valid_extensions='txt').docs) == 3
    assert len(Corpus.from_folder('examples/data/gutenberg', valid_extensions='foo').docs) == 0
    assert len(Corpus.from_folder('examples/data/gutenberg', valid_extensions=('foo', 'txt')).docs) == 3


def test_corpus_from_folder_not_existent():
    with pytest.raises(IOError):
        Corpus.from_folder('not_existent')


def test_corpus_get_doc_labels():
    c = Corpus.from_folder('examples/data/gutenberg')
    assert set(c.docs.keys()) == set(c.get_doc_labels())


def test_corpus_sample():
    c = Corpus.from_folder('examples/data/gutenberg')
    assert len(c.sample(2).docs) == 2


def test_corpus_filter_by_min_length():
    c = Corpus.from_folder('examples/data/gutenberg')
    assert len(c.filter_by_min_length(1).docs) == 3
    assert len(c.filter_by_min_length(142694).docs) == 1
    assert len(c.filter_by_min_length(142695).docs) == 0
    assert len(c.filter_by_min_length(1).docs) == 0


def test_corpus_filter_by_max_length():
    c = Corpus.from_folder('examples/data/gutenberg')
    assert len(c.filter_by_max_length(999999).docs) == 3
    assert len(c.filter_by_max_length(142694).docs) == 3
    assert len(c.filter_by_max_length(142693).docs) == 2
    assert len(c.filter_by_max_length(0).docs) == 0
    assert len(c.filter_by_max_length(999999).docs) == 0


def test_corpus_split_by_paragraphs():
    c = Corpus.from_folder('examples/data/gutenberg', doc_label_fmt=u'{basename}')

    orig_docs = c.docs
    par_docs = c.split_by_paragraphs().docs

    assert len(par_docs) >= len(orig_docs)

    for k, d in orig_docs.items():
        assert k in ('goethe_werther1', 'goethe_werther2', 'kafka_verwandlung')
        pars = [par_docs[par_k] for par_k in sorted(par_docs.keys()) if par_k.startswith(k)]
        assert len(pars) > 0

        pars_ = paragraphs_from_lines(d)
        assert len(pars_) == len(pars)
        assert set(pars_) == set(pars)


def test_corpus_split_by_paragraphs_rejoin():
    # TODO: better tests here
    c = Corpus.from_folder('examples/data/gutenberg', doc_label_fmt=u'{basename}')
    c2 = Corpus.from_folder('examples/data/gutenberg', doc_label_fmt=u'{basename}')

    orig_docs = c.docs
    #par_docs = c.split_by_paragraphs().docs
    par_docs_joined = c2.split_by_paragraphs(join_paragraphs=5).docs

    assert len(par_docs_joined) >= len(orig_docs)

    for k, d in orig_docs.items():
        assert k in ('goethe_werther1', 'goethe_werther2', 'kafka_verwandlung')
        pars = [par_docs_joined[par_k] for par_k in sorted(par_docs_joined.keys()) if par_k.startswith(k)]
        assert len(pars) > 0
