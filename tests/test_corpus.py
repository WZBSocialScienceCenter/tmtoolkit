import pytest

from tmtoolkit.corpus import path_recursive_split, paragraphs_from_lines, Corpus


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
        paragraphs_from_lines(u"foo")

    assert len(paragraphs_from_lines([])) == 0
    assert len(paragraphs_from_lines([u""])) == 0
    assert len(paragraphs_from_lines([u"", u"", u""])) == 0

    pars = paragraphs_from_lines([u"foo"])
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

""".split('\n')

    pars = paragraphs_from_lines(testlines1)  # with default break_on_num_newlines=2

    assert len(pars) == 4
    assert pars[0] == u'par1 lorem ipsum'
    assert pars[1] == u'par2 lorem'
    assert pars[2] == u'par3 ipsum lorem dorem'
    assert pars[3] == u'par4'

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


def test_empty_corpora():
    c1 = Corpus()
    c2 = Corpus.from_files([])
    c3 = Corpus.from_files([]).add_files([])
    assert c1.docs == c2.docs == c3.docs == {}
