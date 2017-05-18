import pytest
import hypothesis.strategies as st
from hypothesis import given, example

from tmtoolkit.utils import (pickle_data, unpickle_file, require_listlike, require_dictlike, require_types,
                             remove_tokens_from_list)


@given(st.lists(st.text()))
def test_remove_tokens_from_list1(l):
    assert len(l) == len(remove_tokens_from_list(l, []))


@given(st.lists(st.text()))
def test_remove_tokens_from_list2(l):
    assert len(remove_tokens_from_list(l, l)) == 0


@given(st.lists(st.text()), st.lists(st.text()))
@example(list('abc'), list('c'))
def test_remove_tokens_from_list3(l, rm):
    l_ = remove_tokens_from_list(l, rm)

    assert len(l_) <= len(l)
    assert all(t not in rm for t in l_)


def test_pickle_unpickle():
    pfile = 'tests/data/test_pickle_unpickle.pickle'
    input_data = ('foo', 123, [])
    pickle_data(input_data, pfile)

    output_data = unpickle_file(pfile)

    for i, o in zip(input_data, output_data):
        assert i == o


def test_require_listlike():
    require_listlike([])
    require_listlike([123])
    require_listlike(tuple())
    require_listlike((1, 2, 3))
    require_listlike(set())
    require_listlike({1, 2, 3})

    with pytest.raises(ValueError): require_listlike({})
    with pytest.raises(ValueError): require_listlike({'x': 'y'})
    with pytest.raises(ValueError): require_listlike('a string')


def test_require_dictlike():
    from collections import  OrderedDict
    require_dictlike({})
    require_dictlike(OrderedDict())

    with pytest.raises(ValueError): require_dictlike(set())


def test_require_types():
    types = (set, tuple, list, dict)
    for t in types:
        require_types(t(), (t, ))

    types_shifted = types[1:] + types[:1]

    for t1, t2 in zip(types, types_shifted):
        with pytest.raises(ValueError): require_types(t1, (t2, ))
