import pytest
import hypothesis.strategies as st
from hypothesis import given

from tmtoolkit.utils import (pickle_data, unpickle_file, require_listlike, require_dictlike, require_types,
                             simplified_pos, filter_elements_in_dict)


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


def test_simplified_pos():
    assert simplified_pos('') is None
    assert simplified_pos('N') == 'N'
    assert simplified_pos('V') == 'V'
    assert simplified_pos('ADJ') == 'ADJ'
    assert simplified_pos('ADV') == 'ADV'
    assert simplified_pos('AD') is None
    assert simplified_pos('ADX') is None
    assert simplified_pos('PRP') is None
    assert simplified_pos('XYZ') is None
    assert simplified_pos('NN') == 'N'
    assert simplified_pos('NNP') == 'N'
    assert simplified_pos('VX') == 'V'
    assert simplified_pos('ADJY') == 'ADJ'
    assert simplified_pos('ADVZ') == 'ADV'

    assert simplified_pos('NNP', tagset='penn') == 'N'
    assert simplified_pos('VFOO', tagset='penn') == 'V'
    assert simplified_pos('JJ', tagset='penn') == 'ADJ'
    assert simplified_pos('JJX', tagset='penn') == 'ADJ'
    assert simplified_pos('RB', tagset='penn') == 'ADV'
    assert simplified_pos('RBFOO', tagset='penn') == 'ADV'


@given(example_list=st.lists(st.text()), example_matches=st.lists(st.booleans()), negate=st.booleans())
def test_filter_elements_in_dict(example_list, example_matches, negate):
    d = {'foo': example_list}
    matches = {'foo': example_matches}

    if len(example_list) != len(example_matches):
        with pytest.raises(ValueError):
            filter_elements_in_dict(d, matches, negate_matches=negate)
    else:
        d_ = filter_elements_in_dict(d, matches, negate_matches=negate)
        if negate:
            n = len(example_matches) - sum(example_matches)
        else:
            n = sum(example_matches)
        assert len(d_['foo']) == n


def test_filter_elements_in_dict_differentkeys():
    with pytest.raises(ValueError):
        filter_elements_in_dict({'foo': []}, {'bar': []})
    filter_elements_in_dict({'foo': []}, {'bar': []}, require_same_keys=False)
