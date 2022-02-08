import string

from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, array_shapes


def strategy_2d_array(dtype, minval=0, maxval=None, **kwargs):
    if 'min_side' in kwargs:
        min_side = kwargs.pop('min_side')
    else:
        min_side = 1

    if 'max_side' in kwargs:
        max_side = kwargs.pop('max_side')
    else:
        max_side = None

    if dtype is int:
        elems = st.integers(minval, maxval, **kwargs)
    elif dtype is float:
        elems = st.floats(minval, maxval, **kwargs)
    elif dtype is str:
        elems = st.text(min_size=minval, max_size=maxval, **kwargs)
    else:
        raise ValueError('no elements strategy for dtype', dtype)

    return arrays(dtype, array_shapes(min_dims=2, max_dims=2, min_side=min_side, max_side=max_side), elements=elems)


def strategy_dtm():
    return strategy_2d_array(int, 0, 10000)


def strategy_dtm_small():
    return strategy_2d_array(int, 0, 10, min_side=2, max_side=6)


def strategy_2d_prob_distribution():
    return strategy_2d_array(float, 0, 1, allow_nan=False, allow_infinity=False)


def strategy_tokens(*args, **kwargs):
    return st.lists(st.text(*args, **kwargs))


def strategy_lists_of_tokens(*args, **kwargs):
    return st.lists(st.lists(st.text(*args, **kwargs)))


def strategy_texts(*args, **kwargs):
    return st.lists(st.text(*args, **kwargs))


def strategy_texts_printable():
    return strategy_texts(string.printable)


def strategy_str_str_dict(keys_args, keys_kwargs, values_args, values_kwargs):
    return st.dictionaries(st.text(*keys_args, **keys_kwargs), st.text(*values_args, **values_kwargs))


def strategy_str_str_dict_printable():
    return st.dictionaries(st.text(string.printable), st.text(string.printable))
