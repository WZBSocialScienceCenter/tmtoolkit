import pytest

from tmtoolkit.corpus import path_recursive_split, paragraphs_from_lines


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


