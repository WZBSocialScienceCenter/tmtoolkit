# -*- coding: utf-8 -*-
import pickle

from nltk.corpus import wordnet as wn


def pickle_data(data, picklefile):
    """Helper function to pickle `data` in `picklefile`."""
    with open(picklefile, 'wb') as f:
        pickle.dump(data, f, protocol=2)


def unpickle_file(picklefile):
    """Helper function to unpickle data from `picklefile`."""
    with open(picklefile, 'rb') as f:
        return pickle.load(f)


def require_types(x, valid_types):
    if all(isinstance(x, t) is False for t in valid_types):
        raise ValueError('requires type:', str(valid_types))


def require_listlike(x):
    require_types(x, (set, tuple, list))


def require_dictlike(x):
    require_types(x, (dict,))


def pos_tag_convert_penn_to_wn(tag):
    if tag in ['JJ', 'JJR', 'JJS']:
        return wn.ADJ
    elif tag in ['RB', 'RBR', 'RBS']:
        return wn.ADV
    elif tag in ['NN', 'NNS', 'NNP', 'NNPS']:
        return wn.NOUN
    elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return wn.VERB
    return None


def simplified_wn_pos(pos):
    """
    Return a simplified POS tag for a full WordNet POS tag `pos`.
    Does the following conversion:
    - all N... (noun) tags to 'N'
    - all V... (verb) tags to 'V'
    - all ADJ... (adjective) tags to 'ADJ'
    - all ADV... (adverb) tags to 'ADV'
    - all other to None
    """
    if pos.startswith('N') or pos.startswith('V'):
        return pos[0]
    elif pos.startswith('ADJ') or pos.startswith('ADV'):
        return pos[:3]
    else:
        return None


def filter_elements_in_dict(d, matches, negate_matches=False, require_same_keys=True):
    """
    For an input dict `d` with key K -> elements list E, and a dict `matches` with K -> match list M, where M is list
    of booleans that denote which element to take from E, this function will return a dict `d_` that for each K in
    `matches` only contains the elements from `d` that were marked with True in the respective match list M.
    """
    d_ = {}
    for key, takelist in matches.items():
        if key not in d:
            if require_same_keys:
                raise ValueError("`d` and `matches` must contain the same dict keys. Key '%s' not found in `d`." % key)
            else:
                continue

        if negate_matches:
            takelist = [not take for take in takelist]

        if len(d[key]) != len(takelist):
            raise ValueError("number of elements in input list is inequal to number of elements in takelist for key '%s'" % key)

        filtered = [x for x, take in zip(d[key], takelist) if take]
        assert len(filtered) == sum(takelist)
        d_[key] = filtered

    return d_
