# -*- coding: utf-8 -*-
import re

import six

from .utils import simplified_pos


def filter_for_token(tokens, search_token, ignore_case=False, remove_found_token=False, remove_empty_docs=True):
    return filter_for_tokenpattern(tokens, search_token, fixed=True, ignore_case=ignore_case,
                                   remove_found_token=remove_found_token, remove_empty_docs=remove_empty_docs)


def filter_for_tokenpattern(tokens, tokpattern, fixed=False, ignore_case=False, remove_found_token=False,
                            remove_empty_docs=True):
    if not tokpattern:
        raise ValueError('empty `tokpattern` passed')

    # compile regular expression
    re_flags = re.IGNORECASE if ignore_case else 0
    if fixed:  # exact match pattern
        pattern = re.compile('^%s$' % re.escape(tokpattern), flags=re_flags)
    else:      # arbitrary regular expression pattern
        pattern = re.compile(tokpattern, flags=re_flags)

    filtered_docs = {}
    for dl, dt in tokens.items():
        tok_match = [pattern.search(tup[0]) is not None for tup in dt]   # list of boolean match result per token
        if any(tok_match):   # if any of the tokens matched the patterns, add it to the filtered documents
            if remove_found_token:   # filter `dt` so that only tokens are left that did not match the pattern
                filtered_tokens = [tup for tup, match in zip(dt, tok_match) if not match]
            else:                    # use all tokens in `dt`
                filtered_tokens = dt

            filtered_docs[dl] = filtered_tokens
        elif not remove_empty_docs:
            filtered_docs[dl] = []

    if remove_empty_docs:
        assert len(filtered_docs) <= len(tokens)
    else:
        assert len(filtered_docs) == len(tokens)

    return filtered_docs


def filter_for_pos(tokens, required_pos, simplify_pos=True, simplify_pos_tagset=None):
    if isinstance(required_pos, six.string_types):
        required_pos = (required_pos,)

    if simplify_pos:
        simplify_fn = lambda x: simplified_pos(x, tagset=simplify_pos_tagset)
    else:
        simplify_fn = lambda x: x

    filtered_docs = {}
    for dl, dt in tokens.items():
        filtered_docs[dl] = [tup for tup in dt if simplify_fn(tup[1]) in required_pos]   # tup[1] is the POS for a token

    assert len(filtered_docs) == len(tokens)

    return filtered_docs
