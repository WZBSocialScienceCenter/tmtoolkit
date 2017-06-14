# -*- coding: utf-8 -*-
import re

from .utils import simplified_wn_pos


def filter_for_token(tokens, search_token, ignore_case=False, remove_found_token=False):
    return filter_for_tokenpattern(tokens, search_token, fixed=True, ignore_case=ignore_case,
                                   remove_found_token=remove_found_token)


def filter_for_tokenpattern(tokens, tokpattern, fixed=False, ignore_case=False, remove_found_token=False):
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
        tok_match = [pattern.search(t) is not None for t in dt]   # list of boolean match result per token
        if any(tok_match):   # if any of the tokens matched the patterns, add it to the filtered documents
            if remove_found_token:   # filter `dt` so that only tokens are left that did not match the pattern
                filtered_tokens = [t for t, match in zip(dt, tok_match) if not match]
            else:                    # use all tokens in `dt`
                filtered_tokens = dt

            filtered_docs[dl] = filtered_tokens

    assert len(filtered_docs) <= len(tokens)

    return filtered_docs


def filter_for_pos(tokens, tokens_pos_tags, required_pos, simplify_wn_pos=True):
    if type(required_pos) is str:
        required_pos = (required_pos,)

    if simplify_wn_pos:
        simplify_fn = simplified_wn_pos
    else:
        simplify_fn = lambda x: x

    filtered_docs = {}
    for dl, dt in tokens.items():
        dpos = tokens_pos_tags[dl]
        if len(dpos) != len(dt):
            raise ValueError("number of tokens does not match number of POS tags for document '%s'" % dl)

        filtered_docs[dl] = [t for t, pos in zip(dt, dpos) if simplify_fn(pos) in required_pos]

    assert len(filtered_docs) <= len(tokens)

    return filtered_docs
