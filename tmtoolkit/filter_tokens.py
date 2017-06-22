# -*- coding: utf-8 -*-
import re

from .utils import simplified_pos, filter_elements_in_dict


def filter_for_token(tokens, search_token, ignore_case=False, remove_found_token=False):
    return filter_for_tokenpattern(tokens, search_token, fixed=True, ignore_case=ignore_case,
                                   remove_found_token=remove_found_token)


def filter_for_tokenpattern(tokens, tokpattern, fixed=False, ignore_case=False, remove_found_token=False,
                            return_matches=False):
    if not tokpattern:
        raise ValueError('empty `tokpattern` passed')

    # compile regular expression
    re_flags = re.IGNORECASE if ignore_case else 0
    if fixed:  # exact match pattern
        pattern = re.compile('^%s$' % re.escape(tokpattern), flags=re_flags)
    else:      # arbitrary regular expression pattern
        pattern = re.compile(tokpattern, flags=re_flags)

    filtered_docs = {}
    matches = {}
    for dl, dt in tokens.items():
        tok_match = [pattern.search(t) is not None for t in dt]   # list of boolean match result per token
        if any(tok_match):   # if any of the tokens matched the patterns, add it to the filtered documents
            if remove_found_token:   # filter `dt` so that only tokens are left that did not match the pattern
                filtered_tokens = [t for t, match in zip(dt, tok_match) if not match]
            else:                    # use all tokens in `dt`
                filtered_tokens = dt

            filtered_docs[dl] = filtered_tokens
            if return_matches:
                matches[dl] = tok_match

    assert len(filtered_docs) <= len(tokens)
    if return_matches:
        assert len(filtered_docs) == len(matches)

    if return_matches:
        return filtered_docs, matches
    else:
        return filtered_docs


def filter_for_pos(tokens, tokens_pos_tags, required_pos,
                   simplify_pos=True, simplify_pos_tagset=None,
                   return_matches=False):
    if type(required_pos) is str:
        required_pos = (required_pos,)

    if simplify_pos:
        simplify_fn = lambda x: simplified_pos(x, tagset=simplify_pos_tagset)
    else:
        simplify_fn = lambda x: x

    matches = {}
    for dl, postags in tokens_pos_tags.items():
        matches[dl] = [simplify_fn(pos) in required_pos for pos in postags]
    filtered_docs = filter_elements_in_dict(tokens, matches)

    assert len(filtered_docs) <= len(tokens)
    assert len(filtered_docs) == len(matches)

    if return_matches:
        return filtered_docs, matches
    else:
        return filtered_docs
