"""
Module with default settings that are mostly used in the functional preprocessing API (see :mod:`tmtoolkit.preprocess`)
and which can be changed during runtime, e.g.::

    import tmtoolkit

    tmtoolkit.defaults.language = 'de'
    # -> the language parameter in `tokenize` is German by default now:
    tmtoolkit.preprocess.tokenize(['Ein Dokument auf Deutsch.'])
"""

language = 'en'
