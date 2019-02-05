"""
Topic modeling sub-package with modules for model evaluation, model I/O, model statistics, parallel computation and
visualization. Functions and classes in `tm_gensim`, `tm_lda` and `tm_sklearn` implement parallel processing with
popular topic modeling packages.

Markus Konrad <markus.konrad@wzb.eu>
"""


import importlib.util

from . import evaluate, model_io, model_stats, parallel, visualize

# conditional imports

# lda package
if importlib.util.find_spec('lda'):
    from . import tm_lda

# sklearn package
if importlib.util.find_spec('sklearn'):
    from . import tm_sklearn

# gensim package
if importlib.util.find_spec('gensim'):
    from . import tm_gensim
