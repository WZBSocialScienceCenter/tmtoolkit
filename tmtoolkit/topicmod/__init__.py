# -*- coding: utf-8 -*-
"""
Topic modeling sub-package with modules for model evaluation, model I/O, model statistics, parallel computation and
visualization. Functions and classes in `tm_gensim`, `tm_lda` and `tm_sklearn` implement parallel processing with
popular topic modeling packages.

Markus Konrad <markus.konrad@wzb.eu>
"""


import imp

from . import evaluate, model_io, model_stats, parallel, visualize

# conditional imports

# lda package
try:
    imp.find_module('lda')
    from . import tm_lda
except: pass

# sklearn package
try:
    imp.find_module('sklearn')
    from . import tm_sklearn
except: pass

# gensim package
try:
    imp.find_module('gensim')
    from . import tm_gensim
except: pass
