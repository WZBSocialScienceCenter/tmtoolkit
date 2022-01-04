"""
Topic modeling sub-package with modules for model evaluation, model I/O, model statistics, parallel computation and
visualization.

Functions and classes in :mod:`~tmtoolkit.topicmod.tm_gensim`, :mod:`~tmtoolkit.topicmod.tm_lda` and
:mod:`~tmtoolkit.topicmod.tm_sklearn` implement parallel model computation and evaluation using popular topic modeling
packages. You need to install the respective packages (*lda*, *scikit-learn* or *gensim*) in order to use them.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
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
