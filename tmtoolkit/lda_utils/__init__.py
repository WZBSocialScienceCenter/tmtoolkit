import imp

from . import common

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
