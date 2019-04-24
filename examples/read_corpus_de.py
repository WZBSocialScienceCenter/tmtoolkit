"""
An example for constructing a corpus of texts from files and passing them to the preprocessing step.

**Important note for Windows users:**
You need to wrap all of the following code in a `if __name__ == '__main__'` block (just as in `lda_evaluation.py`).
"""
from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc

#%%

# if __name__ == '__main__':   # this is necessary for multiprocessing on Windows!

corpus = Corpus.from_folder('data/gutenberg')
print("all loaded documents:")
print(corpus.docs.keys())
print("-----")

corpus.split_by_paragraphs()
print("documents split into paragraphs")
print(corpus.docs.keys())
print("-----")

print("first 5 paragraphs of Werther:")
for par_num in range(1, 6):
    doclabel = 'werther-goethe_werther1-%d' % par_num
    print("par%d (document label '%s'):" % (par_num, doclabel))
    print(corpus.docs[doclabel])
print("-----")

#%%

preproc = TMPreproc(corpus, language='german')
preproc.tokens_to_lowercase()

#%%

print("tokenized first 5 paragraphs of Werther, lowercase:")
for par_num in range(1, 6):
    doclabel = 'werther-goethe_werther1-%d' % par_num
    print("par%d (document label '%s'):" % (par_num, doclabel))
    print(preproc.tokens[doclabel])

#%%

preproc.generate_ngrams(2)
preproc.use_joined_ngrams_as_tokens()

#%%

print("bigrams from first 5 paragraphs of Werther:")
for par_num in range(1, 6):
    doclabel = 'werther-goethe_werther1-%d' % par_num
    print("par%d (document label '%s'):" % (par_num, doclabel))
    print(preproc.tokens[doclabel])

