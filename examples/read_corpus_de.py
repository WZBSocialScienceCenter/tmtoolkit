"""
An example for constructing a corpus of texts from files and passing them to the preprocessing step.
"""
from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc


if __name__ == '__main__':   # this is necessary for multiprocessing on Windows!
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

    preproc = TMPreproc(corpus.docs, language='german')
    preproc.tokenize().tokens_to_lowercase()

    print("tokenized first 5 paragraphs of Werther:")
    for par_num in range(1, 6):
        doclabel = 'werther-goethe_werther1-%d' % par_num
        print("par%d (document label '%s'):" % (par_num, doclabel))
        print(preproc.tokens[doclabel])


    preproc.generate_ngrams(2, join=False).use_ngrams_as_tokens(join=True)

    print("bigrams from first 5 paragraphs of Werther:")
    for par_num in range(1, 6):
        doclabel = 'werther-goethe_werther1-%d' % par_num
        print("par%d (document label '%s'):" % (par_num, doclabel))
        print(preproc.tokens[doclabel])

