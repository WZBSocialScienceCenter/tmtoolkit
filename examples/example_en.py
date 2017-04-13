from tmprep.preprocess import TMPreproc

corpus = {
    'doc1': 'A simple example in English',
    'doc2': 'With only two documents',
}

preproc = TMPreproc(corpus)

preproc.load_stemmer()
