import lda
import numpy as np

from tmtoolkit.topicmod import model_io


def test_save_load_ldamodel_pickle():
    pfile = 'tests/data/test_pickle_unpickle_ldamodel.pickle'

    dtm = np.array([[0, 1], [2, 3], [4, 5], [6, 0]])
    doc_labels = ['doc_' + str(i) for i in range(dtm.shape[0])]
    vocab = ['word_' + str(i) for i in range(dtm.shape[1])]

    model = lda.LDA(2, n_iter=1)
    model.fit(dtm)

    model_io.save_ldamodel_to_pickle(pfile, model, vocab, doc_labels)

    unpickled = model_io.load_ldamodel_from_pickle(pfile)

    assert np.array_equal(model.doc_topic_, unpickled['model'].doc_topic_)
    assert np.array_equal(model.topic_word_, unpickled['model'].topic_word_)
    assert vocab == unpickled['vocab']
    assert doc_labels == unpickled['doc_labels']