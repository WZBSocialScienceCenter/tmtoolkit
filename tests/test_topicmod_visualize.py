import os

import six
import PIL

from tmtoolkit.topicmod import model_io, visualize


try:
    from wordcloud import WordCloud


    def test_generate_wordclouds_for_topic_words():
        py3file = '.py3' if six.PY3 else ''
        data = model_io.load_ldamodel_from_pickle('tests/data/tiny_model_reuters_5_topics%s.pickle' % py3file)
        model = data['model']
        vocab = data['vocab']

        phi = model.topic_word_
        assert phi.shape == (5, len(vocab))

        topic_word_clouds = visualize.generate_wordclouds_for_topic_words(phi, vocab, 10)
        assert len(topic_word_clouds) == 5
        assert set(topic_word_clouds.keys()) == set('topic_%d' % i for i in range(1, 6))
        assert all(isinstance(wc, PIL.Image.Image) for wc in topic_word_clouds.values())

        topic_word_clouds = visualize.generate_wordclouds_for_topic_words(phi, vocab, 10,
                                                                          which_topics=('topic_1', 'topic_2'),
                                                                          return_images=False,
                                                                          width=640, height=480)
        assert set(topic_word_clouds.keys()) == {'topic_1', 'topic_2'}
        assert all(isinstance(wc, WordCloud) for wc in topic_word_clouds.values())
        assert all(wc.width == 640 and wc.height == 480 for wc in topic_word_clouds.values())


    def test_generate_wordclouds_for_document_topics():
        py3file = '.py3' if six.PY3 else ''
        data = model_io.load_ldamodel_from_pickle('tests/data/tiny_model_reuters_5_topics%s.pickle' % py3file)
        model = data['model']
        doc_labels = data['doc_labels']

        theta = model.doc_topic_
        assert theta.shape == (len(doc_labels), 5)

        doc_topic_clouds = visualize.generate_wordclouds_for_document_topics(theta, doc_labels, 3)
        assert len(doc_topic_clouds) == len(doc_labels)
        assert set(doc_topic_clouds.keys()) == set(doc_labels)
        assert all(isinstance(wc, PIL.Image.Image) for wc in doc_topic_clouds.values())

        which_docs = doc_labels[:2]
        assert len(which_docs) == 2
        doc_topic_clouds = visualize.generate_wordclouds_for_document_topics(theta, doc_labels, 3,
                                                                             which_documents=which_docs,
                                                                             return_images=False,
                                                                             width=640, height=480)
        assert set(doc_topic_clouds.keys()) == set(which_docs)
        assert all(isinstance(wc, WordCloud) for wc in doc_topic_clouds.values())
        assert all(wc.width == 640 and wc.height == 480 for wc in doc_topic_clouds.values())


    def test_write_wordclouds_to_folder(tmpdir):
        path = tmpdir.mkdir('wordclouds').dirname

        py3file = '.py3' if six.PY3 else ''
        data = model_io.load_ldamodel_from_pickle('tests/data/tiny_model_reuters_5_topics%s.pickle' % py3file)
        model = data['model']
        vocab = data['vocab']

        phi = model.topic_word_
        assert phi.shape == (5, len(vocab))

        topic_word_clouds = visualize.generate_wordclouds_for_topic_words(phi, vocab, 10)

        visualize.write_wordclouds_to_folder(topic_word_clouds, path, 'cloud_{label}.png')

        for label in topic_word_clouds.keys():
            assert os.path.exists(os.path.join(path, 'cloud_{label}.png'.format(label=label)))
except:
    # wordcloud module not found
    pass
