import os
import logging

import numpy as np

from tmtoolkit.lda_utils.common import top_n_from_distribution


logger = logging.getLogger('tmtoolkit')


#
# word clouds from topic models
#

def _wordcloud_color_func_black(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'rgb(0,0,0)'


DEFAULT_WORDCLOUD_KWARGS = {   # default wordcloud settings for transparent background and black font
    'width': 800,
    'height': 600,
    'mode': 'RGBA',
    'background_color': None,
    'color_func': _wordcloud_color_func_black
}


def write_wordclouds_to_folder(wordclouds, folder, file_name_fmt='{label}.png', **save_kwargs):
    if not os.path.exists(folder):
        raise ValueError('target folder `%s` does not exist' % folder)

    for label, wc in wordclouds.items():
        file_name = file_name_fmt.format(label=label)
        file_path = os.path.join(folder, file_name)
        logger.info('writing wordcloud to file `%s`' % file_path)

        wc.save(file_path, **save_kwargs)


def generate_wordclouds_for_topic_words(phi, vocab, top_n, topic_labels='topic_{i1}', which_topics=None,
                                        return_images=True, wordcloud_instance=None, **wordcloud_kwargs):
    return generate_wordclouds_from_distribution(phi, row_labels=topic_labels, val_labels=vocab, top_n=top_n,
                                                 which_rows=which_topics, return_images=return_images,
                                                 wordcloud_instance=wordcloud_instance, **wordcloud_kwargs)


def generate_wordclouds_for_document_topics(theta, doc_labels, top_n, topic_labels='topic_{i1}', which_documents=None,
                                            return_images=True, wordcloud_instance=None, **wordcloud_kwargs):
    return generate_wordclouds_from_distribution(theta, row_labels=doc_labels, val_labels=topic_labels, top_n=top_n,
                                                 which_rows=which_documents, return_images=return_images,
                                                 wordcloud_instance=wordcloud_instance, **wordcloud_kwargs)



def generate_wordclouds_from_distribution(distrib, row_labels, val_labels, top_n, which_rows=None, return_images=True,
                                          wordcloud_instance=None, **wordcloud_kwargs):
    prob = top_n_from_distribution(distrib, top_n=top_n, row_labels=row_labels, val_labels=None)
    words = top_n_from_distribution(distrib, top_n=top_n, row_labels=row_labels, val_labels=val_labels)

    if which_rows:
        prob = prob.loc[which_rows, :]
        words = words.loc[which_rows, :]

        assert prob.shape == words.shape

    wordclouds = {}
    for (p_row_name, p), (w_row_name, w) in zip(prob.iterrows(), words.iterrows()):
        assert p_row_name == w_row_name
        logger.info('generating wordcloud for `%s`' % p_row_name)
        wc = generate_wordcloud_from_probabilities_and_words(p, w,
                                                             return_image=return_images,
                                                             wordcloud_instance=wordcloud_instance,
                                                             **wordcloud_kwargs)
        wordclouds[p_row_name] = wc

    return wordclouds


def generate_wordcloud_from_probabilities_and_words(prob, words, return_image=True, wordcloud_instance=None,
                                                    **wordcloud_kwargs):
    if len(prob) != len(words):
        raise ValueError('`distrib` and `labels` must have the name length')
    if hasattr(prob, 'ndim') and prob.ndim != 1:
        raise ValueError('`distrib` must be a 1D array or sequence')
    if hasattr(words, 'ndim') and words.ndim != 1:
        raise ValueError('`labels` must be a 1D array or sequence')

    weights = dict(zip(words, prob))

    return generate_wordcloud_from_weights(weights, return_image=return_image,
                                           wordcloud_instance=wordcloud_instance, **wordcloud_kwargs)


def generate_wordcloud_from_weights(weights, return_image=True, wordcloud_instance=None, **wordcloud_kwargs):
    if not isinstance(weights, dict) or not weights:
        raise ValueError('`weights` must be a non-empty dictionary')

    if not wordcloud_instance:
        from wordcloud import WordCloud

        use_wc_kwargs = DEFAULT_WORDCLOUD_KWARGS.copy()
        use_wc_kwargs.update(wordcloud_kwargs)
        wordcloud_instance = WordCloud(**use_wc_kwargs)

    wordcloud_instance.generate_from_frequencies(weights)

    if return_image:
        return wordcloud_instance.to_image()
    else:
        return wordcloud_instance



#
# plotting of evaluation results #
#


def plot_eval_results(plt, eval_results, metric=None, normalize_y=None):
    if type(eval_results) not in (list, tuple) or not eval_results:
        raise ValueError('`eval_results` must be a list or tuple with at least one element')

    if type(eval_results[0]) not in (list, tuple) or len(eval_results[0]) != 2:
        raise ValueError('`eval_results` must be a list or tuple containing a (param, values) tuple. '
                         'Maybe `eval_results` must be converted with `results_by_parameter`.')

    if normalize_y is None:
        normalize_y = metric is None

    if metric == 'cross_validation':
        plotting_res = []
        for k, folds in eval_results:
            plotting_res.extend([(k, val, f) for f, val in enumerate(folds)])
        x, y, f = zip(*plotting_res)
        fig, ax = plt.subplots()
        ax.scatter(x, y, c=f, alpha=0.5)
    else:
        if metric is not None and type(metric) not in (list, tuple):
            metric = [metric]
        elif metric is None:
            # remove special evaluation result 'model': the calculated model itself
            all_metrics = set(next(iter(eval_results))[1].keys()) - {'model'}
            metric = sorted(all_metrics)

        if normalize_y:
            res_per_metric = {}
            for m in metric:
                params = list(zip(*eval_results))[0]
                unnorm = np.array([metric_res[m] for _, metric_res in eval_results])
                unnorm_nonnan = unnorm[~np.isnan(unnorm)]
                vals_max = np.max(unnorm_nonnan)
                vals_min = np.min(unnorm_nonnan)

                if vals_max != vals_min:
                    rng = vals_max - vals_min
                else:
                    rng = 1.0   # avoid division by zero

                if vals_max < 0:
                    norm = -(vals_max - unnorm) / rng
                else:
                    norm = (unnorm - vals_min) / rng
                res_per_metric[m] = dict(zip(params, norm))

            eval_results_tmp = []
            for k, _ in eval_results:
                metric_res = {}
                for m in metric:
                    metric_res[m] = res_per_metric[m][k]
                eval_results_tmp.append((k, metric_res))
            eval_results = eval_results_tmp

        fig, ax = plt.subplots()
        x = list(zip(*eval_results))[0]
        for m in metric:
            y = [metric_res[m] for _, metric_res in eval_results]
            ax.plot(x, y, label=m)
        ax.legend(loc='best')
