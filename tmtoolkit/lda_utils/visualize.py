import numpy as np


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
