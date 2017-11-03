# -*- coding: utf-8 -*-
import gensim

from .common import MultiprocModelsRunner, MultiprocModelsWorkerABC, dtm_to_gensim_corpus


class MultiprocModelsWorkerGensim(MultiprocModelsWorkerABC):
    package_name = 'gensim'

    def fit_model_using_params(self, params):
        data = dtm_to_gensim_corpus(self.data.tocsr())
        model = gensim.models.ldamodel.LdaModel(data, **params)

        return model


def compute_models_parallel(data, varying_parameters, constant_parameters=None, n_max_processes=None):
    mp_models = MultiprocModelsRunner(MultiprocModelsWorkerGensim, data, varying_parameters, constant_parameters,
                                      n_max_processes=n_max_processes)

    return mp_models.run()

