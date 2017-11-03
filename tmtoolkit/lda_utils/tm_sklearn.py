# -*- coding: utf-8 -*-
from sklearn.decomposition.online_lda import LatentDirichletAllocation

from .common import MultiprocModelsRunner, MultiprocModelsWorkerABC


class MultiprocModelsWorkerSklearn(MultiprocModelsWorkerABC):
    package_name = 'sklearn'

    def fit_model_using_params(self, params):
        data = self.data.tocsr()

        lda_instance = LatentDirichletAllocation(**params)
        lda_instance.fit(data)

        return lda_instance


def compute_models_parallel(data, varying_parameters, constant_parameters=None, n_max_processes=None):
    mp_models = MultiprocModelsRunner(MultiprocModelsWorkerSklearn, data, varying_parameters, constant_parameters,
                                      n_max_processes=n_max_processes)

    return mp_models.run()

