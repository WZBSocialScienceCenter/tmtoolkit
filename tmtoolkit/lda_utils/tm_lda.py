# -*- coding: utf-8 -*-
import logging

from lda import LDA

from .common import MultiprocModelsRunner, MultiprocModelsWorkerABC


class MultiprocModelsWorkerLDA(MultiprocModelsWorkerABC):
    package_name = 'lda'

    def fit_model_using_params(self, params):
        lda_instance = LDA(**params)
        lda_instance.fit(self.data)

        return lda_instance


def compute_models_parallel(data, varying_parameters, constant_parameters=None, n_max_processes=None):
    mp_models = MultiprocModelsRunner(MultiprocModelsWorkerLDA, data, varying_parameters, constant_parameters,
                                      n_max_processes=n_max_processes)

    return mp_models.run()

