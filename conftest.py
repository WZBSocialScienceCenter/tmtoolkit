"""
Configuration for tests with pytest

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

from hypothesis import settings, HealthCheck


# set default timeout deadline
settings.register_profile('default', deadline=5000)

# profile for CI runs on GitHub machines, which may be slow from time to time so we disable the "too slow" HealthCheck
# and set the timeout deadline very high (60 sec.)
settings.register_profile('ci', suppress_health_check=(HealthCheck.too_slow, ), deadline=60000)

# load default settings profile
settings.load_profile('default')
