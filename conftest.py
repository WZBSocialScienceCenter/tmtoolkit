"""
Configuration for tests with pytest

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

from hypothesis import settings, HealthCheck

# profile for CI runs on GitHub machines, which may be slow from time to time so we disable the "too slow" HealthCheck
settings.register_profile('ci', suppress_health_check=(HealthCheck.too_slow, ))
