import numpy as np

from tolteca_kids.core import Kids


def test_kids():
    kids = Kids({})
    assert kids.config.sweep_check.chan_range_db_min == 0.1
    assert kids.config.sweep_check.chan_range_db_max == np.inf

    kids = Kids({"kids": {"sweep_check": {"chan_range_db_max": 2}}})
    assert kids.config.sweep_check.chan_range_db_min == 0.1
    assert kids.config.sweep_check.chan_range_db_max == 2


def test_kids_cache():
    kids = Kids({"kids": {"sweep_check": {"chan_range_db_max": 2}}})
    assert kids.sweep_check.config.chan_range_db_max == 2

    kids.update_config({"sweep_check": {"chan_range_db_max": 4}})
    assert kids.config.sweep_check.chan_range_db_max == 4
