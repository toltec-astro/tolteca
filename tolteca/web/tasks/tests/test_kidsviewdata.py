#! /usr/bin/env python

from ..kidsviewdata import KidsViewData
from tollan.utils.fmt import pformat_yaml


def test_kidsviewdata():
    info = {
            "ObsType": "TUNE",
            "Obsnum": 8925,
            "SubObsNum": 0,
            "ScanNum": 0,
            "Master": "ICS",
            "interfaces": ['toltec0', 'toltec4', 'toltec5'],
            "interface": "toltec5",
            }

    kd = KidsViewData(info, lambda *a, **k: None)
    print(kd._model_params.model)
    print(pformat_yaml(kd.to_dict()))
