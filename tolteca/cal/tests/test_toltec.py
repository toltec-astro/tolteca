#! /usr/bin/env python


from ..toltec import ToltecArrayProp
from ...utils import get_pkg_data_path


def test_array_prop():
    filepath = get_pkg_data_path().joinpath(
            'tests/toltec_wyattmap/index.yaml')

    cal = ToltecArrayProp.from_indexfile(filepath)
    apt = cal.get(array_name=None)

    assert len(apt) == 7718
