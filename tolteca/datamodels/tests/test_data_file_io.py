#! /usr/bin/env python


import pytest
from ..io.base import DataFileIO, DataFileIOError
from ...utils import get_pkg_data_path


def test_data_file_io():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_010943_000_0000_2020_07_13_22_32_19_targsweep.nc')

    meta = {'test': 'test_value'}

    df = DataFileIO()
    df._setup(file_loc=filepath, file_obj=None, meta=meta)

    assert df.file_loc.is_local
    assert df.file_loc.path == filepath
    assert df.file_obj is None

    # this is a dummy object so open will raise
    with pytest.raises(DataFileIOError, match='file object not available.+'):
        df.open()

    assert df.meta['test'] == 'test_value'

    df = DataFileIO()
    df._setup(file_loc='clipa:/toltec0.nc', file_obj=None, meta=None)
    assert df.filepath.as_posix() == '/toltec0.nc'
