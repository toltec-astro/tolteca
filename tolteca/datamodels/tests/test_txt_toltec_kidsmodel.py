#! /usr/bin/env python


from ..io.toltec.table import TableIO, KidsModelParams, KidsModelParamsIO
from ...utils import get_pkg_data_path
from astropy.table import Table
import pytest
import pickle
from kidsproc import kidsmodel


def test_txt_file_io_basic_open():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_011536_000_0000_2020_07_18_18_02_31_tune.txt')

    # this is a bare object which does not have tied to an open file yet
    # and is set to not load meta data automatically when open.
    df = TableIO(source=None, open_=False)

    assert df._source is None
    assert df._file_loc is None
    assert df.file_loc is df._file_loc
    assert df._file_obj is None
    assert df.file_obj is df._file_obj

    # now we open the object
    # source has to be specified
    with pytest.raises(ValueError, match='source is not specified'):
        df.open()

    with df.open(source=filepath) as dff:
        assert dff is df
        assert df._source is None
        assert df._file_loc.path == filepath
        assert isinstance(df._file_obj, Table)
        assert df._open_state == {
                'data': df.file_obj,
                'data_loc': df.file_loc
                }

    assert df._source is None
    assert df._file_loc is None
    assert df._file_obj is None

    # test open with set source
    df = TableIO(source=filepath, open_=False)
    assert df._source.path == filepath
    assert df._file_loc is df._source
    assert df._file_obj is None

    # open cannot accept source anymore
    with pytest.raises(ValueError, match='source needs to be None for.+'):
        df.open(source=filepath)

    # open the dataset
    with df.open() as dff:
        assert dff is df
        assert df._source.path == filepath
        assert df._file_loc.path == filepath
        assert isinstance(df._file_obj, Table)
        assert df._open_state == {
                'data': df.file_obj,
                'data_loc': df.file_loc
                }

    # dataset is closed
    assert df._source.path == filepath
    assert df._file_loc is df._source
    assert df._file_obj is None

    # test open at construction time
    df = TableIO(source=filepath, open_=True)
    assert df._source.path == filepath
    assert df._file_loc.path == filepath
    assert df._open_state == {
            'data': df.file_obj,
            'data_loc': df.file_loc
            }
    # assert df.meta['obsnum'] == 11536

    df.close()
    # dataset is closed
    assert df._source.path == filepath
    assert df._file_loc is df._source
    assert df._file_obj is None


def test_text_file_io_pickle():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_011536_000_0000_2020_07_18_18_02_31_tune.txt')

    with TableIO(source=filepath) as df:
        blob = pickle.dumps(df)

    df = pickle.loads(blob)

    with df.open():
        assert isinstance(df._file_obj, Table)
        assert len(df.file_obj) == 648


def test_text_file_kidsmodel():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_011536_000_0000_2020_07_18_18_02_31_tune.txt')

    with KidsModelParamsIO(source=filepath) as df:
        m = df.read()
        assert isinstance(m, KidsModelParams)
        assert m.n_models == 648
        assert m.model_cls is kidsmodel.KidsSweepGainWithLinTrend
        assert m.meta['obsnum'] == 11536
