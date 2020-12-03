#! /usr/bin/env python


from ..io.toltec.kidsdata import (
        NcFileIO, KidsDataKind, _KidsDataAxisSlicer)
from ...utils import get_pkg_data_path
from tollan.utils.nc import NcNodeMapper, NcNodeMapperError
import netCDF4
import pytest
import pickle
import numpy as np
import astropy.units as u
import kidsproc.kidsdata as kd


def test_nc_file_io_basic_open():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_010943_000_0000_2020_07_13_22_32_19_targsweep.nc')

    # this is a bare object which does not have tied to an open file yet
    # and is set to not load meta data automatically when open.
    df = NcFileIO(source=None, open_=False, load_meta_on_open=False)

    assert 'ncopen' in df.node_mappers.keys()
    assert isinstance(df.node_mappers['ncopen'], NcNodeMapper)

    with pytest.raises(NcNodeMapperError, match='no netCDF dataset'):
        df.nc_node
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
        assert isinstance(df._file_obj, netCDF4.Dataset)
        # access the raw object rought the mapper
        assert df.node_mappers['ncopen'].getany('Header.Toltec.ObsType') == 3

    # now df is closed after the with context:
    with pytest.raises(NcNodeMapperError, match='no netCDF dataset'):
        df.nc_node
    assert df._source is None
    assert df._file_loc is None
    assert df._file_obj is None

    # test open with set source
    df = NcFileIO(source=filepath, open_=False, load_meta_on_open=False)
    # not opened yet
    with pytest.raises(NcNodeMapperError, match='no netCDF dataset'):
        df.nc_node
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
        assert isinstance(df._file_obj, netCDF4.Dataset)
        # access the raw object rought the mapper
        assert df.node_mappers['ncopen'].getany('Header.Toltec.ObsType') == 3

    # dataset is closed
    with pytest.raises(NcNodeMapperError, match='no netCDF dataset'):
        df.nc_node
    assert df._source.path == filepath
    assert df._file_loc is df._source
    assert df._file_obj is None

    # test open at construction time
    df = NcFileIO(source=filepath, open_=True, load_meta_on_open=False)
    assert df._source.path == filepath
    assert df._file_loc.path == filepath
    assert isinstance(df._file_obj, netCDF4.Dataset)
    # access the raw object rought the mapper
    assert df.node_mappers['ncopen'].getany('Header.Toltec.ObsType') == 3

    df.close()
    # dataset is closed
    with pytest.raises(NcNodeMapperError, match='no netCDF dataset'):
        df.nc_node
    assert df._source.path == filepath
    assert df._file_loc is df._source
    assert df._file_obj is None

    # all above does not tried to load any data yet
    assert len(df._meta_cached) == 0

    # test open at construction time as context
    with NcFileIO(source=filepath, open_=True) as df:
        assert df._source.path == filepath
        assert df._file_loc.path == filepath
        assert isinstance(df._file_obj, netCDF4.Dataset)
        assert df.node_mappers['ncopen'].getany('Header.Toltec.ObsType') == 3

    # closed
    with pytest.raises(NcNodeMapperError, match='no netCDF dataset'):
        df.nc_node

    # test open and open again
    df = NcFileIO(source=filepath)
    with df.open() as dff:
        assert df is dff


def test_nc_file_io_kind_and_meta():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_010943_000_0000_2020_07_13_22_32_19_targsweep.nc')

    # this is a bare object which does not have tied to an open file yet
    # and is set to not load meta data automatically when open.
    # this will close the file at the end
    with NcFileIO(source=filepath, load_meta_on_open=False) as df:
        assert isinstance(df._file_obj, netCDF4.Dataset)
        assert df.data_kind == KidsDataKind.TargetSweep

    # now we should have something in meta cache
    assert df._meta_cached['obs_type'] == 3
    # but the file is closed
    assert df._file_obj is None
    # cached property works
    assert df.data_kind == KidsDataKind.TargetSweep

    # we can try reopen the file
    with df.open():
        assert isinstance(df._file_obj, netCDF4.Dataset)
        assert df.data_kind == KidsDataKind.TargetSweep
        # now we load the meta
        assert df.meta['obsnum'] == 10943

    # now we should have something more in meta cache
    assert df._meta_cached['roachid'] == 0
    # but the file is closed
    assert df._file_obj is None
    # cached property works
    assert df.meta['obsnum'] == 10943


def test_nc_file_io_target_sweep_block_info():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_010943_000_0000_2020_07_13_22_32_19_targsweep.nc')

    with NcFileIO(source=filepath) as df:

        meta = df.meta
        assert meta['n_blocks'] == 1
        assert meta['n_blocks_max'] == 1
        assert meta['n_timesperblock'] == 1760
        assert df._resolve_block_index(None) == (0, 1, 1)
        assert df._resolve_block_index(-1) == (0, 1, 1)


def test_nc_file_io_tune_block_info():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_011536_000_0000_2020_07_18_18_02_31_tune.nc')

    with NcFileIO(source=filepath, load_meta_on_open=False) as df:

        meta = df.meta

    assert meta['n_blocks'] == 2
    assert meta['n_blocks_max'] == 2
    assert meta['n_timesperblock'] == 1760
    assert 'flos' in meta
    assert meta['file_loc'] == df.file_loc
    assert df._resolve_block_index(None) == (1, 2, 2)
    assert df._resolve_block_index(-1) == (1, 2, 2)


def test_nc_file_io_tone_axis_data():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_011536_000_0000_2020_07_18_18_02_31_tune.nc')

    with NcFileIO(source=filepath) as df:

        tone_axis_data = df._tone_axis_data

    assert len(tone_axis_data) == 2
    assert tone_axis_data[-1].colnames == ['id', 'f_tone', 'f_center']
    assert len(tone_axis_data[-1]) == 648


def test_nc_file_io_sweep_axis_data():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_011536_000_0000_2020_07_18_18_02_31_tune.nc')

    with NcFileIO(source=filepath) as df:

        sweep_axis_data = df._sweep_axis_data

    assert len(sweep_axis_data) == 2
    assert sweep_axis_data[-1].colnames == [
            'id', 'f_sweep', 'n_samples', 'sample_start', 'sample_end']
    assert len(sweep_axis_data[-1]) == 176
    assert np.all(sweep_axis_data[-1]['n_samples'] == df.meta['n_sweepreps'])


def test_nc_file_pickle():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_010943_000_0000_2020_07_13_22_32_19_targsweep.nc')

    with NcFileIO(
            source=filepath, load_meta_on_open=False,
            auto_close_on_pickle=False) as df:
        assert isinstance(df._file_obj, netCDF4.Dataset)
        assert df.data_kind == KidsDataKind.TargetSweep

        # pickle does not here because the file is open
        with pytest.raises(
                AttributeError, match="Can't pickle local object.+"):
            pickle.dumps(df)

    # file is closed, we try to pickle and check roundtrip
    blob = pickle.dumps(df)
    df = pickle.loads(blob)

    with df.open():
        assert isinstance(df._file_obj, netCDF4.Dataset)
        assert df.data_kind == KidsDataKind.TargetSweep

    # now with auto close
    with NcFileIO(
            source=filepath, load_meta_on_open=False,
            auto_close_on_pickle=True) as df:
        assert isinstance(df._file_obj, netCDF4.Dataset)
        assert df.data_kind == KidsDataKind.TargetSweep

        blob = pickle.dumps(df)
        # load back inside of context
        df = pickle.loads(blob)

        assert isinstance(df._file_obj, netCDF4.Dataset)
        assert df.data_kind == KidsDataKind.TargetSweep

        # pickle again
        blob = pickle.dumps(df)

    # load back outside of context
    df = pickle.loads(blob)

    assert isinstance(df._file_obj, netCDF4.Dataset)
    assert df.data_kind == KidsDataKind.TargetSweep

    df.close()

    assert df._file_obj is None
    # this property is cached
    assert df.data_kind == KidsDataKind.TargetSweep


def test_nc_file_io_kids_data_slicer_tune():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_011536_000_0000_2020_07_18_18_02_31_tune.nc')

    with NcFileIO(source=filepath) as df:

        pass

    assert isinstance(df.block_loc, _KidsDataAxisSlicer)
    assert df.axis_types == {'block', 'tone', 'sweep', 'sample'}

    for t in ['block', 'tone', 'sweep', 'sample']:
        loc = getattr(df, f'{t}_loc')
        assert isinstance(loc, _KidsDataAxisSlicer)
        assert loc._file_obj is df
        assert loc._axis_type == t
        assert loc._args == {}

    # this is a tune so no time axis
    with pytest.raises(ValueError, match="time axis is not available"):
        df.time_loc

    # check chaining
    assert isinstance(df.block_loc.tone_loc, _KidsDataAxisSlicer)
    bl = df.block_loc
    assert bl.tone_loc is bl  # chained builder pattern

    for t in ['block', 'tone', 'sweep', 'sample']:
        loc = getattr(df.tone_loc, f'{t}_loc')
        assert isinstance(loc, _KidsDataAxisSlicer)
        assert loc._file_obj is df
        assert loc._axis_type == t
        assert loc._args == {}

    # check slice args
    assert df.tone_loc[0].sweep_loc[:].block_loc[-1]._args == {
            'tone': [(0, ), {}],
            'sweep': [(slice(None, None, None), ), {}],
            'block': [(-1, ), {}],
            }

    # some invalid slice args
    with pytest.raises(ValueError, match='block loc does not accept keyword'):
        df.tone_loc[0].sweep_loc[:].block_loc(a=1)[None].read()

    with pytest.raises(ValueError, match='block loc can only be integer'):
        df.tone_loc[0].sweep_loc[:].block_loc[:].read()

    with pytest.raises(ValueError, match='time axis is not available for'):
        df.time_loc[0].block_loc[None, ].read()

    with pytest.raises(
            ValueError, match='can only slice on one of sample or sweep'):
        df.sample_loc[0].sweep_loc[None, ].read()

    # check resolve slice

    with df.open():
        slicer = df.tone_loc[:10].sweep_loc[-10:].block_loc[-1]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(3420, 3520, None)
        assert df._resolve_slice(
                slicer)['tone_slice'] == slice(None, 10, None)

        slicer = df.tone_loc[:10].sweep_loc[[0, -1]]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(1760, 3520, None)
        assert df._resolve_slice(
                slicer)['tone_slice'] == slice(None, 10, None)

        # str slicer
        slicer = df.tone_loc['id < 10'].sweep_loc['id < 10']
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(1760, 1860, None)
        # this is a mask with the first 10 true
        assert np.all(df._resolve_slice(slicer)['tone_slice'][:10])

        # check read
        swp = df.tone_loc['id < 10'].sweep_loc['id < 20'].read()
        assert isinstance(swp, kd.MultiSweep)
        assert swp.S21.shape == (10, 20)
        assert swp.S21.unit.is_equivalent(u.adu)
        assert swp.frequency.shape == (10, 20)
        assert swp.frequency.unit.is_equivalent(u.Hz)
        assert swp.meta['data_kind'] is KidsDataKind.Tune


def test_nc_file_io_kids_data_slicer_timesteam():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_011367_000_0000_2020_07_16_20_41_03_timestream.nc')

    with NcFileIO(source=filepath) as df:

        pass

    assert isinstance(df.block_loc, _KidsDataAxisSlicer)
    assert df.axis_types == {'block', 'tone', 'time', 'sample'}

    for t in ['block', 'tone', 'time', 'sample']:
        loc = getattr(df, f'{t}_loc')
        assert isinstance(loc, _KidsDataAxisSlicer)
        assert loc._file_obj is df
        assert loc._axis_type == t
        assert loc._args == {}

    # this is a timestream so no sweep axis
    with pytest.raises(ValueError, match="sweep axis is not available"):
        df.sweep_loc

    # check chaining
    assert isinstance(df.block_loc.tone_loc, _KidsDataAxisSlicer)
    bl = df.block_loc
    assert bl.tone_loc is bl  # chained builder pattern

    for t in ['block', 'tone', 'time', 'sample']:
        loc = getattr(df.tone_loc, f'{t}_loc')
        assert isinstance(loc, _KidsDataAxisSlicer)
        assert loc._file_obj is df
        assert loc._axis_type == t
        assert loc._args == {}

    # check slice args
    assert df.tone_loc[0].time_loc[:].block_loc[-1]._args == {
            'tone': [(0, ), {}],
            'time': [(slice(None, None, None), ), {}],
            'block': [(-1, ), {}],
            }

    # some invalid slice args
    with pytest.raises(ValueError, match='block loc does not accept keyword'):
        df.tone_loc[0].sample_loc[:].block_loc(a=1)[None].read()

    with pytest.raises(ValueError, match='block loc can only be integer'):
        df.tone_loc[0].sample_loc[:].block_loc[:].read()

    with pytest.raises(ValueError, match='sweep axis is not available for'):
        df.sweep_loc[0].block_loc[None].read()

    with pytest.raises(
            ValueError, match='can only slice on one of sample or time'):
        df.sample_loc[0].time_loc[:].read()

    with pytest.raises(
            ValueError, match='time loc can only be slice'):
        df.time_loc[0].read()

    # check resolve slice

    with df.open():
        slicer = df.tone_loc[:10].time_loc[-10:].block_loc[-1]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(-4882, None, None)
        assert df._resolve_slice(
                slicer)['tone_slice'] == slice(None, 10, None)

        # time of quantity
        slicer = df.tone_loc['id < 10'].time_loc[-10::1]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(-4882, None, 488)

        slicer = df.tone_loc['id < 10'].time_loc[-10 << u.s::1]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(-4882, None, 488)

        slicer = df.tone_loc['id < 10'].time_loc[0:1 * u.min:1 * u.s]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(0, 29296, 488)

        with pytest.raises(
                ValueError, match='invalid time slice step'):
            df.time_loc[::-1].read()

        with pytest.raises(
                ValueError, match='invalid time slice step'):
            df.time_loc[::1 * u.us].read()

        # check read
        ts = df.tone_loc['id < 10'].time_loc[-10::1].read()
        assert ts.I.shape == (10, 11)

        # check read
        ts = df.tone_loc['id < 10'].time_loc[:1].read()
        assert ts.I.shape == (10, 488)


def test_nc_file_io_kids_data_slicer_timesteam_processed():

    # local file
    filepath = get_pkg_data_path().joinpath(
        'tests/basic_obs_data/'
        'toltec3_012759_000_0000_2020_09_24_17_05_02_timestream_processed.nc')

    with NcFileIO(source=filepath) as df:

        pass

    assert isinstance(df.block_loc, _KidsDataAxisSlicer)
    assert df.axis_types == {'block', 'tone', 'time', 'sample'}

    for t in ['block', 'tone', 'time', 'sample']:
        loc = getattr(df, f'{t}_loc')
        assert isinstance(loc, _KidsDataAxisSlicer)
        assert loc._file_obj is df
        assert loc._axis_type == t
        assert loc._args == {}

    # this is a timestream so no sweep axis
    with pytest.raises(ValueError, match="sweep axis is not available"):
        df.sweep_loc

    # check chaining
    assert isinstance(df.block_loc.tone_loc, _KidsDataAxisSlicer)
    bl = df.block_loc
    assert bl.tone_loc is bl  # chained builder pattern

    for t in ['block', 'tone', 'time', 'sample']:
        loc = getattr(df.tone_loc, f'{t}_loc')
        assert isinstance(loc, _KidsDataAxisSlicer)
        assert loc._file_obj is df
        assert loc._axis_type == t
        assert loc._args == {}

    # check slice args
    assert df.tone_loc[0].time_loc[:].block_loc[-1]._args == {
            'tone': [(0, ), {}],
            'time': [(slice(None, None, None), ), {}],
            'block': [(-1, ), {}],
            }

    # some invalid slice args
    with pytest.raises(ValueError, match='block loc does not accept keyword'):
        df.tone_loc[0].sample_loc[:].block_loc(a=1)[None].read()

    with pytest.raises(ValueError, match='block loc can only be integer'):
        df.tone_loc[0].sample_loc[:].block_loc[:].read()

    with pytest.raises(ValueError, match='sweep axis is not available for'):
        df.sweep_loc[0].block_loc[None].read()

    with pytest.raises(
            ValueError, match='can only slice on one of sample or time'):
        df.sample_loc[0].time_loc[:].read()

    with pytest.raises(
            ValueError, match='time loc can only be slice'):
        df.time_loc[0].read()

    # check resolve slice

    with df.open():
        slicer = df.tone_loc[:10].time_loc[-10:].block_loc[-1]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(-4882, None, None)
        assert df._resolve_slice(
                slicer)['tone_slice'] == slice(None, 10, None)

        # time of quantity
        slicer = df.tone_loc['id < 10'].time_loc[-10::1]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(-4882, None, 488)

        slicer = df.tone_loc['id < 10'].time_loc[-10 << u.s::1]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(-4882, None, 488)

        slicer = df.tone_loc['id < 10'].time_loc[0:1 * u.min:1 * u.s]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(0, 29296, 488)

        with pytest.raises(
                ValueError, match='invalid time slice step'):
            df.time_loc[::-1].read()

        with pytest.raises(
                ValueError, match='invalid time slice step'):
            df.time_loc[::1 * u.us].read()

        # check read every 1 second
        pts = df.tone_loc['id < 10'].time_loc[-10::1].read()
        assert pts.x.shape == (10, 11)

        # check read
        pts = df.tone_loc['id < 10'].time_loc[:1].read()
        assert pts.r.shape == (10, 488)
        # check psds shape
        pts.meta['f_psd'].shape == (1025, )
        pts.meta['I_psd'].shape == (10, 1025)


def test_nc_file_io_kids_data_slicer_tune_processed():

    # local file
    filepath = get_pkg_data_path().joinpath(
        'tests/basic_obs_data/'
        'toltec3_012758_000_0000_2020_09_24_17_04_06_tune_processed.nc')

    with NcFileIO(source=filepath) as df:

        pass

    assert isinstance(df.block_loc, _KidsDataAxisSlicer)
    assert df.axis_types == {'block', 'tone', 'sweep', 'sample'}

    for t in ['block', 'tone', 'sweep', 'sample']:
        loc = getattr(df, f'{t}_loc')
        assert isinstance(loc, _KidsDataAxisSlicer)
        assert loc._file_obj is df
        assert loc._axis_type == t
        assert loc._args == {}

    # this is a tune so no time axis
    with pytest.raises(ValueError, match="time axis is not available"):
        df.time_loc

    # check chaining
    assert isinstance(df.block_loc.tone_loc, _KidsDataAxisSlicer)
    bl = df.block_loc
    assert bl.tone_loc is bl  # chained builder pattern

    for t in ['block', 'tone', 'sweep', 'sample']:
        loc = getattr(df.tone_loc, f'{t}_loc')
        assert isinstance(loc, _KidsDataAxisSlicer)
        assert loc._file_obj is df
        assert loc._axis_type == t
        assert loc._args == {}

    # check slice args
    assert df.tone_loc[0].sweep_loc[:].block_loc[-1]._args == {
            'tone': [(0, ), {}],
            'sweep': [(slice(None, None, None), ), {}],
            'block': [(-1, ), {}],
            }

    # some invalid slice args
    with pytest.raises(ValueError, match='block loc does not accept keyword'):
        df.tone_loc[0].sweep_loc[:].block_loc(a=1)[None].read()

    with pytest.raises(ValueError, match='block loc can only be integer'):
        df.tone_loc[0].sweep_loc[:].block_loc[:].read()

    with pytest.raises(ValueError, match='time axis is not available for'):
        df.time_loc[0].block_loc[None, ].read()

    with pytest.raises(
            ValueError, match='can only slice on one of sample or sweep'):
        df.sample_loc[0].sweep_loc[None, ].read()

    # check resolve slice

    with df.open():
        slicer = df.tone_loc[:10].sweep_loc[-10:].block_loc[-1]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(166, 176, None)
        assert df._resolve_slice(
                slicer)['tone_slice'] == slice(None, 10, None)

        slicer = df.tone_loc[:10].sweep_loc[[0, -1]]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(0, 176, None)
        assert df._resolve_slice(
                slicer)['tone_slice'] == slice(None, 10, None)

        # str slicer
        slicer = df.tone_loc['id < 10'].sweep_loc['id < 10']
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(0, 10, None)
        # this is a mask with the first 10 true
        assert np.all(df._resolve_slice(slicer)['tone_slice'][:10])

        # check read
        swp = df.tone_loc['id < 10'].sweep_loc['id < 20'].read()
        assert isinstance(swp, kd.MultiSweep)
        assert swp.S21.shape == (10, 20)
        assert swp.S21.unit.is_equivalent(u.adu)
        assert swp.frequency.shape == (10, 20)
        assert swp.frequency.unit.is_equivalent(u.Hz)
        assert swp.meta['data_kind'] is KidsDataKind.ReducedSweep


def test_nc_file_io_kids_data_slicer_vna_processed():

    # local file
    filepath = get_pkg_data_path().joinpath(
        'tests/basic_obs_data/'
        'toltec0_012673_000_0000_2020_09_23_12_41_06_vnasweep_processed.nc')

    with NcFileIO(source=filepath) as df:

        pass

    assert isinstance(df.block_loc, _KidsDataAxisSlicer)
    assert df.axis_types == {'block', 'tone', 'sweep', 'sample'}

    for t in ['block', 'tone', 'sweep', 'sample']:
        loc = getattr(df, f'{t}_loc')
        assert isinstance(loc, _KidsDataAxisSlicer)
        assert loc._file_obj is df
        assert loc._axis_type == t
        assert loc._args == {}

    # this is a tune so no time axis
    with pytest.raises(ValueError, match="time axis is not available"):
        df.time_loc

    # check chaining
    assert isinstance(df.block_loc.tone_loc, _KidsDataAxisSlicer)
    bl = df.block_loc
    assert bl.tone_loc is bl  # chained builder pattern

    for t in ['block', 'tone', 'sweep', 'sample']:
        loc = getattr(df.tone_loc, f'{t}_loc')
        assert isinstance(loc, _KidsDataAxisSlicer)
        assert loc._file_obj is df
        assert loc._axis_type == t
        assert loc._args == {}

    # check slice args
    assert df.tone_loc[0].sweep_loc[:].block_loc[-1]._args == {
            'tone': [(0, ), {}],
            'sweep': [(slice(None, None, None), ), {}],
            'block': [(-1, ), {}],
            }

    # some invalid slice args
    with pytest.raises(ValueError, match='block loc does not accept keyword'):
        df.tone_loc[0].sweep_loc[:].block_loc(a=1)[None].read()

    with pytest.raises(ValueError, match='block loc can only be integer'):
        df.tone_loc[0].sweep_loc[:].block_loc[:].read()

    with pytest.raises(ValueError, match='time axis is not available for'):
        df.time_loc[0].block_loc[None, ].read()

    with pytest.raises(
            ValueError, match='can only slice on one of sample or sweep'):
        df.sample_loc[0].sweep_loc[None, ].read()

    # check resolve slice

    with df.open():
        slicer = df.tone_loc[:10].sweep_loc[-10:].block_loc[-1]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(481, 491, None)
        assert df._resolve_slice(
                slicer)['tone_slice'] == slice(None, 10, None)

        slicer = df.tone_loc[:10].sweep_loc[[0, -1]]
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(0, 491, None)
        assert df._resolve_slice(
                slicer)['tone_slice'] == slice(None, 10, None)

        # str slicer
        slicer = df.tone_loc['id < 10'].sweep_loc['id < 10']
        assert df._resolve_slice(
                slicer)['sample_slice'] == slice(0, 10, None)
        # this is a mask with the first 10 true
        assert np.all(df._resolve_slice(slicer)['tone_slice'][:10])

        # check read
        swp = df.tone_loc['id < 10'].sweep_loc['id < 20'].read()
        assert isinstance(swp, kd.MultiSweep)
        assert swp.S21.shape == (10, 20)
        assert swp.S21.unit.is_equivalent(u.adu)
        assert swp.frequency.shape == (10, 20)
        assert swp.frequency.unit.is_equivalent(u.Hz)
        assert swp.meta['data_kind'] is KidsDataKind.ReducedSweep
        assert swp.unified.D21.shape == (492733, )
        assert swp.unified.D21.unit == u.adu / u.Hz
        assert swp.unified.D21_cov.shape == (492733, )
        assert swp.unified.D21_mean.unit == u.adu / u.Hz
        assert swp.unified.meta['candidates'].shape == (643, )
        # slice the d21
        d = swp.unified[:10]
        assert d.frequency.shape == (10, )
        assert d.D21.shape == (10, )
        assert d.D21_cov.shape == (10, )
