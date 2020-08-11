#! /usr/bin/env python

from ..toltec.basic_obs_data import BasicObsData, BasicObsDataset
from astropy.table import Table
import numpy as np
import re
import pytest
from ...utils import get_pkg_data_path
from tollan.utils import fileloc
from kidsproc.kidsdata import MultiSweep


def test_basic_obs_data():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_010943_000_0000_2020_07_13_22_32_19_targsweep.nc')
    d = BasicObsData(filepath)
    assert re.match(f'file:///.*{filepath.name}', d.file_uri)
    assert d.file_loc.uri == d.file_uri
    assert d.file_loc.netloc == ''
    assert d.file_loc.path == d.filepath
    assert d.file_loc.path.name == filepath.name

    # remote file
    d = BasicObsData('clipa:/toltec0.nc')
    assert re.match(r'file://clipa/.*toltec0.nc', d.file_uri)
    assert d.file_loc.uri == d.file_uri
    assert d.file_loc.netloc == 'clipa'
    assert d.file_loc.path == d.filepath
    assert d.file_loc.path.name == 'toltec0.nc'

    # remote file with parent path
    with pytest.raises(ValueError, match='remote path shall be absolute'):
        d = BasicObsData('clipa:toltec0.nc')

    d = BasicObsData(
            fileloc('clipa:toltec0.nc', remote_parent_path='/data/data_toltec')
            )
    assert d.file_loc.netloc == 'clipa'
    assert d.file_loc.path.name == 'toltec0.nc'
    assert d.file_loc.path.parent.as_posix() == '/data/data_toltec'


def test_basic_obs_data_meta():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_010943_000_0000_2020_07_13_22_32_19_targsweep.nc')
    d = BasicObsData(filepath)
    # meta is available after construction
    assert d.meta['obsnum'] == 10943


def test_basic_obs_data_read():

    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_010943_000_0000_2020_07_13_22_32_19_targsweep.nc')
    d = BasicObsData(filepath)
    with d.open() as fo:
        kidsdata = fo.read()
        assert kidsdata.__class__ is MultiSweep

    with BasicObsData.open(filepath) as fo:
        kidsdata = fo.read()
        assert kidsdata.__class__ is MultiSweep

    kidsdata = BasicObsData.read(filepath)
    assert kidsdata.__class__ is MultiSweep


def test_basic_obs_dataset_from_files():

    filepaths = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/').glob("*.nc")
    dataset = BasicObsDataset.from_files(
            filepaths, include_meta_cols='intersection')
    assert dataset[0]['interface'] == 'toltec0'
    assert dataset[1]['interface'] == 'toltec0'
    assert dataset[0]['obsnum'] == 11367
    assert len(dataset) == 3
    assert next(iter(dataset)) is dataset._bod_list[0]
    assert dataset['interface'][0] == 'toltec0'

    for kd in dataset.read():
        print(kd)


def test_basic_obs_dataset_from_index_table():

    index_table = Table(rows=[
        {
            'interface': 'toltec1',
            'obsnum': 1000,
            'subobsnum': 1000,
            'scannum': 1000,
            'master': 'ICS',
            'repeat': 0,
            'some_other_data': 'test_data',
            'source': f'mock_file_0',
            },
        {
            'interface': 'toltec2',
            'obsnum': 1000,
            'subobsnum': 1000,
            'scannum': 1000,
            'master': 'ICS',
            'repeat': 0,
            },
        ])
    dataset = BasicObsDataset.from_index_table(
            index_table, copy=False, meta={
                'name': 'test_dataset'
                })
    assert dataset.index_table is index_table
    assert dataset.index_table.meta['name'] == 'test_dataset'
    assert dataset[0]['interface'] == 'toltec1'
    assert dataset[1]['interface'] == 'toltec2'
    assert dataset[0]['obsnum'] == 1000
    assert dataset[0]['source'] == 'mock_file_0'
    assert len(dataset) == 2
    assert next(iter(dataset)) is dataset._bod_list[0]
    assert dataset['interface'][0] == 'toltec1'


def test_basic_obs_dataset_select():

    index_table = Table(rows=[
        {
            'interface': 'toltec1',
            'obsnum': obsnum,
            'subobsnum': 0,
            'scannum': 0,
            'master': 'ICS',
            'repeat': 0,
            'source': f'mock_file_{obsnum}'
            }
        for obsnum in range(10)
        ])
    dataset = BasicObsDataset.from_index_table(
            index_table, copy=False, meta={
                'name': 'test dataset'
                })
    subset = dataset.select('obsnum < 2')
    assert len(subset) == 2
    assert dataset.history[-1] == {
            'bods_select': {
                'cond_str': 'obsnum < 2',
                'description': ""
                }
            }

    subset = dataset.select(np.s_[:3])
    assert len(subset) == 3
    assert dataset.history[-1] == {
            'bods_select': {
                'cond_str': '[:3]',
                'description': ""
                }
            }
