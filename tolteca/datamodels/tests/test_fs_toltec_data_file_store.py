#! /usr/bin/env python

from ..fs.toltec import meta_from_source, ToltecDataFileStore
from ..io.toltec.kidsdata import KidsDataKind
from ...utils import get_pkg_data_path


def test_meta_from_source():

    # local file
    filepath = get_pkg_data_path().joinpath(
            'tests/basic_obs_data/'
            'toltec0_010943_000_0000_2020_07_13_22_32_19_targsweep.nc')

    meta = meta_from_source(filepath)

    assert meta['instru'] == 'toltec'
    assert meta['interface'] == 'toltec0'
    assert meta['obsnum'] == 10943
    assert meta['data_kind'] == KidsDataKind.TargetSweep
    assert meta['file_loc'].netloc == ''


def test_meta_from_source_remote():

    filepath = (
            f'clipa:/data/data_toltec/ics/toltec0/'
            f'toltec0_010943_000_0000_2020_07_13_22_32_19_targsweep.nc')

    meta = meta_from_source(filepath)

    assert meta['instru'] == 'toltec'
    assert meta['interface'] == 'toltec0'
    assert meta['obsnum'] == 10943
    assert meta['data_kind'] == KidsDataKind.TargetSweep
    assert meta['file_loc'].netloc == 'clipa'


def test_toltec_data_file_store():
    d = ToltecDataFileStore('taco:/data/data_toltec')

    assert not d.is_local
    assert d.rootpath.name == 'data_toltec'

    d = ToltecDataFileStore(get_pkg_data_path())
    assert d.is_local
    assert d.rootpath.name == 'data'
