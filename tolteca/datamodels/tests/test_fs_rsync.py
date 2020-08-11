#! /usr/bin/env python


from ..fs.rsync import RsyncAccessor
from ...utils import get_pkg_data_path
from pathlib import Path


def test_rsync_accessor_glob():

    paths = RsyncAccessor.glob(get_pkg_data_path().joinpath('tests/rsync'))

    assert len(paths) == 4
    names = {Path(path).name for path in paths}
    assert names == {'a', 'b', 'c', 'e'}


def test_rsync_accessor_glob_taco():

    paths = RsyncAccessor.glob('zma@taco:/home/zma/test_rsync_accessor')

    assert len(paths) == 4
    names = {Path(path).name for path in paths}
    assert names == {'a', 'b', 'c', 'e'}
