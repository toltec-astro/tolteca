#! /usr/bin/env python


from ..base import CalibBase
import tempfile
import yaml
from pathlib import Path
from tollan.utils.sys import touch_file


def test_calib_base():
    with tempfile.TemporaryDirectory() as rootpath:
        rootpath = Path(rootpath)
        filepath = rootpath / 'index.yaml'
        with open(filepath, 'w') as fo:
            yaml.dump({
                'test_key': 'test_value',
                'test_path': 'test_path_value'
                }, fo)
        calobj = CalibBase.from_indexfile(filepath)
        assert calobj.index['test_key'] == 'test_value'
        assert calobj.rootpath == filepath.parent == rootpath
        assert calobj.rootpath / 'test_path_value' == \
            calobj.resolve_path(calobj.index['test_path'], validate=False)
        touch_file(calobj.rootpath / 'test_path_value')
        assert calobj.resolve_path(calobj.index['test_path']) == \
            calobj.rootpath / 'test_path_value' 
