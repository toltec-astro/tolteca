#! /usr/bin/env python


from ..base import CalibBase
import tempfile
import yaml
from pathlib import Path


def test_calib_base():
    with tempfile.NamedTemporaryFile('w') as fo:
        filepath = Path(fo.name)
        yaml.dump({
            'test_key': 'test_value',
            'test_path': 'test_path_value'
            }, fo)
        calobj = CalibBase.from_indexfile(filepath)
        assert calobj.index['test_key'] == 'test_value'
        assert calobj.rootpath == filepath.parent
        print(calobj.resolve_path(calobj.index['test_path']))
        # assert calobj.resolve_path(calobj.index['test_path'])
