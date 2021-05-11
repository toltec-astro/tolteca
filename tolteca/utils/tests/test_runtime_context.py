#! /usr/bin/env python

import tempfile
import pytest
from pathlib import Path
from tollan.utils.sys import touch_file
from schema import SchemaMissingKeyError
from copy import deepcopy
from .. import RuntimeContext, RuntimeContextError
from ... import version


def test_runtime_context():

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        with pytest.raises(
                RuntimeContextError, match='missing runtime context contents'
                ):
            rc = RuntimeContext(rootpath=tmp)
        rc = RuntimeContext.from_dir(tmp, create=True)
        assert rc.rootpath == tmp
        assert rc.to_dict() == {
                'rootpath': tmp,
                'bindir': tmp / 'bin',
                'caldir': tmp / 'cal',
                'logdir': tmp / 'log',
                'setup_file': tmp / '50_setup.yaml',
                }
        assert all(getattr(rc, f).exists() for f in [
            'bindir', 'caldir', 'logdir', 'setup_file'])
        with open(tmp / '00_a.yaml', 'w') as fa, \
                open(tmp / '10_b.yaml', 'w') as fb:
            fa.write('''
---
a: 1
b:
  c: 2
''')
            fb.write('''
---
b:
  c: 1
d: 'test'
''')
        touch_file(tmp / "not_a_config.yaml")
        assert [f.name for f in rc.config_files] == \
            ['00_a.yaml', '10_b.yaml', '50_setup.yaml']

        with pytest.raises(
                SchemaMissingKeyError,
                match="Missing key: 'setup'"
                ):
            assert rc.config
        # need setup
        rc.setup()
        assert rc.config
        assert rc.config['setup']['version'] == version.version
        # create directly
        rc2 = RuntimeContext(tmp)
        assert rc2.config == rc.config

        # create from config
        rc3 = RuntimeContext(config=rc.config)
        assert rc3.config == rc.config
        assert not rc3.is_persistent
        assert rc3.rootpath == rc.rootpath == tmp

        # create from config, but with runtime removed
        cfg = deepcopy(rc.config)
        cfg.pop('runtime')
        rc4 = RuntimeContext(config=cfg)
        assert rc4.config
        assert rc4.config['runtime']['bindir'] is None
        assert not rc4.is_persistent
        assert rc4.rootpath is None

        # create from config, but with setupremoved
        cfg = deepcopy(rc.config)
        cfg.pop('setup')
        rc5 = RuntimeContext(config=cfg)
        with pytest.raises(
                SchemaMissingKeyError,
                match="Missing key: 'setup'"
                ):
            assert rc5.config
