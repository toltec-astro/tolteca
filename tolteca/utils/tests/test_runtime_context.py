#!/usr/bin/env python


import tempfile
from pathlib import Path
from ..runtime_context import (
    RuntimeInfo, SetupInfo,
    ConfigInfo,
    ConfigBackend,
    DirConf, FileConf, DictConf,
    RuntimeContextError, RuntimeContext
    )

from tollan.utils.fmt import pformat_yaml
from tollan.utils.sys import touch_file
from tollan.utils.log import get_logger
# from schema import SchemaMissingKeyError
from ... import version
import pytest
from copy import deepcopy


def test_config_backend():
    logger = get_logger()
    runtime_info = ConfigBackend._get_runtime_info_from_config(dict())
    assert isinstance(runtime_info, RuntimeInfo)
    assert isinstance(runtime_info.config_info, ConfigInfo)
    assert isinstance(runtime_info.setup_info, SetupInfo)
    assert runtime_info.config_info.load_user_config
    assert runtime_info.config_info.load_sys_config
    assert runtime_info.setup_info.config == {}
    logger.debug(pformat_yaml(runtime_info.to_dict()))


def test_dirconf():
    logger = get_logger()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        dc = DirConf(DirConf.populate_dir(tmp, create=True))
        assert dc.rootpath == tmp
        assert dc.get_content_paths() == {
                'rootpath': tmp,
                'bindir': tmp / 'bin',
                'logdir': tmp / 'log',
                'caldir': tmp / 'cal',
                'setup_file': tmp / '40_setup.yaml',
                }
        assert all(
            getattr(dc, f).exists() for f in dc.get_content_paths().keys())
        assert dc.runtime_info.bindir == dc.bindir
        assert dc.runtime_info.logdir == dc.logdir
        assert dc.runtime_info.caldir == dc.caldir
        # no setup yet
        assert not dc.setup_info.config
        # print(RuntimeInfo.schema.pformat())
        # print(ConfigInfo.schema.pformat())
        assert dc.rootpath == tmp
        assert dc.is_persistent
        # no user config yet
        assert dc.config == {dc._runtime_info_key: dc.runtime_info.to_dict()}

        # check default
        dc.set_override_config({'a': 'b'})
        assert dc.config['a'] == 'b'
        # check create setup_config
        setup_config = dc._make_config_for_setup(runtime_info_only=True)
        setup_config[dc._runtime_info_key][dc._setup_info_key]['config'] == {
            dc._runtime_info_key: dc.runtime_info.to_dict()}
        setup_config = dc._make_config_for_setup(runtime_info_only=False)
        assert setup_config[dc._runtime_info_key][
            dc._setup_info_key]['config'] == {
            dc._runtime_info_key: dc.runtime_info.to_dict(),
            'a': 'b'}
        # setup file meta
        assert dc._make_setup_file_meta()[
            dc._setup_file_meta_key]['created_at'] == \
            dc.runtime_info.setup_info.created_at
        # update setup file
        dc.update_setup_file({'a': 1})
        with open(dc.setup_file, 'r') as fo:
            logger.debug(f"{fo.read()}")
        # this is because the override config
        assert dc.config['a'] == 'b'
        dc.set_override_config({})
        assert dc.config['a'] == 1


def test_fileconf():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        f = tmp / 'some_config.yaml'
        with open(f, 'w') as fo:
            fo.write("""
---
a: 1
b:
  c: 'some_value'
""")
        fc = FileConf(filepath=f)
        assert fc.config == {
            'a': 1,
            'b': {'c': 'some_value'},
            fc._runtime_info_key: fc.runtime_info.to_dict()}
        assert fc.is_persistent
        assert fc.rootpath == f
        # check override dict with runtime info
        fc._override_config = {'runtime_info': {'bindir': 'abc'}}
        # the config and runtime info is cached
        assert fc.config['runtime_info']['bindir'] is None
        assert fc.runtime_info.bindir is None
        fc.load()
        assert fc.config['runtime_info']['bindir'] == Path('abc').resolve()
        assert fc.runtime_info.bindir == Path('abc').resolve()

        # set dict with method will directly invalid the cache
        fc.set_default_config({'runtime_info': {'logdir': 'abc'}})
        # the config and runtime info is cached
        assert fc.config['runtime_info']['logdir'] == Path('abc').resolve()
        assert fc.runtime_info.logdir == Path('abc').resolve()

        # override config precedes default config
        fc.set_default_config({'runtime_info': {'logdir': 'abc'}, 'd': 1})
        fc.set_override_config({'runtime_info': {'logdir': 'def'}, 'd': 2})
        assert fc.runtime_info.logdir == Path('def').resolve()
        assert fc.config['d'] == 2


def test_dictconf():

    dc = DictConf({'a': 1, 'b': 'some_value', 'runtime_info': {}})
    assert not dc.is_persistent
    assert dc.runtime_info.bindir is None
    assert dc.runtime_info.config_info.runtime_context_dir is None
    assert dc.rootpath is None
    dc = DictConf({
        'a': 1,
        'b': 'some_value',
        'runtime_info': {'config_info': {'runtime_context_dir': 'abc'}}})
    assert dc.runtime_info.bindir is None
    assert dc.runtime_info.config_info.runtime_context_dir == \
        Path('abc').resolve()
    assert dc.rootpath == Path('abc').resolve()


def test_runtime_context():

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        with pytest.raises(
                RuntimeContextError, match='unable to load config from workdir'
                ):
            rc = RuntimeContext(tmp)
        rc = RuntimeContext.from_dir(tmp, create=True)
        assert rc.rootpath == tmp
        assert all(getattr(rc, f).exists() for f in [
            'bindir', 'caldir', 'logdir'])
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
        assert [f.name for f in rc.config_backend.config_files] == \
            ['00_a.yaml', '10_b.yaml', '40_setup.yaml']

        assert rc.runtime_info.bindir == tmp / 'bin'
        assert rc.runtime_info.version == version.version
        # no setup config
        assert rc.get_setup_rc() is None

        # re-create this rc with the dirpath will work fine
        rc2 = RuntimeContext(tmp)
        # the time stamp would be different so the config will not equal
        assert rc2.config != rc.config

        # if we skip checking the times they would be fine
        def _remove_times(c):
            c = deepcopy(c)
            c['runtime_info'].pop('created_at')
            c['runtime_info']['setup_info'].pop('created_at')
            return c
        assert _remove_times(rc2.config) == _remove_times(rc.config)

        # create from config
        rc3 = RuntimeContext(rc.config)
        assert rc3.config == rc.config
        assert rc3.runtime_info.logdir == tmp / 'log'
        assert not rc3.is_persistent
        assert rc3.rootpath == rc.rootpath == tmp

        # create from config, but with runtime removed
        cfg = deepcopy(rc.config)
        cfg.pop('runtime_info')
        rc4 = RuntimeContext(cfg)
        assert rc4.config != rc.config
        assert rc4.runtime_info.bindir is None
        assert not rc4.is_persistent
        assert rc4.rootpath is None

        # try create from dir again, this will error with force=False
        with pytest.raises(RuntimeContextError, match='force=True to proceed'):
            RuntimeContext.from_dir(tmp, force=False)
        # without create this works the same as the constructor
        rc5 = RuntimeContext.from_dir(
            tmp, force=True, init_config={'some_another_key': 'abc'})
        # no backup created
        assert len(list(tmp.glob('40_setup.yaml.*'))) == 0
        assert rc5.config['some_another_key'] == 'abc'
        # but if we provide init_config, backup is created according
        # to the disable_backup kwarg
        rc6 = RuntimeContext.from_dir(
            tmp, force=True, init_config={'some_another_key': 1})
        assert len(list(tmp.glob('40_setup.yaml.*'))) == 1
        assert rc6.config['some_another_key'] == 1

        # this will create backup for setup file again
        # to see this we need to touch the file because the backup
        # file name is based on the modified time
        rc6.config_backend.update_setup_file({'stuff': 1}, disable_backup=True)
        # assert rc6.config_backend.get_config_from_file(
        #     rc6.config_backend.setup_file)['stuff'] == 1
        # TODO for some reason this does not work
        # RuntimeContext.from_dir(tmp, force=True, create=True)
        # assert len(list(tmp.glob('40_setup.yaml.*'))) == 2


def test_runtime_context_setup_for_dirconf():
    logger = get_logger()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        rc = RuntimeContext.from_dir(tmp, create=True)
        meta_key = rc.config_backend._setup_file_meta_key
        assert meta_key not in rc.config
        assert rc.setup_info.config == dict()
        assert rc.get_setup_rc() is None
        # setup for the first time, this populates setup_file
        rc.setup(overwrite=False)
        logger.debug(pformat_yaml(rc.config))
        assert meta_key in rc.config
        assert rc.setup_info.config != dict()
        assert rc.get_setup_rc().runtime_info.setup_info.config == dict()

        # setup for the second time, this will error
        with pytest.raises(
                RuntimeContextError, match='has existing setup config'):
            rc.setup(overwrite=False)
        # setup with overwrite will create backup by default
        rc.setup(overwrite=True)
        assert len(list(tmp.glob('40_setup.yaml.*'))) == 1
        # setup with given setup file path works and does not create backup
        rc.setup(overwrite=True, setup_filepath=tmp / 'another_setup.yaml')
        assert len(list(tmp.glob('40_setup.yaml.*'))) == 1
        assert (tmp / 'another_setup.yaml').exists()
        # setup again with given setup file path fails when not
        # overwrite_setup_file
        with pytest.raises(
                RuntimeContextError, match='setup file path .+ exists'):
            rc.setup(overwrite=True, setup_filepath=tmp / 'another_setup.yaml')
        # with overwrite_setup_file:
        rc.setup(
            overwrite=True, setup_filepath=tmp / 'another_setup.yaml',
            overwrite_setup_file=True)
        assert len(list(tmp.glob('40_setup.yaml.*'))) == 1
        assert (tmp / 'another_setup.yaml').exists()
        assert len(list(tmp.glob('another_setup.yaml.*'))) == 1


def test_runtime_context_setup_for_dictconf():
    logger = get_logger()

    rc = RuntimeContext({'a': 1, 'b': {'c': False}})
    assert rc.setup_info.config == dict()
    assert rc.get_setup_rc() is None
    # setup for the first time, this populates setup config
    rc.setup(overwrite=False)
    logger.debug(pformat_yaml(rc.config))
    assert rc.setup_info.config != dict()
    assert rc.get_setup_rc().runtime_info.setup_info.config == dict()

    # setup for the second time, this will error
    with pytest.raises(
            RuntimeContextError, match='has existing setup config'):
        rc.setup(overwrite=False)
    # setup with overwrite will create backup by default
    rc.setup(overwrite=True)
    assert rc.setup_info.config != dict()
    # the set up info now is nested twice
    assert rc.get_setup_rc().get_setup_rc().runtime_info.setup_info.config == \
        dict()
    # # setup with given setup file path works and does not create backup
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        rc.setup(overwrite=True, setup_filepath=tmp / 'another_setup.yaml')
        assert (tmp / 'another_setup.yaml').exists()
        # setup again with given setup file path fails when not
        # overwrite_setup_file
        with pytest.raises(
                RuntimeContextError, match='setup file path .+ exists'):
            rc.setup(overwrite=True, setup_filepath=tmp / 'another_setup.yaml')
        # with overwrite_setup_file:
        rc.setup(
            overwrite=True, setup_filepath=tmp / 'another_setup.yaml',
            overwrite_setup_file=True)
        assert (tmp / 'another_setup.yaml').exists()
        assert len(list(tmp.glob('another_setup.yaml.*'))) == 1
