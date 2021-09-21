#!/usr/bin/env python


from .. import ConfigLoaderError, ConfigLoader, RuntimeContext
from tollan.utils import ensure_abspath
from tollan.utils.sys import touch_file
import tempfile
import pytest


def test_config_loader():
    # with all enabled
    cl = ConfigLoader(
        load_sys_config=True, load_user_config=True, runtime_context_dir='.',
        files=['a.yaml', 'b.yaml'],
        env_files=['a_env'],
        )
    assert cl.standalone_config_files == [
        ensure_abspath('a.yaml'), ensure_abspath('b.yaml')]
    assert cl.load_user_config == cl.user_config_path.exists()
    assert cl.load_sys_config == cl.sys_config_path.exists()
    assert cl.runtime_context_dir == ensure_abspath('.')
    assert cl.env_files == [ensure_abspath('a_env')]

    with pytest.raises(ConfigLoaderError, match='does not exist'):
        cl.get_config()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = ensure_abspath(tmp)
        with open(tmp / 'a.yaml', 'w') as fa, \
                open(tmp / 'b.yaml', 'w') as fb, \
                open(tmp / 'not.yaml', 'w') as fc:
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
            fc.write('Not a yaml config')
        touch_file(tmp / "empty.yaml")
        cl = ConfigLoader(
            load_sys_config=False, load_user_config=False,
            runtime_context_dir=tmp,
            files=[
                tmp / 'a.yaml',
                tmp / 'b.yaml',
                tmp / 'not.yaml',
                tmp / 'empty.yaml']
            )
        assert cl.standalone_config_files == [
                tmp / 'a.yaml',
                tmp / 'b.yaml',
                tmp / 'not.yaml',
                tmp / 'empty.yaml']
        assert not cl.load_user_config
        assert not cl.load_sys_config
        assert cl.runtime_context_dir == tmp
        with pytest.raises(ConfigLoaderError, match='no valid config dict'):
            cl.get_config()
        with pytest.raises(
                ConfigLoaderError, match='invalid runtime context dir'):
            cl.get_runtime_context()
        cl = ConfigLoader(
            load_sys_config=False, load_user_config=False,
            runtime_context_dir=tmp / 'not_exist',
            files=[
                tmp / 'a.yaml',
                tmp / 'b.yaml',
                tmp / 'empty.yaml']
            )
        assert cl.standalone_config_files == [
                tmp / 'a.yaml',
                tmp / 'b.yaml',
                tmp / 'empty.yaml']
        assert cl.get_config() == {'a': 1, 'b': {'c': 1}, 'd': 'test'}
        with pytest.raises(
                ConfigLoaderError, match='does not exist'):
            cl.get_runtime_context()

        cl = ConfigLoader(
            load_sys_config=False, load_user_config=False,
            runtime_context_dir=tmp / 'not.yaml',
            files=[
                tmp / 'a.yaml',
                tmp / 'b.yaml',
                tmp / 'empty.yaml']
            )
        with pytest.raises(
                ConfigLoaderError, match='is not a dir'):
            cl.get_runtime_context()

        # create runtime context with tmp
        RuntimeContext.from_dir(tmp, create=True, force=True)
        cl = ConfigLoader(
            load_sys_config=False, load_user_config=False,
            runtime_context_dir=tmp,
            files=[
                tmp / 'a.yaml',
                tmp / 'b.yaml',
                tmp / 'empty.yaml']
            )
        rc = cl.get_runtime_context(include_config_as_default=True)
        # not setup
        assert rc.runtime_info.setup_info.config == dict()
        # config should be propagated in include_config_as_default
        rc.config['a'] == 1
        rc.config['b'] == {'c': 1}
        rc.config['d'] == 'test'

        # without config default
        rc = cl.get_runtime_context(include_config_as_default=False)
        # not setup
        assert rc.runtime_info.setup_info.config == dict()
        assert 'a' not in rc.config
