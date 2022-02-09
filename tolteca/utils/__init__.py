#!/usr/bin/env python


from .misc import (get_pkg_data_path, get_user_data_dir)
from .runtime_context import (
    ConfigInfo, yaml_load, yaml_dump,
    RuntimeContext, RuntimeContextError, RuntimeBase, RuntimeBaseError)
from tollan.utils import ensure_abspath, rupdate
from tollan.utils.sys import parse_systemd_envfile
from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml
from wrapt import ObjectProxy
import re
import scalpl
import argparse


__all__ = [
    'get_user_data_dir', 'get_pkg_data_path',
    'yaml_load', 'yaml_dump',
    'RuntimeContext', 'RuntimeContextError',
    'RuntimeBase', 'RuntimeBaseError',
    'ConfigLoaderError', 'ConfigLoader',
    'dict_from_cli_args',
           ]


class ConfigLoaderError(RuntimeError):
    """Raise when error in `ConfigLoader`."""
    pass


class ConfigLoader(ObjectProxy):
    """A helper class to load config files for tolteca.

    The config paths are loaded in the following order:

        system_config_path < user_config_path < standalone_config_files

    Parameters
    ----------

    files : list
        A list of YAML config file paths. The config dict get
        merged in their order in the list.

    load_sys_config : bool
        If True, load the built-in config file (:attr:`sys_config_path`).

    load_user_config : bool
        If True, load the user config file (:attr:`user_config_path`).

    runtime_context_dir : str, `pathlib.Path`, optional
        If specified, load the runtime context from this path.

    env_files : list
        A list of systemd env file paths. The env dict get
        merged in their order in the list.
    """

    logger = get_logger()
    sys_config_path = get_pkg_data_path().joinpath("tolteca.yaml")
    user_config_path = get_user_data_dir().joinpath("tolteca.yaml")

    def __init__(
            self,
            files=None,
            load_sys_config=True,
            load_user_config=True,
            runtime_context_dir=None,
            env_files=None,
            ):
        # build the list of paths
        load_sys_config = load_sys_config and self.sys_config_path.exists()
        load_user_config = load_user_config and self.user_config_path.exists()

        # build a config info object and initialize the proxy.
        super().__init__(ConfigInfo.schema.load({
            'standalone_config_files': files or list(),
            'user_config_path': self.user_config_path,
            'sys_config_path': self.sys_config_path,
            'load_user_config': load_user_config,
            'load_sys_config': load_sys_config,
            'runtime_context_dir': runtime_context_dir,
            'env_files': env_files or list(),
            }))

    def get_config(self):
        """Return the merged config dict from all config file paths.

        Note that this only compile the user supplied config files via
        `filepaths` in the constructor and the sys/user config files.

        It does not load the config from the runtime context dir.
        """
        return self.load_config_from_files(self.get_config_paths())

    def get_config_paths(self):
        """Return the merged config paths list from all config file paths.
        """
        paths = list()
        if self.load_sys_config:
            paths.append(self.sys_config_path)
        if self.load_user_config:
            paths.append(self.user_config_path)
        for p in self.standalone_config_files:
            paths.append(ensure_abspath(p))
        return paths

    def get_runtime_context(
            self,
            include_config_as_default=True,
            include_config_as_override=False,
            ):
        """Return the runtime context.

        Returns None when the ``runtime_context_dir`` is not set.

        The `include_config_as_*` can be set to True to include the config dict
        from the :meth:`get_config`. When as default, the config dict is made
        available to the runtime context config through the
        `ConfigBackend.set_default_config`, when as override, it is made
        available to the runtime context config trough the
        `ConfigBackend.set_override_config`.
        """
        # when include as override, include as default is ignored because
        # it gets override anyway
        if include_config_as_override:
            include_config_as_default = False
        dirpath = self.runtime_context_dir
        if dirpath is None:
            return None
        if not dirpath.exists():
            raise ConfigLoaderError(
                f"runtime context dir {dirpath} does not exist.")
        if not dirpath.is_dir():
            raise ConfigLoaderError(
                f"runtime context dir {dirpath} is not a directory.")
        try:
            rc = RuntimeContext(dirpath)
        except RuntimeContextError as e:
            raise ConfigLoaderError(
                f"invalid runtime context dir {dirpath}: {e}")
        if include_config_as_default:
            rc.config_backend.set_default_config(self.get_config())
        elif include_config_as_override:
            rc.config_backend.set_override_config(self.get_config())
        return rc

    @classmethod
    def load_config_from_files(cls, filepaths, schema=None):
        cfg = dict()
        for p in filepaths:
            rupdate(cfg, cls._load_config_from_file(p))
        if schema is None:
            return cfg
        return schema.validate(cfg)

    @staticmethod
    def _load_config_from_file(filepath):
        filepath = ensure_abspath(filepath)
        if filepath.exists():
            try:
                with open(filepath, 'r') as fo:
                    cfg = yaml_load(fo)
                if cfg is None:
                    # empty file is fine
                    cfg = dict()
                elif not isinstance(cfg, dict):
                    raise ConfigLoaderError(
                        f"no valid config dict found in {filepath}.")
            except Exception as e:
                raise ConfigLoaderError(
                    f"cannot load config from {filepath}: {e}")
        else:
            raise ConfigLoaderError(f"{filepath} does not exist.")
        return cfg

    def get_env(self):
        """Return the merged env dict from all env file paths."""
        env = dict()
        # we use update here because env file is only one level.
        for path in self.env_files or list():
            env.update(parse_systemd_envfile(path))
        if len(env) > 0:
            self.logger.debug(f"loaded env:\n{pformat_yaml(env)}")
        return env


default_config_loader = ConfigLoader(runtime_context_dir='.')
"""The tolteca config loader with default settings."""


def dict_from_cli_args(args):
    """Return a nested dict composed from CLI arguments.

    This is used to compose config dict on-the-fly. Nested keys can
    be specified using syntax like ``--a.b.c``. Nested lists are
    supported with the index as the key: ``--a.0.c``. The values
    of the options are parsed as YAML string.
    """

    logger = get_logger()

    logger.debug(f"parse command line args: {args}")

    parser = argparse.ArgumentParser()
    re_arg = re.compile(r'^--(?P<key>[a-zA-Z_]([a-zA-z0-9_.\[\]])*)')
    n_args = len(args)
    for i, arg in enumerate(args):
        # collect all items that are argument keys.
        m = re_arg.match(arg)
        if m is None:
            continue
        if i + 1 < n_args:
            val = args[i + 1]
        else:
            # the last item
            val = None
        arg_kwargs = dict()
        if val is None or re_arg.match(val) is not None:
            # the next item is a validate arg, this is a flag
            arg_kwargs['action'] = 'store_true'
        else:
            # parse the item with config yaml loader
            arg_kwargs['type'] = yaml_load
        parser.add_argument(arg, **arg_kwargs)
    args = parser.parse_args(args)
    # create the nested dict
    d = scalpl.Cut(dict())
    _missing = object()
    for k, v in args.__dict__.items():
        v0 = d.get(k, _missing)
        if v0 is _missing:
            d.setdefault(k, v)
            continue
        # update v to v0 if dict
        if isinstance(v0, dict):
            rupdate(v0, v)
        else:
            d[k] = v
    d = d.data
    logger.debug(f'dict parsed from CLI args: {pformat_yaml(d)}')
    return d
