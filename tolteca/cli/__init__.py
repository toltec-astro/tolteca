#! /usr/bin/env python

import sys

from ..utils import (
    get_pkg_data_path, get_user_data_dir, RuntimeContext, RuntimeContextError)
from .. import version
import argparse
from tollan.utils.log import init_log, get_logger
from tollan.utils.cli.multi_action_argparser import \
        MultiActionArgumentParser
from tollan.utils.cli.path_type import PathType
from tollan.utils.sys import parse_systemd_envfile
from tollan.utils import rupdate
from tollan.utils.fmt import pformat_yaml
from astropy.io.misc import yaml
import os
import shlex
from wrapt import ObjectProxy
from pathlib import Path


__all__ = [
    'main_parser', 'config', 'base_runtime_context', 'main']


main_parser = ObjectProxy(None)
"""
A proxy to the
`~tollan.utils.cli.multi_action_argparser.MultiActionArgumentParser`
instance, which is made available when `~tolteca.cli.main` is
called.

This can be used to register subcommands.

"""

config = ObjectProxy(None)
"""
A proxy to the loaded config YAML dict when running in CLI mode.

"""

base_runtime_context = ObjectProxy(None)
"""
A proxy to the loaded base runtime context in CLI mode.

"""


def _load_config_from_file(filepath, schema=None):
    empty = dict()
    if filepath.exists():
        with open(filepath, 'r') as fo:
            cfg = yaml.load(fo)
        if cfg is None:
            cfg = empty
        elif not isinstance(cfg, dict):
            raise ValueError(f"no valid config dict found in {filepath}.")
    else:
        cfg = empty
    if schema is None:
        return cfg
    return schema.validate(cfg)


def load_config_from_files(
        paths=None,
        load_sys_config=True,
        load_user_config=True,
        runtime_context_dir=None):
    """Load config from files.

    Parameters
    ----------
    filepaths : list, optional
        The paths of the YAML config files to load.
    load_sys_config : bool
        If True, load the system wide tolteca.yaml in the package data.
    load_app_config : bool
        If True, load the tolteca.yaml in the user app directory.
    runtime_context_dir : str, `pathlib.Path`, optional
        If set, create `tolteca.utils.RuntimeContext` instance from it.
    """
    logger = get_logger()
    _paths = list()
    sys_config_path = get_pkg_data_path().joinpath("tolteca.yaml")
    user_config_path = get_user_data_dir().joinpath("tolteca.yaml")
    if load_sys_config:
        _paths.append(sys_config_path)
    if load_user_config:
        _paths.append(user_config_path)
    if paths is not None:
        for p in paths:
            _paths.append(Path(p))
    cfg = dict()
    for p in _paths:
        rupdate(cfg, _load_config_from_file(p))
    # handle runtime context
    if runtime_context_dir is not None:
        runtime_context_dir = Path(runtime_context_dir).expanduser().resolve()
        try:
            rc = RuntimeContext(runtime_context_dir)
        except RuntimeContextError:
            logger.debug(f"invalid runtime context dir {runtime_context_dir}")
            rc = None
    else:
        rc = None
    # add some runtime info
    rupdate(cfg, {
        'runtime': {
            'config_info': {
                'sources': _paths,
                'sys_config_path': sys_config_path,
                'user_config_path': user_config_path,
                'sys_config_loaded': (
                    load_sys_config and sys_config_path.exists()),
                'user_config_loaded': (
                    load_user_config and user_config_path.exists()),
                'runtime_context_dir': runtime_context_dir,
                'base_runtime_context_loaded': rc
                }
            }
        })
    return cfg


def _make_runtime_context(runtime_context_cls):
    """A helper class to create runtime context object from
    `config` or `base_runtime_context`"""
    logger = get_logger()
    if base_runtime_context:
        rt = runtime_context_cls(base_runtime_context.rootpath)
        logger.info(
            f'created {runtime_context_cls} object from base '
            f'runtime context {base_runtime_context}')
    else:
        # need the __wrapped__ to allow deepcopy
        rt = runtime_context_cls.from_config(config.__wrapped__)
        config_files = config["runtime"]['config_info']['sources']
        logger.info(
            f'created {runtime_context_cls} from config files\n'
            f'{pformat_yaml(config_files)}')
    logger.debug(f'runtime context created: {rt}')
    return rt


def _make_banner(title, subtitle):

    from art import text2art
    title_lines = text2art(title, "avatar").split('\n')

    pad = 4
    width = len(title_lines[0]) + pad * 2

    st_line = f'{{:^{width}s}}'.format(subtitle)

    return '\n{}\n{}\n{}\n{}\n'.format(
        '.~' * (width // 2),
        '\n'.join([f'{" " * pad}{title_line}' for title_line in title_lines]),
        st_line,
        '~.' * (width // 2)
        )


def main(args=None):
    """The CLI entry point."""

    prog_name = 'TolTECA'
    prog_desc = 'TolTEC Data Analysis All-in-one!'

    parser = main_parser.__wrapped__ = MultiActionArgumentParser(
            description=f"{prog_name} v{version.version}"
                        f" - {prog_desc}"
            )

    parser.add_argument(
            "-c", "--config",
            nargs='+',
            help="The path to additional YAML config file(s) for tolteca. "
                 "By default, the sys config and user config are loaded, "
                 "which can be disabled by the switch -n.",
            metavar='FILE',
            type=PathType(exists=True, type_="file")
            )
    parser.add_argument(
            "-d", "--runtime_context_dir",
            nargs='?',
            default=argparse.SUPPRESS,
            help="The path to look for runtime context directory.",
            metavar='DIR',
            type=PathType(exists=True, type_='dir'),
            )

    parser.add_argument(
            "-n", "--no_persistent_config",
            help="If set, skip loading the sys config and user config.",
            action='store_true',
            )
    parser.add_argument(
            "--no_banner",
            help="If set, the banner will not be shown.",
            action='store_true',
            )
    parser.add_argument(
            '-e', '--env_files',
            metavar='ENV_FILE', nargs='*',
            help='Path to systemd env file. '
                 'Multiple files are merged in order.',
            type=PathType(exists=True, type_="file")
            )
    parser.add_argument(
            "-q", "--quiet",
            help="Suppress debug logging messages.",
            action='store_true')
    parser.add_argument(
            '-v', '--version', action='store_true',
            help='Print the version info and exit.'
            )

    # import subcommand modules:
    from .check import cmd_check  # noqa: F401
    from .db import cmd_migrate  # noqa: F401
    from .run import cmd_run  # noqa: F401
    from .setup import cmd_setup  # noqa: F401
    from .simu import cmd_simu  # noqa: F401
    from .reduce import cmd_reduce  # noqa: F401

    # parse and handle global args:
    option, unknown_args = parser.parse_known_args(args or sys.argv[1:])

    if option.quiet:
        loglevel = 'INFO'
    else:
        loglevel = 'DEBUG'
    init_log(level=loglevel)

    logger = get_logger()

    if not option.no_banner:
        print(_make_banner(prog_name, 'http://toltecdr.astro.umass.edu'))

    if option.version:
        print(version.version)
        sys.exit(0)

    # load config
    cfg_kwargs = dict(
        load_sys_config=True, load_user_config=True,
        )
    if 'runtime_context_dir' in option:
        cfg_kwargs['runtime_context_dir'] = \
            option.runtime_context_dir or Path('.')
    else:
        pass
    if option.no_persistent_config:
        cfg_kwargs.update(load_sys_config=False, load_user_config=False)
    cfg = option.config = config.__wrapped__ = load_config_from_files(
            paths=option.config, **cfg_kwargs)
    option.base_runtime_context = base_runtime_context.__wrapped__ = \
        cfg['runtime']['config_info']['base_runtime_context_loaded']

    # load env
    # this will combine the env defined with the config and those
    # provided with the --env-files.
    env = option.env = cfg.get('env', dict())
    for path in option.env_files or tuple():
        env.update(parse_systemd_envfile(path))
    if len(env) > 0:
        logger.debug(f"loaded env:\n{pformat_yaml(env)}")
    # add the env to the system env
    for k, v in env.items():
        os.environ[k] = v or ''
    # update the runtime with some more info
    rupdate(
            cfg, {
                'runtime': {
                    'config_info': {
                        'env_files': option.env_files,
                        },
                    'env': env,
                    'exec_path': sys.argv[0],
                    'cmd': shlex.join(sys.argv),
                    'version': version.version,
                    }
                })

    # handle subcommands
    parser.bootstrap_actions(option, unknown_args=unknown_args)
