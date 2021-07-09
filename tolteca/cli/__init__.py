#! /usr/bin/env python

import sys

from ..utils import get_pkg_data_path, get_user_data_dir
from .. import version
from tollan.utils.log import init_log, get_logger
from tollan.utils.cli.multi_action_argparser import \
        MultiActionArgumentParser
from tollan.utils.sys import parse_systemd_envfile
from tollan.utils import rupdate
from tollan.utils.fmt import pformat_yaml
from astropy.io.misc import yaml
import os
import shlex
from wrapt import ObjectProxy
from pathlib import Path


__all__ = ['main_parser', 'config', 'main']


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


def _load_config_from_file(filepath, schema=None):
    empty = dict()
    if filepath.exists():
        with open(filepath, 'r') as fo:
            cfg = yaml.load(fo)
        if cfg is None:
            cfg = empty
    else:
        cfg = empty
    if schema is None:
        return cfg
    return schema.validate(cfg)


def load_config_from_files(
        paths=None,
        load_sys_config=True, load_user_config=True):
    """Load config from files.

    Parameters
    ----------
    filepaths : list, optional
        The paths of the YAML config files to load.
    load_sys_config : bool
        If True, load the system wide tolteca.yaml in the package data.
    load_app_config : bool
        If True, load the tolteca.yaml in the user app directory.
    """
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
                }
            }
        })
    return cfg


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
                 "which can be disabled by the switch -n",
            metavar='FILE',
            )
    parser.add_argument(
            "-n", "--no_persistent_config",
            help="If set, skip loading the sys config and user config.",
            action='store_true',
            )
    parser.add_argument(
            '-e', '--env_files',
            metavar='ENV_FILE', nargs='*',
            help='Path to systemd env file. '
                 'Multiple files are merged in order')
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

    if option.version:
        print(version.version)
        sys.exit(0)

    # load config
    cfg_kwargs = dict(load_sys_config=True, load_user_config=True)
    if option.no_persistent_config:
        cfg_kwargs.update(load_sys_config=False, load_user_config=False)
    cfg = option.config = config.__wrapped__ = load_config_from_files(
            paths=option.config, **cfg_kwargs)

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
                    'env': env,
                    'exec_path': sys.argv[0],
                    'cmd': shlex.join(sys.argv),
                    'version': version.version,
                    }
                })

    # handle subcommands
    parser.bootstrap_actions(option, unknown_args=unknown_args)
