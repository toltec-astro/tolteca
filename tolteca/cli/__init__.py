#! /usr/bin/env python

import sys

from ..utils import get_pkg_data_path
from .. import version
from tollan.utils.log import init_log, get_logger
from tollan.utils.cli.multi_action_argparser import \
        MultiActionArgumentParser
from tollan.utils.sys import parse_systemd_envfile
from tollan.utils import rupdate
from tollan.utils.fmt import pformat_yaml
import yaml
import os
from wrapt import ObjectProxy


__all__ = ['main_parser', 'main']


main_parser = ObjectProxy(None)
"""
A proxy to the
`~tollan.utils.cli.multi_action_argparser.MultiActionArgumentParser`
instance, which is made available when `~tolteca.cli.main` is
called.

This can be used to register subcommands.

"""


def main(args=None):
    """The CLI entry point."""

    prog_name = 'TolTECA'
    prog_desc = 'TolTEC Data Analysis All-in-one!'
    config_default = get_pkg_data_path().joinpath("tolteca.yaml")

    parser = main_parser.__wrapped__ = MultiActionArgumentParser(
            description=f"{prog_name} v{version.version}"
                        f" - {prog_desc}"
            )

    parser.add_argument(
            "-c", "--config",
            nargs='+',
            help="The path to the config file(s). "
                 "Multiple config files are merged in order.",
            metavar='FILE',
            default=[config_default, ])
    parser.add_argument(
            '--env_files', '-e',
            metavar='ENV_FILE', nargs='*',
            help='Path to systemd env file.')
    parser.add_argument(
            "-q", "--quiet",
            help="Suppress debug logs.",
            action='store_true')
    parser.add_argument(
            '-v', '--version', action='store_true',
            help='Print the version info and exit.'
            )

    # import subcommand modules:
    from .db import cmd_migrate  # noqa: F401
    from .run import cmd_run  # noqa: F401

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
    _config = None
    for c in option.config:
        with open(c, 'r') as fo:
            if _config is None:
                _config = yaml.safe_load(fo)
            else:
                rupdate(_config, yaml.safe_load(fo))
    option.config = _config

    # load env
    envs = dict()
    for path in option.env_files or tuple():
        envs.update(parse_systemd_envfile(path))
    if len(envs) > 0:
        logger.debug(f"loaded envs:\n{pformat_yaml(envs)}")
    for k, v in envs.items():
        os.environ[k] = v or ''

    # handle subcommands
    parser.bootstrap_actions(option, unknown_args=unknown_args)
