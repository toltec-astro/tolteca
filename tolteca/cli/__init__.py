#! /usr/bin/env python

import sys

from ..utils import get_pkg_data_path
from .. import version
from tollan.utils.log import init_log
from tollan.utils.cli.multi_action_argparser import \
        MultiActionArgumentParser
from tollan.utils import rupdate
import yaml
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
            "-q", "--quiet",
            help="Suppress debug logs.",
            action='store_true')
    parser.add_argument(
            '-v', '--version', action='store_true',
            help='Print the version info and exit.'
            )

    # import subcommand modules:
    from .db import cmd_migrate  # noqa: F401

    # parse and handle global args:
    option = parser.parse_args(args or sys.argv[1:])

    if option.quiet:
        loglevel = 'INFO'
    else:
        loglevel = 'DEBUG'
    init_log(level=loglevel)

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

    # handle subcommands
    parser.bootstrap_actions(option)
