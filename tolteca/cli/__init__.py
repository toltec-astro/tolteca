#! /usr/bin/env python

import sys

from .. import version
from wrapt import ObjectProxy
import argparse


__all__ = ['main_parser', 'config_loader', 'main']


main_parser = ObjectProxy(None)
"""
A proxy to the
`~tollan.utils.cli.multi_action_argparser.MultiActionArgumentParser`
instance, which is made available when `~tolteca.cli.main` is
called.

This can be used to register subcommands.

"""

config_loader = ObjectProxy(None)
"""
A proxy to the `ConfigLoader` instance created via CLI.
"""


def main(args=None):
    """The CLI entry point."""

    prog_name = 'TolTECA'
    prog_desc = 'TolTEC Data Analysis All-in-one!'
    # prog_url = 'http://toltecdr.astro.umass.edu'
    description = f"{prog_name} v{version.version} - {prog_desc}"
    banner = r"""
.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~
     _____  ____  _     _____  _____ ____  ____
    /__ __\/  _ \/ \   /__ __\/  __//   _\/  _ \
      / \  | / \|| |     / \  |  \  |  /  | / \|
      | |  | \_/|| |_/\  | |  |  /_ |  \_ | |-||
      \_/  \____/\____/  \_/  \____\\____/\_/ \|


          http://toltecdr.astro.umass.edu
~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
"""

    # make a "pre-parser" without loading subcommands.
    # this is useful to speed things up for something like `tolteca -v`
    def _add_pre_parser_arguments(parser):
        parser.add_argument(
                "--no_banner",
                help="If set, the banner will not be shown.",
                action='store_true',
                )
        parser.add_argument(
                '-v', '--version', action='store_true',
                help='Print the version info and exit.'
                )
        return parser
    parser = _add_pre_parser_arguments(
        argparse.ArgumentParser(
            description=description, add_help=False))
    option, unknown_args = parser.parse_known_args(args or sys.argv[1:])
    if not option.no_banner:
        # generating the ascii art is too slow.
        # since it is static we just created that inline.
        # from ..utils.misc import make_ascii_banner
        print(banner)

    if option.version:
        print(version.version)
        sys.exit(0)

    # now we create the actual parser with subcommands loaded

    from tollan.utils.cli.multi_action_argparser import \
        MultiActionArgumentParser
    from tollan.utils.cli.path_type import PathType
    from pathlib import Path

    from tollan.utils.log import init_log, get_logger
    from ..utils import ConfigLoader
    from tollan.utils.fmt import pformat_yaml

    parser = main_parser.__wrapped__ = _add_pre_parser_arguments(
        MultiActionArgumentParser(description=description))

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
            type=PathType(exists=False, type_='dir'),
            )
    parser.add_argument(
            "-n", "--no_persistent_config",
            help="If set, skip loading the sys config and user config.",
            action='store_true',
            )
    parser.add_argument(
            "-w", "--no_cwd",
            help="If set, skip automatically set -d to cwd when applicable.",
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
            "-g", "--debug",
            help="Show debug logging messages.",
            action='store_true')

    # import subcommand modules
    # these has to go here because they rely on the main_parser object
    # defined above.
    from .check import cmd_check  # noqa: F401
    from .setup import cmd_setup  # noqa: F401
    from .db import cmd_migrate  # noqa: F401
    # from .run import cmd_run  # noqa: F401
    from .simu0 import cmd_simu0  # noqa: F401
    from .simu import cmd_simu  # noqa: F401
    from .reduce import cmd_reduce  # noqa: F401

    option, unknown_args = parser.parse_known_args(args or sys.argv[1:])

    if option.debug:
        loglevel = 'DEBUG'
    else:
        loglevel = 'INFO'

    init_log(level=loglevel)
    logger = get_logger()

    # create config loader
    cl_kwargs = dict(
        load_sys_config=True,
        load_user_config=True,
        runtime_context_dir=None,
        files=option.config,
        env_files=option.env_files,
        )
    if 'runtime_context_dir' in option:
        cl_kwargs['runtime_context_dir'] = \
            option.runtime_context_dir or Path('.')
    if option.no_persistent_config:
        cl_kwargs.update(load_sys_config=False, load_user_config=False)
    cl = option.config_loader = config_loader.__wrapped__ = \
        ConfigLoader(**cl_kwargs)
    logger.debug(f"config info:{pformat_yaml(cl.to_dict())}")
    # load all the items
    # config.__wrapped__ = cl.get_config()
    # base_runtime_context.__wrapped__ = cl.get_runtime_context(
    #     include_config_as_default=False, include_config_as_override=False)
    # env.__wrapped__ = cl.get_env()
    # handle subcommands
    parser.bootstrap_actions(option, unknown_args=unknown_args)
