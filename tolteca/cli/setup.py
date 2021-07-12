#! /usr/bin/env python

from tollan.utils.log import get_logger
from tollan.utils import rupdate
from tollan.utils.cli.path_type import PathType

from .. import version
from . import main_parser
from .check import register_cli_checker
from ..utils import RuntimeContext

import sys


@register_cli_checker('setup')
def check_setup(result):
    return result


@main_parser.register_action_parser(
        'setup',
        help="Setup a pipeline/simu workdir."
        )
def cmd_setup(parser):

    parser.add_argument(
            'workdir',
            type=PathType(exists=None, type_="dir"),
            metavar="DIR",
            help="The workdir to setup.",
            )
    parser.add_argument(
            "-f", "--force", action="store_true",
            help="Force the setup even if DIR is not empty",
            )
    parser.add_argument(
            "-o", "--overwrite", action="store_true",
            help="Overwrite any existing file without backup in case "
                 "a forced setup is requested"
            )
    parser.add_argument(
            "-n", "--dry_run", action="store_true",
            help="Run without actually create files."
            )

    @parser.parser_action
    def action(option, unknown_args=None):
        logger = get_logger()

        logger.debug(f"option: {option}")
        logger.debug(f"unknown_args: {unknown_args}")

        config = option.config or dict()
        rupdate(
            config,
            {
                'setup': {
                    'jobkey': option.workdir.resolve().name,
                    'exec_path': sys.argv[0],
                    'cmd': ' '.join(sys.argv),
                    'version': version.version
                    }
            })

        ctx = RuntimeContext.from_dir(
                option.workdir,
                create=True,
                force=option.force,
                overwrite=option.overwrite,
                dry_run=option.dry_run,
                init_config=config
                )
        logger.debug(f"runtime context: {ctx}")
        ctx.config  # load config
