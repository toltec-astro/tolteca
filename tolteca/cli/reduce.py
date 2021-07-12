#! /usr/bin/env python

from tollan.utils.log import get_logger
from tollan.utils.cli.path_type import PathType

from pathlib import Path
import argparse
from . import main_parser
from .check import register_cli_checker


@register_cli_checker('reduce')
def check_reduce(result):
    return result


@main_parser.register_action_parser(
        'reduce',
        help="Run tolteca.reduce CLI."
        )
def cmd_reduce(parser):

    parser.add_argument(
            '--dir', '-d',
            type=PathType(exists=True, type_="dir"),
            metavar="DIR",
            help="The work dir to use",
            )

    logger = get_logger()

    @parser.parser_action
    def action(option, unknown_args=None):

        from ..reduce import PipelineRuntime
        from ..utils import RuntimeContextError

        workdir = option.dir or Path.cwd()

        try:
            ctx = PipelineRuntime.from_dir(
                workdir,
                create=False,
                force=True,
                overwrite=False,
                dry_run=False
                )
        except RuntimeContextError as e:
            raise argparse.ArgumentTypeError(f"invalid workdir {workdir}: {e}")
        logger.debug(f"pipeline ctx: {ctx}")
        ctx.run()
