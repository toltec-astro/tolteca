#! /usr/bin/env python

from tollan.utils.log import get_logger
from tollan.utils import ensure_abspath

import argparse
from . import main_parser, config_loader
from .check import register_cli_checker


@register_cli_checker('reduce')
def check_reduce(result, option):
    return result


@main_parser.register_action_parser(
        'reduce',
        help="Run tolteca.reduce CLI."
        )
def cmd_reduce(parser):

    logger = get_logger()

    @parser.parser_action
    def action(option, unknown_args=None):

        logger.debug(f"option: {option}")
        logger.debug(f"unknown_args: {unknown_args}")

        workdir = config_loader.runtime_context_dir

        if workdir is None:
            # in this special case we just use the current directory
            workdir = ensure_abspath('.')

        from ..reduce import PipelineRuntime
        from ..utils import RuntimeContextError

        try:
            rc = PipelineRuntime(workdir)
        except RuntimeContextError as e:
            raise argparse.ArgumentTypeError(f"invalid workdir {workdir}: {e}")
        logger.debug(f"pipeline rc: {rc}")
        rc.cli_run(args=unknown_args)
