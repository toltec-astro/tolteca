#! /usr/bin/env python

from tollan.utils.log import get_logger
from tollan.utils import ensure_abspath

import argparse
from . import main_parser, config_loader


@main_parser.register_action_parser(
        'simu0',
        help="Run the legacy tolteca.simu0 CLI."
        )
def cmd_simu0(parser):

    logger = get_logger()

    @parser.parser_action
    def action(option, unknown_args=None):

        logger.debug(f"option: {option}")
        logger.debug(f"unknown_args: {unknown_args}")

        workdir = config_loader.runtime_context_dir

        if workdir is None:
            # in this special case we just use the current directory
            workdir = ensure_abspath('.')

        from ..simu0 import SimulatorRuntime
        from ..utils import RuntimeContextError

        try:
            rc = SimulatorRuntime(workdir)
        except RuntimeContextError as e:
            raise argparse.ArgumentTypeError(f"invalid workdir {workdir}: {e}")
        logger.debug(f"simu rc: {rc}")
        rc.cli_run(args=unknown_args)