#! /usr/bin/env python

from tollan.utils.log import get_logger

from . import main_parser, config_loader
from .utils import load_runtime


@main_parser.register_action_parser(
        'dp',
        help="CLI to manage data products."
        )
def cmd_dp(parser):

    logger = get_logger()

    @parser.parser_action
    def action(option, unknown_args=None):

        logger.debug(f"option: {option}")
        logger.debug(f"unknown_args: {unknown_args}")

        from ..db import DatabaseRuntime
        from ..datamodels.db.toltec import data_prod

        dbrt = load_runtime(
                DatabaseRuntime,
                config_loader=config_loader,
                no_cwd=option.no_cwd,
                runtime_context_dir_only=True,
                runtime_cli_args=unknown_args)
        logger.debug(f'dbrt config: {dbrt.config.to_dict()}')
        dpdb = dbrt.dpdb
        data_prod.init_db(
            dpdb,
            create_tables=True
            )
        data_prod.init_orm(dpdb)
        # TODO
        # implement dp related operations.
