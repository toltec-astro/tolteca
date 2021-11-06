#! /usr/bin/env python

import argparse
from tollan.utils.log import get_logger
from tollan.utils import ensure_abspath
# from tollan.utils.fmt import pformat_yaml

from . import main_parser, config_loader
from .check import (
    register_cli_checker,
    _MISSING,
    _error_invalid_rcdir
    )


@register_cli_checker('reduce')
def check_reduce(result, option):

    from ..reduce import ReduConfig

    add_reduce_instruction = False

    # load the config
    try:
        rc = load_pipeline_runtime(config_loader, no_cwd=option.no_cwd)
    except Exception as e:
        _error_invalid_rcdir(result, e)
        return
    if rc is not _MISSING and rc is not None:
        rc_config = rc.config
        if ReduConfig.config_key in rc_config:
            d = rc_config[ReduConfig.config_key]
            result.add_item(
                result.S.info,
                f'Found reduction config in {rc}',
                details=(
                    f'n_inputs: {len(d["inputs"])}\n'
                    f'n_steps: {len(d["steps"])}'
                    )
                )
            add_reduce_instruction = False
        else:
            result.add_item(
                result.S.info,
                'No reduction config found')
    else:
        result.add_item(
            result.S.info,
            'Skipped checking reduction config in runtime context '
            '(not loaded).')
    if add_reduce_instruction:
        result.add_item(
            result.S.note,
            'Valid reduction config is required to run ``tolteca reduce``.'
            'To setup, add in the config file with key "reduce" and '
            'value being a dict following one of the examples in the '
            'workdir/doc folder.'
            )
    return result


def load_pipeline_runtime(config_loader, no_cwd=False):

    logger = get_logger()

    workdir = config_loader.runtime_context_dir

    if workdir is None and not no_cwd:
        # in this special case we just use the current directory
        workdir = ensure_abspath('.')

    from ..utils import RuntimeContextError
    from ..reduce import PipelineRuntime
    try:
        rc = PipelineRuntime(workdir)
    except RuntimeContextError as e:
        if config_loader.runtime_context_dir is not None:
            # raise when user explicitly specified the workdir
            raise argparse.ArgumentTypeError(
                f"invalid workdir {workdir}: {e}")
        else:
            logger.debug("no valid runtime context in current directory")
            # create rc from config
            rc = PipelineRuntime(config_loader.get_config())
    logger.debug(f"pipeline rc: {rc}")
    return rc


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

        rc = load_pipeline_runtime(config_loader, no_cwd=option.no_cwd)
        rc.cli_run(args=unknown_args)
