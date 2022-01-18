#! /usr/bin/env python

import argparse
from tollan.utils.log import get_logger
from tollan.utils import ensure_abspath, getname
from tollan.utils.fmt import pformat_yaml

from . import main_parser, config_loader
from .check import (
    register_cli_checker,
    _MISSING,
    _error_invalid_rcdir
    )


@register_cli_checker('simu')
def check_simu(result, option):

    from ..simu import SimuConfig

    add_simu_instruction = False

    # load the config
    try:
        rc = load_simulator_runtime(config_loader, no_cwd=option.no_cwd)
    except Exception as e:
        _error_invalid_rcdir(result, e)
        return
    if rc is not _MISSING and rc is not None:
        rc_config = rc.config
        if SimuConfig.config_key in rc_config:
            d = rc_config[SimuConfig.config_key]
            result.add_item(
                result.S.info,
                f'Found simu config in {rc}',
                details=(
                    f'instrument: {d["instrument"]["name"]}\n'
                    f'mapping: {d["mapping"]["type"]}\n'
                    f'sources: {[s["type"] for s in d["sources"]]}'
                    )
                )
            add_simu_instruction = False
        else:
            result.add_item(
                result.S.info,
                'No simu config found')
    else:
        result.add_item(
            result.S.info,
            'Skipped checking simu config in runtime context (not loaded).')
    if add_simu_instruction:
        result.add_item(
            result.S.note,
            'Valid simulator config is required to run ``tolteca simu``.'
            'To setup, add in the config file with key "simu" and '
            'value being a dict following one of The examples in the '
            'workdir/doc folder.'
            )
    return result


def load_simulator_runtime(
        config_loader, no_cwd=False, runtime_context_dir_only=False):

    logger = get_logger()

    workdir = config_loader.runtime_context_dir

    if workdir is None and not no_cwd:
        # in this special case we just use the current directory
        workdir = ensure_abspath('.')

    from ..utils import RuntimeContextError
    from ..simu import SimulatorRuntime
    try:
        rc = SimulatorRuntime(workdir)
    except RuntimeContextError as e:
        if (
                config_loader.runtime_context_dir is not None
                or runtime_context_dir_only):
            # raise when user explicitly specified the workdir
            # or requested to load only from rc dir.
            raise argparse.ArgumentTypeError(
                f"invalid workdir {workdir}\n"
                f"{getname(e.__class__)}: {e}")
        else:
            logger.debug("no valid runtime context in current directory")
            # create rc from config
            cfg = config_loader.get_config()
            logger.warning(
                f"create pipeline runtime from config dict:\n"
                f"{pformat_yaml(cfg or dict())}")
            rc = SimulatorRuntime(cfg)
    logger.debug(f"simu rc: {rc}")
    return rc


@main_parser.register_action_parser(
        'simu',
        help="Run tolteca.simu CLI."
        )
def cmd_simu(parser):

    logger = get_logger()

    @parser.parser_action
    def action(option, unknown_args=None):

        logger.debug(f"option: {option}")
        logger.debug(f"unknown_args: {unknown_args}")

        rc = load_simulator_runtime(
            config_loader, no_cwd=option.no_cwd,
            runtime_context_dir_only=True)
        rc.cli_run(args=unknown_args)
