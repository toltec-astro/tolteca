#! /usr/bin/env python

from tollan.utils.log import get_logger

from . import main_parser, config_loader
from .utils import load_runtime
from .check import (
    register_cli_checker,
    _check_load_rc_config,
    )


@register_cli_checker('simu')
def check_simu(result, option):

    def load_config_cls(result, rc):
        from ..simu import SimuConfig
        return SimuConfig

    def make_config_details(result, rc, config_dict):
        d = config_dict
        return (
                    f'instrument: {d["instrument"]["name"]}\n'
                    f'mapping: {d["mapping"]["type"]}\n'
                    f'sources: {[s["type"] for s in d["sources"]]}'
                    )

    _check_load_rc_config(
        result,
        config_name='simu',
        load_config_cls=load_config_cls,
        make_config_details=make_config_details,
        config_instruction=(
            'Valid simulator config is required to run ``tolteca simu``.'
            'To setup, add in the config file with key "simu" and '
            'value being a dict following one of The examples in the '
            'workdir/doc folder.'
            )
        )
    return result


def load_simulator_runtime(config_loader, **kwargs):

    logger = get_logger()

    from ..simu import SimulatorRuntime

    rt = load_runtime(
        runtime_cls=SimulatorRuntime, config_loader=config_loader, **kwargs)
    logger.debug(f"simu runtime: {rt}")
    return rt


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

        rt = load_simulator_runtime(
            config_loader,
            no_cwd=option.no_cwd,
            runtime_context_dir_only=True,
            runtime_cli_args=unknown_args)
        rt.run()
