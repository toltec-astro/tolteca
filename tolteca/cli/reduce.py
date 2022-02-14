#! /usr/bin/env python

from tollan.utils.log import get_logger

from . import main_parser, config_loader
from .utils import load_runtime
from .check import (
    register_cli_checker,
    _check_load_rc_config,
    )


@register_cli_checker('reduce')
def check_reduce(result, option):

    def load_config_cls(result, rc):
        from ..reduce import ReduConfig
        return ReduConfig

    def make_config_details(result, rc, config_dict):
        d = config_dict
        return (
                    f'n_inputs: {len(d["inputs"])}\n'
                    f'n_steps: {len(d["steps"])}'
                    )
    _check_load_rc_config(
        result,
        config_name='reduction',
        load_config_cls=load_config_cls,
        make_config_details=make_config_details,
        config_instruction=(
            'Valid reduction config is required to run ``tolteca reduce``.'
            'To setup, add in the config file with key "reduce" and '
            'value being a dict following one of the examples in the '
            'workdir/doc folder.'
            )
        )
    return result


def load_pipeline_runtime(
        config_loader, **kwargs):

    logger = get_logger()

    from ..reduce import PipelineRuntime

    rt = load_runtime(
        runtime_cls=PipelineRuntime, config_loader=config_loader, **kwargs)
    logger.debug(f"pipeline runtime: {rt}")
    return rt


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

        rt = load_pipeline_runtime(
            config_loader,
            no_cwd=option.no_cwd,
            runtime_context_dir_only=True,
            runtime_cli_args=unknown_args)
        rt.run()
