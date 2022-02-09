#! /usr/bin/env python

from tollan.utils.log import get_logger
from tollan.utils import ensure_abspath

from . import main_parser, config_loader
from .utils import load_runtime
from .check import (
    register_cli_checker,
    _MISSING,
    _check_load_rc,
    )


@register_cli_checker('web')
def check_web(result, option):

    # load the rc
    rc = _check_load_rc(result)

    add_web_instruction = False

    from ..web import WebConfig
    config_cls = WebConfig

    if rc is not _MISSING and rc is not None:
        rc_config = rc.config
        if config_cls.config_key in rc_config:
            d = rc_config[config_cls.config_key]
            result.add_item(
                result.S.info,
                f'Found web config in {rc}',
                details=(
                    f'n_apps: {len(d["apps"])}\n'
                    )
                )
            add_web_instruction = False
        else:
            result.add_item(
                result.S.info,
                'No web config found')
    else:
        result.add_item(
            result.S.info,
            'Skipped checking web config in runtime context '
            '(not loaded).')
    if add_web_instruction:
        result.add_item(
            result.S.note,
            'Valid web config is required to run ``tolteca web``.'
            'To setup, add in the config file with key "web" and '
            'value being a dict following one of the examples in the '
            '{workdir}/doc folder.'
            )
    return result


def load_web_runtime(
        config_loader, no_cwd=False,
        runtime_context_dir_only=False,
        runtime_cli_args=None):

    # logger = get_logger()

    workdir = config_loader.runtime_context_dir

    if workdir is None and not no_cwd:
        # in this special case we just use the current directory
        workdir = ensure_abspath('.')

    from ..web import WebRuntime

    rc = load_runtime(
        runtime_cls=WebRuntime, config_loader=config_loader,
        no_cwd=no_cwd, runtime_context_dir_only=runtime_context_dir_only,
        runtime_cli_args=runtime_cli_args)
    return rc


@main_parser.register_action_parser(
        'web',
        help="Run tolteca.web CLI."
        )
def cmd_web(parser):

    logger = get_logger()

    parser.add_argument(
            '-a', '--app', metavar='APP', required=True,
            help='The name of app to run.'
            )
    parser.add_argument(
            '--ext_proc', metavar='EXT', default='flask',
            choices=['flask', 'celery', 'beat', 'flower'],
            help='The extension process to run. See DashA doc for details.'
            )

    @parser.parser_action
    def action(option, unknown_args=None):

        logger.debug(f"option: {option}")
        logger.debug(f"unknown_args: {unknown_args}")

        rt = load_web_runtime(
            config_loader,
            no_cwd=option.no_cwd,
            runtime_context_dir_only=False,
            runtime_cli_args=unknown_args)
        rt.run(
            app_name=option.app, ext_proc_name=option.ext_proc)
