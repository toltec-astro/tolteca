#! /usr/bin/env python

from tollan.utils.log import get_logger

from . import main_parser, config_loader
from .utils import load_runtime
from .check import (
    register_cli_checker,
    _check_load_rc_config,
    )


@register_cli_checker('web')
def check_web(result, option):

    def load_config_cls(result, rc):
        from ..web import WebConfig
        return WebConfig

    def make_config_details(result, rc, config_dict):
        apps = config_dict.get('apps', list())
        return f'n_apps: {len(apps)}\n'

    _check_load_rc_config(
        result,
        config_name='web',
        load_config_cls=load_config_cls,
        make_config_details=make_config_details,
        config_instruction=(
            'Web config allows customizing Web-based tools invoked by '
            '``tolteca web``. '
            'Add in the config file with key "web" and '
            'value being a dict following one of the examples in the '
            '{workdir}/doc folder.')
        )
    return result


def load_web_runtime(config_loader, **kwargs):

    from ..web import WebRuntime

    return load_runtime(
        runtime_cls=WebRuntime, config_loader=config_loader, **kwargs)


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
