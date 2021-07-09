#!/usr/bin/env python


from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml
from tollan.utils.registry import Registry, register_to
from . import main_parser, config
from astropy.utils.console import terminal_size
from textwrap import TextWrapper


__all__ = ['register_cli_checker', ]


_checkers = Registry.create()
"""This holds the checkers of supported submodules."""


class CheckerResult(object):
    """A class to provide feedback for checker result."""

    _symbols = {
            'ok': '✓',
            'error': '✗',
            'info': '•',
            'note': '✎',
            }

    def __init__(self, name):
        self._name = name
        self._lines = list()

    def add_line(self, accent, message):
        self._lines.append((accent, message))

    def pformat(self):
        _, w = terminal_size()
        if w > 120:
            w = 120
        wrapper = TextWrapper(width=w - 2)
        body = []
        note = []
        for accent, message in self._lines:
            wrapper.initial_indent = f'{self._symbols[accent]} '
            wrapper.subsequent_indent = '  '
            message = wrapper.fill(message)
            dest = note if accent == 'note' else body
            dest.append(message)
        name = self._name
        if not body and not note:
            return f'\n{name}: N/A'
        header = f'\n{name}\n{"-" * len(name)}'
        if body:
            body = '\n{}\n'.format('\n'.join(body))
        else:
            body = ''
        if note:
            note = '\n{}\n'.format('\n'.join(note))
        else:
            note = ''
        return '{}{}{}'.format(header, body, note).rstrip('\n')


def register_cli_checker(name):

    result = CheckerResult(name=name)

    def decorator(func):

        @register_to(_checkers, name)
        def wrapped():
            return func(result)

        return wrapped
    return decorator


@register_cli_checker('config')
def check_config(result):

    logger = get_logger()
    logger.debug(f"loaded config:\n{pformat_yaml(config)}")

    cfg_info = config['runtime']['config_info']

    if cfg_info:
        # report in the summary the loaded config files
        for key in ['sys', 'user']:
            result.add_line(
                    'ok' if cfg_info[f'{key}_config_loaded']
                    else 'info',
                    '{} {} config {}{}'.format(
                        'Loaded' if cfg_info[f'{key}_config_loaded']
                        else 'Skipped',
                        key,
                        cfg_info[f'{key}_config_path'],
                        "" if cfg_info[f'{key}_config_path'].exists()
                        else " (non-exist)"
                        )
                    )
        cli_config_paths = set(cfg_info['sources']) - {
                cfg_info['sys_config_path'], cfg_info['user_config_path']}
        if cli_config_paths:
            result.add_line(
                    'ok',
                    f"Loaded {len(cli_config_paths)} config files from CLI"
                    )
        else:
            result.add_line(
                    'info',
                    "No config file loaded from CLI")
        if cfg_info['user_config_loaded'] or cfg_info['sys_config_loaded']:
            result.add_line(
                    'note',
                    'Use "-n" to skip loading sys/user config'
                    )
        else:
            result.add_line(
                    'note',
                    'Remove "-n" to load sys/user config'
                    )
    return result


@main_parser.register_action_parser(
        'check',
        help="Run a check of the runtime environment for troubleshooting."
        )
def cmd_check(parser):
    logger = get_logger()

    parser.add_argument(
            'item',
            nargs='*',
            metavar='ITEM',
            help='The item(s) to check.'
                 ' When not specified, check all of them.'
            )

    @parser.parser_action
    def action(option, unknown_args=None):

        logger.debug(f"option: {option}")
        logger.debug(f"unknown_args: {unknown_args}")

        items = option.item
        if not option.item:
            items = _checkers.keys()

        # check if the items are valid
        items = set(items)
        results = list()
        if items <= set(_checkers.keys()):
            for item in sorted(items):
                results.append(_checkers[item]())
                # summary.append(f'\n{item}\n{"-" * len(item)}')
                # summary.append(f'{_checkers[item]()}')
        logger.info('tolteca check summary:\n{}'.format('\n'.join(
            [r.pformat() for r in results])))
