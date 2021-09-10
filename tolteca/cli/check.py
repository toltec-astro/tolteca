#!/usr/bin/env python


from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml
from tollan.utils.registry import Registry, register_to
from . import main_parser, config, base_runtime_context
from astropy.utils.console import terminal_size
import sys
from textwrap import TextWrapper, indent
from enum import Enum, auto

__all__ = ['register_cli_checker', ]


_checkers = Registry.create()
"""This holds the checkers of supported submodules."""


class CheckerResult(object):
    """A class to collate feedbacks from checkers."""

    class S(Enum):
        ok = auto()
        error = auto()
        info = auto()
        note = auto()

    dispatch_style = {
        S.ok: {
            'symbol': '✓',
            },
        S.error: {
            'symbol': '✗',
            },
        S.info: {
            'symbol': '•',
            },
        S.note: {
            'symbol': '✎',
            },
        }

    def __init__(self, name):
        self._name = name
        self._items = list()

    def add_item(self, style, message, details=None):
        self._items.append((style, message, details))

    def pformat(self):
        _, w = terminal_size()
        if w > 120:
            w = 120
        if w < 30:
            w = 30
        wrapper = TextWrapper(width=w - 2)
        body = []
        note = []
        for style, message, details in self._items:
            s = self.dispatch_style[style]
            wrapper.initial_indent = f'{s["symbol"]} '
            wrapper.subsequent_indent = '  '
            message = wrapper.fill(message)
            if details is not None:
                details = indent(details.strip(), " " * 4)
                message = f'{message}\n{details}'
            dest = note if style is self.S.note else body
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


@register_cli_checker('runtime')
def check_runtime(result):
    rt = config['runtime']
    for key in ['exec_path', 'cmd', 'version']:
        result.add_item(
            result.S.info,
            f'{key}: {rt[key]}'
            )
    return result


@register_cli_checker('config')
def check_config(result):

    logger = get_logger()
    logger.debug(f"loaded config:\n{pformat_yaml(config)}")

    cfg_info = config['runtime']['config_info']

    if cfg_info:
        # report in the summary the loaded config files
        for key in ['sys', 'user']:
            result.add_item(
                    result.S.ok if cfg_info[f'{key}_config_loaded']
                    else result.S.info,
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
            result.add_item(
                    result.S.ok,
                    f"Loaded {len(cli_config_paths)} config files from CLI\n",
                    details=f"{pformat_yaml(cli_config_paths)}"
                    )
        else:
            result.add_item(
                    result.S.info,
                    "No config file loaded from CLI")
        if cfg_info['user_config_loaded'] or cfg_info['sys_config_loaded']:
            result.add_item(
                    result.S.note,
                    'Use "-n" to skip loading sys/user config'
                    )
        else:
            result.add_item(
                    result.S.note,
                    'Remove "-n" to load sys/user config'
                    )
        if base_runtime_context:
            rc = base_runtime_context
            result.add_item(
                result.S.ok,
                f'Found runtime context in {rc.rootpath}',
                details=f"config files:{pformat_yaml(rc.config_files)}"
            )
            result.add_item(
                result.S.note,
                'Remove "-d" to skip loading runtime context dir.'
                )
        else:
            if cfg_info['runtime_context_dir']:
                result.add_item(
                    result.S.info,
                    f'No runtime context found in '
                    f'{cfg_info["runtime_context_dir"]}',
                    )
            else:
                result.add_item(
                    result.S.info,
                    'Skipped loading runtime context.'
                    )
            result.add_item(
                result.S.note,
                'Use "-d" to specify a runtime context directory'
                ' to load'
                    )
            result.add_item(
                    result.S.info,
                    'Refer to "tolteca setup -h" for how to create a '
                    'runtime context directory'
                    )
        if cfg_info['env_files'] is not None:
            env_file_paths = cfg_info['env_files']
            result.add_item(
                result.S.ok,
                f"Loaded {len(env_file_paths)} env files from CLI\n",
                details=f"{pformat_yaml(env_file_paths)}"
                )
        else:
            result.add_item(
                result.S.info,
                "No env files specified"
                )
            result.add_item(
                result.S.note,
                'Use "-e" to specify env files'
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
                 ' When not specified, check all of them. Use "-l" to see'
                 ' the list of available checkers.'
            )
    parser.add_argument(
            '-l', '--list',
            help='List the available checkers.',
            action='store_true'
            )

    @parser.parser_action
    def action(option, unknown_args=None):

        logger.debug(f"option: {option}")
        logger.debug(f"unknown_args: {unknown_args}")

        if option.list:
            print(
                'Available tolteca check items:\n{}'.format(
                    pformat_yaml(list(_checkers.keys()))
                    ))
            sys.exit(0)
        items = option.item
        if not option.item:
            items = _checkers.keys()

        # check if the items are valid
        items = set(items)
        results = list()
        _checker_keys = set(_checkers.keys())
        if items <= _checker_keys:
            for item in sorted(items):
                results.append(_checkers[item]())
                # summary.append(f'\n{item}\n{"-" * len(item)}')
                # summary.append(f'{_checkers[item]()}')
        else:
            unknown_keys = items - _checker_keys
            raise parser.error(
                    f"unknown item to check: {unknown_keys}")
        logger.info('tolteca check summary:\n{}'.format(
            '\n'.join(
                [r.pformat() for r in results])))
