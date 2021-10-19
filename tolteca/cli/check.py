#!/usr/bin/env python


from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml
from tollan.utils.registry import Registry, register_to
from ..utils.runtime_context import RuntimeInfo
from ..utils import RuntimeContext
from . import main_parser, config_loader
from astropy.utils.console import terminal_size
import sys
from textwrap import TextWrapper, indent
import pathlib
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


def _note_specify_runtime_context_dir(result):
    result.add_item(
        result.S.note,
        'Use "-d" to specify a runtime context directory to load',
        details="IMPORTANT: the `--` separator may be needed between the "
                "options and the subcommand, e.g., "
                "`$ tolteca -d -- check`"
        )
    return result


def _error_invalid_config_files(result, exc):
    result.add_item(
        result.S.error,
        'Failed to load config from files.',
        details=f'Reason: {exc}.'
        )
    result.add_item(
        result.S.note,
        'The config files have to be YAML format and have a top-level dict.'
        )
    return result


def _error_invalid_rcdir(result, exc):
    result.add_item(
        result.S.error,
        f'Failed to load runtime context from '
        f'{config_loader.runtime_context_dir}',
        details=f'Reason: {exc}'
        )
    rcd = config_loader.runtime_context_dir.relative_to(
        pathlib.Path.cwd()).as_posix()
    result.add_item(
        result.S.note,
        f'Use `tolteca -d "{rcd}" setup` to properly create tolteca workdir '
        f'for persisted runtime context.'
        )
    return result


def _error_no_rc_setup(result, rc):
    result.add_item(
        result.S.error,
        f'Runtime context {rc} is not setup.'
        )
    result.add_item(
        result.S.note,
        'Run `tolteca setup` to properly setup the tolteca workdir.'
        )


_MISSING = object()  # a marker to mark a failed loading object.


def _check_load_config(result):
    try:
        config = config_loader.get_config()
    except Exception as e:
        config = _MISSING
        _error_invalid_config_files(result, e)
    return config


def _check_load_rc(result):
    try:
        rc = config_loader.get_runtime_context(
            include_config_as_default=False,
            include_config_as_override=False,
            )
    except Exception as e:
        rc = _MISSING
        _error_invalid_rcdir(result, e)
    return rc


@register_cli_checker('runtime')
def check_runtime(result):

    runtime_info_keys = ['exec_path', 'cmd', 'version']

    def _runtime_info_details(rc):
        return pformat_yaml({
            f'{key}': getattr(rc.runtime_info, key)
            for key in runtime_info_keys
            })

    # check default runtime info
    r = RuntimeInfo.schema.load({})
    for k in runtime_info_keys:
        result.add_item(
            result.S.ok,
            f'{k}: {getattr(r, k)}'
            )

    # try load config and check the runtime info
    config = _check_load_config(result)
    if config is not _MISSING:
        # check runtime info in config
        result.add_item(
            result.S.ok,
            'Runtime info found in loaded config files:',
            # unwrap because deep copy is required on config
            details=_runtime_info_details(RuntimeContext(config))
            )
    # try load rc
    rc = _check_load_rc(result)
    if rc is not _MISSING:
        if rc is None:
            # no rc specified in loader
            result.add_item(
                result.S.info,
                'Skipped check runtime info in runtime context (not loaded).'
                )
            _note_specify_runtime_context_dir(result)
        else:
            result.add_item(
                result.S.ok,
                f'Runtime info found in {rc}:',
                details=_runtime_info_details(rc)
                )
            # check setup rc.
            setup_rc = rc.get_setup_rc()
            if setup_rc is not None:
                result.add_item(
                    result.S.ok,
                    f'Found setup info in {rc}:',
                    details=_runtime_info_details(setup_rc)
                    )
            else:
                _error_no_rc_setup(result, rc)
    return result


@register_cli_checker('config')
def check_config(result):

    for key in ['sys', 'user']:
        loaded = getattr(config_loader, f'load_{key}_config')
        path = getattr(config_loader, f'{key}_config_path')
        result.add_item(
                result.S.ok if loaded
                else result.S.info,
                '{} {} config {}{}'.format(
                    'Loaded' if loaded else 'Skipped',
                    key,
                    path,
                    "" if path.exists() else " (non-exist)"
                    )
                )
    scf = config_loader.standalone_config_files
    if scf:
        result.add_item(
                result.S.ok,
                f"Loaded {len(scf)} config files from CLI\n",
                details=f"{pformat_yaml(scf)}"
                )
    else:
        result.add_item(
                result.S.info,
                "No config file loaded from CLI")
    if config_loader.load_user_config or config_loader.load_sys_config:
        result.add_item(
                result.S.note,
                'Use "-n" to skip loading sys/user config'
                )
    else:
        result.add_item(
                result.S.note,
                'Remove "-n" to load sys/user config'
                )
    rcdir = config_loader.runtime_context_dir
    if rcdir is None:
        result.add_item(
            result.S.info,
            'Skipped loading runtime context.'
            )
        _note_specify_runtime_context_dir(result)
    else:
        rc = _check_load_rc(result)
        if rc is not _MISSING:
            cb = rc.config_backend
            result.add_item(
                result.S.ok,
                f'Found runtime context in {rc.rootpath}',
                details=f"config files:{pformat_yaml(cb.config_files)}"
                )
        result.add_item(
            result.S.note,
            'Remove "-d" to skip loading runtime context dir.'
            )

    # env
    env_files = config_loader.env_files
    if env_files:
        result.add_item(
            result.S.ok,
            f"Loaded {len(env_files)} env files from CLI\n",
            details=f"{pformat_yaml(env_files)}"
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
        else:
            unknown_keys = items - _checker_keys
            raise parser.error(
                f"unknown item to check: {unknown_keys}."
                f" Available items: {_checker_keys}")
        summary = '\n'.join(r.pformat() for r in results)
        print(summary)
        # print(inspect.cleandoc(
        #     f"""
# =============
# tolteca check
# =============
# {summary}
        #     """))
        # logger.info('tolteca check summary:\n{}'.format(
        #     )))
