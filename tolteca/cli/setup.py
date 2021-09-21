#! /usr/bin/env python

from tollan.utils.log import get_logger, logit
from tollan.utils import ensure_abspath
from tollan.utils.fmt import pformat_yaml

from textwrap import indent
from . import main_parser, config_loader
from .check import (
    register_cli_checker,
    _check_load_rc, _MISSING, _note_specify_runtime_context_dir,
    _error_no_rc_setup)
from ..utils import RuntimeContext, RuntimeContextError


@register_cli_checker('setup')
def check_setup(result):

    runtime_info_keys = ['exec_path', 'cmd', 'version']

    def _runtime_info_details(rc):
        return pformat_yaml({
            f'{key}': getattr(rc.runtime_info, key)
            for key in runtime_info_keys
            })

    rc = _check_load_rc(result)
    if rc is _MISSING:
        return result
    if rc is None:
        result.add_item(
            result.S.info,
            'Nothing to check (no runtime context)'
            )
        _note_specify_runtime_context_dir(result)
        return result
    # check setup info
    setup_rc = rc.get_setup_rc()
    if setup_rc is not None:
        result.add_item(
            result.S.ok,
            f'Found setup info in {rc}:',
            details=_runtime_info_details(setup_rc)
            )
    else:
        _error_no_rc_setup(result, rc)
        _note_specify_runtime_context_dir(result)
    return result


@main_parser.register_action_parser(
        'setup',
        help="Setup a pipeline/simu workdir."
        )
def cmd_setup(parser):

    logger = get_logger()

    parser.add_argument(
            "-f", "--force", action="store_true",
            help="Force the setup even if DIR is not empty",
            )
    parser.add_argument(
            "-o", '--overwrite', action="store_true",
            help="If set, overwrite exist setup info."
            )
    parser.add_argument(
            "--no_backup", action="store_true",
            help="Disable creation of backup files when `-f` is used."
            )
    parser.add_argument(
        "-n", "--no_config_file", action="store_true",
        help='If set, the config in config files (system, user, and '
             'standalone via `-c`) will not be included in the setup.yaml.'
            )
    parser.add_argument(
            "--dry_run", action="store_true",
            help="Run without actually create/modify files."
            )

    @parser.parser_action
    def action(option, unknown_args=None):

        logger.debug(f"option: {option}")
        logger.debug(f"unknown_args: {unknown_args}")

        rcdir = config_loader.runtime_context_dir

        if rcdir is None:
            # in this special case we just use the current directory
            rcdir = ensure_abspath('.')
        with logit(logger.info, f'create/setup workdir {rcdir}'):
            if option.no_config_file:
                init_config = None
            else:
                init_config = config_loader.get_config()
            rc = RuntimeContext.from_dir(
                    rcdir,
                    create=True,
                    force=option.force,
                    disable_backup=option.no_backup,
                    dry_run=option.dry_run,
                    init_config=init_config
                    )
            setup_rc = rc.get_setup_rc()
            if setup_rc is not None:
                def _make_setup_details(rc):
                    return {
                        'created_at': rc.setup_info.created_at,
                        'created_by': f"{setup_rc.runtime_info.username}@"
                                      f"{setup_rc.runtime_info.hostname}",
                        'created_in': f"{setup_rc.rootpath}",
                        }
                setup_details = indent(
                    pformat_yaml(_make_setup_details(setup_rc)), ' ' * 4)
                if not option.overwrite:
                    raise RuntimeContextError(
                        f"the workdir is already setup:\n{setup_details}\n"
                        f"use `-o` to proceed anyways.")
                # proceed with overwriting
                logger.info(f"overwrite existing setup:\n{setup_details}")

            # the setup file get backup already in the from_dir
            # so no need to do it again.
            rc.setup(
                overwrite=True,
                runtime_info_only=False,
                setup_filepath=None,
                backup_setup_file=False,
                )
        logger.debug(f"runtime context: {rc}")
        logger.debug(f"rc config: {rc.config}")
        logger.info(
            f"""

~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

The following path is sucessfully set up as a tolteca workdir:

    {rcdir}

View the README.md for a brief tour of the contents within.

.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~
""")
