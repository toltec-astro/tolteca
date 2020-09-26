#! /usr/bin/env python

from tollan.utils.log import get_logger, logit, init_log
from tollan.utils import getobj
import inspect
from . import main_parser


@main_parser.register_action_parser(
        'run',
        help="Run a submodule script."
        )
def cmd_run(parser):

    parser.add_argument(
            'script',
            help='The script to execute.'
            )

    @parser.parser_action
    def action(option, unknown_args=None):
        logger = get_logger()
        m = f'tolteca.{option.script}'
        try:
            obj = getobj(m)
        except Exception as e:
            raise ValueError(f'error in module {m}: {e}')
        # this may get the logger reset so we'd better reset the logger
        # again.
        if option.quiet:
            loglevel = 'INFO'
        else:
            loglevel = 'DEBUG'
        init_log(level=loglevel)

        if inspect.ismodule(obj):
            func = getattr(obj, 'main', None)
            if func is None:
                raise ValueError(f"no entry point found in {m}")
        else:
            func = obj
        with logit(logger.debug, "run {} {}".format(
            option.script, ' '.join(unknown_args)
                ).strip()):
            if len(unknown_args) == 0:
                return func()
            return func(unknown_args)
