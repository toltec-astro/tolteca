#! /usr/bin/env python
"""The CLI entry point."""

import os
import sys
import click
import signal
import textwrap
import traceback

from . import version, PROGNAME
from .utils.fmt import pformat_obj
from .utils.log import get_logger, logit
from .utils.cli import click_helpers as ch
from .utils.cli import click_log, cli_header
from .utils.cli.click_helpers import (
        ctx_no_recreate_obj,
        )
from .clicmd import CliRuntime

SHOW_BANNER = False
CONTEXT_SETTINGS = dict(
        **ch.CONTEXT_SETTINGS,
        auto_envvar_prefix=PROGNAME.lower()
        )
OPTION_SETTINGS = ch.OPTION_SETTINGS


@click.group(
        context_settings=CONTEXT_SETTINGS,
        help=cli_header(),
        invoke_without_command=True,
        chain=True,
        )
@click.option(
        '--debug',
        is_flag=True,
        default=False,
        help='Enable debug messages.',
        **OPTION_SETTINGS)
@click.option(
        '--logfile',
        metavar='FILE',
        type=click.STRING,
        help='Write logging messages to file.',
        **OPTION_SETTINGS)
@click.version_option(
        version.version,
        '-v', '--version',
        message='%(version)s',
        **OPTION_SETTINGS,
        )
@click.pass_context
def cli(ctx, debug, logfile):
    # the below will check if the command is invoke from other command,
    # if so we reuse the existing runtime object.
    if ctx_no_recreate_obj(ctx):
        ctx.obj.reset()
        return
    # here the command is invoked for the first time
    click_log.init(
            level='INFO' if not debug else 'DEBUG',
            file_=logfile
            )
    logger = get_logger()
    if ctx.invoked_subcommand is None:
        raise click.UsageError(
                "No command specified. Run -h/--help for usage.")

    rt = ctx.ensure_object(CliRuntime)
    rt._ctx = ctx
    # register the close method
    ctx.call_on_close(rt.close)
    logger.debug(f"cli runtime: {pformat_obj(rt)}")


@cli.resultcallback()
def process_callbacks(cbs, *args, **kwargs):
    logger = get_logger()
    cbs = [cb for cb in cbs if cb is not None]
    if cbs:
        logger.debug(f"run registered callbacks: {cbs}")
        for cb in cbs:
            cb()
    return 0


class OnExitHandler(object):

    logger = get_logger()
    ctx = None
    msg = None
    code = 0
    exception_to_raise = None

    # avoid calling this multiple times
    _called_once = False

    def __call__(self, msg=None, frame=None):
        if self._called_once:
            return
        self._called_once = True
        if msg is not None:
            self.msg = msg
        elif isinstance(msg, int):
            self.msg = signal.Signals(msg).name
        self.logger.debug(
                f'exit {self.code}: {self.msg} '
                f'frame={frame} exc={self.exception_to_raise}')
        if self.ctx is not None and self.ctx.obj is not None:
            # this should in general not happen
            with logit(self.logger.debug, "final cleanup"):
                self.ctx.obj.close()
        if self.exception_to_raise is not None:
            raise self.exception_to_raise
        else:
            tb = traceback.format_exc()
            if not tb.startswith("NoneType:"):
                self.logger.debug(f"traceback: {tb}")
            if SHOW_BANNER:
                banner = (
                    "",
                    "*************************",
                    "|                       |",
                    "|      Jobs Done!       |",
                    "|                       |",
                    "*+++++++++++++++++++++++*")
                indent = 0
                bw = len(banner[-1])
                w, _ = click.get_terminal_size()
                indent = (w - bw) // 2
                if indent > bw:
                    indent = bw
                self.logger.debug(textwrap.indent(
                            '\n'.join(banner), ' ' * indent))


# ################################################
# commands
from .clicmd.dot import cmd_dot  # noqa: F401
from .clicmd.ipy import cmd_ipy  # noqa: F401
from .clicmd.setup import cmd_setup  # noqa: F401
# ################################################


def main():

    on_exit_handler = OnExitHandler()
    try:
        ctx = cli.make_context(os.path.basename(sys.argv[0]), sys.argv[1:])
        on_exit_handler.ctx = ctx
        on_exit_handler.code = cli.invoke(ctx)
    except click.exceptions.Exit as e:
        on_exit_handler.msg = e.__class__.__name__
        on_exit_handler.code = 0
    except (Exception, KeyboardInterrupt) as e:
        # logger = getNamedLogger()
        # tb = traceback.format_exc()
        # logger.debug(f'{tb}')
        on_exit_handler.msg = e.__class__.__name__
        on_exit_handler.code = 1
        if isinstance(e, click.ClickException):
            e.show()
        elif isinstance(e, KeyboardInterrupt):
            pass
        else:
            on_exit_handler.exception_to_raise = e
        # raise e
    except BaseException as e:
        on_exit_handler.msg = e.__class__.__name__
        raise e
    finally:
        on_exit_handler()


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
