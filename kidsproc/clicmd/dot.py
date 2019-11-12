#! /usr/bin/env python

"""Command to introspect the CLI runtime and invoke functions."""


import click
import inspect
import itertools
from ..utils.log import get_logger, timeit
from ..utils.cli.click_helpers import varg_command, OPTION_SETTINGS
from ..utils.fmt import pformat_obj
from ..utils import rgetattr
from ..cli import cli


@varg_command(
    cli, '.',
    click.argument('obj')
)
@click.option(
        '-c', '--cast_type',
        is_flag=True,
        default=False,
        help="Do type cast for input args.",
        **dict(
            OPTION_SETTINGS,
            show_default=True
            )
        )
@click.option(
        '-n', '--no_parse',
        is_flag=True,
        default=False,
        help="Run command with arguments without varstore/atsyntax parsing.",
        **dict(
            OPTION_SETTINGS,
            show_default=True
            )
        )
@click.pass_context
@timeit
def cmd_dot(ctx, obj, args, cast_type=False, no_parse=False):
    """Invoke function or inspect object."""
    logger = get_logger()

    rt = ctx.obj

    def resolve_type(arg):
        try:
            return int(arg)
        except ValueError:
            try:
                return float(arg)
            except ValueError:
                return arg

    def resolve_args(func, args):
        if len(func) > 1:
            return resolve_args(func[1:], resolve_args(func[:1], args))
        else:
            func = func[0]
            if func is None:
                return args
            logger.debug(f'resolve args {args} with {func}')
            result = []
            for arg in map(func, args):
                if isinstance(arg, list):
                    result.extend(arg)
                else:
                    result.append(arg)
            return result

    if obj == "":
        # if no obj is set, inspect the runtime
        m = rt
    else:
        # TODO make this handle more levels of modules
        rt.load_module(obj.lstrip(".").split(".", 1)[0])
        m = rgetattr(rt, obj)

    if inspect.isclass(m):
        click.echo(pformat_obj(m))
    if callable(m):
        args = resolve_args([
            resolve_type if cast_type else None,
            ], args)
        # the resolved args can be a list of args
        # we issue a series of calls with the permutations of them.
        allargs = [a if isinstance(a, tuple)
                   else (a, ) for a in args]
        for args_ in itertools.product(*allargs):
            logger.debug(f'dot cmd args: {args_}')
            r = m(*args_)
            if r is not None:
                click.echo(r)
    else:
        # regular object
        click.echo(pformat_obj(m))
