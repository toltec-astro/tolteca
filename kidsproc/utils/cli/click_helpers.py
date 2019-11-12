#! /usr/bin/env python

"""CLI helpers."""

import re
import os
import click
from contextlib import contextmanager

CONTEXT_SETTINGS = dict(
            help_option_names=('-h', '--help'),
            )
OPTION_SETTINGS = dict(
        allow_from_autoenv=False,
        show_envvar=True
        )
CTX_NO_RECREATE_OBJ = '_ctx_no_recreate_obj'


class OptionalArgumentGroup(click.Command):
    def parse_args(self, ctx, args):
        if args[0] in self.commands:
            if len(args) == 1 or args[1] not in self.commands:
                args.insert(0, '')
        super().parse_args(ctx, args)


class CommandAfterArgs(click.Command):

    split_arg = 'args'
    split_token = '^'

    def parse_args(self, ctx, args):
        parsed_args = super().parse_args(ctx, args)
        if self.split_arg in ctx.params and (
            ctx.params[self.split_arg] is not None) and (
                self.split_token in ctx.params[self.split_arg]):
            isplit = ctx.params['args'].index(self.split_token)
            args = ctx.params['args'][:isplit]
            rest = list(ctx.params['args'][isplit + 1:])
            ctx.params['args'] = args
            ctx.args = rest
            parsed_args = ctx.args
        return parsed_args


def varg_command(cli, name, *others):
    def decorator(func):
        r = click.argument(CommandAfterArgs.split_arg, nargs=-1)(func)
        for o in reversed(others):
            r = o(r)
        return cli.command(
                name,
                cls=CommandAfterArgs,
                context_settings={"ignore_unknown_options": True}
                )(r)
    return decorator


def split_paths(ctx, param, value):
    try:
        vs = []
        for v in value:
            vs.extend(re.split(r'[:;,]+', v))
        paths = tuple(param.type.convert(v, param, ctx) for v in vs)
        return paths
    except ValueError:
        raise click.BadParameter('invalid argument')


def resolve_path(ctx, param, value):
    if value is None:
        value = '.'
    try:
        value = param.type.convert(os.path.expanduser(value), param, ctx)
        return value
    except ValueError:
        raise click.BadParameter('invalid argument')


def getctxroot(ctx):
    parent = ctx.parent
    while parent.parent is not None:
        parent = parent.parent
    return parent


@contextmanager
def hook_ctx_make_context(group):
    # hook the make context function so that we can setup flag
    # for reusing the objects
    _make_context = group.make_context

    def _make_context_hooked(*args, **kwargs):
        ctx = _make_context(*args, **kwargs)
        setattr(ctx, CTX_NO_RECREATE_OBJ, True)
        return ctx
    group.make_context = _make_context_hooked
    try:
        yield group
    finally:
        group.make_context = _make_context


def ctx_no_recreate_obj(ctx):
    return getattr(ctx, CTX_NO_RECREATE_OBJ, False)
