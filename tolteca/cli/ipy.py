#! /usr/bin/env python
"""Command to drop into IPython interactive mode."""


import click
from tollan.utils import hookit
from tollan.utils.log import get_logger, timeit, logit
from tollan.utils.fmt import pformat_obj, pformat_dict
from . import cli, OPTION_SETTINGS, cli_header
from tollan.utils.cli.click_helpers import split_option_arg
# from functools import lru_cache
# import inspect


@cli.command('ipy')
@click.option(
        '-n', '--no_default_modules', 'ipy_no_default_modules',
        is_flag=True,
        default=False,
        help='Do not load default modules.',
        **OPTION_SETTINGS)
@click.option(
        '-m', '--load_modules', 'ipy_load_modules',
        help='Load specified modules (comma separated list)'
             ' in addition to the default ones',
        **OPTION_SETTINGS)
@click.option(
        '-u', '--use_modules', 'ipy_use_modules',
        is_flag=True,
        default=True,
        help='Update the locals() with modules so they'
             ' can be accessed directly',
        **OPTION_SETTINGS)
@click.pass_obj
@timeit
def cmd_ipy(rt, ipy_no_default_modules, ipy_load_modules, ipy_use_modules):

    logger = get_logger()
    locals_ = locals()

    with hookit(rt, 'load_module') as hook:
        if ipy_use_modules:
            def post(self):
                nonlocal locals_
                locals_.update(rt.modules)
                logger.debug(f"locals updated")
            hook.set_post_func(post)
        if not ipy_no_default_modules:
            rt.load_modules()
        if ipy_load_modules:
            for m in split_option_arg(ipy_load_modules):
                rt.load_module(m)
            del m  # this clashes with the ipy magic

        # init ipython
        import IPython
        from IPython.core import magic_arguments
        from IPython.core.magic import (
                Magics, magics_class, line_magic)
        from IPython.terminal.embed import InteractiveShellEmbed

        shortcuts = dict()

        def register_shortcut(func):
            if func not in shortcuts:
                shortcuts[func] = func.__doc__
            return func

        def list_shortcuts():
            return dict(v.split(':', 1) for v in shortcuts.values())

        def header():
            return (
                f"{cli_header()}\n\nRuntime: {pformat_obj(rt)}\n\n"
                f"Magics: {pformat_dict(list_shortcuts(), minw=0)}")

        @magics_class
        class RuntimeMagics(Magics):

            @line_magic
            @register_shortcut
            def h(self, parameter_s):
                """%h: show this message"""
                print(f"{header()}")

            @line_magic
            @register_shortcut
            def m(self, parameter_s):
                """%m: list the current loaded modules"""
                print("Loaded modules:\n{}".format(pformat_obj(rt.modules)))

            @line_magic
            @magic_arguments.magic_arguments()
            @magic_arguments.argument(
                    'modules', nargs='+',
                    help='The modules to load. If empty, load the defaults')
            @register_shortcut
            def ml(self, parameter_s):
                """%ml ...: load module(s) to the runtime."""
                args = magic_arguments.parse_argstring(
                        self.ml, parameter_s).modules
                rt.load_modules(*args, with_default=False)

            @line_magic
            @magic_arguments.magic_arguments()
            @magic_arguments.argument(
                    'modules', nargs='*',
                    help='The modules to reload. If empty, reload all')
            @register_shortcut
            def mrl(self, parameter_s):
                """%mrl ...: reload module(s) attached to the runtime."""
                args = magic_arguments.parse_argstring(
                        self.mrl, parameter_s).modules
                if not rt.modules:
                    logger.debug("no module is loaded, nothing to do")
                    return
                if not args:
                    rt.reload_modules()
                else:
                    for arg in args:
                        try:
                            rt.reload_module(arg)
                        except AttributeError:
                            logger.debug(
                                    f"skip reloading invalid module {arg}")

        kwargs = dict(colors="neutral", header=header())

        with logit(logger.debug, "interactive mode"):
            with hookit(InteractiveShellEmbed, 'init_magics') as hook:
                hook.set_post_func(
                        lambda self: self.register_magics(RuntimeMagics))
                IPython.embed(**kwargs)
