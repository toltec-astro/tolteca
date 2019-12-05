#! /usr/bin/env python


"""This module manages work spaces."""


import click
from ..utils import hookit
from ..utils.log import get_logger, timeit, logit
from ..utils.fmt import pformat_obj, pformat_dict
from ..cli import cli, OPTION_SETTINGS
from ..utils.cli import cli_header
from ..utils.cli.click_helpers import split_option_arg
# from functools import lru_cache
# import inspect


@cli.command('workon')
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


