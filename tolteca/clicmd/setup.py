#! /usr/bin/env python

"""This module is used to setup workdir for the reduction pipeline."""


import click
from ..utils import hookit
from ..utils.log import get_logger, timeit, logit
from ..utils.fmt import pformat_obj, pformat_dict
from ..cli import cli, OPTION_SETTINGS
from ..utils.cli import cli_header
from ..utils.cli.click_helpers import split_option_arg, resolve_path
# from functools import lru_cache
# import inspect
from ..pipeline import PipelineRuntime
from pathlib import Path


@cli.command('setup')
@click.argument(
        'workdir',
        type=click.Path(
            exists=False,
            file_okay=False, dir_okay=True,
            writable=True, readable=True),
        required=True,
        callback=resolve_path,
        metavar='DIR',
        )
@click.option(
        '--pipeline_bindir',
        default=None,
        type=click.Path(
            exists=True,
            file_okay=False, dir_okay=True,
            writable=False, readable=True),
        callback=resolve_path,
        metavar='PATH',
        help='The directory of the pipeline executable.',
        **OPTION_SETTINGS)
@click.option(
        '--calib_dir',
        default=None,
        type=click.Path(
            exists=False,
            file_okay=False, dir_okay=True,
            writable=False, readable=True),
        callback=resolve_path,
        metavar='PATH',
        help='The path to the calibration object.',
        **OPTION_SETTINGS)
@click.option(
        "-f", "--force",
        is_flag=True,
        default=False,
        help="Force the setup even if DIR is not empty.",
        )
@click.option(
        "-o", "--overwrite",
        is_flag=True,
        default=False,
        help="Overwrite any existing files without backup in case "
             "a forced setup is requested"
        )
@click.option(
        "-n", "--dry_run",
        is_flag=True,
        default=False,
        help="Run without actually create files.",
        )
@click.pass_obj
@timeit
def cmd_setup(
        rt, workdir, pipeline_bindir, calib_dir, force, overwrite, dry_run):
    """Setup DIR as the workdir for the reduction pipeline.

    The current dir is used if DIR is not specified.',
    """
    logger = get_logger()
    # logger.info(
    #     f"setup {workdir} as workdir with pipeline_bindir={pipeline_bindir}"
    #     f" calib_dir={calib_dir}"
    #     )

    prt = PipelineRuntime.from_dir(
            Path(workdir),
            empty_only=not force,
            backup=not overwrite,
            create=True,
            dry_run=dry_run,
            )
    print(prt)
    # setup_workdir(
    #         Path(workdir),
    #         pipeline_bindir=pipeline_bindir,
    #         calib_dir=calib_dir,
    #         )
