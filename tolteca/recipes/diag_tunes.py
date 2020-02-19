#! /usr/bin/env python

# Author:
#   Zhiyuan Ma

"""This recipe makes use of various KIDs related module and classes
to make diagnostic plot for a collection of tune files.

The code requires an external tool `stilts` to match the tones in
different files. This code will try download it automatically if not
already installed. Please refer to http://www.star.bris.ac.uk/~mbt/stilts/
for more information.
"""

import numpy as np
import subprocess
import shutil
from tolteca.recipes import get_extern_dir, get_logger, logit
import os
import stat
import re
import tempfile
from contextlib import ExitStack
from astropy.table import Table
from tolteca.io.toltec import NcFileIO


def ensure_stilts():
    logger = get_logger()
    extern_dir = get_extern_dir()
    which_path = f"{extern_dir.resolve().as_posix()}:{os.getenv('PATH')}"
    # which_path = f"{extern_dir.resolve().as_posix()}"
    # logger.debug(f"extern search paths: {which_path}")
    stilts_cmd = shutil.which("stilts", path=which_path)
    if stilts_cmd is None:
        logger.warning("unable to find stilts, download from internet")
        with logit(logger.debug, "setup stilts"):
            # retrieve stilts
            from astropy.utils.data import download_file
            stilts_jar_tmp = download_file(
                    "http://www.star.bris.ac.uk/%7Embt/stilts/stilts.jar",
                    cache=True)
            stilts_jar = extern_dir.joinpath('stilts.jar')
            shutil.copyfile(stilts_jar_tmp, stilts_jar)
            stilts_cmd = extern_dir.joinpath('stilts')
            with open(stilts_cmd, 'w') as fo:
                fo.write("""#!/bin/sh
java -Xmx4000M -classpath "{0}:$CLASSPATH" uk.ac.starlink.ttools.Stilts "$@"
""".format(stilts_jar.resolve()))
            os.chmod(
                    stilts_cmd,
                    os.stat(stilts_cmd).st_mode | stat.S_IEXEC)
    # verify that stilts works
    try:
        output = subprocess.check_output(
                (stilts_cmd, '-version'),
                stderr=subprocess.STDOUT
                ).decode().strip('\n')
    except Exception as e:
        raise RuntimeError(f"error when run stilts {stilts_cmd}: {e}")
    else:
        logger.debug(f"\n\n{output}\n")
    return stilts_cmd


def run_stilts(cmd, *tbls):
    logger = get_logger()
    with ExitStack() as es:
        for i, c in enumerate(cmd):
            s = re.match(r'(.+)=\$(\d+)', c)
            if s is not None:
                a = int(s.group(2)) - 1
                t = tbls[a]
                if not isinstance(t, str):
                    f = es.enter_context(
                            tempfile.NamedTemporaryFile())
                    logger.debug(f"write table to {f.name}")
                    t.write(
                            f.name,
                            format='ascii.commented_header', overwrite=True)
                    t = f.name
                cmd[i] = f"{s.group(1)}={t}"
        logger.debug("run stilts: {}".format(' '.join(cmd)))
        exitcode = subprocess.check_call(cmd)
    return exitcode


def stilts_match1d(tbl1, tbl2, colname, radius, stilts_cmd=None):
    cmd = [
        stilts_cmd or 'stilts',
        "tmatch2",
        "in1=$1", "ifmt1=ascii",
        "in2=$2", "ifmt2=ascii",
        "matcher=1d", f"params={radius}", f"values1='{colname}'",
        f"values2='{colname}'",
        # "action=keep1",
        "out=$3", "ofmt=ascii"]

    f = tempfile.NamedTemporaryFile()

    try:
        run_stilts(cmd, tbl1, tbl2, f.name)
    except Exception as e:
        raise RuntimeError(f"failed run {''.join(cmd)}: {e}")
    else:
        tbl = Table.read(f.name, format='ascii.commented_header')
        return tbl


def main():
    logger = get_logger()
    stilts_cmd = ensure_stilts()
    logger.debug(f'use stilts: "{stilts_cmd}"')


def build_index(files, keys=None):
    """Return a table of meta data for given TolTEC *.nc files."""
    logger = get_logger()

    logger.debug(f"build index for files {files}")

    files = list(map(NcFileIO, files))

    cols = [
            ('ut', lambda f: f.meta['ut'].strftime('%Y_%m_%d_%H_%M_%S')),
            'roachid', 'obsid', 'subobsid', 'scanid', 'kindstr']
    if keys is not None:
        for k in keys:
            if k not in cols:
                cols.append(k)

    cols.append(('filepath', lambda f: f.filepath))
    cols = [(c, lambda f, c=c: f.meta[c]) if isinstance(c, str) else c
            for c in cols]
    rows = []
    for _, f in enumerate(files):
        row = [None] * len(cols)
        for j, c in enumerate(cols):
            row[j] = c[1](f)
        rows.append(row)
    tbl = Table(rows=rows, names=[c[0] for c in cols])
    pformat_tbl = '\n'.join(tbl[tbl.colnames[:-1]].pformat(max_width=-1))
    logger.debug(f"index:\n{pformat_tbl}")
    return tbl, files


def match_tunes(tbl, files):
    return tbl


if __name__ == "__main__":
    import sys
    import argparse
    args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="tune diagnostics."
        )
    subparsers = parser.add_subparsers(
            title="actions",
            help="available actions")
    parser_index = subparsers.add_parser("index", help="create index file.")
    parser_index.add_argument(
            "files",
            metavar="FILE",
            nargs='+',
            help="The files to use.",
            )
    parser_index.add_argument(
            "-o", "--output",
            metavar="OUTPUT_FILE",
            required=True,
            help="The output index file.",
            )
    parser_index.add_argument(
            "-f", "--overwrite",
            action='store_true',
            help="If set, overwrite the existing index file.",
            )

    def f_index(option):
        tbl, files = build_index(option.files, keys=['atten_out', ])
        if np.all(tbl['kindstr'] == 'tune'):
            tbl = match_tunes(tbl, files)
        tbl.write(
                option.output, overwrite=option.overwrite,
                format='ascii.commented_header')
    parser_index.set_defaults(func=f_index)
    option = parser.parse_args(args)
    # execute the action
    if hasattr(option, 'func'):
        option.func(option)
    else:
        parser.print_help()
