#! /usr/bin/env python


"""This recipe shows how to use rsync with `tolteca.fs.ToltecDataset` to
select data files.
"""

from tollan.utils.log import init_log
import subprocess
import tempfile
import sys
from tolteca.fs.toltec import ToltecDataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Rsync and filter dataset.')
    parser.add_argument(
            "path",
            nargs='+',
            help='Path(s) of the remote data files.'
            )
    parser.add_argument(
            "-s", "--select",
            metavar="COND",
            help='A selection predicate, e.g.,:'
            '"(obsid>8900) & (nwid==3) & (fileext=="nc")"',
            )
    parser.add_argument(
            "-f", "--overwrite",
            action='store_true',
            help="If set, overwrite the existing index file",
            )
    args = parser.parse_args()

    init_log(level='DEBUG')

    def _filter_rsync_ln(ln):
        return 'skipping non-regular file' not in ln

    result = set()
    with tempfile.TemporaryDirectory() as tmp:
        for path in args.path:
            hostname = path.split(":", 1)[0]

            def _map_rsync_ln(ln):
                return f'{hostname}:/{ln}'

            # get file list
            rsync_cmd = [
                    'rsync', '-rn', '--info=name', '--relative',
                    path, tmp]
            for p in map(
                _map_rsync_ln,
                filter(
                    _filter_rsync_ln,
                    subprocess.check_output(
                        rsync_cmd).decode().strip().split('\n'))):
                result.add(p)

    dataset = ToltecDataset.from_files(*result)
    if args.select:
        dataset = dataset.select(args.select)
    for ln in map(str, dataset.index_table['source']):
        sys.stdout.write(f"{ln}\n")
        sys.stdout.flush()
