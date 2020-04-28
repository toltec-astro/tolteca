#! /usr/bin/env python

# Author:
#   Zhiyuan Ma
#
# History:
#   2020/04/21 Zhiyuan Ma:
#       - Created.

"""
This recipe shows how to use the `tolteca.fs.rsync_accessor.RsyncAccessor`
to select and retrieve TolTEC data files from remote.

The ``-s`` switch makes use of the ``panads.DataFrame.query``, and the
documentation on composing the query string could be found here:
https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-query

Example usage:

1. The following prints a list of files according to the selection string::

    # noqa: E501
    $ python dataset_rsync.py taco:/data/data_toltec/reduced/ -s "(obsid>10232) & (fileext=='txt')"
    <some logging messages>
    taco:/data/data_toltec/reduced/toltec3_010860_00_0000_2020_03_23_18_51_39_tune.txt
    taco:/data/data_toltec/reduced/toltec3_010885_00_0000_2020_03_23_22_31_43_tune.txt
    taco:/data/data_toltec/reduced/toltec3_010886_00_0000_2020_03_23_22_34_41_tune.txt
    taco:/data/data_toltec/reduced/toltec3_010888_00_0000_2020_03_24_01_13_47_tune.txt
    taco:/data/data_toltec/reduced/toltec3_010890_00_0000_2020_03_24_03_55_18_tune.txt
    taco:/data/data_toltec/reduced/toltec3_010892_00_0000_2020_03_24_06_36_33_tune.txt
    taco:/data/data_toltec/reduced/toltec3_010894_00_0000_2020_03_24_09_19_45_tune.txt
    taco:/data/data_toltec/reduced/toltec3_010896_00_0000_2020_03_24_12_00_58_tune.txt
    taco:/data/data_toltec/reduced/toltec3_010898_00_0000_2020_03_24_14_44_00_tune.txt

2. The following will download the matched files to specified directory.
   Optionally, specify [-f] -o <FILENAME> to also same an dataset index table
   for later use::

    $ python dataset_rsync.py taco:/data/data_toltec/reduced/ -s "(obsid>10232) & (nwid==3) & (fileext=='txt')" -d my_local_data_dir -f -o dataset_for_bla.asc

   The files matching the criteria will be mirrored to the folder
   `./my_local_data_dir`, and an index table named `dataset_for_bla.asc` is
   created in the current directory.
"""

import sys
from tolteca.recipes import get_logger
from tolteca.fs.toltec import ToltecDataset
from tolteca.fs.rsync_accessor import RsyncAccessor
from pathlib import Path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
            description='Build local dataset index using rsync as backend.')
    parser.add_argument(
            "paths",
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
            "-d", "--download_dir",
            metavar="DIR",
            help="If set, the files are rsynced into this directory.",
            )
    parser.add_argument(
            "-o", "--output",
            metavar="OUTPUT_FILE",
            required=False,
            help="The output index filename. "
                 "Only used when download dir is set.",
            )
    parser.add_argument(
            "-f", "--overwrite",
            action='store_true',
            help="If set, overwrite the existing index file",
            )
    option = parser.parse_args()
    # check that -fo is not set if -d is not
    if option.download_dir is None and option.output is not None:
        parser.error("-f/-o can only be used with -d.")
    logger = get_logger()
    # we get a list of remote files from querying the remote file system
    # This by default uses the RsyncAccessor. Since will be using the
    # same access to actually download the files, we just make that
    # explicit
    rsync = RsyncAccessor()
    dataset = ToltecDataset.from_remote_files(
            *option.paths, accessor=rsync)
    if option.select:
        dataset = dataset.select(option.select)

    if option.download_dir is None:
        logger.debug(f"list {len(dataset)} remote files")
        for ln in map(str, dataset.index_table['source']):
            sys.stdout.write(f"{ln}\n")
            sys.stdout.flush()
    else:
        logger.debug(f"download {len(dataset)} remote files")
        # download the files using the rsync accessor
        # and build the dataset using downloaded files
        filepaths = rsync.rsync(
                dataset['source'], option.download_dir)
        dataset = dataset.from_files(*filepaths)
        dispatch_fmt = {
                '.ecsv': 'ascii.ecsv',
                '.asc': 'ascii.commented_header',
                '.fits': 'fits',
                '.adsf': 'adsf',
                }
        fmt = dispatch_fmt.get(
                Path(option.output).suffix, 'ascii.commented_header')
        dataset.write_index_table(
                option.output, overwrite=option.overwrite,
                format=fmt)
