#! /usr/bin/env python

# Author:
#   Zhiyuan Ma
#
# History:
#   2020/04/21 Zhiyuan Ma:
#       - Created.

"""
This recipe shows how to use the `tolteca.datamodels.fs.rsync.RsyncAccessor`
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
from tolteca.datamodels.toltec import BasicObsDataset
from tolteca.datamodels.fs.rsync import RsyncAccessor
from tollan.utils.fmt import pformat_yaml
from pathlib import Path
import argparse


def main(args):

    parser = argparse.ArgumentParser(
            description='Build local dataset index using rsync as backend.')
    parser.add_argument(
            "paths",
            nargs='+',
            help='Path(s) of the remote data files.'
            )
    parser.add_argument(
            "--load_meta_from_file",
            action='store_true',
            help="If set, open source file to load meta.",
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
            "-m", "--transfer_mode",
            choices=['mirror', 'lmt_archive'],
            default='archive',
            help="If mirror, the orignal path is retained; if archive, "
                 "the files are organized following the LMT raw data "
                 "convention",
            )
    parser.add_argument(
            "-o", "--output",
            metavar="OUTPUT_FILE",
            required=False,
            help="The output index filename."
            )
    parser.add_argument(
            "-f", "--overwrite",
            action='store_true',
            help="If set, overwrite the existing index file",
            )
    option = parser.parse_args(args)
    logger = get_logger()
    # we get a list of remote files from querying the remote file system
    # This by default uses the RsyncAccessor. Since will be using the
    # same access to actually download the files, we just make that
    # explicit
    accessor = RsyncAccessor()
    dataset = BasicObsDataset.from_files(accessor.glob(*option.paths), open_=option.load_meta_from_file)
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
        rsync_kwargs = {}
        if option.transfer_mode == 'mirror':
            rsync_dest = option.download_dir
        elif option.transfer_mode == 'lmt_archive':
            # insert the ./ in the source path at the master level
            def path_filter(p):
                p = Path(p)
                for pp in p.parents:
                    if pp.name == 'data_lmt':
                        p0 = pp
                        break
                else:
                    raise ValueError("not a valid data_lmt tree")
                p1 = p.relative_to(p0)
                # p0 = p.parents[3]  # upto the instru  {}toltec/ics/toltec0/tolte0.nc
                # p1 = p.relative_to(p0)
                return f'{p0}/./{p1}'
            rsync_kwargs['path_filter'] = path_filter
            rsync_dest = Path(option.download_dir)
        else:
            raise NotImplementedError
        filepaths = accessor.rsync(
                dataset['source'], rsync_dest, **rsync_kwargs)
        logger.debug(f"local filepaths:\n{pformat_yaml(filepaths)}")
        # make a table with remote solource
        ukeys = ['obsnum', 'subobsnum', 'scannum']

        local_tbl=BasicObsDataset.from_files(filepaths, open_=False)
        ccols = set(dataset.colnames).intersection(set(local_tbl))
        if 'roachid' in ccols:
            ukeys.append('roachid')
        remote_source_tbl = dataset.index_table[ukeys + ['source', ]]
        remote_source_tbl.rename_column('source', 'source_remote')
        dataset = local_tbl.join(
                remote_source_tbl, keys=ukeys, join_type='outer'
                )
    if option.output:
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


if __name__ == "__main__":
    main(sys.argv[1:])
