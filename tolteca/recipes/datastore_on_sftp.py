#! /usr/bin/env python

"""
This recipes shows how to use the `tolteca.fs.ToltecDataFileStore` class.
"""

import yaml
from tolteca.fs.toltec import ToltecDataFileStore
from tollan.utils.log import init_log
from tollan.utils import rupdate


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
            description='Rsync from remote data and run some reduction')

    parser.add_argument("path", help='Remote data file root path.')

    parser.add_argument(
            '-l', '--localpath',
            help='Path to store local copy of data files.')

    parser.add_argument(
            "-c", "--config",
            nargs='+',
            help="The path to tolteca config file(s). "
                 "Multiple config files are merged.")

    args = parser.parse_args()

    init_log(level='DEBUG')

    conf = None
    for c in args.config:
        with open(c, 'r') as fo:
            if conf is None:
                conf = yaml.safe_load(fo)
            else:
                rupdate(conf, yaml.safe_load(fo))

    rootpath = args.path
    if ':' in rootpath:
        hostname, rootpath = rootpath.split(':', 1)
        config = conf['fs'].get(hostname, None)
    else:
        hostname = None
        rootpath = args.rootpath
        config = None

    datastore = ToltecDataFileStore(
            rootpath=rootpath, hostname=hostname,
            local_rootpath=args.localpath, config=config)

    if hostname in ['clipa', 'clipo']:
        # define some filters so we don't load the old files
        def inc_filter(p):
            p = p.as_posix()
            if any(m in p for m in ['clip/', 'tcs/']):
                return False
            return True
    else:
        inc_filter = None

    print(datastore.glob(
        '**/*.nc',
        inc=inc_filter,
        ))
