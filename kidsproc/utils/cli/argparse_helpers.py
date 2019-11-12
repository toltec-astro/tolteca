#! /usr/bin/env python

import argparse
from .. import version


def argparser_with_common_options(desc=None):
    desc_base = f"Kidsproc v{version.version} {version.timestamp}"
    if desc:
        desc = "{}\n  - {}".format(desc_base, desc)
    else:
        desc = desc_base
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--quiet", help="disable debug logs",
                        action="store_true")

    parser.add_argument(
            "--figsize", help="plot window size in inches.",
            nargs=2, type=int, default=(10, 10))

    def parse(parser=parser):
        args, unparsed_args = parser.parse_known_args()
        level = 'DEBUG'
        if args.quiet:
            level = 'INFO'
        from astropy import log
        log.setLevel(level)
        return args, unparsed_args
    return parser, parse
