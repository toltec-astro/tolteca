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

from tolteca.recipes import get_logger
from tolteca.fs.toltec import ToltecDataset
from tollan.utils.cli import get_action_argparser
from tollan.utils.wraps.stilts import ensure_stilts


def main():
    logger = get_logger()
    stilts_cmd = ensure_stilts()
    logger.debug(f'use stilts: "{stilts_cmd}"')


def match_tones(tunes):
    logger = get_logger()
    logger.debug(f"input tune data: {tunes}")
    return tunes


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    parser, add_action_parser, set_parser_action = get_action_argparser(
            description="Diagnostics for a set of TolTEC KIDs tune files."
            )
    # Set up the `index` action.
    # The purpose of this is to collect a set of tune files and
    # try match the tones among them. This is necessary because
    # consecutive tune files does not necessarily result in
    # the same set of tune positions.
    # The end result of this action group is to dump an index
    # file that contains a list of tune files, such that neighbouring
    # ones have at least 10 tones found to be referring to the same
    # detector.
    act_index = add_action_parser(
            "index",
            help="build an index file that have tones"
            " matched for a set of tune files")
    act_index.add_argument(
            "files",
            metavar="FILE",
            nargs='+',
            help="The files to use",
            )
    act_index.add_argument(
            "-s", "--select",
            metavar="COND",
            help='A selection predicate, e.g.,:'
            '"(obsid>8900) & (nwid==3) & (fileext==b"nc")"',
            )
    act_index.add_argument(
            "-o", "--output",
            metavar="OUTPUT_FILE",
            required=True,
            help="The output index file",
            )
    act_index.add_argument(
            "-f", "--overwrite",
            action='store_true',
            help="If set, overwrite the existing index file",
            )

    @set_parser_action(act_index)
    def act_index(option):
        # This function is called when `index` is specified in the cmd
        # Collect the dataset from the command line arguments
        dataset = ToltecDataset.from_files(*option.files).select(
                '(kindstr==b"tune")'
                )
        # Apply any selection filtering
        if option.select:
            dataset = dataset.select(option.select).open_files()
        else:
            dataset = dataset.open_files()
        # Run the tone matching algo.
        dataset = match_tones(dataset)
        # Dump the result file.
        dataset.write_index_table(
                option.output, overwrite=option.overwrite,
                format='ascii.commented_header')

    # bootstrap the parser actions
    option = parser.parse_args(args)
    if hasattr(option, 'func'):
        option.func(option)
    else:
        parser.print_help()
