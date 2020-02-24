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
from tollan.utils.cli.multi_action_argparser import \
        MultiActionArgumentParser
from tollan.utils.wraps.stilts import ensure_stilts
import numpy as np


def main():
    logger = get_logger()
    stilts_cmd = ensure_stilts()
    logger.debug(f'use stilts: "{stilts_cmd}"')


def diqs_df(iqs, fs):
    diqs = np.empty_like(iqs)
    for i in range(iqs.shape[0]):
        diqs[i] = np.gradient(iqs[i], fs[i])
    return diqs


def match_tones(tunes):
    logger = get_logger()
    logger.debug(f"tunes: {tunes}")

    fos = tunes.file_objs

    tones = list(range(10))
    swps = [fo.itone[tones].isweep(index=1)[:].read() for fo in fos]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(tones))
    atten_in = []
    atten_out = []
    atten_total = []
    d21max = []
    for swp in swps:
        atten_in.append(swp.meta['atten_in'])
        atten_out.append(swp.meta['atten_out'])
        atten_total.append(swp.meta['atten_in'] + swp.meta['atten_out'])
        fs = np.tile(
            swp.meta['sweeps']['flo'].T,
            (len(swp.meta['tones']), 1)
            ) + swp.meta['tones']['fc'][:, None]
        iqs = swp.data
        adiqs = np.abs(diqs_df(iqs, fs))
        jd21max = np.argmax(adiqs, axis=-1)
        d21max.append([
            adiqs[i, jd21max[i]]
            for i in range(fs.shape[0])
            ])
        for i in range(axes.shape[-1]):
            ax = axes[0, i]
            ax.plot(
                iqs[i, :].real,
                iqs[i, :].imag,
                marker='.')
            ax = axes[1, i]
            ax.plot(fs[i, :], adiqs[i, :])
    fig, axes = plt.subplots(len(tones))
    for i in range(axes.shape[-1]):
        ax = axes[i]
        ax.plot(
            atten_out,
            [
                d21max[j][i]
                for j in range(len(swps))
                ])
    plt.show()

    logger.debug(f"swps: {swps}")
    return tunes


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    maap = MultiActionArgumentParser(
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
    act_index = maap.add_action_parser(
            'index',
            help="Build an index file that have tones"
            " matched for a set of tune files"
            )
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
            '"(obsid>8900) & (nwid==3) & (fileext=="nc")"',
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

    @act_index.parser_action
    def index_action(option):
        # This function is called when `index` is specified in the cmd
        # Collect the dataset from the command line arguments
        dataset = ToltecDataset.from_files(*option.files).select(
                '(kindstr=="tune")'
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

    act_plot = maap.add_action_parser(
            'plot',
            help="Make diagnostic plots"
            )
    act_plot.add_argument(
            'something', nargs='*'
            )

    @act_plot.parser_action
    def plot_action(option):
        print(option)

    option = maap.parse_args(args)
    maap.bootstrap_actions(option)
