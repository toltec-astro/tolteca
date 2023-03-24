#!/usr/bin/env python

import argparse
from astropy.table import Table
import numpy as np
from pathlib import Path


def make_ampcor(adrv_file, perc, plot=False):
    perc = float(perc)  # the percnetile goes from 0 to 100
    if str(adrv_file).endswith('_adrv.csv'):
        a_drv_tbl = Table.read(adrv_file, format='ascii.csv')
        a_drv_tbl.sort('tone_num')
        print(a_drv_tbl)
        a_drv_bests = a_drv_tbl['drive_atten']
        a_drv_bests[a_drv_bests == 0] = np.nan
    else:
        a_drv_bests = Table.read(adrv_file, format='ascii.no_header')['col1']
    m = np.isnan(a_drv_bests)
    print(f'{np.sum(m)}/{len(m)} has no a_drv_best')
    med = np.nanmedian(a_drv_bests)
    print(f'replace with a_drv_med = {med}')
    a_drv_bests[m] = med
    a_drv_ref = np.nanpercentile(a_drv_bests, perc)
    print(f'use a_drv_ref={a_drv_ref}')
    ampcors = []
    for a in a_drv_bests:
        ampcor = 10 ** ((a_drv_ref - a) / 20.)
        ampcors.append(ampcor)
    ampcors = np.array(ampcors)
    m = ampcors > 1.0
    print(f'{np.sum(m)}/{len(m)} has ampcor cutoff at 1.0')
    ampcors[m] = 1
    outfile = Path(adrv_file).with_suffix(f'.p{perc:.0f}.txt')
    print(f"save to file {outfile}")
    np.savetxt(outfile, ampcors)
    if plot:
        import matplotlib.pyplot as plt
        fig, (ax, bx, cx) = plt.subplots(1, 3)
        ax.hist(a_drv_bests)
        ax.axvline(a_drv_ref)
        bx.hist(ampcors)
        bx.axvline(1)
        cx.plot(ampcors)
        plt.show()


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '-p', '--perc',
            default=None, type=int,
            help='The output reference driving atten.'
            )
    parser.add_argument(
            '--plot',
            action='store_true',
            help='make hist plot.'
            )
    parser.add_argument(
            'files', nargs='+',
            help='the input a_drv files')

    option = parser.parse_args(args)
    print(option)

    for adrv_file in option.files:
        print(adrv_file)
        make_ampcor(adrv_file, option.perc, plot=option.plot)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
