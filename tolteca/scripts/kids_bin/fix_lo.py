#!/usr/bin/env python

import numpy as np
import netCDF4

_ = {
     "lofreq": "Header.Toltec.LoCenterFreq"
     }


def get_lofreq(sweepfile):
    nc = netCDF4.Dataset(sweepfile)

    lofreq = nc[_['lofreq']][:].item()
    print("subtract lo_freq={} Hz".format(lofreq))
    return lofreq


if __name__ == "__main__":
    from astropy.table import Table, Column
    import sys
    sweepfile, reportfile, tonefile = sys.argv[1:]
    report = Table.read(reportfile, format='ascii')
    flo = get_lofreq(sweepfile)
    fs = report['f_out'] - flo
    report.add_column(Column(fs, name='f_centered'), 0)
    report.meta[_['lofreq']] = flo
    tones = report['f_centered']
    max_ = 2.1 * np.max(tones)
    isort = sorted(
            range(len(tones)),
            key=lambda i: tones[i] + max_ if tones[i] < 0 else tones[i])
    report = report[isort]
    # import matplotlib.pyplot as plt
    # plt.plot(report['f_centered'])
    # plt.show()
    report.write(tonefile, format='ascii.ecsv', overwrite=True)
