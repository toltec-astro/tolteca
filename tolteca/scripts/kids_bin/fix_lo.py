#!/usr/bin/env python

import numpy as np
import netCDF4
from numpy.lib.recfunctions import append_fields


def get_lofreq(sweepfile):
    nc = netCDF4.Dataset(sweepfile)
    _ = {
         "lofreq": "Header.Toltec.LoCenterFreq"
         }
    
    lofreq = np.asscalar(nc[_['lofreq']][:])
    print("subtract lo_freq={} Hz".format(lofreq))
    return lofreq


if __name__ == "__main__":
    import sys
    sweepfile, reportfile, tonefile = sys.argv[1:]
    report = np.genfromtxt(reportfile, names=True)
    fs = report['f_out'] - get_lofreq(sweepfile)
    report = append_fields(report, 'f_centered', fs) 
    names =  report.dtype.names[-1:] + report.dtype.names[:-1]
    tones = report['f_centered']
    max_ = 2.1 * np.max(tones)
    isort = sorted(
            range(len(tones)), key=lambda i: tones[i] + max_ if tones[i] < 0 else tones[i])
    report = report[list(names)][isort]
    # import matplotlib.pyplot as plt
    # plt.plot(report['f_centered'])
    # plt.show()
    np.savetxt(tonefile, report, header= ' '.join(names))
