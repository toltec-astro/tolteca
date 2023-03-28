#!/usr/bin/env python

import argparse
from astropy.table import Table
import numpy as np
from pathlib import Path
from netCDF4 import Dataset


if __name__ == "__main__":
    import sys

    tune = Dataset(sys.argv[1])

    tone_amps_lut = tune.variables['Header.Toltec.ToneAmps'][:]

    tone_amps_lut[tone_amps_lut <= 0] = 1.
    # print(tone_amps_lut)
    tone_amps = Table.read(sys.argv[2], names=['amp'], format='ascii.no_header')['amp']

    # print(tone_amps)

    tone_amps = tone_amps * tone_amps_lut
    tone_amps_norm = np.max(tone_amps)
    tone_amps_norm_db = -10 * np.log10(tone_amps_norm)
    print(f'norm factor {tone_amps_norm} ({tone_amps_norm_db} db)')
    tone_amps = tone_amps / tone_amps_norm

    # print(tone_amps)
    Table([tone_amps]).write(sys.argv[2].replace('.txt', '.lut.txt'), format='ascii.no_header', overwrite=True)
