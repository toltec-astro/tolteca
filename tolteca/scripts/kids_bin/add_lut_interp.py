#!/usr/bin/env python

import argparse
from astropy.table import Table
import numpy as np
from pathlib import Path
from netCDF4 import Dataset


if __name__ == "__main__":
    import sys

    tune_path = sys.argv[1]
    tune = Dataset(tune_path)

    tone_amps_lut = tune.variables['Header.Toltec.ToneAmps'][:]

    tone_amps_lut[tone_amps_lut <= 0] = 1.
    print(tone_amps_lut)
    tone_amps = Table.read(sys.argv[2], names=['amp'], format='ascii.no_header')['amp']

    # print(tone_amps)

    tone_amps = tone_amps * tone_amps_lut
    tone_amps_norm = np.max(tone_amps)
    tone_amps_norm_db = -10 * np.log10(tone_amps_norm)
    print(f'norm factor {tone_amps_norm} ({tone_amps_norm_db} db)')
    tone_amps = tone_amps / tone_amps_norm

    f_tones = tune.variables['Header.Toltec.ToneFreq'][0,:] + tune.variables['Header.Toltec.LoCenterFreq'][:].item()

    # i_sort = np.argsort(f_tones)

    # f_tones_sorted = f_tones[i_sort]
    # tone_amps_sorted = tone_amps[i_sort]
    print(f_tones.shape)

    current_targ_freqs = Table.read(sys.argv[3], format='ascii.ecsv')['f_out']
    current_tone_amps = []
    for i, f in enumerate(current_targ_freqs):
        j = np.argmin(np.abs(f_tones - f))
        # print(f'{i=} {j=} {f=} -> {f_tones[j]}')
        current_tone_amps.append(tone_amps[j])

    print(current_tone_amps)
    # print(tone_amps)
    Table([tone_amps]).write(sys.argv[2].replace('.txt', '.lut.txt'), format='ascii.no_header', overwrite=True)
