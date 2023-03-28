#!/usr/bin/env python

from astropy.table import Table, Column, vstack, hstack
import astropy.units as u
from pathlib import Path
import numpy as np

def fix_apt_cols(apt):

    dispatch_cols = {
        'uid': {
            'dtype': int,
            'unit': None,
            },
        'tone_freq': {
            'dtype': float,
            'unit': u.Hz
            },
        'array': {
            'dtype': int,
            'unit': None,
            },
        'nw': {
            'dtype': int,
            'unit': None,
            },
        'fg': {
            'dtype': int,
            'unit': None,
            },
        'pg': {
            'dtype': int,
            'unit': None,
            },
        'ori': {
            'dtype': int,
            'unit': None,
            },
        'loc': {
            'dtype': int,
            'unit': None,
            },
        'responsivity': {
            'dtype': float,
            'unit': 1 / u.pW,
            },
        'flxscale': {
            'dtype': float,
            'unit': u.MJy/u.sr
            },
        'sens': {
            'dtype': float,
            'unit': u.mJy * u.s ** 0.5
            },
        'derot_elev': {
            'dtype': float,
            'unit': u.rad
            },
        'amp': {
            'dtype': float,
            'unit': u.mJy / u.beam
            },
        'amp_err': {
            'dtype': float,
            'unit': u.mJy / u.beam
            },
        'x_t': {
            'dtype': float,
            'unit': u.arcsec
            },
        'x_t_err': {
            'dtype': float,
            'unit': u.arcsec
            },
        'y_t': {
            'dtype': float,
            'unit': u.arcsec
            },
        'y_t_err': {
            'dtype': float,
            'unit': u.arcsec
            },
        'a_fwhm': {
            'dtype': float,
            'unit': u.arcsec
            },
        'a_fwhm_err': {
            'dtype': float,
            'unit': u.arcsec
            },
        'b_fwhm': {
            'dtype': float,
            'unit': u.arcsec
            },
        'b_fwhm_err': {
            'dtype': float,
            'unit': u.arcsec
            },
        'angle': {
            'dtype': float,
            'unit': u.rad
            },
        'angle_err': {
            'dtype': float,
            'unit': u.rad
            },
        'converge_iter': {
            'dtype': int,
            'unit': None
            },
        'flag': {
            'dtype': int,
            'unit': None
            },
        'sig2noise': {
            'dtype': float,
            'unit': None
            },
        'x_t_raw': {
            'dtype': float,
            'unit': u.arcsec
            },
        'y_t_raw': {
            'dtype': float,
            'unit': u.arcsec
            },
        'x_t_derot': {
            'dtype': float,
            'unit': u.arcsec
            },
        'x_t_derot': {
            'dtype': float,
            'unit': u.arcsec
            },
        }
    for c, kw in dispatch_cols.items():
        if c in apt.colnames:
            apt[c] = Column(apt[c].astype(kw['dtype']), **kw)
    return apt


reduced_dir = Path("/data_lmt/toltec/reduced")

def collect_kids_model_params(obsnum):
    t = list()
    for nw in range(13):
        kmfile = list(reduced_dir.glob(
            f"toltec{nw}_{obsnum:06d}_*tune.txt"))
        if len(kmfile) != 1:
            print(f'skip nw={nw}')
            continue
        print(kmfile)
        kt = Table.read(kmfile[0], format='ascii.ecsv')
        kt['nw'] = nw
        kt['tone'] = range(len(kt))
        t.append(kt)
    t = vstack(t)
    # rename columns with prefix kids_
    for c in t.colnames:
        t.rename_column(c, f'kids_{c}')
    t['kids_flag'] = Column(t['kids_flag'], dtype=int)
    return vstack(t)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("aptfile")
    parser.add_argument("--obsnum")
    parser.add_argument("--output_dir")

    option = parser.parse_args()


    apt = Table.read(option.aptfile, format='ascii')
    if option.obsnum is not None:
        apt.meta['obsnum'] = int(option.obsnum)

    apt = fix_apt_cols(apt)
    # print(apt)
    # print(np.unique(apt['nw']))
    obsnum = apt.meta['obsnum']
    kt = collect_kids_model_params(obsnum=obsnum)
    # print(kt)
    # print(np.unique(kt['nw']))

    apt = hstack([apt, kt], join_type='exact')

    print(apt)
    output_path = Path(option.output_dir).joinpath(f"apt_{obsnum}_cleaned.ecsv")
    apt.write(output_path, format='ascii.ecsv', overwrite=True)
