#!/usr/bin/env python
import dill
from pathlib import Path
from tolteca.datamodels.io.toltec.kidsdata import NcFileIO
from tolteca.datamodels.io.toltec.table import KidsModelParamsIO
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from tollan.utils.log import init_log, get_logger


def make_d21_mask(swp, kmp, Qr_cut=2e4, n_max=10, n_fwhm=3):
    logger = get_logger()
    obsnum = swp.meta['obsnum']

    kt = kmp.table.copy()
    kt['tone'] = range(len(kt))
    kt.sort('Qr')
    m = kt['Qr'] > Qr_cut
    logger.debug(f'detectors with highest Qrs:\n{kt[-n_max:]}')
    i_dark = np.where(m)[0]
    if m.sum() > n_max:
        i_dark = i_dark[-n_max:]
    logger.debug(f'mask {len(i_dark)} out of {len(kt)} dark detectors')
    f = swp.unified.frequency
    mask = np.ones(f.shape, dtype=bool)
    dark_det_lims = list()
    for i in kt[i_dark]['tone']:
        km = kmp.get_model(i)
        fwhm = km.fr / km.Qr
        fmin = km.fr - n_fwhm * fwhm
        fmax = km.fr + n_fwhm * fwhm
        logger.debug(f'mask FWHM={fwhm} fmin={fmin} fmax={fmax}')
        dark_det_lims.append((fmin, fmax))
        # f is sorted so we can find the switch of sign
        imin = np.where(f - fmin >= 0)[0][0]
        imax = np.where(f - fmax <= 0)[0][-1]
        mask[imin:imax] = 0
    # save the list of dark detectors
    # kt[i_dark].write(f'{e["interface"]}_{e["obsnum"]}_dark_detectors.txt', overwrite=True, format='ascii.ecsv')
    return mask, dark_det_lims


def make_ref_data(sweep_data, model_data, debug_plot=False):

    logger = get_logger()

    swp = sweep_data.read()
    kmp = model_data.read()

    d21_kw = dict(
        fstep=1000 << u.Hz, flim=(4.0e8 << u.Hz, 1.0e9 << u.Hz)
        )
    logger.info(f"calc d21 for {swp.meta['file_loc']} {d21_kw=}")
    # extract d21 spectrum
    swp.make_unified(**d21_kw)

    d21_mask_kw = dict(
        Qr_cut=2e4,
        n_max=10,
        n_fwhm=3
        )
    # make d21 mask that exclude dark detectors
    logger.info(f"calc d21 mask with {d21_mask_kw=}")
    d21_mask, dark_det_lims = make_d21_mask(
        swp=swp, kmp=kmp, **d21_mask_kw)
    nw = swp.meta['nwid']
    obsnum = swp.meta['obsnum']
    if debug_plot:
        fig, ax = plt.subplots(
                1, 1,
                figsize=(10, 4),
                constrained_layout=True,
                )
        fig.suptitle(f'{obsnum=} {nw=}')
        # plot dark det data
        d21 = swp.unified
        ax.plot(
            d21.frequency.to_value(u.MHz),
            d21.D21.to_value(u.adu / u.Hz),
            )
        ax.plot(
            d21.frequency.to_value(u.MHz),
            d21_mask * 10
            )
        for l, r in dark_det_lims:
            ax.axvline(l.to_value(u.MHz), linestyle=':')
            ax.axvline(r.to_value(u.MHz), linestyle=':')
        plt.show()
    return {
        'swp': swp,
        'kmp': kmp,
        'd21_kw': d21_kw,
        'd21_mask_kw': d21_mask_kw,
        'd21_mask': d21_mask,
        'dark_det_lims': dark_det_lims,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'sweep_file', help='sweep data'
        )
    parser.add_argument(
        'model_file', help='kids model file.'
        )
    parser.add_argument(
        '--debug_plot', action='store_true'
        )
    parser.add_argument(
        '--log_level', default='INFO'
        )
    # parser.add_argument("--config", "-c", help='YAML config file.')
    option = parser.parse_args()

    init_log(level=option.log_level)
    logger = get_logger()

    sweep_file = Path(option.sweep_file)
    model_file = Path(option.model_file)

    logger.info(f"make ref data with {sweep_file} and {model_file}")

    sweep_data = NcFileIO(sweep_file).open()
    model_data = KidsModelParamsIO(model_file).open()

    output_file = model_file.with_suffix(".refdata")
    ref_data = make_ref_data(
        sweep_data=sweep_data,
        model_data=model_data,
        debug_plot=option.debug_plot)
    with open(output_file, 'wb') as fo:
        dill.dump(ref_data, fo)
    logger.info(f'ref_data written to {output_file}')
