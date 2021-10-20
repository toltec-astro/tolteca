#!/usr/bin/env python

from tolteca.simu import SimulatorRuntime
from reproject import reproject_interp
import yaml


lss_plan = {

    'ECDFS': {
        't0': [
            '2021-11-01T04:00:00',
            '2021-11-01T05:00:00',
            '2021-11-01T06:00:00',
            '2021-11-01T07:00:00',
            '2021-11-01T08:00:00',
            '2021-11-01T09:00:00',

            '2021-11-02T04:00:00',
            '2021-11-02T05:00:00',
            '2021-11-02T06:00:00',
            '2021-11-02T07:00:00',
            '2021-11-02T08:00:00',
            '2021-11-02T09:00:00',

            '2021-11-03T04:00:00',
            '2021-11-03T05:00:00',
            '2021-11-03T06:00:00',
            '2021-11-03T07:00:00',
            '2021-11-03T08:00:00',
            '2021-11-03T09:00:00',

            '2021-11-04T04:00:00',
            '2021-11-04T05:00:00',
            '2021-11-04T06:00:00',
            '2021-11-04T07:00:00',
            '2021-11-04T08:00:00',
            '2021-11-04T09:00:00',

            ],
        'mapping': {
            'type': 'tolteca.simu:SkyRasterScanModel',
            'length': '200 arcmin',
            'space': '2 arcmin',
            'n_scans': 100,
            'speed': '500 arcsec/s',
            't_turnaround': '5s',
            'target': '53.0d -28.1d',
            'ref_frame': 'altaz',
            }  # this mapping pattern is 48min
        },
    'XMM-LSS': {
        't0': [
            '2021-11-01T03:00:00',
            '2021-11-01T04:00:00',
            '2021-11-01T05:00:00',
            '2021-11-01T06:00:00',
            '2021-11-01T07:00:00',
            '2021-11-01T08:00:00',

            '2021-11-02T03:00:00',
            '2021-11-02T04:00:00',
            '2021-11-02T05:00:00',
            '2021-11-02T06:00:00',
            '2021-11-02T07:00:00',
            '2021-11-02T08:00:00',

            '2021-11-03T03:00:00',
            '2021-11-03T04:00:00',
            '2021-11-03T05:00:00',
            '2021-11-03T06:00:00',
            '2021-11-03T07:00:00',
            '2021-11-03T08:00:00',

            '2021-11-04T03:00:00',
            '2021-11-04T04:00:00',
            '2021-11-04T05:00:00',
            '2021-11-04T06:00:00',
            '2021-11-04T07:00:00',
            '2021-11-04T08:00:00',
            ],
        'mapping': {
            'type': 'tolteca.simu:SkyRasterScanModel',
            'length': '200 arcmin',
            'space': '2 arcmin',
            'n_scans': 100,
            'speed': '500 arcsec/s',
            't_turnaround': '5s',
            'target': '35.39d -4.6d',
            'ref_frame': 'altaz',
            }  # this mapping pattern is 48min
        },
    'COSMOS-L': {
        't0': [
            '2022-02-01T04:00:00',
            '2022-02-01T05:00:00',
            '2022-02-01T06:00:00',
            '2022-02-01T07:00:00',
            '2022-02-01T08:00:00',
            '2022-02-01T09:00:00',
            '2022-02-01T10:00:00',

            '2022-02-02T04:00:00',
            '2022-02-02T05:00:00',
            '2022-02-02T06:00:00',
            '2022-02-02T07:00:00',
            '2022-02-02T08:00:00',
            '2022-02-02T09:00:00',
            '2022-02-02T10:00:00',

            '2022-02-03T04:00:00',
            '2022-02-03T05:00:00',
            '2022-02-03T06:00:00',
            '2022-02-03T07:00:00',
            '2022-02-03T08:00:00',
            '2022-02-03T09:00:00',
            '2022-02-03T10:00:00',

            '2022-02-04T04:00:00',
            '2022-02-04T05:00:00',
            '2022-02-04T06:00:00',
            '2022-02-04T07:00:00',
            '2022-02-04T08:00:00',
            '2022-02-04T09:00:00',
            '2022-02-04T10:00:00',
            ],
        'mapping': {
            'type': 'tolteca.simu:SkyRasterScanModel',
            'length': '150 arcmin',
            'space': '2 arcmin',
            'n_scans': 75,
            'speed': '300 arcsec/s',
            't_turnaround': '5s',
            'target': '150.1d 2.2d',
            'ref_frame': 'altaz',
            }  # this mapping pattern is 48min
        },
    'COSMOS': {
        't0': [
            '2022-02-01T04:00:00',
            '2022-02-01T05:00:00',
            '2022-02-01T06:00:00',
            '2022-02-01T07:00:00',
            '2022-02-01T08:00:00',
            '2022-02-01T09:00:00',
            '2022-02-01T10:00:00',

            '2022-02-02T04:00:00',
            '2022-02-02T05:00:00',
            '2022-02-02T06:00:00',
            '2022-02-02T07:00:00',
            '2022-02-02T08:00:00',
            '2022-02-02T09:00:00',
            '2022-02-02T10:00:00',

            '2022-02-03T04:00:00',
            '2022-02-03T05:00:00',
            '2022-02-03T06:00:00',
            '2022-02-03T07:00:00',
            '2022-02-03T08:00:00',
            '2022-02-03T09:00:00',
            '2022-02-03T10:00:00',

            '2022-02-04T04:00:00',
            '2022-02-04T05:00:00',
            '2022-02-04T06:00:00',
            '2022-02-04T07:00:00',
            '2022-02-04T08:00:00',
            '2022-02-04T09:00:00',
            '2022-02-04T10:00:00',
            ],
        'mapping': {
            'type': 'tolteca.simu:SkyRasterScanModel',
            'length': '100 arcmin',
            'space': '2 arcmin',
            'n_scans': 75,
            'speed': '200 arcsec/s',
            't_turnaround': '5s',
            'target': '150.1d 2.2d',
            'ref_frame': 'altaz',
            }  # this mapping pattern is 48min
        },
    'Bootes': {
        't0': [
            '2022-04-01T05:00:00',
            '2022-04-01T06:00:00',
            '2022-04-01T07:00:00',
            '2022-04-01T08:00:00',
            '2022-04-01T09:00:00',
            '2022-04-01T10:00:00',

            '2022-04-02T05:00:00',
            '2022-04-02T06:00:00',
            '2022-04-02T07:00:00',
            '2022-04-02T08:00:00',
            '2022-04-02T09:00:00',
            '2022-04-02T10:00:00',

            '2022-04-03T05:00:00',
            '2022-04-03T06:00:00',
            '2022-04-03T07:00:00',
            '2022-04-03T08:00:00',
            '2022-04-03T09:00:00',
            '2022-04-03T10:00:00',

            '2022-04-04T05:00:00',
            '2022-04-04T06:00:00',
            '2022-04-04T07:00:00',
            '2022-04-04T08:00:00',
            '2022-04-04T09:00:00',
            '2022-04-04T10:00:00',
            ],
        'mapping': {
            'type': 'tolteca.simu:SkyRasterScanModel',
            'length': '200 arcmin',
            'space': '2 arcmin',
            'n_scans': 100,
            'speed': '500 arcsec/s',
            't_turnaround': '5s',
            'target': '217.9d 34.1d',
            'ref_frame': 'altaz',
            }  # this mapping pattern is 48min
        },
    'NEP': {
        't0': [
            '2022-05-01T05:00:00',
            '2022-05-01T06:00:00',
            '2022-05-01T07:00:00',
            '2022-05-01T08:00:00',
            '2022-05-01T09:00:00',
            '2022-05-01T10:00:00',

            '2022-05-02T05:00:00',
            '2022-05-02T06:00:00',
            '2022-05-02T07:00:00',
            '2022-05-02T08:00:00',
            '2022-05-02T09:00:00',
            '2022-05-02T10:00:00',

            '2022-05-03T05:00:00',
            '2022-05-03T06:00:00',
            '2022-05-03T07:00:00',
            '2022-05-03T08:00:00',
            '2022-05-03T09:00:00',
            '2022-05-03T10:00:00',

            '2022-05-04T05:00:00',
            '2022-05-04T06:00:00',
            '2022-05-04T07:00:00',
            '2022-05-04T08:00:00',
            '2022-05-04T09:00:00',
            '2022-05-04T10:00:00',
            ],
        'mapping': {
            'type': 'tolteca.simu:SkyRasterScanModel',
            'length': '200 arcmin',
            'space': '2 arcmin',
            'n_scans': 100,
            'speed': '500 arcsec/s',
            't_turnaround': '5s',
            'target': '269.7d 66.0d',
            'ref_frame': 'altaz',
            }  # this mapping pattern is 48min
        },

    }


def get_simu_rt():
    rt = SimulatorRuntime('./simu')
    return rt


def run_simu_grid(target_name):
    rt = get_simu_rt()

    grid = lss_plan[target_name]

    grid['target_name'] = target_name
    with open(f'{target_name}_plan.yaml', 'w') as fo:
        yaml.safe_dump(grid, fo)
    # update mapping pattern
    rt.update({
        'simu': {
            'mapping': grid['mapping']
            }
        })

    hls = []
    for t0 in grid['t0']:
        rt.update({
            'simu': {
                'mapping': {
                    't0': t0
                    }
                }
            })
        hl = rt.run_coverage_only(
                write_output=False, mask_with_holdflags=True)
        mapping = rt.get_mapping_model()
        hl.writeto(
                f'{target_name}_'
                f'{mapping.t0.datetime.strftime("%Y_%m_%d_%H_%M_%S")}_'
                f'cov.fits', overwrite=True)
        hls.append(hl)
    # make a coadded cov image
    chl = hls[0]
    for i, hl in enumerate(hls[1:]):
        for j, (chdu, hdu) in enumerate(zip(chl, hl)):
            if j > 0:
                # data
                data, _ = reproject_interp(hdu, chdu.header)
                chdu.data = chdu.data + data
            else:
                # pdu header
                chdu.header.append((
                    f'COADD{i + 1:03d}', hdu.header['EXPTIME'],
                    'Coadded exposure time (s).'
                    ))
    chl.writeto(f'{target_name}_coaded_cov.fits', overwrite=True)


if __name__ == "__main__":

    from tollan.utils.log import init_log
    init_log(level='DEBUG')
    import sys
    field = sys.argv[1]
    run_simu_grid(field)
