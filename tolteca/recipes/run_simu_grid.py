#!/usr/bin/env python

from tolteca.simu0 import SimulatorRuntime
from reproject import reproject_interp
import yaml
import astropy.units as u


lss_plan = {

    'ECDFS': {
        't0': [
            '2024-11-01T04:00:00',
            '2024-11-01T05:00:00',
            '2024-11-01T06:00:00',
            '2024-11-01T07:00:00',
            '2024-11-01T08:00:00',
            '2024-11-01T09:00:00',

            '2024-11-02T04:00:00',
            '2024-11-02T05:00:00',
            '2024-11-02T06:00:00',
            '2024-11-02T07:00:00',
            '2024-11-02T08:00:00',
            '2024-11-02T09:00:00',

            '2024-11-03T04:00:00',
            '2024-11-03T05:00:00',
            '2024-11-03T06:00:00',
            '2024-11-03T07:00:00',
            '2024-11-03T08:00:00',
            '2024-11-03T09:00:00',

            '2024-11-04T04:00:00',
            '2024-11-04T05:00:00',
            '2024-11-04T06:00:00',
            '2024-11-04T07:00:00',
            '2024-11-04T08:00:00',
            '2024-11-04T09:00:00',

            ],
        'mapping': {
            'type': 'tolteca.simu0:SkyRasterScanModel',
            'length': '210 arcmin',
            'space': '2 arcmin',
            'n_scans': 104,
            'speed': '300 arcsec/s',
            't_turnaround': '5s',
            'target': '52.95d -28.1d',
            'ref_frame': 'altaz',
            }  # this mapping pattern is 48min
        },
    'XMM-LSS': {
        't0': [
            '2024-11-02T03:00:00',
            '2024-11-02T04:00:00',
            '2024-11-02T05:00:00',
            '2024-11-02T06:00:00',
            '2024-11-02T07:00:00',
            '2024-11-02T08:00:00',

            '2024-11-03T03:00:00',
            '2024-11-03T04:00:00',
            '2024-11-03T05:00:00',
            '2024-11-03T06:00:00',
            '2024-11-03T07:00:00',
            '2024-11-03T08:00:00',

            '2024-11-04T04:30:00',
            '2024-11-04T05:00:00',
            '2024-11-04T06:00:00',
            '2024-11-04T07:00:00',
            '2024-11-04T08:00:00',
            ],
        'mapping': {
            'type': 'tolteca.simu0:SkyRasterScanModel',
            'length': '210 arcmin',
            'space': '2 arcmin',
            'n_scans': 105,
            'speed': '300 arcsec/s',
            't_turnaround': '5s',
            'target': '35.440d -4.6500d',
            'ref_frame': 'altaz',
            }  # this mapping pattern is 48min
        },
    'COSMOS': {
        't0': [
            '2025-02-01T04:00:00',
            '2025-02-01T05:00:00',
            '2025-02-01T06:00:00',
            '2025-02-01T07:00:00',
            '2025-02-01T08:00:00',
            '2025-02-01T09:00:00',
            '2025-02-01T10:00:00',

            '2025-02-02T04:00:00',
            '2025-02-02T05:00:00',
            '2025-02-02T06:00:00',
            '2025-02-02T07:00:00',
            '2025-02-02T08:00:00',
            '2025-02-02T09:00:00',
            '2025-02-02T10:00:00',

            '2025-02-03T04:00:00',
            '2025-02-03T05:00:00',
            '2025-02-03T06:00:00',
            '2025-02-03T07:00:00',
            '2025-02-03T08:00:00',
            '2025-02-03T09:00:00',
            '2025-02-03T10:00:00',

            '2025-02-04T04:00:00',
            '2025-02-04T05:00:00',
            '2025-02-04T06:00:00',
            '2025-02-04T07:00:00',
            '2025-02-04T08:00:00',
            '2025-02-04T09:00:00',
            '2025-02-04T10:00:00',
            ],
        'mapping': {
            'type': 'tolteca.simu0:SkyRasterScanModel',
            'length': '120 arcmin',
            'space': '2 arcmin',
            'n_scans': 60,
            'speed': '300 arcsec/s',
            't_turnaround': '5s',
            'target': '150.131d 2.200d',
            'ref_frame': 'altaz',
            }  # this mapping pattern is 48min
        },
    # 'COSMOS': {
    #     't0': [
    #         '2022-02-01T04:00:00',
    #         '2022-02-01T05:00:00',
    #         '2022-02-01T06:00:00',
    #         '2022-02-01T07:00:00',
    #         '2022-02-01T08:00:00',
    #         '2022-02-01T09:00:00',
    #         '2022-02-01T10:00:00',
    #
    #         '2022-02-02T04:00:00',
    #         '2022-02-02T05:00:00',
    #         '2022-02-02T06:00:00',
    #         '2022-02-02T07:00:00',
    #         '2022-02-02T08:00:00',
    #         '2022-02-02T09:00:00',
    #         '2022-02-02T10:00:00',
    #
    #         '2022-02-03T04:00:00',
    #         '2022-02-03T05:00:00',
    #         '2022-02-03T06:00:00',
    #         '2022-02-03T07:00:00',
    #         '2022-02-03T08:00:00',
    #         '2022-02-03T09:00:00',
    #         '2022-02-03T10:00:00',
    #
    #         '2022-02-04T04:00:00',
    #         '2022-02-04T05:00:00',
    #         '2022-02-04T06:00:00',
    #         '2022-02-04T07:00:00',
    #         '2022-02-04T08:00:00',
    #         '2022-02-04T09:00:00',
    #         '2022-02-04T10:00:00',
    #         ],
    #     'mapping': {
    #         'type': 'tolteca.simu0:SkyRasterScanModel',
    #         'length': '100 arcmin',
    #         'space': '2 arcmin',
    #         'n_scans': 75,
    #         'speed': '200 arcsec/s',
    #         't_turnaround': '5s',
    #         'target': '150.1d 2.2d',
    #         'ref_frame': 'altaz',
    #         }  # this mapping pattern is 48min
    #     },
    'Bootes': {
        't0': [
            '2022-04-02T07:00:00',
            '2022-04-02T08:00:00',
            '2022-04-02T09:00:00',
            '2022-04-02T10:00:00',

            '2022-04-03T07:00:00',
            '2022-04-03T08:00:00',
            '2022-04-03T09:00:00',
            '2022-04-03T10:00:00',

            '2022-04-04T07:00:00',
            '2022-04-04T08:00:00',
            '2022-04-04T09:00:00',
            '2022-04-04T10:00:00',
            ],
        'mapping': {
            'type': 'tolteca.simu0:SkyRasterScanModel',
            'length': '180 arcmin',
            'space': '2 arcmin',
            'n_scans': 90,
            'speed': '300 arcsec/s',
            't_turnaround': '5s',
            # 'target': '217.9d 34.1d',
            # 'target': '218.46d 34.45d',
            'target': '218.46d 34.3d',
            'ref_frame': 'altaz',
            }  # this mapping pattern is 48min
        },
    'Bootes_radec': {
        't0': [
            '2022-04-02T07:00:00',
            '2022-04-02T08:00:00',
            '2022-04-02T09:00:00',
            '2022-04-02T10:00:00',

            '2022-04-03T07:00:00',
            '2022-04-03T08:00:00',
            '2022-04-03T09:00:00',
            '2022-04-03T10:00:00',

            '2022-04-04T07:00:00',
            '2022-04-04T08:00:00',
            '2022-04-04T09:00:00',
            '2022-04-04T10:00:00',
            ],
        'rot': [
            "0 deg", "45 deg", "90 deg", "135 deg",
            "10 deg", "55 deg", "100 deg", "145 deg",
            "20 deg", "65 deg", "110 deg", "155 deg",
            "30 deg", "75 deg", "120 deg", "165 deg",
        ],
        'mapping': {
            'type': 'tolteca.simu0:SkyRasterScanModel',
            'length': '180 arcmin',
            'space': '2 arcmin',
            'n_scans': 90,
            'speed': '300 arcsec/s',
            't_turnaround': '5s',
            # 'target': '217.9d 34.1d',
            # 'target': '218.46d 34.45d',
            'target': '218.46d 34.3d',
            'ref_frame': 'icrs',
            }  # this mapping pattern is 48min
        },
    'XMM-LSS_radec': {
        't0': [
            '2024-11-02T04:00:00',
            '2024-11-02T05:00:00',
            '2024-11-02T06:00:00',
            '2024-11-02T07:00:00',

            '2024-11-03T04:00:00',
            '2024-11-03T05:00:00',
            '2024-11-03T06:00:00',
            '2024-11-03T07:00:00',

            '2024-11-04T04:00:00',
            '2024-11-04T05:00:00',
            '2024-11-04T06:00:00',
            '2024-11-04T07:00:00',

            ],
        'rot': [
            "0 deg", "45 deg", "90 deg", "135 deg",
            "10 deg", "55 deg", "100 deg", "145 deg",
            "20 deg", "65 deg", "110 deg", "155 deg",
            "30 deg", "75 deg", "120 deg", "165 deg",
        ],
        'mapping': {
            'type': 'tolteca.simu0:SkyRasterScanModel',
            'length': '210 arcmin',
            'space': '2 arcmin',
            'n_scans': 105,
            'speed': '300 arcsec/s',
            't_turnaround': '5s',
            'target': '35.440d -4.6500d',
            'ref_frame': 'icrs',
            }  # this mapping pattern is 48min
        },

    'NEP': {
        't0': [
            '2025-05-01T06:00:00',
            '2025-05-01T07:30:00',
            '2025-05-01T09:00:00',
            '2025-05-01T10:30:00',

            '2025-05-02T06:00:00',
            '2025-05-02T07:30:00',
            '2025-05-02T09:00:00',
            '2025-05-02T10:30:00',

            '2025-05-03T06:00:00',
            '2025-05-03T07:30:00',
            '2025-05-03T09:00:00',
            '2025-05-03T10:30:00',

            '2025-05-04T06:00:00',
            '2025-05-04T07:30:00',
            '2025-05-04T09:00:00',
            '2025-05-04T10:30:00',

            ],
        'mapping': {
            'type': 'tolteca.simu0:SkyRasterScanModel',
            'length': '210 arcmin',
            'space': '2 arcmin',
            'n_scans': 105,
            'speed': '300 arcsec/s',
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
    for ir, t0 in enumerate(grid['t0']):
        if 'rot' in grid:
            rot = grid["rot"][ir]
        else:
            rot = 0 << u.deg
        rt.update({
            'simu': {
                'mapping': {
                    't0': t0,
                    'rot': rot,
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
