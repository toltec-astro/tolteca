#!/usr/bin/env python


import astropy.units as u


__all__ = ['toltec_info', ]


toltec_info = {
    'instru': 'toltec',
    'name': 'TolTEC',
    'name_long': 'TolTEC Camera',
    'array_physical_diameter': 127.049101 << u.mm,
    'fov_diameter': 4. << u.arcmin,
    'fg_names': ['fg0', 'fg1', 'fg2', 'fg3'],
    'fg0': {
        'index': 0,
        'det_pa': 0. << u.deg,
        },
    'fg1': {
        'index': 1,
        'det_pa': 45. << u.deg,
        },
    'fg2': {
        'index': 2,
        'det_pa': 90. << u.deg,
        },
    'fg3': {
        'index': 3,
        'det_pa': 135. << u.deg,
        },
    'array_names': ['a1100', 'a1400', 'a2000'],
    'a1100': {
        'index': 0,
        'name': 'a1100',
        'name_long': 'TolTEC 1.1 mm array',
        'n_dets': 4012,
        'wl_center': 1.1 << u.mm,
        'array_mounting_angle': 90. << u.deg
        },
    'a1400': {
        'index': 1,
        'name': 'a1400',
        'name_long': 'TolTEC 1.4 mm array',
        'n_dets': 2534,
        'wl_center': 1.4 << u.mm,
        'array_mounting_angle': -90. << u.deg
        },
    'a2000': {
        'index': 2,
        'name': 'a2000',
        'name_long': 'TolTEC 2.0 mm array',
        'n_dets': 1172,
        'wl_center': 2.0 << u.mm,
        'array_mounting_angle': -90. << u.deg
        },
    'nws': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'interfaces': [
        'toltec0', 'toltec1', 'toltec2', 'toltec3',
        'toltec4', 'toltec5', 'toltec6',
        'toltec7', 'toltec8', 'toltec9', 'toltec10',
        'toltec11', 'toltec12',
        'hwpr', 'wyatt', 'tel', 'toltec_hk'],
    'toltec0': {
        'name': 'toltec0',
        'nw': 0,
        'array_name': 'a1100',
        },
    'toltec1': {
        'name': 'toltec1',
        'nw': 1,
        'array_name': 'a1100',
        },
    'toltec2': {
        'name': 'toltec2',
        'nw': 2,
        'array_name': 'a1100',
        },
    'toltec3': {
        'name': 'toltec3',
        'nw': 3,
        'array_name': 'a1100',
        },
    'toltec4': {
        'name': 'toltec4',
        'nw': 4,
        'array_name': 'a1100',
        },
    'toltec5': {
        'name': 'toltec5',
        'nw': 5,
        'array_name': 'a1100',
        },
    'toltec6': {
        'name': 'toltec6',
        'nw': 6,
        'array_name': 'a1100',
        },
    'toltec7': {
        'name': 'toltec7',
        'nw': 7,
        'array_name': 'a1400',
        },
    'toltec8': {
        'name': 'toltec8',
        'nw': 8,
        'array_name': 'a1400',
        },
    'toltec9': {
        'name': 'toltec9',
        'nw': 9,
        'array_name': 'a1400',
        },
    'toltec10': {
        'name': 'toltec10',
        'nw': 10,
        'array_name': 'a1400',
        },
    'toltec11': {
        'name': 'toltec11',
        'nw': 11,
        'array_name': 'a2000',
        },
    'toltec12': {
        'name': 'toltec12',
        'nw': 12,
        'array_name': 'a2000',
        },
    'hwpr': {
        'name': 'hwpr',
        },
    'wyatt': {
        'name': 'wyatt',
        },
    'tel': {
        'name': 'tel',
        },
    'toltec_hk': {
        'name': 'toltec_hk'
        },
    }
