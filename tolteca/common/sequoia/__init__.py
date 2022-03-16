#!/usr/bin/env python


import astropy.units as u


__all__ = ['sequoia_info', ]


sequoia_info = {
    'instru': 'sequoia',
    'name': 'SEQUOIA',
    'name_long': 'SEQUOIA',
    'fov_diameter': 2. << u.arcmin,
    'pixel_space': 27.8 << u.arcsec,
    'pa0': 46 << u.deg,
    'band_names': ['b1075', 'b925'],
    'b1075': {
        'index': 0,
        'name': '100-115GHz',
        'name_long': 'SEQUOIA 100-115GHz',
        'wl_center': 107.5 << u.GHz,
        'hpbw': 13.0 << u.arcsec,
        },
    'b925': {
        'index': 1,
        'name': '85-100GHz',
        'name_long': 'SEQUOIA 85-100GHz',
        'wl_center': 92.5 << u.GHz,
        'hpbw': 15.1 << u.arcsec,
        },
    'mode_names': ['s_wide', 's_intermediate', 's_narrow'],
    's_wide': {
        'name': 'wide',
        'bw_total': 800 << u.MHz,
        'n_chans': 2048,
        'bw_chan': 391 << u.kHz
        },
    's_intermediate': {
        'name': 'intermediate',
        'bw_total': 400 << u.MHz,
        'n_chans': 4096,
        'bw_chan': 98 << u.kHz
        },
    's_narrow': {
        'name': 'narrow',
        'bw_total': 200 << u.MHz,
        'n_chans': 8192,
        'bw_chan': 24 << u.kHz
        },
    }
