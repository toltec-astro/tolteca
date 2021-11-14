#!/usr/bin/env python


import astropy.units as u


__all__ = ['lmt_info', ]


lmt_info = {
    'instru': 'lmt',
    'name': 'LMT',
    'name_long': "Large Millimeter Telescope",
    'location': {
        'lon': '-97d18m52.6s',
        'lat': '+18d59m10s',
        'height': 4640 << u.m,
        },
    'timezone_local': 'America/Mexico_City'
    }
