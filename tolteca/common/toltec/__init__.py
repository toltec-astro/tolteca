#!/usr/bin/env python


from tolteca.utils import get_pkg_data_path
from astropy.io.misc import yaml


__all__ = ['info', ]

_info_yaml = get_pkg_data_path().joinpath('common/toltec.yaml')

with open(_info_yaml, 'r') as fo:
    info = yaml.load(fo)
