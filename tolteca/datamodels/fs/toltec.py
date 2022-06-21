#! /usr/bin/env python

"""
This module provides handling of the TolTEC data storage and data file naming
conventions.

"""

import re
from datetime import datetime
from tollan.utils.log import get_logger
from astropy.time import Time
from tollan.utils.registry import Registry, register_to
from tollan.utils import fileloc, dict_from_regex_match
from ..io.toltec.kidsdata import KidsDataKind
from ..io.toltec.table import TableKind


__all__ = ['meta_from_source', 'ToltecDataFileStore', ]


_filepath_meta_parsers = Registry.create()
"""A registry to hold functions that extract info from file path."""


@register_to(_filepath_meta_parsers, 'basic_obs_data')
def _meta_from_bod_filename(file_loc):
    """Return the meta data parsed from the filename of BOD file."""

    path = file_loc.path
    filename = path.name

    re_toltec_file = re.compile(
        r'^(?P<interface>toltec(?P<roachid>\d+))_(?P<obsnum>\d+)_'
        r'(?P<subobsnum>\d+)_(?P<scannum>\d+)_'
        r'(?P<ut>\d{4}_\d{2}_\d{2}(?:_\d{2}_\d{2}_\d{2}))'
        r'(?:_(?P<filesuffix>[^\/.]+))?'
        r'\.(?P<fileext>.+)$')

    def parse_ut(v):
        result = Time(
            datetime.strptime(v, '%Y_%m_%d_%H_%M_%S'),
            scale='utc')
        result.format = 'isot'
        return result

    type_dispatcher = {
        'roachid': int,
        'obsnum': int,
        'subobsnum': int,
        'scannum': int,
        'ut': parse_ut,
        'fileext': lambda s: s.lower()
        }

    meta = dict_from_regex_match(
            re_toltec_file, filename, type_dispatcher)
    if meta is None:
        return None

    # add more items to the meta
    meta['file_loc'] = file_loc
    meta['instru'] = 'toltec'

    data_kind_mapper = {
            ('vnasweep', 'nc'): KidsDataKind.VnaSweep,
            ('targsweep', 'nc'): KidsDataKind.TargetSweep,
            ('tune', 'nc'): KidsDataKind.Tune,
            ('timestream', 'nc'): KidsDataKind.RawTimeStream,
            ('vnasweep_processed', 'nc'):
            KidsDataKind.ReducedSweep,
            ('targsweep_processed', 'nc'):
            KidsDataKind.ReducedSweep,
            ('tune_processed', 'nc'):
            KidsDataKind.ReducedSweep,
            ('timestream_processed', 'nc'):
            KidsDataKind.SolvedTimeStream,
            ('vnasweep', 'txt'):
            TableKind.KidsModelParams,
            ('targsweep', 'txt'):
            TableKind.KidsModelParams,
            ('tune', 'txt'):
            TableKind.KidsModelParams,
        }
    meta['data_kind'] = data_kind_mapper.get(
            (meta['filesuffix'], meta['fileext']), None)

    # one can infer the master if the immediate parent of the file is
    # the interface
    if path.parent.name == meta['interface']:
        master_name = path.parent.parent.name
        if master_name in ['ics', 'tcs', 'clip']:
            meta['master_name'] = master_name
    return meta


@register_to(_filepath_meta_parsers, 'wyatt')
def _meta_from_wyatt_filename(file_loc):
    """Return the meta data parsed from the Wyatt filename."""

    path = file_loc.path
    filename = path.name

    re_wyatt_file = (
        r'^(?P<interface>wyatt)'
        r'_(?P<ut>\d{4}-\d{2}-\d{2})'
        r'_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)'
        r'\.(?P<fileext>.+)$')

    def parse_ut(v):
        result = Time(
            datetime.strptime(v, '%Y-%m-%d'),
            scale='utc')
        result.format = 'isot'
        return result

    type_dispatcher = {
        'obsnum': int,
        'subobsnum': int,
        'scannum': int,
        'ut': parse_ut,
        'fileext': lambda s: s.lower()
        }

    meta = dict_from_regex_match(
            re_wyatt_file, filename, type_dispatcher)
    if meta is None:
        return None

    # add more items to the meta
    meta['file_loc'] = file_loc
    meta['interface'] = 'wyatt'
    meta['instru'] = 'wyatt'
    meta['master_name'] = 'ics'
    return meta



@register_to(_filepath_meta_parsers, 'toltec_hk')
def _meta_from_hk_filename(file_loc):
    """Return the meta data parsed from the HK filename."""

    path = file_loc.path
    filename = path.name

    re_hk_file = (
        r'^(?P<interface>toltec_hk)'
        r'_(?P<ut>\d{4}-\d{2}-\d{2})'
        r'_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)'
        r'\.(?P<fileext>.+)$')

    def parse_ut(v):
        result = Time(
            datetime.strptime(v, '%Y-%m-%d'),
            scale='utc')
        result.format = 'isot'
        return result

    type_dispatcher = {
        'obsnum': int,
        'subobsnum': int,
        'scannum': int,
        'ut': parse_ut,
        'fileext': lambda s: s.lower()
        }

    meta = dict_from_regex_match(
            re_hk_file, filename, type_dispatcher)
    if meta is None:
        return None

    # add more items to the meta
    meta['file_loc'] = file_loc
    meta['interface'] = 'hk'
    meta['instru'] = 'toltec'
    meta['master_name'] = 'ics'
    return meta


@register_to(_filepath_meta_parsers, 'lmt_tel')
def _meta_from_lmt_tel_filename(file_loc):
    """Return the meta data parsed from the LMT telescope filename."""

    path = file_loc.path
    filename = path.name

    re_lmt_tel_file = (
        r'^(?P<interface>tel_\w+)'
        r'_(?P<ut>\d{4}-\d{2}-\d{2})'
        r'_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)'
        r'\.(?P<fileext>.+)$')

    def parse_ut(v):
        result = Time(
            datetime.strptime(v, '%Y-%m-%d'),
            scale='utc')
        result.format = 'isot'
        return result

    type_dispatcher = {
        'obsnum': int,
        'subobsnum': int,
        'scannum': int,
        'ut': parse_ut,
        'fileext': lambda s: s.lower(),
        'interface': lambda s: ('lmt' if s == 'tel' else s)
        }

    meta = dict_from_regex_match(
            re_lmt_tel_file, filename, type_dispatcher)
    if meta is None:
        return None

    # add more items to the meta
    meta['file_loc'] = file_loc
    meta['instru'] = 'lmt'
    meta['master_name'] = 'tcs'
    return meta


@register_to(_filepath_meta_parsers, 'tolteca.simu')
def _meta_from_simu_filename(file_loc):
    """Return the meta data parsed from the tolteca.simu results."""

    path = file_loc.path
    filename = path.name

    re_simu = re.compile(
        r'^(?P<interface>tel|apt)_(?P<obsnum>\d+)_'
        r'(?P<subobsnum>\d+)_(?P<scannum>\d+)_'
        r'(?P<ut>\d{4}_\d{2}_\d{2}(?:_\d{2}_\d{2}_\d{2}))'
        r'\.(?P<fileext>.+)$')

    def parse_ut(v):
        result = Time(
            datetime.strptime(v, '%Y_%m_%d_%H_%M_%S'),
            scale='utc')
        result.format = 'isot'
        return result

    type_dispatcher = {
        'obsnum': int,
        'subobsnum': int,
        'scannum': int,
        'ut': parse_ut,
        'fileext': lambda s: s.lower(),
        'interface': lambda s: ('lmt' if s == 'tel' else s)
        }

    meta = dict_from_regex_match(
            re_simu, filename, type_dispatcher)
    if meta is None:
        return None

    # add more items to the meta
    meta['file_loc'] = file_loc
    meta['instru'] = 'tolteca.simu'
    return meta


def meta_from_source(source):
    """Extract meta data from `source` according to the file naming
    conventions.

    Parameters
    ----------
    source : str, `~pathlib.Path`, `~tollan.utils.FileLoc`
        The location of the file.
    """

    file_loc = fileloc(source)
    for type_, parser in _filepath_meta_parsers.items():
        meta = parser(file_loc)
        if meta is None:
            continue
        break
    if meta is None:
        raise ValueError(f"unable to parse meta from source {source}")
    return meta


class ToltecDataFileStore(object):
    """A helper class to work with directories that store the
    TolTEC data files.
    """
    logger = get_logger()

    def __init__(self, root):
        self._root_loc = fileloc(root)

    @property
    def rootpath(self):
        return self._root_loc.path

    @property
    def is_local(self):
        return self._root_loc.is_local
