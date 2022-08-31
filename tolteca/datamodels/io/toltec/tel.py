#! /usr/bin/env python

from ..nc import SimpleNcFileIO
from cached_property import cached_property
from tollan.utils.nc import ncopen
from tollan.utils.log import get_logger
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord


__all__ = ['LmtTelFileIO']


def identify_lmt_tel_nc_file(filepath):
    """Check if `filepath` points to an LMT telescope pointing file."""
    logger = get_logger()
    attrs_to_check = ['Data.TelescopeBackend.TelTime', ]
    try:
        with ncopen(filepath) as ds:
            return any(a in ds.variables for a in attrs_to_check)
    except Exception as e:
        logger.debug(f"unable to open file {filepath} as netCDF dataset: {e}")
        return False
    return True


class LmtTelData(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class LmtTelFileIO(SimpleNcFileIO):
    """A class to read the LMT TolTEC tel files.

    Parameters
    ----------
    source : str, `pathlib.Path`, `FileLoc`, `netCDF4.Dataset`
        The data file location or netCDF dataset.
        This is passed to `~tollan.utils.nc.NcNodeMapper`.
    """

    _nc_node_map = {
            'time': 'Data.TelescopeBackend.TelTime',
            'az': 'Data.TelescopeBackend.TelAzAct',
            'alt': 'Data.TelescopeBackend.TelElAct',
            'ra': 'Data.TelescopeBackend.SourceRaAct',
            'dec': 'Data.TelescopeBackend.SourceDecAct',
            'holdflag': 'Data.TelescopeBackend.Hold',
            'mapping_type': 'Header.Dcs.ObsPgm',
            'target_ra': 'Header.Source.Ra',
            'target_dec': 'Header.Source.Dec',
            'target_az': 'Header.Source.Az',
            'target_alt': 'Header.Source.El',
            'obs_goal': 'Header.Dcs.ObsGoal',
            "mastervar": "Header.Toltec.Master",
            "repeatvar": "Header.Toltec.RepeatLevel",
            }

    @cached_property
    def meta(self):
        m = dict()
        m['file_loc'] = self.file_loc
        m['mapping_type'] = self.getstr('mapping_type')
        m['obs_goal'] = self.getstr('obs_goal').lower()
        m['master'] = 0
        m['master_name'] = 'tcs'
        m['interface'] = 'lmt'

        # target info
        t_ra, t_ra_off = self.getvar('target_ra')[:]
        t_dec, t_dec_off = self.getvar('target_dec')[:]
        t_az, t_az_off = self.getvar('target_az')[:]
        t_alt, t_alt_off = self.getvar('target_alt')[:]

        t0 = Time(self.getvar('time')[0], format='unix')
        t0.format = 'isot'
        m['t0'] = t0
        m['target'] = SkyCoord(
                t_ra << u.rad, t_dec << u.rad, frame='icrs')
        m['target_off'] = SkyCoord(
                t_ra_off << u.rad, t_dec_off << u.rad, frame='icrs')
        m['target_frame'] = 'icrs'
        return m

    # registry info to the DataFileIO.open interface
    io_registry_info = {
            'label': 'nc.lmt.tel',
            'identifier': identify_lmt_tel_nc_file
            }

    def read(self):
        meta = self.meta
        t = self.getvar('time')[:]
        t = (t - t[0]) << u.s
        ra = self.getvar('ra')[:] << u.rad
        dec = self.getvar('dec')[:] << u.rad
        az = self.getvar('az')[:] << u.rad
        alt = self.getvar('alt')[:] << u.rad
        holdflag = self.getvar('holdflag')[:].astype(int)
        return LmtTelData(
                time=t,
                ra=ra,
                dec=dec,
                az=az,
                alt=alt,
                t0=meta['t0'],
                meta=meta,
                holdflag=holdflag,
                target=meta['target'],
                ref_frame='icrs',
                )
