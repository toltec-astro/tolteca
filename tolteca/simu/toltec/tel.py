#! /usr/bin/env python


from ..base import SkyICRSTrajModel  # , SkyAltAzTrajModel

from ...datamodels.io.nc import SimpleNcFileIO
from cached_property import cached_property
from tollan.utils.nc import ncopen
from tollan.utils.log import get_logger
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
# from astropy.modeling import models


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


class LmtTelFileIO(SimpleNcFileIO):
    """A class to read the TolTEC dilution fridge data files.

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
            }

    @cached_property
    def meta(self):
        m = dict()
        m['file_loc'] = self.file_loc
        m['mapping_type'] = self.getstr('mapping_type')

        # target info
        t_ra, t_ra_off = self.getvar('target_ra')[:]
        t_dec, t_dec_off = self.getvar('target_dec')[:]
        t_az, t_az_off = self.getvar('target_az')[:]
        t_alt, t_alt_off = self.getvar('target_alt')[:]

        m['target'] = SkyCoord(
                t_ra << u.deg, t_dec << u.deg, frame='icrs')
        m['target_off'] = SkyCoord(
                t_ra_off << u.deg, t_dec_off << u.deg, frame='icrs')
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
        t0 = Time([0], format='unix')
        t = t - t[0]
        ra = self.getvar('ra')[:]
        dec = self.getvar('dec')[:]
        az = self.getvar('az')[:]
        alt = self.getvar('alt')[:]
        holdflag = self.getvar('holdflag')[:].astype(int)
        m1 = SkyICRSTrajModel(
                time=t,
                ra=ra,
                dec=dec,
                az=az,
                alt=alt,
                t0=t0,
                meta=meta,
                holdflag=holdflag,
                target=meta['target'],
                ref_frame='icrs',
                )
        # m2 = SkyAltAzTrajModel(
        #         time=t,
        #         az=az,
        #         alt=alt,
        #         t0=t0,
        #         )
        # make a composite to eval for all coords
        # return models.Mapping((0, 0)) | (m1 & m2)
        return m1
