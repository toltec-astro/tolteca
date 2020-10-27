#! /usr/bin/env python


from ...datamodels.io.nc import SimpleNcFileIO
from cached_property import cached_property
from tollan.utils.nc import ncopen
from tollan.utils.log import get_logger


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
            'el': 'Data.TelescopeBackend.TelElAct',
            'ra': 'Data.TelescopeBackend.SourceRaAct',
            'dec': 'Data.TelescopeBackend.SourceDecAct',
            }

    @cached_property
    def meta(self):
        m = dict()
        m['file_loc'] = self.file_loc
        return m

    # registry info to the DataFileIO.open interface
    io_registry_info = {
            'label': 'nc.lmt.tel',
            'identifier': identify_lmt_tel_nc_file
            }
