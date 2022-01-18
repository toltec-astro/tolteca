#! /usr/bin/env python

from tollan.utils.log import get_logger
from ..base import DataFileIO
from cached_property import cached_property
from astropy.table import Table
from astropy.table import meta as table_meta
from astropy.io.ascii.ecsv import EcsvHeader
from enum import Flag, auto
from kidsproc import kidsmodel
from kidsproc.kidsdata import MultiSweep
import astropy.units as u
import numpy as np

__all__ = ['KidsModelParamsIO', ]


class TableKind(Flag):

    ArrayProp = auto()
    KidsModelParams = auto()

    @classmethod
    def identify(cls, tbl):
        colnames = set(tbl.colnames)
        dispatch = {
                cls.ArrayProp: {
                    'uid', 'nw', 'pg', 'loc', 'ori', 'fg',
                    'x', 'y', 'f',
                    },
                cls.KidsModelParams: {
                    'fp', 'fr', 'Qr',
                    }
                }
        flag = None
        for k, required_colnames in dispatch.items():
            if required_colnames.issubset(colnames):
                if flag is None:
                    flag = k
                else:
                    flag = flag | k
        return flag


class KidsModelParams(object):
    """This class manages a set of KIds model params."""

    logger = get_logger()

    def __init__(self, table, meta=None):
        self._table = table
        self._meta = meta

    @property
    def table(self):
        return self._table

    @property
    def meta(self):
        return self._meta

    @property
    def n_models(self):
        return len(self.model)

    @cached_property
    def model_cls(self):
        return self._get_model_cls(self.table)

    @cached_property
    def model(self):
        return self._get_model(self.model_cls, self.table)

    def get_model(self, i):
        m = self.model
        kwargs = {}
        # print(m)
        for name in m.param_names:
            param = getattr(m, name)
            v = param[i]
            if param.unit is not None:
                v = v << param.unit
            kwargs[name] = v
        return self.model_cls(**kwargs)

    @staticmethod
    def _get_model_cls(tbl):
        dispatch = {
                kidsmodel.KidsSweepGainWithLinTrend: {
                    'columns': [
                        'fp', 'Qr',
                        'Qc', 'fr', 'A',
                        'normI', 'normQ', 'slopeI', 'slopeQ',
                        'interceptI', 'interceptQ'
                        ],
                    }
                }
        for cls, v in dispatch.items():
            # check column names
            cols_required = set(v['columns'])
            if cols_required.issubset(set(tbl.colnames)):
                return cls
        raise NotImplementedError

    @staticmethod
    def _get_model(model_cls, tbl):
        if model_cls is kidsmodel.KidsSweepGainWithLinTrend:
            # print(model_cls.model_params)
            # print(model_cls)
            dispatch = [
                    ('fr', 'fr', u.Hz),
                    ('Qr', 'Qr', None),
                    ('g0', 'normI', None),
                    ('g1', 'normQ', None),
                    ('g', 'normI', None),
                    ('phi_g', 'normQ', None),
                    ('f0', 'fp', u.Hz),
                    ('k0', 'slopeI', u.s),
                    ('k1', 'slopeQ', u.s),
                    ('m0', 'interceptI', None),
                    ('m1', 'interceptQ', None),
                    ]
            args = []
            for k, kk, unit in dispatch:
                if kk is None:
                    args.append(None)
                elif unit is None:
                    args.append(np.asanyarray(tbl[kk]))
                else:
                    args.append(np.asanyarray(tbl[kk]) * unit)
            kwargs = dict(
                    n_models=len(tbl)
                    )
            return model_cls(*args, **kwargs)
        raise NotImplementedError

    def __repr__(self):
        return f"{self.model_cls.__name__}({self.n_models})"

    def make_sweep(self, frequency):
        """Return a `MultiSweep` object given the frequency."""
        return MultiSweep(
                frequency=frequency, S21=self.model(frequency) * u.adu)

    def derotate(self, sweep):
        """Return a `MultiSweep` object that has de-rotated S21."""
        S21_derot = self.model.derotate(
                sweep.S21.to_value(u.adu),
                sweep.frequency
                ).value << u.adu
        return MultiSweep(
                frequency=sweep.frequency,
                S21=S21_derot
                )


class TableIO(DataFileIO):
    """A class to read tables.

    Parameters
    ----------
    source : str, `pathlib.Path`, `FileLoc`, `astropy.table.Table`
        The data file location or model parameter table.
    open_ : bool
        If True and `source` is set, open the file.
    """

    logger = get_logger()

    def __init__(self, source=None, open_=True):
        source = self._source = self._normalize_file_loc(source)
        # init the exit stack
        super(DataFileIO, self).__init__()

        # if source is given, we just open it right away
        if self._source is not None and open_:
            self.open()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.filepath})'

    def open(self, source=None):
        """Return a context to operate on `source`.

        Parameters
        ----------
        source : str, `pathlib.Path`, `FileLoc`, `astropy.table.Table`, optional  # noqa: E501
            The data file location or netCDF dataset. If None, the
            source passed to constructor is used. Noe that source has to
            be None if it has been specified in the constructor.
        """
        # ensure that we don't have source set twice.
        if source is not None and self._source is not None:
            raise ValueError(
                    'source needs to be None for '
                    'object with source set at construction time.')
        # use the constructor source
        if source is None:
            source = self._source
        if source is None:
            raise ValueError('source is not specified')
        if isinstance(source, Table):
            data_loc, data = None, source
        else:
            data_loc = self._normalize_file_loc(source)
            data = Table.read(data_loc.path, format='ascii')
        self._open_state = {
                'data_loc': data_loc,
                'data': data
                }
        # make the cached meta available
        _ = self.meta  # noqa: F841
        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        # reset the nc_node so that this object can be pickled if
        # not bind to open dataset.
        del self._open_state

    @property
    def _file_obj(self):
        # we expose the data table as the low level file object.
        # this returns None if no dataset is open.
        if hasattr(self, '_open_state'):
            return self._open_state['data']
        return None

    @property
    def _file_loc(self):
        # here we return the _source if it is passed to the constructor
        # we had ensured in open that if self._source is given,
        # the source passed to open can only be None.
        # so that source is always the same as self.nc_node.file_loc
        if self._source is not None:
            return self._source
        # if no dataset is open, we just return None
        if self._file_obj is None:
            return None
        # the opened dataset file loc.
        return self._open_state['data_loc']

    def _get_meta(self):
        return dict()

    @cached_property
    def meta(self):
        return self._get_meta()


def identify_txt_kidsmodel(filepath):
    """Check if `filepath` points to a TolTEC model parameter file."""
    logger = get_logger()
    header = EcsvHeader()
    try:
        with open(filepath, 'r') as fo:
            meta = table_meta.get_header_from_yaml(
                    header.process_lines(fo))['meta']
    except Exception as e:
        logger.debug(f"unable to get yaml header from {filepath}: {e}")
        return False
    attrs_to_check = ['Header.Toltec.ObsNum', ]
    for attr in attrs_to_check:
        if attr not in meta:
            return False
    return True


class KidsModelParamsIO(TableIO):
    """"A class to read KIDs model parameters."""

    _meta_mapper = {
            'obsnum': 'Header.Toltec.ObsNum',
            'subobsnum': 'Header.Toltec.SubObsNum',
            'scannum': 'Header.Toltec.ScanNum',
            }

    def read(self):
        """Read the file and return a data object.

        """
        return KidsModelParams(self.file_obj, meta=self.meta)

    # registry info to the DataFileIO.open interface
    io_registry_info = {
            'label': 'txt.toltec.kidsmodel',
            'identifier': identify_txt_kidsmodel
            }

    def _get_meta(self):
        result = super()._get_meta()
        for n, m in self._meta_mapper.items():
            result[n] = self.file_obj.meta.get(m, None)
        return result
