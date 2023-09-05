import functools
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table
from tollan.utils.fileloc import FileLoc
from tollan.utils.fmt import pformat_yaml
from tollan.utils.general import add_to_dict
from tollan.utils.log import logger

from tolteca_kidsproc import kidsdata, kidsmodel

from .core import ToltecDataFileIO, format_doc, base_doc
from .types import ToltecDataKind


@format_doc(base_doc)
class TableIO(ToltecDataFileIO):
    """A class to read TolTEC table."""

    # TODO get rid of this when all table writers produce the correct units.
    _toltec_tbl_col_default_unit = {
        ("f_out", u.Hz),
        ("f_in", u.Hz),
        ("fp", u.Hz),
        ("fr", u.Hz),
        ("slopeI", u.s),
        ("slopeQ", u.s),
    }

    _meta_mappers = {
        ToltecDataKind.TableData: {
            "obsnum": "Header.Toltec.ObsNum",
            "subobsnum": "Header.Toltec.SubObsNum",
            "scannum": "Header.Toltec.ScanNum",
        },
        ToltecDataKind.KidsTableData: {
            "roach": "Header.Toltec.RoachIndex",
            "accumlen": ("Header.Toltec.AccumLen", 524288),
        },
    }

    @classmethod
    def _get_source_file_info(cls, source, source_loc):
        if isinstance(source, Table):
            file_loc_orig = None
            file_loc = source_loc
            file_obj = source.copy()
        elif isinstance(source, (str, Path, FileLoc)):
            file_loc_orig = source_loc
            file_loc = source
            file_obj = None
        else:
            raise TypeError(f"invalid source {source}")
        file_loc_orig = cls._validate_file_loc(file_loc_orig)
        file_loc = cls._validate_file_loc(file_loc)
        # TODO add logic to handle remote file
        if file_loc.is_remote:
            raise ValueError(f"remote file is not supported: {file_loc}")
        return {
            "file_loc_orig": file_loc_orig,
            "file_loc": file_loc,
            "file_obj": file_obj,
        }

    def open(self):
        """Open the TolTEC table file."""
        file_loc = self.file_loc
        if self.io_state.is_open():
            return self
        tbl = Table.read(file_loc.path, format="ascii")
        # attach unit to data column if not already
        for c, unit in self._toltec_tbl_col_default_unit:
            if c in tbl.colnames and tbl[c].unit is None:
                tbl[c].unit = unit
        # update the underlying io obj
        self._set_open_state(tbl)
        return self

    def _load_meta_from_io_obj(self):
        """Return meta data from the table."""
        meta = self._meta
        tbl = self.io_state.io_obj
        data_kind = self.data_kind

        for k, m in self._meta_mappers.items():
            if k & data_kind:
                # read all entries in mapper
                for kk, tbl_meta_key in m.items():
                    if isinstance(tbl_meta_key, tuple):
                        _tbl_meta_key, defval = tbl_meta_key
                        v = tbl.meta.get(_tbl_meta_key, defval)
                    else:
                        v = tbl.meta[tbl_meta_key]
                    meta[kk] = v

        # update meta using the per type registered updater.
        for k, _meta_updater in self._meta_updaters.items():
            if data_kind & k:
                _meta_updater(self)
        logger.debug(f"loaded meta data:\n{pformat_yaml(meta)}")

    # a registry to metadata updaters
    _meta_updaters = {}

    @add_to_dict(_meta_updaters, ToltecDataKind.TableData)
    def _update_derived_info(self):
        meta = self._meta
        meta["instru"] = "toltec"

    @add_to_dict(_meta_updaters, ToltecDataKind.KidsTableData)
    def _update_derived_info_kids(self):
        meta = self._meta
        meta["interface"] = f'toltec{meta["roach"]}'
        meta["nw"] = meta["roach"]
        # tbl stats
        tbl = self.io_state.io_obj
        meta["n_rows"] = len(tbl)
        for c, cn in [
            ("model_id", "n_models"),
            ("group_id", "n_groups"),
            ("chan_id", "n_chans"),
            ("tone_id", "n_tones"),
        ]:
            if c in tbl.colnames:
                meta[cn] = len(np.unique(tbl[c]))

    def read(self):
        """Return the TolTEC table object."""
        data_kind = self.data_kind
        meta = self._meta
        data = self.io_state.io_obj
        for k, m in self._tbl_obj_makers.items():
            if data_kind & k:
                return m(self.__class__, data_kind, meta, data)
        # generic maker, this is just the underlying table with meta updated
        data.meta.update(meta)
        return data

    _tbl_obj_makers = {}

    @classmethod
    @add_to_dict(_tbl_obj_makers, ToltecDataKind.KidsTableData)
    def _make_kids_table(cls, data_kind, meta, data):
        tbl = QTable(data)
        tbl.meta.update(meta)
        if data_kind & ToltecDataKind.KidsModelParamsTable:
            return KidsModelParamsTable(tbl)
        return tbl

    @classmethod
    def identify(cls, file_loc, file_obj=None):
        """Return if this class can handle ths given file."""
        if file_obj is not None:
            return isinstance(file_obj, Table)
        if file_loc is not None:
            return file_loc.path.suffix in [".ecsv", ".txt"]
        return False


class KidsModelParamsTable(QTable):
    """A class to manage a set of Kids model params."""

    @property
    def n_models(self):
        """The number of KIDs models."""
        return len(self.modelset)

    @functools.cached_property
    def model_cls(self):
        """The model class as infered from the table data."""
        return self._get_model_cls(self)

    @functools.cached_property
    def modelset(self):
        """The modelset instance created from the table."""
        return self._get_modelset(self.model_cls, self)

    def get_model(self, i):
        """Return the i-th model in the model set."""
        m = self.modelset
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
                "columns": [
                    "fp",
                    "Qr",
                    "Qc",
                    "fr",
                    "A",
                    "normI",
                    "normQ",
                    "slopeI",
                    "slopeQ",
                    "interceptI",
                    "interceptQ",
                ],
            },
        }
        for cls, v in dispatch.items():
            # check column names
            cols_required = set(v["columns"])
            if cols_required.issubset(set(tbl.colnames)):
                return cls
        raise ValueError("unable to infer model class.")

    @staticmethod
    def _get_modelset(model_cls, tbl):
        if model_cls is kidsmodel.KidsSweepGainWithLinTrend:
            dispatch = [
                ("fr", "fr", u.Hz),
                ("Qr", "Qr", None),
                ("g0", "normI", None),
                ("g1", "normQ", None),
                ("g", "normI", None),
                ("phi_g", "normQ", None),
                ("f0", "fp", u.Hz),
                ("k0", "slopeI", u.s),
                ("k1", "slopeQ", u.s),
                ("m0", "interceptI", None),
                ("m1", "interceptQ", None),
            ]
            args = []
            for _k, kk, unit in dispatch:
                if kk is None:
                    args.append(None)
                elif tbl[kk].unit is not None or unit is None:
                    args.append(np.asanyarray(tbl[kk]))
                else:
                    args.append(np.asanyarray(tbl[kk]) << unit)
            kwargs = {"n_models": len(tbl)}
            return model_cls(*args, **kwargs)
        raise ValueError(f"unable to create modelset of type {model_cls}.")

    def __repr__(self):
        return f"{self.model_cls.__name__}({self.n_models})"

    def make_sweep(self, frequency):
        """Return a `MultiSweep` object given the frequency."""
        return kidsdata.MultiSweep(
            frequency=frequency,
            S21=self.model(frequency) * u.adu,
        )

    def derotate(self, sweep):
        """Return a `MultiSweep` object that has de-rotated S21."""
        S21_derot = (
            self.model.derotate(sweep.S21.to_value(u.adu), sweep.frequency).value
            << u.adu
        )
        return kidsdata.MultiSweep(frequency=sweep.frequency, S21=S21_derot)

    def rotate(self, sweep):
        """Return a `MultiSweep` object that has rotated S21."""
        S21 = (
            self.model.rotate(sweep.S21.to_value(u.adu), sweep.frequency).value << u.adu
        )
        return kidsdata.MultiSweep(frequency=sweep.frequency, S21=S21)
