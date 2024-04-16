import functools
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import auto
from functools import cached_property
from pathlib import Path
from typing import ClassVar, TypedDict

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.table import Column, QTable, Table
from loguru import logger
from strenum import StrEnum
from tollan.pipeline.context_handler import MetadataContextHandlerMixin
from tollan.utils.np import ensure_unit
from tollan.utils.table import TableValidator

__all__ = [
    "RoachToneProps",
    "TlalocEtcDataStore",
]

FrequencyQuantityType = u.Quantity["frequency"]


def _validate_roach_frequency_data(f_tones, f_chans, f_lo):
    if f_tones is not None and f_chans is not None:
        # update f_lo if both are present
        f_los = np.unique(f_chans - f_tones)
        if len(f_los) != 1:
            raise ValueError("inconsistent LO frequencies in data.")
        if f_lo is None:
            f_lo = f_los[0]
        elif f_lo != f_los[0]:
            raise ValueError("inconsistent LO frequency in metadata and data.")
        else:
            pass
        return f_tones, f_chans, f_lo
    if f_tones is not None and f_lo is not None:
        return f_tones, f_tones + f_lo, f_lo
    if f_chans is not None and f_lo is not None:
        return f_chans - f_lo, f_tones, f_lo
    # incomplete data, just return as is
    return f_tones, f_chans, f_lo


@dataclass(kw_only=True)
class RoachTonePropsMetadata:
    """metadata manged by roach tone props."""

    f_lo: None | FrequencyQuantityType = None
    atten_drive: None | float = None
    table_validated: bool = False


class RoachToneProps(MetadataContextHandlerMixin[str, RoachTonePropsMetadata]):
    """A base class to manage tone properties in ROACH."""

    def __init__(self, table, **kwargs):
        self._table = self._validate_table(table, **kwargs)
        self._enable_mask = True

    _table: QTable
    _enable_mask: bool

    _context_handler_context_cls = RoachTonePropsMetadata
    _tbl_validator: ClassVar = TableValidator(table_cls=QTable)

    @staticmethod
    def _validate_f_lo(f_lo):
        if f_lo is None:
            return f_lo
        return f_lo << u.Hz

    @classmethod
    def _validate_table(  # noqa: PLR0913
        cls,
        tbl: Table | QTable,
        f_lo=None,
        f_lo_key="f_lo_center",
        atten_drive=None,
        atten_drive_key="atten_drive",
    ):
        if cls.has_context(tbl) and cls.get_context(tbl).table_validated:
            # skip validation if already did.
            return tbl

        tpt = QTable(meta=tbl.meta)
        ctx = cls.create_context(tpt)
        tblv = cls._tbl_validator
        f_unit = u.Hz
        ensure_f_unit = functools.partial(ensure_unit, unit=f_unit)

        if f_lo is None:
            f_lo = tpt.meta.get(f_lo_key)
        f_lo = ctx.f_lo = ensure_unit(f_lo, f_unit)
        if atten_drive is None:
            atten_drive = tpt.meta.get(atten_drive_key)
        ctx.atten_drive = atten_drive

        # validate frequency info
        f_tones, f_chans = map(
            ensure_f_unit,
            tblv.get_col_data(tbl, ["f_tone", "f_chan"]),
        )
        f_tones, f_chans, f_lo = _validate_roach_frequency_data(
            f_tones=f_tones,
            f_chans=f_chans,
            f_lo=f_lo,
        )
        if f_tones is None or f_chans is None:
            raise ValueError("missing frequency data.")
        tpt["f_tone"] = f_tones
        tpt["f_chan"] = f_chans
        ctx.f_lo = f_lo
        # TODO: put this in the table validator
        for k, unit, dtype in [
            ("amp_tone", None, None),
            ("phase_tone", u.rad, None),
            ("mask_tone", None, bool),
        ]:
            # TODO: handle digitalization of drive atten
            v = tbl[k]
            if dtype is not None:
                v = v.astype(dtype)
            if unit is not None:
                v = v << unit
            tpt[k] = v
        ctx.table_validated = True
        return tpt

    @property
    def meta(self):
        """The table metadata."""
        return self.get_context(self.table)

    @property
    def table(self):
        """The table."""
        return self._table

    @cached_property
    def table_masked(self):
        """The table with tone mask applied."""
        return self._table[self.mask]

    @contextmanager
    def no_mask(self):
        """Context with mask disabled."""
        self._enable_mask = False
        yield self
        self._enable_mask = True

    def _get_data(self, key):
        if self._enable_mask:
            return self.table_masked[key]
        return self.table[key]

    @property
    def f_lo(self):
        """The LO frequency."""
        return self.meta.f_lo

    @property
    def atten_drive(self):
        """The drive attenuation."""
        return self.meta.atten_drive

    @property
    def mask(self):
        """The tone mask."""
        return self.table["mask_tone"].astype(bool)

    @property
    def f_tones(self):
        """The comb frequency."""
        return self._get_data("f_tone")

    @property
    def f_chans(self):
        """The tone frequency."""
        return self._get_data("f_chan")

    @property
    def amps(self):
        """The amplitudes."""
        return self._get_data("amp_tone")

    @property
    def phases(self):
        """The phases."""
        return self._get_data("phase_tone")

    @property
    def n_chans(self):
        """Number of channels."""
        return len(self.table)

    @property
    def n_tones(self):
        """Number of tones."""
        return len(self.table_masked)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_tones}/{self.n_chans})"

    def __getitem__(self, *args):
        return self.__class__(self.table.__getitem__(*args))

    @staticmethod
    def make_random_phases(n_chans, seed=None) -> npt.NDArray:
        """Return random phases."""
        rng1 = np.random.default_rng(seed=seed)
        return rng1.random(n_chans) * 2 * np.pi


class TlalocRoachInterface:
    """Tlaloc roach interface."""

    roaches: ClassVar = list(range(13))
    roach_interface: ClassVar = {roach: f"toltec{roach}" for roach in roaches}
    interface_roach: ClassVar = {v: k for k, v in roach_interface.items()}
    interfaces: ClassVar = list(roach_interface.values())


class TlalocInterface:
    """Tlaloc interface."""

    interfaces: ClassVar = TlalocRoachInterface.interfaces + ["hwpr"]


class TlalocEtcDataItem(StrEnum):
    """Tlaloc etc data item."""

    targ_freqs = auto()
    targ_amps = auto()
    targ_phases = auto()
    targ_mask = auto()
    atten_drive = auto()
    lut = auto()
    f_lo_last = auto()
    attens_last = auto()


class TlalocEtcDataItemInfo:
    """Tlaloc etc data item info."""

    class InfoDict(TypedDict, total=False):
        """Info dict."""

        filename: str
        table_colname: None | str
        table_meta_key: None | str
        tone_props_attr: None | str
        unit: None | u.Unit
        dtype: None | type

    _T = TlalocEtcDataItem

    items: ClassVar[list[TlalocEtcDataItem]] = list(_T)
    item_info: ClassVar[dict[TlalocEtcDataItem, InfoDict]] = {
        _T.targ_freqs: {
            "filename": "targ_freqs.dat",
        },
        _T.targ_amps: {
            "filename": "default_targ_amps.dat",
            "table_colname": "amp_tone",
            "tone_props_attr": "amps",
        },
        _T.targ_phases: {
            "filename": "random_phases.dat",
            "table_colname": "phase_tone",
            "tone_props_attr": "phases",
            "unit": u.rad,
        },
        _T.targ_mask: {
            "filename": "default_targ_masks.dat",
            "table_colname": "mask_tone",
            "tone_props_attr": "mask",
            "dtype": int,
        },
        _T.atten_drive: {
            "filename": "atten_drive.dat",
        },
        _T.lut: {
            "filename": "amps_correction_lut.csv",
        },
        _T.f_lo_last: {
            "filename": "last_centerlo.dat",
        },
        _T.attens_last: {
            "filename": "last_attens.dat",
        },
    }

    item_filename: ClassVar = {
        item: info["filename"] for item, info in item_info.items()
    }


class TlalocKidsModel(StrEnum):
    """The kids model."""

    gain_with_lintrend = auto()


class TlalocKidsModelInfo:
    """The kids model info."""

    class InfoDict(TypedDict, total=False):
        """The kids model info dict."""

        params: dict[str, "TlalocKidsModelInfo.ParamInfoDict"]

    class ParamInfoDict(TypedDict, total=False):
        """The kids model params info dict."""

        table_colname: None | str
        unit: None | u.Unit
        default_value: float

    _T = TlalocKidsModel
    model_info: ClassVar[dict[TlalocKidsModel, InfoDict]] = {
        _T.gain_with_lintrend: {
            "params": {
                "fr": {
                    "table_colname": "fr",
                    "unit": u.Hz,
                },
                "Qr": {
                    "table_colname": "Qr",
                    "unit": None,
                },
                "g0": {
                    "table_colname": "normI",
                    "unit": None,
                },
                "g1": {
                    "table_colname": "normQ",
                    "unit": None,
                },
                "g": {
                    "table_colname": None,
                    "unit": None,
                },
                "phi_g": {
                    "table_colname": None,
                    "unit": None,
                },
                "f0": {
                    "table_colname": "fp",
                    "unit": u.Hz,
                },
                "k0": {
                    "table_colname": "slopeI",
                    "unit": u.Hz**-1,
                },
                "k1": {
                    "table_colname": "slopeQ",
                    "unit": u.Hz**-1,
                },
                "m0": {
                    "table_colname": "interceptI",
                    "unit": None,
                },
                "m1": {
                    "table_colname": "interceptQ",
                    "unit": None,
                },
                # additional columns in kidscpp v0.x
                # TODO: fix this
                "Qc": {
                    "table_colname": "Qc",
                    "unit": None,
                    "default_value": 0.0,
                },
                "A": {
                    "table_colname": "A",
                    "unit": None,
                    "default_value": 0.0,
                },
            },
        },
    }

    table_model_params_info: ClassVar = {
        m: {
            p: p_info
            for p, p_info in m_info["params"].items()
            if p_info.get("table_colname", None) is not None
        }
        for m, m_info in model_info.items()
    }


@dataclass(kw_only=True)
class TlalocEtcTableMetadata:
    """table metadata manged by tlaloc etc data store."""

    item_validated: dict[TlalocEtcDataItem, bool] = field(default_factory=dict)


class TlalocEtcDataStore(MetadataContextHandlerMixin[str, TlalocEtcTableMetadata]):
    """The data files managed by tlaloc."""

    def __init__(self, path="~/tlaloc/etc"):
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"etc directory does not exist {path}")
        self._path = path
        self._file_suffix = None
        self._do_dry_run = False
        self._interface_subpath = {
            interface: self._path.joinpath(self._interface_subpath_name[interface])
            for interface in TlalocInterface.interfaces
        }

    _path: Path
    _file_suffix: None | str
    _do_dry_run: bool
    _interface_subpath: dict[str, Path]

    _interface_subpath_name: ClassVar = {
        interface: interface for interface in TlalocInterface.interfaces
    }

    n_chans_max: ClassVar[int] = 1000
    """The max number channels allowed."""

    _tbl_validator: ClassVar = TableValidator()

    @property
    def path(self):
        """The etc directory."""
        return self._path

    @property
    def roaches(self):
        """The list of roaches."""
        return TlalocRoachInterface.roaches

    def get_roach_path(self, roach):
        """Return the roach interface subpath."""
        interface = TlalocRoachInterface.roach_interface[roach]
        return self._interface_subpath[interface]

    def iter_interface_paths(self, exist_only=True, roach_only=False):
        """Return the interface subpaths."""
        for interface in TlalocInterface.interfaces:
            roach = TlalocRoachInterface.interface_roach.get(interface, None)
            if roach_only and roach is None:
                continue
            p = self._interface_subpath[interface]
            if exist_only and not p.exists():
                continue
            yield roach, interface, p

    def iter_roach_paths(self, exist_only=True):
        """Return the roach interface subpaths."""
        return self.iter_interface_paths(exist_only=exist_only, roach_only=True)

    @contextmanager
    def set_file_suffix(self, suffix):
        """Context to set filesuffix."""
        self._file_suffix = suffix
        yield self
        self._file_suffix = None

    @contextmanager
    def dry_run(self):
        """Context to set dry run."""
        self._do_dry_run = True
        yield self
        self._do_dry_run = False

    @staticmethod
    def _validate_item_arg(item: TlalocEtcDataItem):
        if item not in TlalocEtcDataItemInfo.items:
            raise ValueError(f"invalid item {item}")
        return TlalocEtcDataItem(item)

    def get_item_path(self, roach, item):
        """Return item path."""
        item = self._validate_item_arg(item)
        filename = TlalocEtcDataItemInfo.item_filename[item]
        s = self._file_suffix
        if s:
            s = s.lstrip(".")
            name = f"{filename}.{s}"
        return self.get_roach_path(roach).joinpath(name)

    @classmethod
    def _get_roach_col_from_table(cls, tbl: Table) -> Column:
        return cls._tbl_validator.get_first_col_data(tbl, ["nw", "roach"])

    @classmethod
    def _get_roach_from_meta(cls, meta: dict) -> None | int:
        return cls._tbl_validator.get_first_meta_value(meta, ["nw", "roach"])

    @classmethod
    def _get_table_for_roach(cls, tbl, roach) -> tuple[Table, int]:
        roach_col = cls._get_roach_col_from_table(tbl)
        if roach is None and roach_col is None:
            # try infer from meta
            roach = cls._get_roach_from_meta(tbl.meta)
        if roach_col is not None and roach is not None:
            # roach will be used to filter the table
            tbl = tbl[roach_col == roach]
        else:
            # the table may contain multiple roaches.
            pass
        return tbl, roach

    @classmethod
    def _validate_data_args(cls, data, item, roach):
        item = cls._validate_item_arg(item)
        item_info = TlalocEtcDataItemInfo.item_info[item]
        col = item_info["table_colname"]
        if isinstance(data, RoachToneProps):
            tbl = data.table
        elif isinstance(data, Table):
            tbl = data
        else:
            # make adhoc table with data.
            data = np.array(data)
            if data.ndim != 1:
                raise ValueError("data has to be 1-d when not a table.")
            tbl = Table({col: data})
        if col not in tbl.colnames:
            raise ValueError(f"{col} not found in data.")
        tbl, roach = cls._get_table_for_roach(tbl, roach)
        return tbl, item_info, roach

    @staticmethod
    def _get_data_col_from_table(
        tbl: Table,
        item_info: TlalocEtcDataItemInfo.InfoDict,
    ) -> Column:
        col = item_info["table_colname"]
        col_unit = item_info["unit"]
        col_dtype = item_info["dtype"]
        v = tbl[col]
        if col_dtype is not None:
            v = v.astype(col_dtype)
        if col_unit is not None:
            v = ensure_unit(v, col_unit)
        return v

    @classmethod
    def _make_simple_data_table(cls, data, item, roach=None):
        item = cls._validate_item_arg(item)
        tbl, item_info, roach = cls._validate_data_args(data, item, roach)
        tbl_out = Table(
            {
                item_info["table_colname"]: cls._get_data_col_from_table(
                    tbl,
                    item_info,
                ),
            },
            meta={
                "roach": roach,
            },
        )
        ctx = cls.get_or_create_context(tbl_out)
        ctx.item_validated[item] = True
        return tbl_out

    @classmethod
    def _gen_data_if_not_given(
        cls,
        data,
        n_chans,
        gen_none: Callable[..., npt.NDArray],
        gen_scalar: Callable[..., npt.NDArray],
    ):
        if isinstance(data, RoachToneProps | Table):
            return data
        if data is None:
            if n_chans is None:
                raise ValueError("n_chans required when data not provided.")
            data = gen_none(n_chans)
        else:
            data = np.array(data)
            if data.ndim == 0:
                if n_chans is None:
                    raise ValueError("n_chans required when data is scalar.")
                data = gen_scalar(data.item(), n_chans)
        return data

    @classmethod
    def make_targ_amps_table(cls, data=None, roach=None, n_chans=None):
        """Return targ amps table."""
        data = cls._gen_data_if_not_given(
            data=data,
            n_chans=n_chans or cls.n_chans_max,
            gen_none=lambda n: np.ones((n,), dtype=float),
            gen_scalar=lambda d, n: np.full((n,), d),
        )
        return cls._make_simple_data_table(
            data,
            TlalocEtcDataItem.targ_amps,
            roach=roach,
        )

    @classmethod
    def make_targ_phases_table(cls, data=None, roach=None, n_chans=None, seed=None):
        """Return targ phases table."""
        data = cls._gen_data_if_not_given(
            data=data,
            n_chans=n_chans or cls.n_chans_max,
            gen_none=functools.partial(RoachToneProps.make_random_phases, seed=seed),
            gen_scalar=lambda d, n: np.full((n,), d),
        )
        return cls._make_simple_data_table(
            data,
            TlalocEtcDataItem.targ_phases,
            roach=roach,
        )

    @classmethod
    def make_targ_mask_table(cls, data, roach=None, n_chans=None):
        """Return targ mask table."""
        data = cls._gen_data_if_not_given(
            data=data,
            n_chans=n_chans or cls.n_chans_max,
            gen_none=lambda n: np.ones((n,), dtype=bool),
            gen_scalar=lambda d, n: np.full((n,), bool(d)),
        )
        return cls._make_simple_data_table(
            data,
            TlalocEtcDataItem.targ_mask,
            roach=roach,
        )

    @staticmethod
    def _get_f_lo_from_table(tbl: Table) -> None | FrequencyQuantityType:
        for k in ["Header.Toltec.LoCenterFreq", "f_lo_center"]:
            v = tbl.meta.get(k)
            if v is not None:
                return ensure_unit(v, u.Hz)
        if RoachToneProps.has_context(tbl):
            return RoachToneProps.get_context(tbl).f_lo
        return None

    @classmethod
    def make_targ_freqs_table(
        cls,
        data,
        roach=None,
        f_lo=None,
    ):
        """Return targ freqs table."""
        tbl = data.table if isinstance(data, RoachToneProps) else QTable(data)
        tbl, roach = cls._get_table_for_roach(tbl, roach)
        f_lo = cls._get_f_lo_from_table(tbl)
        f_unit = u.Hz
        ensure_f_unit = functools.partial(ensure_unit, unit=f_unit)

        # validate frequency info
        tblv = cls._tbl_validator
        f_tones, f_centered, f_chans = map(
            ensure_f_unit,
            tblv.get_col_data(tbl, ["f_tones", "f_centered", "f_chans"]),
        )
        f_tones, f_chans, f_lo = _validate_roach_frequency_data(
            f_tones=f_tones or f_centered,
            f_chans=f_chans,
            f_lo=f_lo,
        )
        if f_tones is None or f_chans is None:
            raise ValueError("missing frequency data.")

        tbl_out = Table(
            {
                # TODO: should standarize these keys.
                "f_centered": f_tones.to_value(f_unit),
                "f_out": f_chans.to_value(f_unit),
            },
            meta={
                "f_lo_center": f_lo.to_value(f_unit),
                "roach": roach,
            },
        )

        # copy over extra model params if not present in table.
        # TODO: allow customize the model
        mdl = TlalocKidsModel.gain_with_lintrend
        for p, param_info in TlalocKidsModelInfo.table_model_params_info[mdl].items():
            c = tblv.get_first_col(tbl, [param_info["table_colname"], p])
            tbl_out[c] = 0.0 if c is None else tbl[c]
        # assumes fr is always present in the kids model.
        tbl_out["f_in"] = tbl_out["fr"]
        # update metadata
        for k, defval in [
            (
                "Header.Toltec.LoCenterFreq",
                f_lo.to_value(f_unit) if f_lo is not None else None,
            ),
            ("Header.Toltec.ObsNum", 99),
            ("Header.Toltec.SubObsNum", 0),
            ("Header.Toltec.ScanNum", 0),
            ("Header.Toltec.RoachIndex", roach),
        ]:
            tbl_out.meta[k] = tbl.meta.get(k, defval)
        ctx = cls.get_or_create_context(tbl_out)
        ctx.item_validated[TlalocEtcDataItem.targ_freqs] = True
        return tbl_out

    def _write_table(  # noqa: PLR0913
        self,
        tbl: Table,
        item,
        tbl_fmt,
        data_maker: Callable[..., Table],
        roach=None,
        **kwargs,
    ):
        item = self._validate_item_arg(item)
        logger.debug(f"write table {item=}\n{tbl=}")
        ctx = self.get_or_create_context(tbl)
        if ctx.item_validated.get(item, False):
            tbl = data_maker(tbl, roach=roach, **kwargs)
        roach = self._get_roach_from_meta(tbl.meta)
        if roach is None:
            raise ValueError("cannot find network id to write to.")
        outpath = self.get_item_path(roach, item)
        logger.debug(f"write validated {item} data to {outpath}\n{tbl}")
        if self._do_dry_run:
            print(f"DRY RUN: {outpath=}\n{tbl}")  # noqa: T201
        else:
            tbl.write(outpath, format=tbl_fmt, overwrite=True)
        return outpath

    def write_targ_amps_table(self, tbl, roach=None):
        """Write targ amps table."""
        return self._write_table(
            tbl,
            TlalocEtcDataItem.targ_amps,
            "ascii.no_header",
            self.make_targ_amps_table,
            roach=roach,
        )

    def write_targ_phases_table(self, tbl, roach=None):
        """Write targ phases."""
        return self._write_table(
            tbl,
            TlalocEtcDataItem.targ_phases,
            "ascii.no_header",
            self.make_targ_phases_table,
            roach=roach,
        )

    def write_targ_mask_table(self, tbl, roach=None):
        """Write targ mask."""
        return self._write_table(
            tbl,
            TlalocEtcDataItem.targ_mask,
            "ascii.no_header",
            self.make_targ_mask_table,
            roach=roach,
        )

    def write_targ_freqs_table(self, tbl, roach=None, f_lo=None):
        """Write targ freqs."""
        return self._write_table(
            tbl,
            TlalocEtcDataItem.targ_freqs,
            "ascii.ecsv",
            self.make_targ_freqs_table,
            roach=roach,
            f_lo=f_lo,
        )

    @classmethod
    def _validate_items_arg(cls, items):
        all_items = [
            TlalocEtcDataItem.targ_freqs,
            TlalocEtcDataItem.targ_amps,
            TlalocEtcDataItem.targ_phases,
            TlalocEtcDataItem.targ_mask,
        ]
        if items is None:
            items = all_items
        else:
            items = [cls._validate_item_arg(item) for item in items]
        if any(i not in all_items for i in items):
            raise ValueError(f"invalid item. Choose from {all_items}")
        return items

    def write_tone_props_table(self, tbl, roach=None, f_lo=None, items=None):
        """Write tone prop table."""
        items = self._validate_items_arg(items)
        paths = {}
        for item, writer, kwargs in [
            (
                TlalocEtcDataItem.targ_freqs,
                self.write_targ_freqs_table,
                {"roach": roach, "f_lo_center": f_lo},
            ),
            (
                TlalocEtcDataItem.targ_amps,
                self.write_targ_amps_table,
                {"roach": roach},
            ),
            (
                TlalocEtcDataItem.targ_phases,
                self.write_targ_phases_table,
                {"roach": roach},
            ),
            (
                TlalocEtcDataItem.targ_mask,
                self.write_targ_mask_table,
                {"roach": roach},
            ),
        ]:
            if item in items:
                paths[item] = writer(tbl, *kwargs)
        return paths

    def _set_file_suffix_for_items(self, roach, old_suffix, suffix, items=None):
        items = self._validate_items_arg(items)
        paths: dict[TlalocEtcDataItem, list[Path]] = {}
        with self.set_file_suffix(old_suffix):
            for item in items:
                paths[item] = [self.get_item_path(roach, item)]
        with self.set_file_suffix(suffix):
            for item in items:
                paths[item].append(self.get_item_path(roach, item))
        for old, new in paths.values():
            logger.debug(f"rename file {old.name} -> {new.name} in {old.parent.name}")
            if self._do_dry_run:
                print(  # noqa: T201
                    f"DRY RUN: rename {old.name} -> {new.name} in {old.parent.name}",
                )
            else:
                old.rename(new)
        return paths

    def backup_files(self, roach, suffix="backup", items=None):
        """Back up files."""
        return self._set_file_suffix_for_items(roach, None, suffix, items=items)

    def restore_files(self, roach, backup_suffix, items=None):
        """Restore files."""
        return self._set_file_suffix_for_items(roach, backup_suffix, None, items=items)

    def _read_table(self, roach, item, tbl_fmt, tbl_kw):
        item = self._validate_item_arg(item)
        path = self.get_item_path(roach, item)
        tbl = QTable.read(path, format=tbl_fmt, **tbl_kw)
        tbl.meta["roach"] = roach
        # data read from the etc folder is assumed to be varified
        ctx = self.get_or_create_context(tbl)
        ctx.item_validated[item] = True
        return tbl

    def read_targ_freqs_table(self, roach):
        """Read targ freqs table."""
        return self._read_table(roach, TlalocEtcDataItem.targ_freqs, "ascii.ecsv", {})

    def read_targ_amps_table(self, roach):
        """Read targ amps table."""
        item = TlalocEtcDataItem.targ_amps
        return self._read_table(roach, item, "ascii.no_header", {"names": [item]})

    def read_targ_phases_table(self, roach):
        """Read targ phases."""
        item = TlalocEtcDataItem.targ_phases
        return self._read_table(roach, item, "ascii.no_header", {"names": [item]})

    def read_targ_mask_table(self, roach):
        """Read targ mask."""
        item = TlalocEtcDataItem.targ_mask
        return self._read_table(roach, item, "ascii.no_header", {"names": [item]})

    def read_tone_props(self, roach):
        """Return tone props instance read from the files."""
        tpt = QTable()

        targ_freqs = self.read_targ_freqs_table(roach)

        n_chans = len(targ_freqs)
        tpt["f_tone"] = targ_freqs["f_centered"]
        tpt["f_chan"] = targ_freqs["f_out"]

        for item, reader, make_default in [
            (
                TlalocEtcDataItem.targ_amps,
                self.read_targ_amps_table,
                lambda: 1.0,
            ),
            (
                TlalocEtcDataItem.targ_phases,
                self.read_targ_phases_table,
                lambda: self.make_random_phases(n_chans),
            ),
            (
                TlalocEtcDataItem.targ_mask,
                self.read_targ_mask_table,
                lambda: 1,
            ),
        ]:
            t = reader(roach)
            if len(t) != n_chans:
                logger.warning(f"inconsistent {item}, use default")
                d = make_default()
            else:
                d = t[item]
            item_info = TlalocEtcDataItemInfo.item_info[item]
            tpt[item_info["table_colname"]] = d
        return RoachToneProps(tpt)

    def get_n_chans(self, roach):
        """Return n_chans."""
        return len(self.read_targ_freqs_table(roach))

    def get_last_f_lo(self, roach):
        """Return last f_lo."""
        with self.get_item_path(roach, "last_flo").open("r") as fo:
            f_lo = float(fo.read()) << u.Hz
            logger.debug(f"get last {f_lo=} in {roach=}")
            return f_lo

    @classmethod
    def create(
        cls,
        tlaloc_etc_path,
        roaches=None,
        exist_ok=False,
        roach_only=False,
    ):
        """Create the directory tree."""
        etcdir = Path(tlaloc_etc_path)
        if etcdir.exists() and not exist_ok:
            raise ValueError("etc directory exists, abort.")
        roaches = roaches or TlalocRoachInterface.roaches
        for roach, _, path in cls.iter_interface_paths(
            exist_only=False,
            roach_only=roach_only,
        ):
            if roach is not None and roach not in roaches:
                continue
            path.mkdir(exist_ok=exist_ok, parents=True)
        return cls(path=etcdir)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path})"
