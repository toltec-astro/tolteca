import itertools
import re
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

import numpy as np
import pandas as pd
import pandas.api.typing as pdt
from pydantic import BaseModel, TypeAdapter, computed_field, model_validator
from pydantic.types import StringConstraints
from tollan.utils.fileloc import FileLoc
from tollan.utils.fmt import pformat_yaml
from tollan.utils.general import dict_from_regex_match
from tollan.utils.log import logger
from tollan.utils.table import TableValidator
from typing_extensions import Self, assert_never

from ..lmt.tel import RE_LMT_TEL_FILE
from .types import ToltecDataKind

__all__ = [
    "SourceInfoModel",
    "guess_info_from_source",
    "guess_info_from_sources",
]


_T = ToltecDataKind

_file_interface_suffix_ext_to_toltec_data_kind = {
    # raw kids files
    (r"toltec(\d+)", "vnasweep", ".nc"): _T.VnaSweep,
    (r"toltec(\d+)", "targsweep", ".nc"): _T.TargetSweep,
    (r"toltec(\d+)", "tune", ".nc"): _T.Tune,
    (r"toltec(\d+)", "timestream", ".nc"): _T.RawTimeStream,
    (r"toltec(\d+)", None, ".nc"): _T.RawTimeStream,
    (r"toltec(\d+)", "(vnasweep|targsweep|tune)_processed", ".nc"): _T.ReducedSweep,
    (r"toltec(\d+)", "timestream_processed", ".nc"): _T.SolvedTimeStream,
    # kids reduction
    (r"toltec(\d+)", "(vnasweep|targsweep|tune)", ".txt"): _T.KidsModelParamsTable,
    (r"toltec(\d+)", "(vnasweep|targsweep|tune)_targamps", ".dat"): _T.TargAmpsDat,
    (
        r"toltec(\d+)",
        "(vnasweep|targsweep|tune)_(kids_find|kids_fit|chan_prop|adrv)",
        ".ecsv",
    ): _T.KidsPropTable,
    # obs reduction
    ("apt", None, ".ecsv"): _T.ArrayPropTable,
    ("ppt", None, ".ecsv"): _T.PointingTable,
    ("tel_toltec", None, ".nc"): _T.LmtTel,
    ("tel_toltec2", None, ".nc"): _T.LmtTel2,
    ("hwpr", None, ".nc"): _T.Hwpr,
    ("toltec_hk", None, ".nc"): _T.HouseKeeping,
    ("wyatt", None, ".nc"): _T.Wyatt,
    ("tolteca", None, ".yaml"): _T.ToltecaConfig,
}

_re_filenames = [
    re.compile(s)
    for s in [
        # raw kids data files
        (
            r"^(?P<interface>toltec(?P<roach>\d+)|hwpr|toltec_hk)_"
            r"(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
            r"(?P<file_timestamp>\d{4}_\d{2}_\d{2}(?:_\d{2}_\d{2}_\d{2}))"
            r"(?:_(?P<file_suffix>[^\/.]+))?"
            r"(?P<file_ext>\..+)$"
        ),
        # tel file
        RE_LMT_TEL_FILE,
        # wyatt
        (
            r"^(?P<interface>wyatt)_"
            r"(?P<file_timestamp>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"(?:_(?P<file_suffix>[^\/.]+))?"
            r"(?P<file_ext>\..+)$"
        ),
    ]
]

_re_named_field_type_dispatcher = {
    "roach": int,
    "obsnum": int,
    "subobsnum": int,
    "scannum": int,
}


class SourceInfoModel(BaseModel):
    """Model for information inferred from source."""

    source: FileLoc
    instru: Literal[None, "toltec", "lmt"] = None
    instru_component: Literal[None, "roach", "hwpr", "tcs"] = None
    interface: None | Annotated[str, StringConstraints(to_lower=True)] = None
    roach: None | int = None
    obsnum: None | int = None
    subobsnum: None | int = None
    scannum: None | int = None
    file_timestamp: None | datetime = None
    file_suffix: None | str = None
    file_ext: None | Annotated[str, StringConstraints(to_lower=True)] = None
    data_kind: None | _T = None

    @computed_field
    @property
    def filepath(self) -> Path:
        """Filepath."""
        return self.source.path

    @computed_field
    @property
    def uid_obs(self) -> Path:
        """Unique id for obs idenfied by an obsnum."""
        return str(self.obsnum)

    @computed_field
    @property
    def uid_raw_obs(self) -> Path:
        """Unique id of idenfied by the (obsnum, subobsnum, scannum) tripplet."""
        return f"{self.obsnum}-{self.subobsnum}-{self.scannum}"

    @computed_field
    @property
    def uid_raw_obs_file(self) -> Path:
        """Unique id of idenfied by (obsnum, subobsnum, scannum, roach)."""
        file_id = self.interface if self.roach is None else self.roach
        return f"{self.obsnum}-{self.subobsnum}-{self.scannum}-{file_id}"

    @model_validator(mode="wrap")
    @classmethod
    def _validate_arg(cls, arg, handler):
        if isinstance(arg, cls):
            return arg
        if isinstance(arg, dict) and "source" in arg:
            # from dumped data or meta
            values = arg
        else:
            values = cls._validate_source(arg)
        return handler(values)

    @classmethod
    def _validate_source(cls, source):
        source = FileLoc(source)
        info = {"source": source}
        for re_filename in _re_filenames:
            d = dict_from_regex_match(
                re_filename,
                source.path_orig.name,
                type_dispatcher=_re_named_field_type_dispatcher,
            )
            if d is None:
                continue
            info.update(d)
            break
        return cls._validate_matched(info)

    @staticmethod
    def _validate_matched(info):
        interface = info.get("interface", None)
        if interface is None:
            raise ValueError("no interface info")
        roach = info.get("roach", None)
        if interface.startswith("toltec") or interface in ["wyatt"]:
            instru = "toltec"
        elif interface in ["tel_toltec", "tel_toltec2"]:
            instru = "lmt"
        else:
            instru = None
        if roach is not None:
            instru_component = "roach"
        else:
            instru_component = {
                "hwpr": "hwpr",
                "toltec_hk": "hk",
                "wyatt": "wyatt",
                "tel_toltec": "tcs",
                "tel_toltec2": "tcs",
            }.get(interface)
        info.update({"instru": instru, "instru_component": instru_component})
        # handle timestamp
        # there are two formats YYYY_mm_dd_HH_MM_SS and YYYY-mm-dd
        v = info["file_timestamp"]
        n_sep_long = 5
        n_sep_short = 2
        if v.count("_") == n_sep_long:
            fmt = "%Y_%m_%d_%H_%M_%S"
        elif v.count("-") == n_sep_short:
            fmt = "%Y-%m-%d"
        else:
            raise ValueError("invalid file timestamp string.")
        info["file_timestamp"] = datetime.strptime(v, fmt).replace(tzinfo=timezone.utc)
        return info

    @model_validator(mode="after")
    def _validate_data_kind(self) -> Self:
        interface = self.interface
        file_suffix = self.file_suffix
        file_ext = self.file_ext

        def _check_match(v, re_v):
            if re_v is None:
                return True
            if v is None:
                return False
            return re.fullmatch(re_v, v)

        for (
            re_interface,
            re_file_suffix,
            re_file_ext,
        ), dk in _file_interface_suffix_ext_to_toltec_data_kind.items():
            if all(
                [
                    _check_match(interface, re_interface),
                    _check_match(file_suffix, re_file_suffix),
                    _check_match(file_ext, re_file_ext),
                ],
            ):
                self.data_kind = dk
                break
        return self


def guess_info_from_source(source):
    """Return file info gussed from parsing source.

    Parameters
    ----------
    source : str, `pathlib.Path`, `FileLoc`
    """
    info = SourceInfoModel.model_validate(source)
    logger.debug(f"guess info from {source}:\n{pformat_yaml(info.model_dump())}")
    return info


def guess_info_from_sources(sources) -> "SourceInfoDataFrame":
    """Return a table of guessed info for a list of sources."""
    info_records = [guess_info_from_source(source).model_dump() for source in sources]
    return pd.DataFrame.from_records(info_records)


@pd.api.extensions.register_dataframe_accessor("toltec_file")
class ToltecFileAccessor:
    _table_validator: ClassVar = TableValidator()
    _obj: pd.DataFrame

    def __init__(self, pandas_obj):
        self._obj = self._validate_obj(pandas_obj)

    @classmethod
    def _validate_obj(cls, obj):
        tblv = cls._table_validator
        if len(obj) == 0:
            raise ValueError("table is empty.")
        if not tblv.has_all_cols(obj, list(SourceInfoModel.model_fields.keys())):
            raise AttributeError("incomplete source info.")
        return obj

    def _make_groups(self, by) -> pdt.DataFrameGroupBy:
        return self._obj.groupby(by, sort=False, group_keys=True, as_index=False)

    def make_obs_groups(self):
        return self._make_groups(["uid_obs"])

    def make_raw_obs_groups(self):
        return self._make_groups(["uid_raw_obs"])

    def _get_obj(self, query=None) -> "SourceInfoDataFrame":
        obj = self._obj
        if query is None:
            return obj
        return obj.query(query)

    _sort_keys: ClassVar = ["obsnum", "subobsnum", "scannum", "file_timestamp"]

    def get_info_latest(self, query=None):
        obj = self._get_obj(query=query)
        obj = obj.sort_values(
            by=self._sort_keys,
            ascending=False,
        )
        return SourceInfoModel.model_validate(obj.iloc[0].to_dict())

    def get_raw_obs_latest(self, query=None):
        info = self.get_info_latest(query=query)
        return self._get_obj(f"uid_raw_obs == '{info.uid_raw_obs}'")

    def get_obs_latest(self, query=None):
        info = self.get_info_latest(query=query)
        return self._get_obj(f"uid_obs == '{info.uid_obs}'")

    _source_info_list_validator = TypeAdapter(list[SourceInfoModel])

    def to_info_list(self, query=None) -> list[SourceInfoModel]:
        obj = self._get_obj(query=query)
        return self._source_info_list_validator.validate_python(
            obj.to_dict(orient="records"),
        )

    _pformat_sort_keys: ClassVar = ["obsnum", "subobsnum", "scannum", "roach"]

    def pformat(
        self,
        type: Literal["full", "long", "short"] = "long",
        sort=True,
        include_cols=None,
    ):
        if type == "full":
            colnames = self._obj.columns
        elif type == "long":
            c_excl = ["source", "file_timestamp"]
            colnames = [c for c in self._obj.columns if c not in c_excl]
        elif type == "short":
            colnames = ["uid_raw_obs", "filepath"]
        else:
            assert_never()
        if include_cols is not None:
            for c in include_cols:
                if c not in colnames:
                    colnames.append(c)
        obj = self._obj.sort_values(by=self._pformat_sort_keys) if sort else self._obj
        return obj.to_string(columns=colnames)

    _io_obj_key = "_io_obj"
    _data_obj_key = "_data_obj"

    @property
    def io_objs(self):
        """The IO objects."""
        io_obj_key = self._io_obj_key
        if io_obj_key not in self._obj.columns:
            return None
        return self._obj[io_obj_key]

    @property
    def data_objs(self):
        """The data objects."""
        data_obj_key = self._data_obj_key
        if data_obj_key not in self._obj.columns:
            return None
        return self._obj[data_obj_key]

    def iter_objs(self):
        """Return a iterator of (entry, io_obj, data_obj)."""
        io_objs = self.io_objs
        if io_objs is None:
            io_objs = []
        data_objs = self.data_objs
        if data_objs is None:
            data_objs = []
        entries = self._obj.itertuples()
        return itertools.zip_longest(entries, io_objs, data_objs, fillvalue=None)

    def _update_from_item_meta(self, items):
        data = []

        def _filter_meta(meta):
            return {k: v for k, v in meta.items() if np.isscalar(v)}

        for item in items:
            d = {} if pd.isna(item) else _filter_meta(item.meta)
            data.append(d)

        data = pd.DataFrame.from_records(data)
        for c in data.columns:
            mask = (~pd.isna(data[c])).to_numpy()
            self._obj.loc[mask, c] = data.loc[mask, c].to_numpy()
        return self._obj

    @contextmanager
    def open(self, raise_on_error=False):
        """Invoke the default file IO to load meta data."""
        obj = self._obj
        io_obj_key = self._io_obj_key
        io_objs = []
        es = ExitStack().__enter__()

        from .ncfile import NcFileIO

        file_ext_io_cls = {
            ".nc": NcFileIO,
        }

        for entry in obj.itertuples():
            # TODO: use the unified IO registry for identifying the io class.
            io_cls = file_ext_io_cls.get(entry.file_ext, None)
            if io_cls is None:
                io_obj = None
            else:
                try:
                    io_obj = io_cls(entry.filepath)
                except Exception as e:
                    if raise_on_error:
                        raise
                    logger.debug(
                        f"unable to read file {entry.filepath} with {io_cls=}: {e}",
                    )
                    io_obj = None
                else:
                    es.enter_context(io_obj.open())
            io_objs.append(io_obj)
        obj[io_obj_key] = io_objs
        self._update_from_item_meta(io_objs)
        yield obj
        es.close()
        del obj[io_obj_key]

    def read(
        self,
        cached=True,
        raise_on_error=False,
    ):
        """Invoke the default file readers to load data objects in to the data frame."""
        data_objs = self.data_objs
        if cached and data_objs is not None:
            return self._obj
        obj = self._obj
        data_obj_key = self._data_obj_key
        data_objs = []
        with self.open(raise_on_error=raise_on_error):
            for io_obj in self.io_objs:
                if pd.isna(io_obj):
                    data_obj = None
                else:
                    try:
                        data_obj = io_obj.read()
                    except Exception as e:
                        if raise_on_error:
                            raise
                        logger.debug(f"failed to read data from {io_obj=}: {e}")
                        data_obj = None
                data_objs.append(data_obj)
        obj[data_obj_key] = data_objs
        self._update_from_item_meta(data_objs)
        return obj


if TYPE_CHECKING:

    class SourceInfoDataFrame(pd.DataFrame):
        toltec_file: ToltecFileAccessor
