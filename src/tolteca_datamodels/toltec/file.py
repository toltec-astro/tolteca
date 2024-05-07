import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

import pandas as pd
import pandas.api.typing as pdt
from pydantic import BaseModel, computed_field, model_validator
from pydantic.types import StringConstraints
from tollan.utils.fileloc import FileLoc
from tollan.utils.fmt import pformat_yaml
from tollan.utils.general import dict_from_regex_match
from tollan.utils.log import logger
from tollan.utils.table import TableValidator
from typing_extensions import Self

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
    (r"toltec(\d+)", "(timestream)?", ".nc"): _T.RawTimeStream,
    (r"toltec(\d+)", "(vnasweep|targsweep|tune)_processed", ".nc"): _T.ReducedSweep,
    (r"toltec(\d+)", "timestream_processed", ".nc"): _T.SolvedTimeStream,
    # kids reduction
    (r"toltec(\d+)", "(vnasweep|targsweep|tune)", ".txt"): _T.KidsModelParamsTable,
    (r"toltec(\d+)", "(vnasweep|targsweep|tune)_targamps", ".dat"): _T.TargAmpsDat,
    (
        r"toltec(\d+)",
        "(vnasweep|targsweep|tune)_(kids_find|kids_fit|tone_prop|chan_prop)",
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
        # tel files
        (
            r"^(?P<interface>tel_toltec|tel_toltec2|wyatt)_"
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

    @model_validator(mode="before")
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


def guess_info_from_sources(sources) -> "DataFrame":
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
        if not tblv.has_all_cols(obj, list(SourceInfoModel.model_fields.keys())):
            raise AttributeError("incomplete source info.")
        return obj

    def _make_groups(self, by) -> pdt.DataFrameGroupBy:
        return self._obj.groupby(by, sort=False, group_keys=True, as_index=False)

    def make_obs_groups(self):
        return self._make_groups(["uid_obs"])

    def make_raw_obs_groups(self):
        return self._make_groups(["uid_raw_obs"])

    def get_latest(self, query=None):
        obj = self._obj
        if query is not None:
            obj = obj.query(query)
        obj = obj.sort_values(
            by=["obsnum", "subobsnum", "scannum", "file_timestamp"],
            ascending=False,
        )
        return SourceInfoModel.model_construct(obj.iloc[0].to_dict())


if TYPE_CHECKING:

    class DataFrame(pd.DataFrame):
        toltec_file: ToltecFileAccessor
