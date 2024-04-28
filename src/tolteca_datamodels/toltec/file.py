import functools
import operator
import re
from datetime import datetime, timezone

from astropy.time import Time
from tollan.utils.fileloc import FileLoc
from tollan.utils.fmt import pformat_yaml
from tollan.utils.general import dict_from_regex_match
from tollan.utils.log import logger

from .types import ToltecDataKind

__all__ = ["guess_meta_from_source"]


_file_suffix_ext_to_toltec_data_kind = {
    (r"toltec(\d+)?", "vnasweep", "nc"): ToltecDataKind.VnaSweep,
    (r"toltec(\d+)?", "targsweep", "nc"): ToltecDataKind.TargetSweep,
    (r"toltec(\d+)?", "tune", "nc"): ToltecDataKind.Tune,
    (r"toltec(\d+)?", "timestream", "nc"): ToltecDataKind.RawTimeStream,
    (r"toltec(\d+)?", "^$", "nc"): ToltecDataKind.RawTimeStream,
    (
        r"toltec(\d+)?",
        "(vnasweep|targsweep|tune)_processed",
        "nc",
    ): ToltecDataKind.ReducedSweep,
    (r"toltec(\d+)?", "timestream_processed", "nc"): ToltecDataKind.SolvedTimeStream,
    (r"toltec(\d+)?", "vnasweep", "txt"): ToltecDataKind.KidsModelParamsTable,
    (r"toltec(\d+)?", "targsweep", "txt"): ToltecDataKind.KidsModelParamsTable,
    (r"toltec(\d+)?", "tune", "txt"): ToltecDataKind.KidsModelParamsTable,
    # (r"toltec(\d+)?", "targfreqs", "dat"): ToltecDataKind.TargFreqsDat,
    (r"toltec(\d+)?", "targamps", "dat"): ToltecDataKind.TargAmpsDat,
    # (r"toltec(\d+)?", "chanflag", "ecsv"): ToltecDataKind.ChanPropTable,
    (
        r"toltec(\d+)?",
        "(vnasweep|targsweep|tune)_kidslist",
        "ecsv",
    ): ToltecDataKind.KidsPropTable,
    (
        r"toltec(\d+)?",
        "(vnasweep|targsweep|tune)_kidsprop",
        "ecsv",
    ): ToltecDataKind.KidsPropTable,
    (
        r"toltec(\d+)?",
        "(vnasweep|targsweep|tune)_toneprop",
        "ecsv",
    ): ToltecDataKind.TonePropTable,
    (
        r"toltec(\d+)?",
        "(vnasweep|targsweep|tune)_chanprop",
        "ecsv",
    ): ToltecDataKind.ChanPropTable,
    # v1 compat
    (
        r"toltec(\d+)?",
        "(vnasweep|targsweep|tune)_tonelist",
        "ecsv",
    ): ToltecDataKind.KidsPropTable,
    (
        r"toltec(\d+)?",
        "(vnasweep|targsweep|tune)_targfreqs",
        "ecsv",
    ): ToltecDataKind.TonePropTable,
    (
        r"toltec(\d+)?",
        "(vnasweep|targsweep|tune)_tonecheck",
        "ecsv",
    ): ToltecDataKind.ChanPropTable,
}

_file_interface_ext_to_toltec_data_kind = {
    ("apt", "ecsv"): ToltecDataKind.ArrayPropTable,
    ("ppt", "ecsv"): ToltecDataKind.PointingTable,
    ("lmt", "nc"): ToltecDataKind.LmtTel,
    ("lmt_tel2", "nc"): ToltecDataKind.LmtTel2,
    ("hwpr", "nc"): ToltecDataKind.Hwpr,
    ("toltec_hk", "nc"): ToltecDataKind.HouseKeeping,
    ("wyatt", "nc"): ToltecDataKind.Wyatt,
    ("tolteca", "yaml"): ToltecDataKind.ToltecaConfig,
}

_filename_parser_defs = [
    {
        "regex": re.compile(
            r"^(?P<interface>toltec(?P<roach>\d+))_(?P<obsnum>\d+)_"
            r"(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
            r"(?P<file_timestamp>\d{4}_\d{2}_\d{2}(?:_\d{2}_\d{2}_\d{2}))"
            r"(?:_(?P<file_suffix>[^\/.]+))?"
            r"\.(?P<file_ext>.+)$",
        ),
        "add_meta": {
            "instru": "toltec",
            "instru_component": "roach",
        },
    },
    {
        "regex": re.compile(
            r"^(?P<interface>hwpr)_(?P<obsnum>\d+)_"
            r"(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
            r"(?P<file_timestamp>\d{4}_\d{2}_\d{2}(?:_\d{2}_\d{2}_\d{2}))"
            r"(?:_(?P<file_suffix>[^\/.]+))?"
            r"\.(?P<file_ext>.+)$",
        ),
        "add_meta": {
            "instru": "toltec",
            "instru_component": "hwpr",
        },
    },
    {
        "regex": re.compile(
            r"^(?P<interface>toltec_hk)_(?P<obsnum>\d+)_"
            r"(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
            r"(?P<file_timestamp>\d{4}_\d{2}_\d{2}(?:_\d{2}_\d{2}_\d{2}))"
            r"(?:_(?P<file_suffix>[^\/.]+))?"
            r"\.(?P<file_ext>.+)$",
        ),
        "add_meta": {
            "instru": "toltec",
            "instru_component": "hk",
        },
    },
    {
        "regex": re.compile(
            r"^(?P<interface>tel_toltec|tel_toltec2)"
            r"_(?P<file_timestamp>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"\.(?P<file_ext>.+)$",
        ),
        "add_meta": {
            "instru": "lmt",
            "instru_component": "tcs",
        },
    },
    {
        "regex": re.compile(
            r"^(?P<interface>wyatt)"
            r"_(?P<file_timestamp>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"\.(?P<file_ext>.+)$",
        ),
        "add_meta": {
            "instru": "toltec",
            "instru_component": "wyatt",
        },
    },
]


def _parse_file_timestamp_str(v):
    # there are two formats YYYY_mm_dd_HH_MM_SS and YYYY-mm-dd
    n_sep_long = 5
    n_sep_short = 2
    if v.count("_") == n_sep_long:
        fmt = "%Y_%m_%d_%H_%M_%S"
    elif v.count("-") == n_sep_short:
        fmt = "%Y-%m-%d"
    else:
        raise ValueError("invalid file timestamp string.")
    result = Time(datetime.strptime(v, fmt).astimezone(timezone.utc), scale="utc")
    result.format = "isot"
    return result


def _normalize_interface(v):
    if v in ["tel", "tel_toltec"]:
        return "lmt"
    if v == "tel_toltec2":
        return "lmt_tel2"
    return v


_filename_parser_type_dispatchers = {
    "file_timestamp": _parse_file_timestamp_str,
    "interface": _normalize_interface,
    "roach": int,
    "obsnum": int,
    "subobsnum": int,
    "scannum": int,
    "file_ext": str.lower,
}


def _guess_data_kind_from_meta(meta):
    interface = meta.get("interface", None)
    file_ext = meta.get("file_ext", None)
    dk_set = set()
    if interface is not None and file_ext is not None:
        for (
            re_interface,
            re_file_ext,
        ), dk in _file_interface_ext_to_toltec_data_kind.items():
            if re.fullmatch(re_interface, interface) and re.fullmatch(
                re_file_ext,
                file_ext,
            ):
                dk_set.add(dk)
                break
    file_suffix = meta.get("file_suffix", None) or ""
    if interface is not None and file_ext is not None:
        for (
            re_interface,
            re_file_suffix,
            re_file_ext,
        ), dk in _file_suffix_ext_to_toltec_data_kind.items():
            if (
                re.fullmatch(re_interface, interface)
                and re.fullmatch(re_file_suffix, file_suffix)
                and re.fullmatch(
                    re_file_ext,
                    file_ext,
                )
            ):
                dk_set.add(dk)
                break
    if len(dk_set) == 0:
        return None
    return functools.reduce(operator.ior, dk_set)


def _guess_data_store_info(meta):
    """Return the infered data storage layout and related info."""
    # TODO : implement this, with respect to taco and data_lmt
    return meta


def guess_meta_from_source(source):
    """Return guessed metadata parsed from data source.

    Parameters
    ----------
    source : str, `pathlib.Path`, `FileLoc`
    """
    file_loc = FileLoc(source)
    filepath = file_loc.path
    meta = {
        "source": source,
        "file_loc": file_loc,
    }
    for parser_def in _filename_parser_defs:
        d = dict_from_regex_match(
            parser_def["regex"],
            filepath.name,
            type_dispatcher=_filename_parser_type_dispatchers,
        )
        if d is None:
            continue
        if "add_meta" in parser_def:
            d.update(parser_def["add_meta"])
        meta.update(d)
        break
    meta.update(_guess_data_store_info(meta))
    data_kind = _guess_data_kind_from_meta(meta)
    if data_kind is not None:
        meta["data_kind"] = data_kind
    logger.debug(f"guess meta data from {source}:\n{pformat_yaml(meta)}")
    return meta
