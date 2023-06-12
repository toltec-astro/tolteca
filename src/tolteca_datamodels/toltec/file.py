import functools
import operator
import re
from datetime import datetime, timezone

from astropy.time import Time
from tollan.utils.fileloc import FileLoc
from tollan.utils.general import dict_from_regex_match

from .types import ToltecDataKind

__all__ = ["guess_meta_from_source"]


_file_suffix_ext_to_toltec_data_kind = {
    ("vnasweep", "nc"): ToltecDataKind.VnaSweep,
    ("targsweep", "nc"): ToltecDataKind.TargetSweep,
    ("tune", "nc"): ToltecDataKind.Tune,
    ("timestream", "nc"): ToltecDataKind.RawTimeStream,
    ("vnasweep_processed", "nc"): ToltecDataKind.ReducedSweep,
    ("targsweep_processed", "nc"): ToltecDataKind.ReducedSweep,
    ("tune_processed", "nc"): ToltecDataKind.ReducedSweep,
    ("timestream_processed", "nc"): ToltecDataKind.SolvedTimeStream,
    ("vnasweep", "txt"): ToltecDataKind.KidsModelParamsTable,
    ("targsweep", "txt"): ToltecDataKind.KidsModelParamsTable,
    ("tune", "txt"): ToltecDataKind.KidsModelParamsTable,
    ("targfreqs", "dat"): ToltecDataKind.TargFreqsDat,
    ("targamps", "dat"): ToltecDataKind.TargAmpsDat,
    ("chanflag", "ecsv"): ToltecDataKind.ChanPropTable,
    ("kidslist", "ecsv"): ToltecDataKind.KidsPropTable,
    ("kidsprop", "ecsv"): ToltecDataKind.KidsPropTable,
    ("toneprop", "ecsv"): ToltecDataKind.TonePropTable,
    ("chanprop", "ecsv"): ToltecDataKind.ChanPropTable,
}

_file_interface_ext_to_toltec_data_kind = {
    ("apt", "ecsv"): ToltecDataKind.ArrayPropTable,
    ("ppt", "ecsv"): ToltecDataKind.PointingTable,
    ("tel_toltec", "nc"): ToltecDataKind.LmtTel,
    ("tel_toltec2", "nc"): ToltecDataKind.LmtTel2,
    ("hwpr", "nc"): ToltecDataKind.Hwpr,
    ("toltec_hk", "nc"): ToltecDataKind.HouseKeeping,
    ("wyatt", "nc"): ToltecDataKind.Wyatt,
    ("tolteca", "yaml"): ToltecDataKind.ToltecaConfig,
}

_filename_parser_defs = [
    {
        "regex": re.compile(
            r"^(?P<interface>toltec(?P<roachid>\d+))_(?P<obsnum>\d+)_"
            r"(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
            r"(?P<ut>\d{4}_\d{2}_\d{2}(?:_\d{2}_\d{2}_\d{2}))"
            r"(?:_(?P<filesuffix>[^\/.]+))?"
            r"\.(?P<fileext>.+)$",
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
            r"(?P<ut>\d{4}_\d{2}_\d{2}(?:_\d{2}_\d{2}_\d{2}))"
            r"(?:_(?P<filesuffix>[^\/.]+))?"
            r"\.(?P<fileext>.+)$",
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
            r"(?P<ut>\d{4}_\d{2}_\d{2}(?:_\d{2}_\d{2}_\d{2}))"
            r"(?:_(?P<filesuffix>[^\/.]+))?"
            r"\.(?P<fileext>.+)$",
        ),
        "add_meta": {
            "instru": "toltec",
            "instru_component": "hk",
        },
    },
    {
        "regex": re.compile(
            r"^(?P<interface>tel_toltec|tel_toltec2)"
            r"_(?P<ut>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"\.(?P<fileext>.+)$",
        ),
        "add_meta": {
            "instru": "lmt",
            "instru_component": "tcs",
        },
    },
    {
        "regex": re.compile(
            r"^(?P<interface>wyatt)"
            r"_(?P<ut>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"\.(?P<fileext>.+)$",
        ),
        "add_meta": {
            "instru": "toltec",
            "instru_component": "wyatt",
        },
    },
]


def _parse_ut_str(v):
    # there are two formats YYYY_mm_dd_HH_MM_SS and YYYY-mm-dd
    n_sep_long = 5
    n_sep_short = 2
    if v.count("_") == n_sep_long:
        fmt = "%Y_%m_%d_%H_%M_%S"
    elif v.count("-") == n_sep_short:
        fmt = "%Y-%m-%d"
    else:
        raise ValueError("invalid UT time string.")
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
    "ut": _parse_ut_str,
    "interface": _normalize_interface,
    "roachid": int,
    "obsnum": int,
    "subobsnum": int,
    "scannum": int,
    "fileext": str.lower,
}


def _guess_data_kind_from_meta(meta):
    interface = meta.get("interface", None)
    fileext = meta.get("fileext", None)
    dk_set = set()
    if interface is not None and fileext is not None:
        dk = _file_interface_ext_to_toltec_data_kind.get((interface, fileext), None)
        if dk is not None:
            dk_set.add(dk)
    filesuffix = meta.get("filesuffix", None)
    if filesuffix is not None and fileext is not None:
        dk = _file_suffix_ext_to_toltec_data_kind.get((filesuffix, fileext), None)
        if dk is not None:
            dk_set.add(dk)
    if len(dk_set) == 0:
        return None
    return functools.reduce(operator.ior, dk_set)


def _guess_data_store_info(meta):
    """Return the infered data storage layout and related info."""
    return meta


def guess_meta_from_source(source):
    """Return guessed metadata parsed from data source.

    Parameters
    ----------
    source : str, `pathlib.Path`, `FileLoc`
    """
    fileloc = FileLoc(source)
    filepath = fileloc.path
    meta = {
        "source": source,
        "fileloc": fileloc,
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
    return meta
