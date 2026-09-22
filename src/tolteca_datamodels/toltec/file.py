"""Register TolTEC acquisition filenames and source classifications."""

from __future__ import annotations

import re

from ..file import (
    DataKindKey,
    data_kind_fields,
    data_kind_map,
    filename_field_types,
    filename_patterns,
)
from ..file import FilepathInfo as FilepathInfoBase
from .types import ToltecDataKind

__all__ = ["FilepathInfo"]


class FilepathInfo(FilepathInfoBase):
    """Filename-derived observation identifiers for a TolTEC ROACH readout."""

    roach: int


filename_field_types.update(roach=int)
data_kind_fields["toltec", "roach"] = ("instru", "instru_component", "suffix", "ext")
data_kind_map.update(
    {
        **{
            DataKindKey(
                instru="toltec",
                instru_component="roach",
                ext=ext,
                suffix=suffix,
            ): kind
            for ext, suffix, kind in (
                ("nc", "vnasweep", ToltecDataKind.VnaSweep),
                ("nc", "targsweep", ToltecDataKind.TargetSweep),
                ("nc", "targetsweep", ToltecDataKind.TargetSweep),
                ("nc", "tune", ToltecDataKind.Tune),
                ("nc", "vnasweep_processed", ToltecDataKind.ReducedVnaSweep),
                ("nc", "targsweep_processed", ToltecDataKind.ReducedTargetSweep),
                ("nc", "tune_processed", ToltecDataKind.ReducedTargetSweep),
                ("nc", "sweep", ToltecDataKind.ReducedSweep),
                ("nc", "timestream_processed", ToltecDataKind.SolvedTimeStream),
                ("nc", "timestream", ToltecDataKind.RawTimeStream),
                ("nc", None, ToltecDataKind.RawTimeStream),
                ("nc", "", ToltecDataKind.RawTimeStream),
                *(
                    ("txt", sweep, ToltecDataKind.KidsModelParamsTable)
                    for sweep in ("vnasweep", "targsweep", "tune")
                ),
                *(
                    ("dat", f"{sweep}_targamps", ToltecDataKind.TargAmpsDat)
                    for sweep in ("vnasweep", "targsweep", "tune")
                ),
                *(
                    ("ecsv", f"{sweep}_{product}", ToltecDataKind.KidsPropTable)
                    for sweep in ("vnasweep", "targsweep", "tune")
                    for product in ("kids_find", "kids_fit", "chan_prop", "adrv")
                ),
            )
        },
        DataKindKey(prefix="hwpr", ext="nc"): ToltecDataKind.Hwpr,
        DataKindKey(prefix="toltec_hk", ext="nc"): ToltecDataKind.HouseKeeping,
        DataKindKey(prefix="wyatt", ext="nc"): ToltecDataKind.Wyatt,
    }
)

filename_patterns.update(
    {
        re.compile(
            r"^(?P<prefix>toltec(?P<roach>\d+))_"
            r"(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
            r"(?P<ut>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ): FilepathInfo,
        re.compile(
            r"^(?P<prefix>hwpr|toltec_hk)_"
            r"(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
            r"(?P<ut>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ): FilepathInfoBase,
        re.compile(
            r"^(?P<prefix>wyatt)_(?P<ut>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ): FilepathInfoBase,
    }
)
