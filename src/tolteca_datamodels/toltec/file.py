"""TolTEC acquisition filename declarations."""

from __future__ import annotations

import re

from ..file import FileKindKey, FileSpec
from ..file import FilepathInfo as FilepathInfoBase
from .types import ToltecDataKind, ToltecInfo

__all__ = ["FilepathInfo", "file_specs"]


class FilepathInfo(FilepathInfoBase):
    """Filename-derived observation identifiers for a TolTEC ROACH readout."""

    roach: int


kind_map = {
    **{
        FileKindKey(
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
    FileKindKey(prefix="hwpr", ext="nc"): ToltecDataKind.Hwpr,
    FileKindKey(prefix="toltec_hk", ext="nc"): ToltecDataKind.HouseKeeping,
    FileKindKey(prefix="wyatt", ext="nc"): ToltecDataKind.Wyatt,
}


file_specs = (
    FileSpec(
        pattern=re.compile(
            r"^(?P<prefix>toltec(?P<roach>\d+))_"
            r"(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
            r"(?P<ut>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ),
        info_cls=FilepathInfo,
        type_dispatcher={**FilepathInfo._type_dispatcher, "roach": int},
        # Captured indices preserve recognition of known ROACHs when the
        # prefix uses an alternate numeric spelling, without changing interface.
        identity_field="roach",
        identity_map={
            roach: ToltecInfo.interface_instru[interface]
            for roach, interface in ToltecInfo.roach_interface.items()
        },
        kind_fields=("instru", "instru_component", "suffix", "ext"),
        kind_map=kind_map,
    ),
    FileSpec(
        pattern=re.compile(
            r"^(?P<prefix>hwpr|toltec_hk)_"
            r"(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
            r"(?P<ut>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ),
        identity_map=ToltecInfo.interface_instru,
        kind_map=kind_map,
    ),
    FileSpec(
        pattern=re.compile(
            r"^(?P<prefix>wyatt)_(?P<ut>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ),
        identity_map=ToltecInfo.interface_instru,
        kind_map=kind_map,
    ),
)
