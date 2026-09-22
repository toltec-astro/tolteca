"""Telescope acquisition filename declarations."""

import re

from ..file import FileKindKey, FileSpec
from ..toltec.types import ToltecDataKind
from .types import LmtInfo

__all__ = ["file_specs"]

file_specs = (
    FileSpec(
        pattern=re.compile(
            r"^(?P<prefix>tel_toltec|tel_toltec2)_"
            r"(?P<ut>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ),
        prefix_interface={"tel_toltec": "tel", "tel_toltec2": "tel"},
        identity_map=LmtInfo.interface_instru,
        kind_map={
            FileKindKey(prefix="tel_toltec", ext="nc"): ToltecDataKind.LmtTel,
            FileKindKey(prefix="tel_toltec2", ext="nc"): ToltecDataKind.LmtTel2,
        },
    ),
)
