"""Register telescope acquisition filenames and source classifications."""

import re

from ..file import (
    DataKindKey,
    FilepathInfo,
    data_kind_map,
    filename_patterns,
    prefix_interface,
)
from ..toltec.types import ToltecDataKind

filename_patterns.update(
    {
        re.compile(
            r"^(?P<prefix>tel_toltec|tel_toltec2)_"
            r"(?P<ut>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ): FilepathInfo,
    }
)
prefix_interface.update(tel_toltec="tel", tel_toltec2="tel")
data_kind_map.update(
    {
        DataKindKey(prefix="tel_toltec", ext="nc"): ToltecDataKind.LmtTel,
        DataKindKey(prefix="tel_toltec2", ext="nc"): ToltecDataKind.LmtTel2,
    }
)
