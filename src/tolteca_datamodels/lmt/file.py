"""LMT telescope file tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pandas as pd
import xarray as xr

from ..file import FileAccessor
from ..lmt_base._filepath import make_filepath_info_column_mapper
from ..lmt_base.funcs import make_uid_columns
from ..lmt_base.obs_spec import ObsSpec
from ..model_table import (
    ModelColumn,
    ModelColumnMapper,
    QuantityColumnRule,
    ScalarColumnRule,
    SkyCoordColumnRule,
    TimeColumnRule,
)
from ._filepath import LmtFilepathInfo
from .metadata import LmtTelMetadata
from .obs_spec import obs_spec_context
from .tel import identify_lmt_tel

if TYPE_CHECKING:
    from ..file import _SourceInfoDiagnostics

__all__ = ["LmtFileAccessor"]

_filepath_info_column_mapper = make_filepath_info_column_mapper()


@pd.api.extensions.register_dataframe_accessor("lmt_file")
class LmtFileAccessor(FileAccessor):
    """Build, query and open LMT telescope file tables.

    Parameters
    ----------
    data : pandas.DataFrame
        File table, as returned by `from_sources`.

    Notes
    -----
    Header reads support local netCDF files and open Dataset or DataTree
    objects. Remote resources must be staged locally. Non-telescope data
    and unsupported formats retain their rows without header enrichment.
    Metadata columns use MJD for times, seconds for durations and degrees
    for coordinates, indicated by column suffixes. Header reads do not load
    pointing streams. Opening requires recognized telescope data, including
    when supplying an already opened object. Observation selectors use the
    telescope interface labels.

    Examples
    --------
    >>> table = LmtFileAccessor.from_sources(paths, mode="read")
    >>> pointing = table.lmt_file.query(
    ...     'obs_goal == "pointing" and obs_pgm == "lissajous"'
    ... )
    >>> selected = pointing.lmt_file.query(
    ...     'target_frame == "icrs" and 70 <= target_ra_deg < 80 '
    ...     'and 60 <= target_dec_deg < 70 and t_int_s > 60 and t_elapsed_s > 65'
    ... )
    >>> with selected.lmt_file.open() as opened:
    ...     az = [ds.lmt_tel.az.isel(time=slice(0, 100)).load() for ds in opened.io_obj]
    """

    _filepath_info_types: ClassVar = (LmtFilepathInfo,)
    _io_obj_types: ClassVar = (xr.Dataset, xr.DataTree)
    _source_opener: ClassVar = xr.open_dataset
    _source_readers: ClassVar = (identify_lmt_tel,)
    _obs_spec_cls: ClassVar = ObsSpec
    _obs_spec_context: ClassVar = obs_spec_context
    _metadata_file_extensions: ClassVar = (".nc",)
    _filepath_info_columns: ClassVar[dict[str, ModelColumn]] = {
        name: column
        for model in _filepath_info_types
        for name, column in _filepath_info_column_mapper(model).items()
    }
    _metadata_columns: ClassVar[dict[str, ModelColumn]] = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            QuantityColumnRule(physical_type="time", unit="s"),
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
            ScalarColumnRule(),
        ],
        include_computed=True,
        default=None,
    )(LmtTelMetadata)

    def _guess_columns(self, *, diagnostics: _SourceInfoDiagnostics) -> set[int]:
        unidentified = super()._guess_columns(diagnostics=diagnostics)
        table = self._obj
        uids = make_uid_columns(
            table.master, table.obsnum, table.subobsnum, table.scannum, table.interface
        )
        table[uids.columns] = uids
        return unidentified
