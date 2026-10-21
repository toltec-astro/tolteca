"""LMT telescope file tables."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
from tollan.accessor.pandas.model_table import (
    Contains,
    ExplicitColumnRule,
    ModelColumn,
    ModelColumnMapper,
    QuantityColumnRule,
    ScalarColumnRule,
    SkyCoordColumnRule,
    TimeColumnRule,
)

from ..core.file import FileAccessor
from ..core.source import SourceReaderRules
from ..lmt_base.funcs import make_uid_columns
from ..lmt_base.obs_spec import ObsSpec
from ._obs_spec import _obs_spec_context
from .metadata import LmtTelMetadata
from .tel import LmtTelAccessor, _get_tel_reader_record
from .types import LmtDataModel

__all__ = ["LmtFileAccessor"]


@pd.api.extensions.register_dataframe_accessor("lmt_file")
class LmtFileAccessor(FileAccessor[LmtDataModel]):
    """Build, query and open LMT telescope file tables.

    Parameters
    ----------
    data : pandas.DataFrame
        File table, as returned by `from_sources`.

    Notes
    -----
    Header reads support local netCDF files and open Dataset or DataTree
    objects. Remote resources must be staged locally. Non-telescope data
    and unsupported formats retain their rows and any registered filename
    information without header enrichment.
    Metadata columns use MJD for times, seconds for durations and degrees
    for coordinates, indicated by column suffixes. Header reads do not load
    pointing streams. Opening requires recognized telescope data, including
    when supplying an already opened object. Observation selectors use the
    telescope interface labels.

    Examples
    --------
    With the reference observation under ``data_lmt`` in the working directory:

    >>> from tolteca_datamodels.lmt.file import LmtFileAccessor
    >>> paths = ["data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc"]
    >>> table = LmtFileAccessor.from_sources(paths, mode="read")
    >>> pointing = table.lmt_file.query(
    ...     'obs_goal == "pointing" and obs_pgm == "lissajous"'
    ... )
    >>> selected = pointing.lmt_file.query(
    ...     'target_frame == "icrs" and 70 <= target_ra_deg < 80 '
    ...     'and 60 <= target_dec_deg < 70 and t_int_s > 60 and t_elapsed_s > 65'
    ... )
    >>> len(selected)
    1
    >>> with selected.lmt_file.open() as opened:
    ...     az = [
    ...         row.accessor_obj.az.isel(time=slice(0, 100)).load()
    ...         for row in opened.itertuples(index=False)
    ...         if row.accessor_obj is not None
    ...     ]
    """

    _reader_rules: ClassVar = SourceReaderRules(
        prepare_record=_get_tel_reader_record,
        rules=(
            (
                {"data_models": Contains(value=LmtDataModel.name), "interface": "tel"},
                LmtTelAccessor,
            ),
        ),
    )
    _obs_spec_cls: ClassVar = ObsSpec
    _obs_spec_context: ClassVar = _obs_spec_context
    _metadata_columns: ClassVar[dict[str, ModelColumn]] = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            ExplicitColumnRule(
                columns={
                    "data_models": ModelColumn(
                        "data_models", dtype=object, converter=list
                    ),
                }
            ),
            QuantityColumnRule.model_validate({"physical_type": "time", "unit": "s"}),
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule.model_validate({"attrs": ["ra", "dec"], "unit": "deg"}),
            ScalarColumnRule(),
        ],
        include_computed=True,
        default=None,
    )(LmtTelMetadata)

    @staticmethod
    def _guess_derived_columns(table: pd.DataFrame) -> pd.DataFrame:
        return make_uid_columns(
            table.master, table.obsnum, table.subobsnum, table.scannum, table.interface
        )
