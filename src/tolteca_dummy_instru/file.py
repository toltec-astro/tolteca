"""File tables for the dummy instrument and its telescope companions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

import pandas as pd
import xarray as xr
from tollan.accessor.pandas.model_table import (
    Contains,
    ExplicitColumnRule,
    ModelColumn,
    ModelColumnMapper,
    QuantityColumnRule,
    RecordRuleMapper,
    ScalarColumnRule,
    SkyCoordColumnRule,
    TimeColumnRule,
)

from tolteca_datamodels.core.file import FileAccessor
from tolteca_datamodels.core.source import SourceReaderRules
from tolteca_datamodels.lmt._obs_spec import _obs_spec_context as _lmt_obs_spec_context
from tolteca_datamodels.lmt.metadata import LmtTelMetadata
from tolteca_datamodels.lmt.tel import LmtTelAccessor, _get_tel_reader_record
from tolteca_datamodels.lmt.types import LmtInfo
from tolteca_datamodels.lmt_base.funcs import make_uid_columns
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec

if TYPE_CHECKING:
    from tolteca_datamodels.core.source import SourceReader

from ._obs_spec import _obs_spec_context
from .data import DummyInstruAccessor, DummyInstruIOMapper
from .metadata import DummyInstruMetadata
from .types import DummyInstruDataKind, DummyInstruDataModel, DummyInstruInfo

__all__ = ["DummyInstruFileAccessor"]

_metadata_mapper = ModelColumnMapper(
    name_rules=[(r".+", r"\g<0>")],
    type_rules=[
        ExplicitColumnRule(
            columns={
                "data_models": ModelColumn("data_models", dtype=object, converter=list),
            }
        ),
        QuantityColumnRule.model_validate({"physical_type": "time", "unit": "s"}),
        TimeColumnRule(format="mjd"),
        SkyCoordColumnRule.model_validate({"attrs": ["ra", "dec"], "unit": "deg"}),
        ScalarColumnRule(),
    ],
    include_computed=True,
    default=None,
)

_instru_data_kind_value = {
    DummyInstruInfo.instru: DummyInstruDataKind.counts.value,
    LmtInfo.instru: DummyInstruDataKind.telescope.value,
}
_data_kind_mapper = RecordRuleMapper[int](
    rules=(
        ({"instru": instru}, kind) for instru, kind in _instru_data_kind_value.items()
    ),
    default=DummyInstruDataKind.unknown.value,
)


def _data_kind_value(instru: str) -> int:
    """Classify the instrument of a dummy-instrument record."""
    return _instru_data_kind_value.get(instru, DummyInstruDataKind.unknown.value)


def _get_reader_record(data_source: object) -> object | None:
    if (record := _get_tel_reader_record(data_source)) is not None:
        return record
    if not isinstance(data_source, xr.Dataset):
        return None
    mapper = DummyInstruIOMapper.from_data_source(data_source)
    if (
        mapper.has_attr(data_source, mapper.schema.instru)
        and mapper.get_str(data_source, mapper.schema.instru) == DummyInstruInfo.instru
    ):
        return {"data_models": [DummyInstruDataModel.name], "interface_kind": "counter"}
    return None


@pd.api.extensions.register_dataframe_accessor("dummy_instru_file")
class DummyInstruFileAccessor(FileAccessor[DummyInstruDataModel]):
    """Build, select and open dummy-instrument file tables.

    Parameters
    ----------
    data : pandas.DataFrame
        File table containing source and source-location columns.

    Notes
    -----
    Filename information uses all registered parsers. Classifications cover
    dummy-instrument files and telescope companions with the ``tel_dummy_instru``
    prefix. Other inputs retain filename information with an unknown dummy data
    kind. Local files are opened as xarray Datasets; supplied Dataset objects
    are borrowed. Current membership and identity columns select readers.
    Without a matching rule, content identification accepts dummy-instrument
    data or telescope data with a dummy-instrument receiver header. Successful
    metadata reads correct guessed affiliation. Numeric interface selectors
    refer to dummy channels. ``data_kind_dummy_instru`` stores integer flag values from
    DummyInstruDataKind: ``counts``, ``telescope`` or ``unknown``. Successful
    header reads replace filename-based classifications. Flag queries use these
    member names directly.

    Examples
    --------
    >>> table = DummyInstruFileAccessor.from_sources([
    ...     "chan01-tcs-1-0-2-20251031T120000.nc",
    ...     "tel_dummy_instru_2025-10-31_000001_00_0002.nc",
    ... ])
    >>> table.dummy_instru_file.query(
    ...     "bits_any_set(data_kind_dummy_instru, counts)"
    ... ).interface.tolist()
    ['chan01']
    """

    _query_namespace: ClassVar = (
        FileAccessor._query_namespace | DummyInstruDataKind.__members__
    )
    _data_kind_type: ClassVar = DummyInstruDataKind
    _data_kind_column: ClassVar = "data_kind_dummy_instru"
    _data_kind_mapper: ClassVar = _data_kind_mapper
    _reader_rules: ClassVar = SourceReaderRules(
        prepare_record=_get_reader_record,
        rules=(
            (
                {
                    "data_models": Contains(value=DummyInstruDataModel.name),
                    "interface_kind": "counter",
                },
                DummyInstruAccessor,
            ),
            (
                {
                    "data_models": Contains(value=DummyInstruDataModel.name),
                    "interface": "tel",
                },
                LmtTelAccessor,
            ),
        ),
    )
    _obs_spec_cls: ClassVar = ObsSpec
    _obs_spec_context: ClassVar = _lmt_obs_spec_context | _obs_spec_context
    _metadata_columns: ClassVar = {
        name: column
        for model in (LmtTelMetadata, DummyInstruMetadata)
        for name, column in _metadata_mapper(model).items()
    }

    @staticmethod
    def _guess_derived_columns(table: pd.DataFrame) -> pd.DataFrame:
        return make_uid_columns(
            table.master, table.obsnum, table.subobsnum, table.scannum, table.interface
        )

    @staticmethod
    def _classify_metadata(
        reader: SourceReader, metadata: object | None
    ) -> DummyInstruDataKind:
        meta = cast("LmtTelMetadata | DummyInstruMetadata", metadata)
        return DummyInstruDataKind(_data_kind_value(meta.instru))
