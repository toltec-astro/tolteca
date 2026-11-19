"""Filename-based dummy-instrument file tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pandas as pd
import xarray as xr
from tollan.accessor.pandas.model_table import Contains

from tolteca_datamodels.core.file import (
    CompositeFileAccessor,
    CompositeFileWorkflow,
    FileAccessorBase,
)
from tolteca_datamodels.core.source import SourceReaderRules
from tolteca_datamodels.lmt.file import LmtFileWorkflow
from tolteca_datamodels.lmt.metadata import LmtTelMetadata
from tolteca_datamodels.lmt.tel import LmtTelAccessor, read_tel_interface_info
from tolteca_datamodels.lmt_base.file import (
    LmtBaseFileWorkflow,
    metadata_column_mapper,
)

from . import filepath as _filepath  # noqa: F401 -- load this instrument's parsers.
from ._data_kind import _data_kind_mapper
from ._obs_spec import _obs_spec_context
from .data import DummyInstruAccessor, DummyInstruIOMapper
from .metadata import DummyInstruMetadata
from .types import DummyInstruDataKind, DummyInstruDataModel, DummyInstruInfo

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from tolteca_datamodels.core.source import SourceReader

__all__ = [
    "DummyInstruCounterFileAccessor",
    "DummyInstruCounterFileWorkflow",
    "DummyInstruFileAccessor",
    "DummyInstruFileWorkflow",
    "DummyInstruTelescopeFileAccessor",
    "DummyInstruTelescopeFileWorkflow",
]


def _get_reader_record(data_source: object) -> object | None:
    """Routing values recorded in an opened dummy or telescope dataset."""
    if (record := read_tel_interface_info(data_source)) is not None:
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


class _DummyInstruFileWorkflowBase(LmtBaseFileWorkflow):
    """Observation columns and classification shared by dummy file policies."""

    _obs_spec_context: ClassVar = _obs_spec_context
    _query_namespace: ClassVar = dict(DummyInstruDataKind.__members__)
    _query_symbols: ClassVar = {"DummyInstruDataKind": DummyInstruDataKind}
    _data_kind_rule_mapper: ClassVar = _data_kind_mapper

    def _prepare_derived_column_defaults(self) -> dict[Hashable, pd.Series]:
        return {
            "data_kind_dummy_instru": pd.Series(
                self._data_kind_rule_mapper.default,
                index=self.table.index,
                dtype="Int64",
            )
        }

    def _prepare_derived_columns(
        self, positions: list[int]
    ) -> dict[Hashable, pd.Series]:
        derived = super()._prepare_derived_columns(positions)
        derived["data_kind_dummy_instru"] = (
            self._map_derived_column_from_current_values(
                positions, derived, self._data_kind_rule_mapper, dtype="Int64"
            )
        )
        return derived

    def _prepare_reader_columns(
        self, reads: Mapping[int, tuple[SourceReader, object | None]]
    ) -> dict[str, pd.Series]:
        return {
            "data_kind_dummy_instru": pd.Series(
                {
                    position: self._data_kind_rule_mapper(metadata)
                    for position, (_, metadata) in reads.items()
                    if metadata is not None
                },
                dtype="Int64",
            )
        }


class DummyInstruCounterFileWorkflow(_DummyInstruFileWorkflowBase):
    """File-handling rules and column preparation for DummyInstruCounterFileAccessor.

    Parameters
    ----------
    file_accessor : FileAccessorBase
        Accessor over the shared table. Construction does not process rows.
    """

    _file_match_rules = (
        {
            "data_models": Contains(value=DummyInstruDataModel.name),
            "interface_kind": "counter",
        },
    )
    _source_reader_rules: ClassVar = SourceReaderRules(
        prepare_record=_get_reader_record,
        rules=((_file_match_rules[0], DummyInstruAccessor),),
    )
    _metadata_columns: ClassVar = {
        DummyInstruMetadata: metadata_column_mapper(DummyInstruMetadata),
    }


class DummyInstruCounterFileAccessor(FileAccessorBase[DummyInstruCounterFileWorkflow]):
    """Collect observation, channel and classification columns for dummy counters.

    Parameters
    ----------
    data : pandas.DataFrame
        Table to access without copying or modifying it. `from_sources` prepares
        filenames and derives UIDs and classification for dummy-instrument counter
        rows. Other rows retain filename metadata and an unknown classification.
    """


class DummyInstruTelescopeFileWorkflow(_DummyInstruFileWorkflowBase):
    """Handle telescope files collected for the dummy instrument.

    Parameters
    ----------
    file_accessor : FileAccessorBase
        Accessor over the shared table. Construction does not process rows.
    """

    _file_match_rules = (
        {
            "data_models": Contains(value=DummyInstruDataModel.name),
            "interface": "tel",
        },
    )
    _supersedes_workflow_types = (LmtFileWorkflow,)
    _source_reader_rules: ClassVar = SourceReaderRules(
        prepare_record=_get_reader_record,
        rules=((_file_match_rules[0], LmtTelAccessor),),
    )
    _metadata_columns: ClassVar = {
        LmtTelMetadata: metadata_column_mapper(LmtTelMetadata),
    }


class DummyInstruTelescopeFileAccessor(
    FileAccessorBase[DummyInstruTelescopeFileWorkflow]
):
    """Collect observation and classification columns for dummy telescope files.

    Parameters
    ----------
    data : pandas.DataFrame
        Table to access without copying or modifying it. `from_sources` prepares
        filenames and derives columns for telescope rows compatible with the dummy
        instrument. Other rows retain filename information and an unknown
        classification. This policy replaces generic LMT processing on shared rows.
    """


class DummyInstruFileWorkflow(CompositeFileWorkflow):
    """File-handling rules and column preparation for DummyInstruFileAccessor.

    Parameters
    ----------
    file_accessor : FileAccessorBase
        Accessor over the shared table. Construction does not process rows.
    """

    _component_accessor_types = (
        DummyInstruCounterFileAccessor,
        DummyInstruTelescopeFileAccessor,
    )


@pd.api.extensions.register_dataframe_accessor("dummy_instru_file")
class DummyInstruFileAccessor(CompositeFileAccessor[DummyInstruFileWorkflow]):
    """Collect dummy counter and companion telescope files in one table.

    Parameters
    ----------
    data : pandas.DataFrame
        Table to access without copying or modifying it. `from_sources` retains
        every supplied row and derives UIDs and classification for dummy files.
        Filename metadata, including counter channels, is available independently
        of the selected children. Unsupported rows retain an unknown classification.
        No resources are opened.

    Notes
    -----
    `partial` selects either concrete child without filtering rows. Queries
    accept ``counts``, ``telescope`` and ``unknown``, or qualified
    DummyInstruDataKind names, with ``bits_any_set`` and ``bits_all_set``.

    Examples
    --------
    >>> file_accessor = DummyInstruFileAccessor.from_sources([
    ...     "chan01-tcs-1-0-2-20251031T120000.nc",
    ...     "tel_dummy_instru_2025-10-31_000001_00_0002.nc",
    ... ])
    >>> file_accessor.table.interface.tolist()
    ['chan01', 'tel']
    >>> selected = file_accessor.partial(
    ...     accessor_types=(DummyInstruCounterFileAccessor,)
    ... )
    >>> [(positions, type(accessor).__name__)
    ...  for positions, accessor in selected.iter_accessors()]
    [([0], 'DummyInstruCounterFileAccessor')]
    """
