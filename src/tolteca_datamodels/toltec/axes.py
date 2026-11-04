"""Columnar acquisition axes with quantity-aware table views."""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cached_property
from typing import TYPE_CHECKING

from tollan.accessor import NameMapping, Schema
from tollan.accessor.xarray import (
    XarrayAccessorBase,
    XarrayMapper,
    dataset_to_qtable,
    select_table_rows,
)

if TYPE_CHECKING:
    from astropy.table import QTable
    from tollan.accessor import Mapping

__all__ = ["ToltecAxisMapper", "ToltecAxisSchema", "ToltecAxisView"]


@dataclass
class ToltecAxisSchema(Schema):
    """Fields shared by columnar channel and probe axes."""

    block: Mapping = NameMapping.var("block")
    chan: Mapping = NameMapping.var("chan")
    step: Mapping = NameMapping.var("step")
    f_tone: Mapping = NameMapping.var("f_tone")
    f_chan: Mapping = NameMapping.var("f_chan")
    tone_mask: Mapping = NameMapping.var("tone_mask")
    tone_amp: Mapping = NameMapping.var("tone_amp")
    tone_phase: Mapping = NameMapping.var("tone_phase")
    f_lo: Mapping = NameMapping.var("f_lo")
    f_sweep: Mapping = NameMapping.var("f_sweep")
    n_samples: Mapping = NameMapping.var("n_samples")
    sample_start: Mapping = NameMapping.var("sample_start")
    sample_end: Mapping = NameMapping.var("sample_end")
    n_rows: Mapping = NameMapping.dim(("chan_entry", "probe"))


class ToltecAxisMapper(XarrayMapper[ToltecAxisSchema]):
    """Mapper for acquisition axis columns."""


class ToltecAxisView(XarrayAccessorBase[ToltecAxisMapper]):
    """View of a channel or probe dataset with one dimension shared by columns.

    Parameters
    ----------
    data_source : xarray.Dataset or xarray.DataTree
        Columnar axis data. Quantity columns use the ``units`` attribute.
    mapper : ToltecAxisMapper, optional
        Existing field mapping for the axis.
    """

    def select_block(self, block: int) -> ToltecAxisView:
        """Select axis rows belonging to a block.

        Parameters
        ----------
        block : int
            Acquisition block identifier.

        Returns
        -------
        ToltecAxisView
            Selected rows, retaining original block/channel/step identifiers.
            Contiguous selections share their underlying array storage.
        """
        schema = self.mapper.schema
        dim = self.mapper.get_name(schema.n_rows)
        if dim is None:
            msg = "Missing axis row dimension"
            raise ValueError(msg)
        selected = select_table_rows(
            self.data_source,
            self.mapper.get_arr(self.data_source, schema.block) == block,
            dim=dim,
        )
        return type(self)(selected, mapper=self.mapper)

    @cached_property
    def axis_table(self) -> QTable:
        """Quantity-aware table of the selected columns, loaded on first access."""
        dim = self.mapper.get_name(self.mapper.schema.n_rows)
        if dim is None:
            msg = "Missing axis row dimension"
            raise ValueError(msg)
        columns = {}
        for field in fields(self.mapper.schema):
            mapping = getattr(self.mapper.schema, field.name)
            if not self.mapper.has_arr(self.data_source, mapping):
                continue
            name = self.mapper.get_name(mapping)
            assert name is not None
            columns[field.name] = name
        return dataset_to_qtable(self.data_source, dim=dim, columns=columns)
