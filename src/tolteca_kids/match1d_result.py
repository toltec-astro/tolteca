"""Persistent pair relations and alignment diagnostics for frequency matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from astropy.io.misc import yaml
from tollan.accessor import NameMapping, Schema, SubpathMapping
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper, dataset_to_qtable

if TYPE_CHECKING:
    from astropy.table import QTable


def _array(value, dims):
    unit = getattr(value, "unit", None)
    return xr.DataArray(
        np.asarray(value), dims=dims, attrs={} if unit is None else {"units": str(unit)}
    )


def _table_dataset(table, *, omit=()):
    columns = [name for name in table.colnames if name not in omit]
    return xr.Dataset(
        {name: _array(table[name], "row") for name in columns},
        attrs={"columns": columns, "table_meta": yaml.dump(table.meta)},
    )


def _make_match_result(value):
    result = xr.DataTree(dataset=xr.Dataset({"shift": _array(value.shift, ())}))
    for name, table in (
        ("pairs", value.matched),
        ("query_matched", value.data["query_matched"]),
        ("ref_matched", value.data["ref_matched"]),
    ):
        # Pair indices already encode the connected membership stored by v2 as
        # Python sets in 'items'. Keep the portable numeric relation once.
        result[name] = xr.DataTree(dataset=_table_dataset(table, omit=("items",)))
    for name in ("query", "ref"):
        result[name] = xr.DataTree(
            dataset=xr.Dataset(
                {
                    "sorted": _array(value.data[f"{name}_sorted"], name),
                    "index": _array(value.data[f"isort_{name}"], name),
                }
            )
        )
    alignment = value.data["alignment"]
    result["alignment"] = xr.DataTree(
        dataset=xr.Dataset(
            {
                "cost": _array(alignment.costMatrix, ("query", "ref")),
            }
        )
    )
    result["path"] = xr.DataTree(
        dataset=xr.Dataset(
            {
                "index_query": _array(alignment.index1, "pair"),
                "index_ref": _array(alignment.index2, "pair"),
            }
        )
    )
    search = value.shiftdata
    trace = search["tracedata"]
    result["shift_search"] = xr.DataTree(
        dataset=xr.Dataset(
            {
                "frequency": _array(search["x"], "bin"),
                "ref_histogram": _array(search["y0"], "bin"),
                "query_histogram": _array(search["y1"], "bin"),
                "lag": _array(trace["dx"], "bin"),
                "correlation": _array(trace["cross_correlation"], "bin"),
                "in_search": _array(trace["mask_max"], "bin"),
                "snr": _array(trace["shift_snr"], ()),
            }
        )
    )
    return result


@dataclass
class Match1DSchema(Schema):
    """Match pairs, ordered inputs and dynamic-time-warping alignment."""

    pairs: SubpathMapping = SubpathMapping.path("pairs")
    query: SubpathMapping = SubpathMapping.path("query")
    query_sorted: NameMapping = query / NameMapping.var("sorted")
    ref: SubpathMapping = SubpathMapping.path("ref")
    ref_sorted: NameMapping = ref / NameMapping.var("sorted")
    alignment: SubpathMapping = SubpathMapping.path("alignment")
    cost: NameMapping = alignment / NameMapping.var("cost")
    path: SubpathMapping = SubpathMapping.path("path")
    index_query: NameMapping = path / NameMapping.var("index_query")
    index_ref: NameMapping = path / NameMapping.var("index_ref")
    shift: NameMapping = NameMapping.var("shift")
    query_matched: SubpathMapping = SubpathMapping.path("query_matched")
    ref_matched: SubpathMapping = SubpathMapping.path("ref_matched")
    shift_search: SubpathMapping = SubpathMapping.path("shift_search")


class Match1DMapper(XarrayMapper[Match1DSchema]):
    """Access to saved matching diagnostics."""


class Match1DView(XarrayAccessorBase[Match1DMapper]):
    """Frequency matches and alignment diagnostics without repeating matching.

    Parameters
    ----------
    data_source : xarray.DataTree
        A saved matching subtree. Arrays load when their properties are read.
    mapper : Match1DMapper, optional
        Mapping resolved against the subtree by default.
    """

    @property
    def data_source(self):
        """Saved result subtree."""
        return self._data_source

    def _validate(self):
        self.mapper.get_node(self.data_source, self.mapper.schema.pairs)

    def _quantity(self, field):
        return self.mapper.get_arr(self.data_source, field).u.quantity

    @property
    def matched(self) -> QTable:
        """All matched query/reference pairs, with the frequency shift."""
        return self._table(self.mapper.schema.pairs)

    def _table(self, field):
        table = _read_table(self.mapper.get_node(self.data_source, field))
        table.meta["shift"] = self._quantity(self.mapper.schema.shift)
        return table

    @property
    def query_matched(self) -> QTable:
        """Best reference match per query, in query order."""
        return self._table(self.mapper.schema.query_matched)

    @property
    def ref_matched(self) -> QTable:
        """Best query match per reference, in reference order."""
        return self._table(self.mapper.schema.ref_matched)

    @property
    def query_sorted(self):
        """Sorted query frequencies, as a Quantity."""
        return self._quantity(self.mapper.schema.query_sorted)

    @property
    def ref_sorted(self):
        """Sorted, unshifted reference frequencies, as a Quantity."""
        return self._quantity(self.mapper.schema.ref_sorted)

    @property
    def cost(self):
        """Accumulated alignment cost, indexed by query and reference."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.cost).to_numpy()

    @property
    def index_query(self):
        """Sorted query indices along the alignment path."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.index_query
        ).to_numpy()

    @property
    def index_ref(self):
        """Sorted reference indices along the alignment path."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.index_ref
        ).to_numpy()

    @property
    def shift_search(self):
        """Histogram, cross-correlation, lag mask and signal-to-noise diagnostic."""
        return self.mapper.get_node(
            self.data_source, self.mapper.schema.shift_search
        ).dataset

    def make_plotly_fig(self, *, type="match", **kwargs):
        """Render saved pair or alignment data without running the matcher.

        Parameters
        ----------
        type : {'match', 'density'}, optional
            Pair correspondences or accumulated DTW cost/path.
        **kwargs
            Figure, panel and labeling arguments for the matching visualization.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure constructed from this result's saved arrays.
        """
        from .match1d import _plot_match_density, _plot_match_pairs  # noqa: PLC0415

        if type == "match":
            return _plot_match_pairs(
                self.matched, self.query_sorted, self.ref_sorted, **kwargs
            )
        if type == "density":
            return _plot_match_density(
                self.cost, self.index_query, self.index_ref, **kwargs
            )
        msg = f"Unknown plot type: {type!r}"
        raise ValueError(msg)


def _read_table(node):
    columns = node.attrs["columns"]
    result = dataset_to_qtable(
        node, dim="row", columns={name: name for name in columns}
    )
    result.meta.clear()
    result.meta.update(yaml.load(node.attrs.get("table_meta", "{}")))
    return result
