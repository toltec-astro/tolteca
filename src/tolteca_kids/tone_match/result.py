"""Saved tone datasets, complete correspondences and pair diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import xarray as xr
from tollan.accessor import NameMapping, Schema, SubpathMapping
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper

from tolteca_kids.match1d_result import Match1DView, _read_table

TONE_MATCH_PRODUCER = "tolteca_kids.tone_match.core:ToneMatch"
"""Producer recorded on the artifacts that `ToneMatch` writes."""

if TYPE_CHECKING:
    from astropy.table import QTable


@dataclass
class ToneMatchOverview:
    """Reference projection and full pair tables for one readout interface.

    Attributes
    ----------
    table : astropy.table.QTable
        Reference-oriented projection with exact input identities in metadata.
    pairs : dict[str, astropy.table.QTable]
        Complete pair relations keyed by the query dataset key. Includes every
        DTW correspondence, including competing assignments. No cost matrices
        are loaded to prepare this overview.
    """

    table: QTable
    pairs: dict[str, QTable]


@dataclass
class ToneMatchPairSchema(Schema):
    """Query/reference projections and the full matching diagnostic."""

    query: SubpathMapping = SubpathMapping.path("query")
    reference: SubpathMapping = SubpathMapping.path("reference")
    matched: SubpathMapping = SubpathMapping.path("matched")


class ToneMatchPairMapper(XarrayMapper[ToneMatchPairSchema]):
    """Access to one pair's saved computation."""


class ToneMatchPairView(XarrayAccessorBase[ToneMatchPairMapper]):
    """A saved pair, with arrays loaded only when requested.

    Parameters
    ----------
    data_source : xarray.DataTree
        One pair subtree. Keep its store open while accessing data.
    mapper : ToneMatchPairMapper, optional
        Mapping resolved against the subtree by default.
    """

    @property
    def data_source(self):
        """Pair result subtree."""
        return self._data_source

    @property
    def query(self) -> QTable:
        """Query-oriented projection, with metadata and units."""
        return _read_table(
            self.mapper.get_node(self.data_source, self.mapper.schema.query)
        )

    @property
    def reference(self) -> QTable:
        """Reference-oriented projection, with metadata and units."""
        return _read_table(
            self.mapper.get_node(self.data_source, self.mapper.schema.reference)
        )

    @property
    def matched(self) -> Match1DView:
        """Full relation, shift and lazily read alignment diagnostics."""
        return Match1DView(
            self.mapper.get_node(self.data_source, self.mapper.schema.matched)
        )


@dataclass
class ToneMatchSchema(Schema):
    """Selected tone datasets and completed pairs for one interface."""

    inputs: SubpathMapping = SubpathMapping.path("inputs")
    pairs: SubpathMapping = SubpathMapping.path("pairs")
    lineage: SubpathMapping = SubpathMapping.path("lineage")
    identities: NameMapping = NameMapping.attr("inputs")
    reference: NameMapping = NameMapping.attr("reference")


class ToneMatchMapper(XarrayMapper[ToneMatchSchema]):
    """Access to saved per-interface tone matching."""


class ToneMatchView(XarrayAccessorBase[ToneMatchMapper]):
    """Saved matching results for one interface.

    Parameters
    ----------
    data_source : xarray.DataTree
        Interface subtree. Keep its store open while reading lazy data.
    mapper : ToneMatchMapper, optional
        Mapping resolved against the subtree by default.
    """

    @property
    def data_source(self):
        """Interface result subtree."""
        return self._data_source

    @property
    def inputs(self) -> list[dict]:
        """Ordered exact dataset identities, with the reference first."""
        return self.mapper.get_value(self.data_source, self.mapper.schema.identities)

    @property
    def reference(self) -> dict:
        """Reference dataset identity."""
        return self.mapper.get_value(self.data_source, self.mapper.schema.reference)

    @property
    def table(self) -> QTable:
        """Reference projection with distinct columns for each selected block."""
        return _read_table(
            self.mapper.get_node(self.data_source, self.mapper.schema.lineage)
        )

    def input(self, key: str) -> QTable:
        """Read the original tone table for a selected dataset.

        Parameters
        ----------
        key : str
            Dataset key from an entry in :attr:`inputs`, including the reference.

        Returns
        -------
        astropy.table.QTable
            Tone rows with their original order, units and metadata.
        """
        node = self.mapper.get_node(self.data_source, self.mapper.schema.inputs)
        return _read_table(node[key])

    def pair(self, key: str) -> ToneMatchPairView:
        """Select a query's match against the reference.

        Parameters
        ----------
        key : str
            Query dataset key from an entry in :attr:`inputs`. The reference
            itself has no pair result.

        Returns
        -------
        ToneMatchPairView
            Lazy view of the projections, full correspondence and diagnostics.
        """
        node = self.mapper.get_node(self.data_source, self.mapper.schema.pairs)
        return ToneMatchPairView(node[key])

    @property
    def overview(self) -> ToneMatchOverview:
        """Loaded projection and every pair edge, without alignment matrices."""
        return ToneMatchOverview(
            self.table,
            {
                entry["key"]: self.pair(entry["key"]).matched.matched
                for entry in self.inputs[1:]
            },
        )


@contextmanager
def open_tone_match(source, *, interface):
    """Open one interface in a saved tone-matching Zarr store.

    Parameters
    ----------
    source : str, pathlib.Path or mapping
        Local/remote Zarr store or filesystem mapper accepted by xarray.
    interface : str
        Saved readout interface, for example ``toltec0``.

    Yields
    ------
    ToneMatchView
        Lazy result view. Read needed data inside the context; the store closes
        on exit. The overview never loads the pair alignment matrices.
    """
    with xr.open_datatree(
        source, engine="zarr", group=interface, chunks=None, consolidated=True
    ) as tree:
        yield ToneMatchView(tree)
