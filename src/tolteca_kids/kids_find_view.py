"""Saved peak detections, grouping and probe-matching diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tollan.accessor import NameMapping, Schema, SubpathMapping
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper

from .kids_find_artifacts import FIND_COLUMNS
from .match1d_result import Match1DView, _read_table

if TYPE_CHECKING:
    from astropy.table import QTable

    from .kids_find import KidsFindConfig


@dataclass
class KidsFindPeakSchema(Schema):
    """Peak samples, their selection masks and the measured peak table."""

    samples: SubpathMapping = SubpathMapping.path("samples")
    x: NameMapping = samples / NameMapping.var("x")
    y: NameMapping = samples / NameMapping.var("y")
    ey: NameMapping = samples / NameMapping.var("ey")
    lookahead: NameMapping = samples / NameMapping.var("lookahead")
    delta: NameMapping = samples / NameMapping.var("delta")
    labels: NameMapping = samples / NameMapping.var("labels")
    mask_not_real: NameMapping = samples / NameMapping.var("mask_not_real")
    mask_dark: NameMapping = samples / NameMapping.var("mask_dark")
    mask_baseline: NameMapping = samples / NameMapping.var("mask_baseline")
    mask_edge: NameMapping = samples / NameMapping.var("mask_edge")
    peaks: SubpathMapping = SubpathMapping.path("peaks")
    bitmask: NameMapping = peaks / NameMapping.var("bitmask")


class KidsFindPeakMapper(XarrayMapper[KidsFindPeakSchema]):
    """Access to saved peak measurements and sample masks."""


class KidsFindPeakView(XarrayAccessorBase[KidsFindPeakMapper]):
    """Peak measurements and masks for D21 or S21.

    Parameters
    ----------
    data_source : xarray.DataTree
        One finding namespace's D21 or S21 subtree. Keep it open while reading.
    mapper : KidsFindPeakMapper, optional
        Mapping resolved against this subtree by default.
    """

    @property
    def data_source(self):
        """Saved result subtree."""
        return self._data_source

    def _validate(self):
        self.mapper.get_node(self.data_source, self.mapper.schema.peaks)
        self.mapper.get_arr(self.data_source, self.mapper.schema.labels)

    def _quantity(self, field):
        return self.mapper.get_arr(self.data_source, field).u.quantity

    def _values(self, field):
        return self.mapper.get_arr(self.data_source, field).to_numpy()

    @property
    def peaks(self) -> QTable:
        """Measured peak table in detection order, with physical units."""
        return _read_table(
            self.mapper.get_node(self.data_source, self.mapper.schema.peaks)
        )

    @property
    def x(self):
        """Sample frequencies, as a Quantity."""
        return self._quantity(self.mapper.schema.x)

    @property
    def y(self):
        """Signal supplied to the peak detector, as a Quantity."""
        return self._quantity(self.mapper.schema.y)

    @property
    def labels(self):
        """Peak labels for each input sample."""
        return self._values(self.mapper.schema.labels)

    @property
    def bitmask(self):
        """Quality and grouping flags for each measured peak."""
        return self._values(self.mapper.schema.bitmask)

    @property
    def mask_not_real(self):
        """Samples within rejected peaks."""
        return self._values(self.mapper.schema.mask_not_real)

    @property
    def mask_dark(self):
        """D21 samples within dark-detector peaks."""
        return self._values(self.mapper.schema.mask_dark)

    @property
    def mask_baseline(self):
        """D21 samples selected as baseline."""
        return self._values(self.mapper.schema.mask_baseline)

    @property
    def mask_edge(self):
        """S21 samples within the sweep edges."""
        return self._values(self.mapper.schema.mask_edge)


@dataclass
class KidsFindSchema(Schema):
    """Finding configuration, detections and their diagnostic subtrees."""

    config: NameMapping = NameMapping.attr("tolteca_kids.kids_find.config")
    identity: NameMapping = NameMapping.attr("identity")
    d21: SubpathMapping = SubpathMapping.path("d21")
    s21: SubpathMapping = SubpathMapping.path("s21")
    samples: SubpathMapping = SubpathMapping.path("samples")
    mask_baseline: NameMapping = samples / NameMapping.var("mask_baseline")
    detected: SubpathMapping = SubpathMapping.path("detected")
    det_grouped: SubpathMapping = SubpathMapping.path("det_grouped")
    det_groups: SubpathMapping = SubpathMapping.path("det_groups")
    bitmask_det: NameMapping = det_groups / NameMapping.var("bitmask")
    mdl_grouped: SubpathMapping = SubpathMapping.path("mdl_grouped")
    mdl_groups: SubpathMapping = SubpathMapping.path("mdl_groups")
    matched: SubpathMapping = SubpathMapping.path("matched")
    matched_ref: SubpathMapping = SubpathMapping.path("matched_ref")


class KidsFindMapper(XarrayMapper[KidsFindSchema]):
    """Access to saved finding datasets and their producing settings."""


class KidsFindView(XarrayAccessorBase[KidsFindMapper]):
    """Read-only finding results for one sweep block.

    Parameters
    ----------
    data_source : xarray.DataTree
        Finding namespace, obtained with ``KidsFind.view(block)``. Keep the
        tree open while accessing lazy arrays. Table properties materialize
        their columns; reading this view does not run detection or matching.
    mapper : KidsFindMapper, optional
        Mapping resolved against the supplied namespace by default.
    """

    @property
    def data_source(self):
        """Saved result subtree."""
        return self._data_source

    def _validate(self):
        self.mapper.get_node(self.data_source, self.mapper.schema.detected)
        self.mapper.get_value(self.data_source, self.mapper.schema.config)

    @property
    def config(self) -> KidsFindConfig:
        """Finding settings used to produce these results."""
        from .kids_find import KidsFindConfig  # noqa: PLC0415

        return KidsFindConfig.model_validate_json(
            self.mapper.get_value(self.data_source, self.mapper.schema.config)
        )

    @property
    def d21_peaks(self) -> KidsFindPeakView:
        """D21 peak measurements and sample masks."""
        return KidsFindPeakView(
            self.mapper.get_node(self.data_source, self.mapper.schema.d21)
        )

    @property
    def s21_peaks(self) -> KidsFindPeakView:
        """S21 peak measurements and sample masks."""
        return KidsFindPeakView(
            self.mapper.get_node(self.data_source, self.mapper.schema.s21)
        )

    @property
    def mask_baseline(self):
        """Baseline samples indexed by channel and sweep probe."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.mask_baseline
        ).to_numpy()

    @property
    def det_groups(self) -> QTable:
        """Detection-group measurements and flags, in group order."""
        return _read_table(
            self.mapper.get_node(self.data_source, self.mapper.schema.det_groups)
        )

    @property
    def det_grouped(self) -> QTable:
        """Peak membership with subdetector, original peak ID and detection group."""
        return _read_table(
            self.mapper.get_node(self.data_source, self.mapper.schema.det_grouped)
        )

    @property
    def mdl_grouped(self) -> QTable:
        """Detection-to-model-group membership, in detection order."""
        return _read_table(
            self.mapper.get_node(self.data_source, self.mapper.schema.mdl_grouped)
        )

    @property
    def mdl_groups(self) -> QTable:
        """Model-group sizes, distance limits and combined flags."""
        return _read_table(
            self.mapper.get_node(self.data_source, self.mapper.schema.mdl_groups)
        )

    @property
    def detected(self) -> QTable:
        """Detected resonances with their model groups and probe matches."""
        return _read_table(
            self.mapper.get_node(self.data_source, self.mapper.schema.detected)
        )

    @property
    def matched(self) -> Match1DView:
        """Matches against the recorded probe comb."""
        return Match1DView(
            self.mapper.get_node(self.data_source, self.mapper.schema.matched)
        )

    @property
    def matched_ref(self) -> Match1DView:
        """Matches against the configured channel, D21 or S21 reference."""
        if self.config.match_ref == "chan":
            return self.matched
        return Match1DView(
            self.mapper.get_node(self.data_source, self.mapper.schema.matched_ref)
        )

    def export_tables(self) -> dict[str, QTable]:
        """Build the detection and channel ECSV tables from saved results.

        Returns
        -------
        dict of str to astropy.table.QTable
            ``kids_find`` and ``chan_prop`` tables with v2 columns and units,
            frequency-shift metadata and available acquisition/file identities.
        """
        matched = self.matched
        identity = self.mapper.get_value(self.data_source, self.mapper.schema.identity)
        tables = {
            "kids_find": matched.query_matched[list(FIND_COLUMNS)],
            "chan_prop": matched.ref_matched[list(FIND_COLUMNS)],
        }
        for table in tables.values():
            table.meta.update(identity)
        return tables
