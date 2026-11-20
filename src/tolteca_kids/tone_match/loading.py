"""Loading-response measurements from completed tone correspondences."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.table import QTable, vstack
from pydantic import Field
from tollan.accessor import Schema, SubpathMapping
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper
from tollan.config import FrozenBaseModel

from tolteca_kids.match1d_result import _read_table, _table_dataset

if TYPE_CHECKING:
    from .result import ToneMatchView


@dataclass
class LoadingResponseResult:
    """Per-dataset summaries, original measurements and complete candidate edges.

    Tables use result-local dataset keys and original row numbers. They do not
    assign persistent detector identities. Frequency changes retain the measured
    common motion; residuals additionally subtract the saved registration shift.
    """

    summary: QTable
    measurements: QTable
    matches: QTable

    def reference_associations(self) -> QTable:
        """Project the saved edges without another frequency search.

        Returns
        -------
        astropy.table.QTable
            One row per reference finding and dataset, with ``row_ref``,
            ``dataset``, ``row`` and ``status`` (unique, ambiguous or missing).
            A single candidate retains its row even when shared by reference
            findings. Multiple candidates and missing edges use row -1;
            all alternatives remain in ``matches``. Rows follow reference
            finding order, then summary dataset order.

        Raises
        ------
        ValueError
            The reference, measurements or saved edge identities disagree.
        """
        reference = self.summary.meta["reference"]["key"]
        datasets = list(map(str, self.summary["dataset"]))
        measurements = {(str(r["dataset"]), int(r["row"])) for r in self.measurements}
        if (
            reference not in datasets
            or len(set(datasets)) != len(datasets)
            or len(measurements) != len(self.measurements)
            or any(key not in datasets for key, _ in measurements)
        ):
            msg = "Loading-response reference and measurement identities disagree"
            raise ValueError(msg)
        refs = [int(r["row"]) for r in self.measurements if r["dataset"] == reference]
        candidates = defaultdict(list)
        degrees = Counter()
        seen = set()
        for edge in self.matches:
            dataset, row, row_ref = (
                str(edge["dataset"]),
                int(edge["row"]),
                int(edge["row_ref"]),
            )
            key = dataset, row, row_ref
            if (
                (dataset, row) not in measurements
                or (reference, row_ref) not in measurements
                or key in seen
            ):
                msg = (
                    "ToneMatch edges contain duplicate or unknown "
                    "measurement identities"
                )
                raise ValueError(msg)
            seen.add(key)
            candidates[dataset, row_ref].append(row)
            degrees[dataset, row] += 1
        rows = []
        for row_ref in refs:
            for dataset in datasets:
                choices = candidates[dataset, row_ref]
                row = choices[0] if len(choices) == 1 else -1
                status = "missing"
                if choices:
                    status = (
                        "unique"
                        if row >= 0 and degrees[dataset, row] == 1
                        else "ambiguous"
                    )
                rows.append((row_ref, dataset, row, status))
        return QTable(
            rows=rows,
            names=("row_ref", "dataset", "row", "status"),
            dtype=(int, str, int, str),
            meta={"reference": reference},
        )

    def to_tree(self) -> xr.DataTree:
        """Return a portable table tree without copying the matching matrices."""
        return xr.DataTree.from_dict(
            {
                name: _table_dataset(getattr(self, name))
                for name in ("summary", "measurements", "matches")
            }
        )


@dataclass
class LoadingResponseSchema(Schema):
    """Saved loading-response tables for one interface."""

    summary: SubpathMapping = SubpathMapping.path("summary")
    measurements: SubpathMapping = SubpathMapping.path("measurements")
    matches: SubpathMapping = SubpathMapping.path("matches")


class LoadingResponseMapper(XarrayMapper[LoadingResponseSchema]):
    """Access to loading summaries, measurements and candidate relations."""


class LoadingResponseView(XarrayAccessorBase[LoadingResponseMapper]):
    """Read saved loading-response tables while their source tree remains open.

    Parameters
    ----------
    data_source : xarray.DataTree
        One interface's loading-response subtree.
    mapper : LoadingResponseMapper, optional
        Mapping resolved from the subtree by default.
    """

    @property
    def summary(self) -> QTable:
        """Conditions, registration and population statistics per dataset."""
        return _read_table(
            self.mapper.get_node(self._data_source, self.mapper.schema.summary)
        )

    @property
    def measurements(self) -> QTable:
        """Measured properties of every input row, including unmatched rows."""
        return _read_table(
            self.mapper.get_node(self._data_source, self.mapper.schema.measurements)
        )

    @property
    def matches(self) -> QTable:
        """Every candidate edge, with degrees, frequency change and residual."""
        return _read_table(
            self.mapper.get_node(self._data_source, self.mapper.schema.matches)
        )

    def read(self) -> LoadingResponseResult:
        """Load all three tables for plotting after the source closes."""
        return LoadingResponseResult(self.summary, self.measurements, self.matches)


def _values(table, name):
    if name not in table.colnames:
        return np.full(len(table), np.nan)
    return np.asarray(np.ma.filled(np.ma.asarray(table[name], dtype=float), np.nan))


class LoadingResponseAnalysis(FrozenBaseModel):
    """Measure response using an existing match and explicit loading coordinates.

    No matching, identity selection, optical-power calibration or classification
    is performed. Population MAD describes scatter among detections; it is not
    an uncertainty on a response estimate.
    """

    frequency_column: str = Field(
        default="f_det",
        description=(
            "Measured resonance-frequency column in every input table. Probe "
            "frequencies are not substituted."
        ),
    )
    coordinate: str = Field(
        default="alt",
        description=(
            "Condition-table column used as the loading coordinate; its units "
            "are retained."
        ),
    )

    def __call__(
        self, matched: ToneMatchView, conditions: QTable
    ) -> LoadingResponseResult:
        """Calculate summaries and edge measurements without changing the inputs.

        Parameters
        ----------
        matched : ToneMatchView
            Completed comparison for one interface. Keep its store open.
        conditions : astropy.table.QTable
            Exactly one row per input, keyed by ``dataset``. Contains the named
            coordinate and optional time/readout/telemetry diagnostics. Missing
            numeric conditions use NaN, not invented defaults.

        Returns
        -------
        LoadingResponseResult
            Eager tables with units. Missing quantities are NaN. Match uniqueness
            requires degree one at both endpoints of the complete saved relation.

        Raises
        ------
        ValueError
            Conditions do not identify every dataset exactly once, or a measured
            frequency column is missing or has incompatible units.
        """
        inputs = matched.inputs
        keys = [row["key"] for row in inputs]
        if (
            "dataset" not in conditions.colnames
            or self.coordinate not in conditions.colnames
            or len(conditions) != len(keys)
            or set(conditions["dataset"]) != set(keys)
        ):
            msg = (
                "Loading conditions require one row per matched dataset and the "
                "selected coordinate"
            )
            raise ValueError(msg)
        order = {str(key): i for i, key in enumerate(conditions["dataset"])}
        summary = conditions[[order[key] for key in keys]].copy()
        tables = {key: matched.input(key) for key in keys}
        frequencies = {}
        for key, table in tables.items():
            if self.frequency_column not in table.colnames:
                msg = (
                    f"{key}: missing measured frequency column "
                    f"{self.frequency_column!r}"
                )
                raise ValueError(msg)
            frequencies[key] = u.Quantity(table[self.frequency_column]).to(u.Hz)
        reference = keys[0]
        measurements, edges, stats = [], [], []
        properties = {
            "Qr": "Qr",
            "depth": "height_db_s21",
            "snr_d21": "snr_d21",
            "snr_s21": "snr_s21",
            "flags": "bitmask_det",
        }
        for key in keys:
            table = tables[key]
            measurement = QTable(
                {
                    "dataset": np.repeat(key, len(table)),
                    "row": np.arange(len(table)),
                    "frequency": frequencies[key],
                    **{
                        name: _values(table, source)
                        for name, source in properties.items()
                    },
                }
            )
            measurement["depth"].unit = u.dB
            measurements.append(measurement)
            if key == reference:
                iq = ir = np.arange(len(table))
                shift = 0 * u.Hz
            else:
                pair = matched.pair(key).matched
                relation = pair.matched
                iq = np.asarray(relation["idx_query"], dtype=int)
                ir = np.asarray(relation["idx_ref"], dtype=int)
                shift = relation.meta["shift"].to(u.Hz)
            dq = np.bincount(iq, minlength=len(table))
            dr = np.bincount(ir, minlength=len(tables[reference]))
            one = (dq[iq] == 1) & (dr[ir] == 1)
            change = frequencies[key][iq] - frequencies[reference][ir]
            edge = QTable(
                {
                    "dataset": np.repeat(key, len(iq)),
                    "row": iq,
                    "row_ref": ir,
                    "degree": dq[iq],
                    "degree_ref": dr[ir],
                    "unique": one,
                    "frequency": frequencies[key][iq],
                    "frequency_ref": frequencies[reference][ir],
                    "frequency_change": change,
                    "fractional_frequency_change": change / frequencies[reference][ir],
                    "registration_shift": np.full(len(iq), shift.to_value(u.Hz)) * u.Hz,
                    "registered_residual": change - shift,
                }
            )
            edges.append(edge)
            stat = {
                "n_detected": len(table),
                "n_edges": len(iq),
                "n_unique": int(one.sum()),
                "n_unmatched": int((dq == 0).sum()),
                "n_ref_missing": int((dr == 0).sum()),
                "registration_shift": shift.to_value(u.Hz),
            }
            for name in ("Qr", "depth", "snr_d21", "snr_s21"):
                values = np.asarray(measurement[name])
                valid = values[np.isfinite(values)]
                median = np.median(valid) if len(valid) else np.nan
                stat[name + "_median"] = median
                stat[name + "_mad"] = (
                    np.median(abs(valid - median)) if len(valid) else np.nan
                )
                stat[name + "_n_valid"] = len(valid)
            stats.append(stat)
        for name in stats[0]:
            summary[name] = [row[name] for row in stats]
        summary["registration_shift"].unit = u.Hz
        for name in ("depth_median", "depth_mad"):
            summary[name].unit = u.dB
        summary.meta.update(
            inputs=inputs,
            reference=matched.reference,
            coordinate=self.coordinate,
            frequency_column=self.frequency_column,
            scatter="Unscaled population MAD; membership can change between datasets.",
        )
        return LoadingResponseResult(summary, vstack(measurements), vstack(edges))
