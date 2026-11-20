"""TolTEC block signals and spectral results through a readonly view.

Use ``tree.toltec_kids.timestream`` or select a block with
``tree.toltec_kids.blocks[block].timestream``. ``TimestreamReduce`` installs
signals and optional PSD results; ``view.psd.psd_analysis`` exposes the producing
spectral configuration. The step context owns execution settings.

Raw samples remain lazy. Keep the source open until computation or export
completes. Compact exports store raw samples once; reopened block views recreate
sample projections. A spectra-only export retains metadata and spectra while
sample properties return None.
"""

# ruff: noqa: E741, E743, N802 -- Preserve the scientific I, Q and S21 names.

from __future__ import annotations

import functools
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from tollan.accessor import NameMapping, SubpathMapping
from tollan.accessor.xarray import (
    XarrayAccessorBase,
    XarrayMapper,
    ensure_dataset,
)

from tolteca_datamodels.toltec.metadata import ToltecTimestreamMetadata
from tolteca_kidsproc.accessors.kids import KidsSchema
from tolteca_kidsproc.analysis import PSDMapper, PSDView

if TYPE_CHECKING:
    from tollan.accessor import Mapping

__all__ = [
    "TimestreamView",
    "ToltecTimestreamMapper",
    "ToltecTimestreamSchema",
]


@dataclass
class ToltecTimestreamSchema(KidsSchema):
    """Schema for TolTEC time-domain signals and power spectra."""

    signals_subpath: SubpathMapping = SubpathMapping.path(("timestream", "_input"))
    chan_dim: Mapping = NameMapping.dim(("chan", "channel", "n_chans", "iqlen"))

    I: Mapping = NameMapping.var("I")
    Q: Mapping = NameMapping.var("Q")
    amplitude: Mapping = NameMapping.var("amplitude")
    phase: Mapping = NameMapping.var("phase")
    sample: Mapping = NameMapping.var("sample")

    # Already-solved numerical input; solving raw I/Q is a separate analysis.
    r: Mapping = NameMapping.var("r")
    x: Mapping = NameMapping.var("x")

    # Spectra and statistics in the shared PSD result format.
    f_psd: Mapping = NameMapping.var("f_psd")
    I_psd: Mapping = NameMapping.var("I_psd")
    Q_psd: Mapping = NameMapping.var("Q_psd")
    r_psd: Mapping = NameMapping.var("r_psd")
    x_psd: Mapping = NameMapping.var("x_psd")

    amplitude_psd: Mapping = NameMapping.var("amplitude_psd")
    phase_psd: Mapping = NameMapping.var("phase_psd")

    # Frequency-band statistics.
    I_psd_median: Mapping = NameMapping.var("I_psd_median")
    Q_psd_median: Mapping = NameMapping.var("Q_psd_median")
    r_psd_median: Mapping = NameMapping.var("r_psd_median")
    x_psd_median: Mapping = NameMapping.var("x_psd_median")
    I_psd_mad_std: Mapping = NameMapping.var("I_psd_mad_std")
    Q_psd_mad_std: Mapping = NameMapping.var("Q_psd_mad_std")
    r_psd_mad_std: Mapping = NameMapping.var("r_psd_mad_std")
    x_psd_mad_std: Mapping = NameMapping.var("x_psd_mad_std")

    # Coordinates (not namespaced)
    time: Mapping = NameMapping.var(("time", "ntimes"))
    chan: Mapping = NameMapping.var(("chan", "channel", "n_chans"))

    # Producing reduction configuration on the prepared signal Dataset.
    meta: Mapping = NameMapping.attr("meta")
    reduction_config: Mapping = NameMapping.attr(
        "tolteca_kids.timestream_reduce.config"
    )


class ToltecTimestreamMapper(XarrayMapper[ToltecTimestreamSchema]):
    """Mapper for TolTEC timestream data."""


class TimestreamView(XarrayAccessorBase[ToltecTimestreamMapper]):
    """View of TolTEC time-domain signals and Python power spectra.

    Parameters
    ----------
    data_source : xarray.Dataset or xarray.DataTree
        Numerical Dataset with I/Q columns, sweep/timestream block, or prepared
        timestream leaf. Run ``BlockCreate`` on native trees before selecting a block.
    mapper : ToltecTimestreamMapper, optional
        Field mapper. If omitted, resolve fields against the available domains.

    Notes
    -----
    ``data_source`` joins the available signal and spectral columns without
    changing the supplied tree. ``psd`` exposes the full spectral result view,
    including common spectra when requested. Recorded samples are unavailable
    in a spectra-only export; their properties return None.
    """

    @functools.cached_property
    def _block(self):
        node = self._data_source
        if isinstance(node, xr.DataTree):
            if node.attrs.get("kids_node_type") == "timestream":
                node = node.parent
            if node is not None and node.attrs.get("kids_node_type") == "block":
                return node.toltec_kids.block
        return None

    @functools.cached_property
    def _signals(self) -> xr.Dataset:
        """Lazy I/Q columns on channel and elapsed-seconds axes, when available."""
        block = self._block
        node = self._data_source
        if isinstance(node, xr.DataTree):
            mapper = ToltecTimestreamMapper.from_data_source(node)
            if mapper.schema.signals_subpath in mapper:
                node = mapper.get_node(node, mapper.schema.signals_subpath)
            node = node.dataset
        ds = ensure_dataset(node)
        mapper = getattr(
            self, "_mapper", None
        ) or ToltecTimestreamMapper.from_data_source(ds)
        names = [
            mapper.get_name(getattr(mapper.schema, name))
            for name in ("I", "Q", "r", "x", "amplitude", "phase")
            if mapper.has_arr(ds, getattr(mapper.schema, name))
        ]
        if block is None:
            return ds[names].copy(deep=False)
        acquisition = block.parent.tree.parent if block.parent is not None else None
        if "raw" not in block.tree.children and (
            acquisition is None or "raw" not in acquisition.children
        ):
            return ds[names].copy(deep=False)
        raw = block.raw
        mapper = raw.toltec_kids.mapper
        time_dim = mapper.get_name(mapper.schema.n_times)
        chan_dim = mapper.get_name(mapper.schema.n_chans)
        shape = {time_dim: "time", chan_dim: "chan"}
        arrays = {
            name: mapper.get_arr(raw, getattr(mapper.schema, name))
            .rename(shape)
            .transpose("chan", "time")
            for name in ("I", "Q")
        }
        # Coordinate labels remain original channel IDs; samples use the
        # acquisition index, while time is seconds from this block's first row.
        start = 0
        if block.parent is not None:
            start = int(block.parent._row(block.meta.block)["sample_start"])
        elif block.probe_axis.data_source.sizes.get("probe", 0):
            start = int(block.probe_axis.data_source["sample_start"][0])
        sample = np.arange(raw.sizes[time_dim]) + start
        if time_dim in raw.coords and raw[time_dim].dtype.kind in "iu":
            sample = raw[time_dim].values
        fs = block.meta.f_smp.to_value("Hz")
        attrs = dict(ds.attrs)
        ds = xr.Dataset(arrays).assign_coords(
            chan=block.chan_axis.data_source["chan"].values,
            time=("time", (sample - start) / fs, {"units": "s"}),
            sample=("time", sample),
        )
        ds.attrs.update(attrs, f_smp=fs)
        return _with_solved_signals(ds, block, sample)

    @functools.cached_property
    def psd(self) -> PSDView | None:
        """Shared spectral-analysis view, or None before analysis."""
        node = self._data_source
        if not isinstance(node, xr.DataTree):
            return None
        if node.attrs.get("kids_node_type") == "timestream":
            node = node.parent
        if node is None:
            return None
        mapper = PSDMapper.from_data_source(node)
        try:
            mapper.get_namespace(node)
        except KeyError:
            return None
        return PSDView(node, mapper=mapper)

    @functools.cached_property
    def _spectra(self) -> xr.Dataset:
        """Spectral outputs projected without changing their stored layout."""
        view = self.psd
        if view is not None:
            data = view.mapper.get_node(
                view.data_source, view.mapper.schema.psd_subpath
            ).dataset
            if view.psd_analysis.compute_common:
                common = view.mapper.get_node(
                    view.data_source, view.mapper.schema.psd_common_subpath
                ).dataset
                return xr.merge([data, common], combine_attrs="override")
            return data
        node = self._data_source
        ds = node.dataset if isinstance(node, xr.DataTree) else ensure_dataset(node)
        mapper = getattr(
            self, "_mapper", None
        ) or ToltecTimestreamMapper.from_data_source(ds)
        names = [
            mapper.get_name(mapping)
            for field in fields(mapper.schema)
            if "psd" in field.name
            and mapper.has_arr(ds, mapping := getattr(mapper.schema, field.name))
        ]
        return ds[names] if names else xr.Dataset()

    @functools.cached_property
    def data_source(self) -> xr.Dataset:
        """Virtual Dataset joining the available time and PSD domains."""
        result = xr.merge([self._signals, self._spectra], combine_attrs="override")
        result.attrs.update(self._spectra.attrs)
        return result

    @property
    def meta(self):
        """Complete source block metadata, when the view has a block scope."""
        if self._block is not None:
            return self._block.meta
        if self.mapper.has_attr(self.data_source, self.mapper.schema.meta):
            return ToltecTimestreamMetadata.model_validate_json(
                self.mapper.get_value(self.data_source, self.mapper.schema.meta)
            )
        return None

    def _field(self, name):
        field = getattr(self.mapper.schema, name)
        if not self.mapper.has_arr(self.data_source, field):
            return None
        return self.mapper.get_arr(self.data_source, field)

    @property
    def I(self):
        """Recorded in-phase samples, or None in a spectra-only export."""
        return self._field("I")

    @property
    def Q(self):
        """Recorded quadrature samples, or None in a spectra-only export."""
        return self._field("Q")

    @property
    def S21(self):
        """Complex recorded signal I + j Q, without model normalization."""
        return None if self.I is None or self.Q is None else self.I + 1j * self.Q

    @property
    def r(self):
        """Already-supplied r samples, or None; this view does not solve I/Q."""
        return self._field("r")

    @property
    def x(self):
        """Already-supplied x samples, or None; this view does not solve I/Q."""
        return self._field("x")

    @property
    def r_psd(self):
        """Welch spectrum of supplied r samples, when computed."""
        return self._field("r_psd")

    @property
    def x_psd(self):
        """Welch spectrum of supplied x samples, when computed."""
        return self._field("x_psd")

    @property
    def amplitude(self):
        """Magnitude of the recorded complex signal, with floating-point arithmetic."""
        if self.I is None or self.Q is None:
            return None
        return np.hypot(self.I.astype(float), self.Q.astype(float))

    @property
    def phase(self):
        """Wrapped raw I/Q phase in radians; no model derotation is applied."""
        if self.I is None or self.Q is None:
            return None
        result = np.arctan2(self.Q, self.I)
        result.attrs["units"] = "rad"
        return result

    @property
    def time(self):
        """Elapsed seconds within the block, not an absolute telescope clock."""
        return self._field("time")

    @property
    def sample(self):
        """Original acquisition sample IDs, when available."""
        return self._field("sample")

    @property
    def chan(self):
        """Original readout channel IDs."""
        return self._field("chan")

    @property
    def amplitude_psd(self):
        """Welch spectrum of raw I/Q magnitude, when computed."""
        return self._field("amplitude_psd")

    @property
    def phase_psd(self):
        """Welch spectrum of unwrapped raw I/Q phase, when computed."""
        return self._field("phase_psd")

    @property
    def f_psd(self) -> xr.DataArray | None:
        """PSD frequency axis [Hz]."""
        return self._field("f_psd")

    @property
    def I_psd(self) -> xr.DataArray | None:
        """In-phase spectrum in squared input units, per Hz for density scaling."""
        return self._field("I_psd")

    @property
    def Q_psd(self) -> xr.DataArray | None:
        """Quadrature spectrum in squared input units, per Hz for density scaling."""
        return self._field("Q_psd")

    @property
    def I_psd_median(self) -> xr.DataArray | None:
        """Median I spectrum within the configured frequency range."""
        return self._field("I_psd_median")

    @property
    def Q_psd_median(self) -> xr.DataArray | None:
        """Median Q spectrum within the configured frequency range."""
        return self._field("Q_psd_median")


def _with_solved_signals(ds, block, sample):
    """Join solved signals while preserving original channel/sample alignment."""
    if "solved" in block.tree.children:
        solved = block.tree["solved"].dataset
        if not np.array_equal(solved.sample_index, sample) or not np.array_equal(
            solved.chan, ds.chan
        ):
            msg = "Solved signal axes differ from the raw block"
            raise ValueError(msg)
        for name in ("r", "x"):
            ds[name] = xr.DataArray(
                solved[name].data.T, dims=("chan", "time"), attrs={"units": ""}
            )
    return ds
