"""KIDs data and analysis views through the xarray ``kids`` accessor.

Dataset views expose I/Q signals with frequency or time coordinates. Sweep and
timestream views accept one-dimensional signals for a single channel or
two-dimensional signals for multiple channels. Time is a one-dimensional
coordinate shared by all channels. Incompatible fields, coordinates, or
dimensions raise ``ValueError`` when the view is constructed.

On a tree, ``kids.sweep`` selects ``sweep``, ``_input``, or the current node, in
that order. ``kids.timestream`` uses ``timestream``, ``_input``, or the current
node. Select a block before requesting its views; the accessor does not search
other branches. Direct view construction uses the supplied Dataset or current
tree node's payload.

Analysis views select a DataTree namespace containing output Datasets and the
producing configuration. Each access to ``kids.d21`` or ``kids.psd`` creates a
new view; an existing view retains the results it selected. Dataset views are
cached on the accessor.

Components
----------
``KidsAccessor`` (``.kids``)
    Entry point for data and analysis views on a Dataset or selected DataTree
    node. ``data_source`` retains that supplied container.

Data views expose DataArrays. The dimensions below use the names assigned by
``make_kids_dataset``; views also accept mapped coordinates on other dimension
names.

``S21View`` (``.kids.s21``)
    ``I``, ``Q``, complex ``S21``, magnitude ``aS21``, and ``aS21_db``, retaining
    the input dimensions.
``SweepView`` (``.kids.sweep``)
    I/Q and derived S21 quantities with physical ``frequency``. A single-channel
    sweep has dimensions ``(sweep,)`` for both signals and frequency.
``MultiSweepView`` (``.kids.multi_sweep``)
    Signals and absolute ``frequency`` with dimensions ``(chan, sweep)``, plus
    ``n_chans``. The sweep-sample dimension is shared by all channels.
``TimestreamView`` (``.kids.timestream``)
    I/Q and derived S21 quantities with a shared one-dimensional ``time``
    coordinate and sampling-frequency quantity ``f_smp``. A single-channel
    signal has dimensions ``(time,)``.
``MultiTimestreamView`` (``.kids.multi_timestream``)
    Signals with dimensions ``(chan, time)`` or ``(time, chan)``, ``time`` with
    dimensions ``(time,)``, and ``n_chans``.

Analyses return DataTrees. Their result views expose arrays or Datasets within
the selected analysis namespace and the configuration that produced them.

``D21Analysis`` / ``D21View`` (``.kids.d21``)
    ``d21`` is the complex derivative on the input sweep dimensions.
    ``d21_unified`` is the channel-averaged magnitude on ``(f_unified,)``;
    ``f_unified`` and ``cov_unified`` give its frequency and channel coverage.
    ``d21_analysis`` is the producing configuration. Matched and unified arrays
    occupy separate Dataset leaves.
``PSDAnalysis`` / ``PSDView`` (``.kids.psd``)
    ``psd`` groups spectra by the semantic signal names ``I`` and ``Q``.
    Spectra retain their non-time dimensions and add ``f_psd``; multi-channel
    timestreams produce ``(chan, f_psd)``. Variables in ``psd_median`` and
    ``psd_mad_std`` retain only non-time dimensions, or are scalar for a single
    series. ``f_psd`` is shared
    and one-dimensional; ``psd_analysis`` is the producing configuration.
    With ``compute_common=True``, ``psd_common`` and ``psd_common_mad_std``
    contain the median and scaled MAD across channels at each frequency,
    with dimensions ``(f_psd,)``.
``KidsSchema`` / ``KidsMapper``
    Common field vocabulary and data access for I/Q, frequency, time, and input
    subtrees. Supply a mapper to use different field names or input paths.

Examples
--------
Create a single-channel sweep and inspect its complex transmission:

>>> import numpy as np
>>> from astropy import units as u
>>> from tolteca_kidsproc.accessors import make_kids_dataset
>>> from tolteca_kidsproc.analysis import D21Analysis, PSDAnalysis
>>> frequency = np.linspace(1e9, 1.1e9, 101)
>>> signal = 1 / (1 + 1j * (frequency - 1.05e9) / 1e6)
>>> sweep = make_kids_dataset(signal.real, signal.imag, frequency=frequency * u.Hz)
>>> sweep.kids.sweep.S21.shape
(101,)
>>> sweep.kids.sweep.aS21_db.shape
(101,)
>>> sweep.kids.s21.data_source is sweep.kids.sweep.data_source
True
>>> sweep.kids.s21.mapper is sweep.kids.sweep.mapper
True

Multiple channels use the same views through ``multi_sweep``:

>>> multi = make_kids_dataset(
...     np.stack([signal.real, signal.real]),
...     np.stack([signal.imag, signal.imag]),
...     frequency=np.stack([frequency, frequency + 1e8]) * u.Hz,
... )
>>> multi.kids.multi_sweep.n_chans
2
>>> multi.kids.multi_sweep.frequency.shape
(2, 101)

A configured analysis returns a tree. Its view selects the analysis namespace
and exposes the matched and unified derivative arrays:

>>> analysis = D21Analysis(method="gradient", smooth=5)
>>> result = analysis(multi.kids.multi_sweep)
>>> view = result.kids.d21
>>> view.d21.shape
(2, 101)
>>> view.d21_unified.ndim
1
>>> view.d21_analysis == analysis
True
>>> view.data_source is result["tolteca_kidsproc.analysis.d21"]
True
>>> result.kids.data_source is result
True
>>> result.kids.multi_sweep.S21.identical(multi.kids.multi_sweep.S21)
True

Select a block to read its sweep and install derivatives beside it:

>>> import xarray as xr
>>> acquisition = xr.DataTree.from_dict({"blocks/0/sweep": multi})
>>> block = acquisition["blocks/0"]
>>> block.kids.sweep.data_source.identical(multi)
True
>>> analysis(block.kids.sweep, output=block) is block
True
>>> block.kids.d21.d21.shape
(2, 101)

Time coordinates select timestream views. Pass the view to PSD analysis:

>>> time = np.arange(128) / 64.0
>>> signal = np.exp(2j * np.pi * 8 * time)
>>> timestream = make_kids_dataset(
...     signal.real, signal.imag, time=time * u.s, f_smp=64 << u.Hz)
>>> timestream.kids.timestream.S21.shape
(128,)
>>> spectra = PSDAnalysis(nperseg=64)(timestream.kids.timestream)
>>> spectra.kids.psd.signal_names
('I', 'Q')
>>> spectra.kids.timestream.S21.identical(timestream.kids.timestream.S21)
True
>>> time_block = xr.DataTree.from_dict({"timestream": timestream})
>>> PSDAnalysis(nperseg=64)(time_block.kids.timestream, output=time_block) is time_block
True
>>> time_block.kids.psd.signal_names
('I', 'Q')
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import xarray as xr
from tollan.accessor import AccessorBase, NameMapping, Schema, SubpathMapping
from tollan.accessor.xarray import XarrayMapper, XarraySourceT

if TYPE_CHECKING:
    from tollan.accessor import Mapping

    from ..analysis.d21 import D21View
    from ..analysis.psd import PSDView
    from .views import (
        MultiSweepView,
        MultiTimestreamView,
        S21View,
        SweepView,
        TimestreamView,
    )

__all__ = [
    "KidsAccessor",
    "KidsMapper",
    "KidsSchema",
]


@dataclass
class KidsSchema(Schema):
    """Field mappings for KIDs data.

    Defines available field mappings. Views determine field requirements.

    Attributes
    ----------
    s21_subpath : SubpathMapping
        I/Q input node, trying ``sweep``, ``timestream``, ``_input``, then the
        current node. Dataset inputs are used directly.
    sweep_subpath : SubpathMapping
        Sweep input node, trying ``sweep``, ``_input``, then the current node.
    timestream_subpath : SubpathMapping
        Time-series input node, trying ``timestream``, ``_input``, then the
        current node. All paths are relative to the supplied tree node.
    I : Mapping
        In-phase component
    Q : Mapping
        Quadrature component
    frequency : Mapping
        Frequency coordinate (required for sweep views)
    time : Mapping
        Shared one-dimensional time coordinate (required for timestream views)
    f_smp : Mapping
        Recorded sampling frequency in Hz, stored as an attribute and required
        for timestream views.
    chan_dim : Mapping
        Channel dimension (required for multi-channel views).
    """

    s21_subpath: SubpathMapping = SubpathMapping.path(("sweep", "timestream", "_input"))
    sweep_subpath: SubpathMapping = SubpathMapping.path(("sweep", "_input"))
    timestream_subpath: SubpathMapping = SubpathMapping.path(("timestream", "_input"))

    I: Mapping = NameMapping.var("I")
    Q: Mapping = NameMapping.var("Q")
    frequency: Mapping = NameMapping.var("frequency")
    time: Mapping = NameMapping.var("time")
    f_smp: Mapping = NameMapping.attr("f_smp")
    chan_dim: Mapping = NameMapping.dim("chan")


class KidsMapper(XarrayMapper[KidsSchema]):
    """Field mapping for I/Q signals and frequency or time coordinates."""


@xr.register_dataset_accessor("kids")
@xr.register_datatree_accessor("kids")
class KidsAccessor(AccessorBase[XarraySourceT, KidsMapper]):
    """Access KIDs signals and analysis results from a Dataset or DataTree.

    Parameters
    ----------
    data_source : xarray.Dataset or xarray.DataTree
        I/Q Dataset, tree node containing I/Q data, or analysis scope containing
        input and result subtrees. Retained as ``data_source``. Analysis views
        require a DataTree.
    mapper : KidsMapper, optional
        Input paths and field mappings for the supplied source. If omitted,
        resolve them when the accessor is constructed. Views of the same payload
        share this mapper; views selecting a child resolve their own mappings.
        Analysis views use their own mappers.

    Notes
    -----
    The ``s21``, ``sweep``, ``multi_sweep``, ``timestream``, and
    ``multi_timestream`` properties select cached data views. The ``d21`` and
    ``psd`` properties select fresh analysis views. After changing input paths or
    field names, construct a new ``KidsAccessor(source)`` to resolve the new
    layout. See the module examples for constructing data, running analyses,
    and accessing results.
    """

    def _make_view_from_mapped_source[ViewT: AccessorBase](
        self, subpath: SubpathMapping, view_cls: type[ViewT]
    ) -> ViewT:
        """Construct a view on a mapped child or the current source.

        A child view creates its own mapper; a view on the current source shares
        this accessor's mapper.
        """
        mapper = self.mapper
        source = self.data_source
        if subpath in mapper:
            return view_cls(mapper.get_node(source, subpath))
        return view_cls(source, mapper)

    # Data View Properties
    # ====================

    @functools.cached_property
    def s21(self) -> S21View:
        """Complex transmission and amplitude view of I/Q data."""
        from .views import S21View  # noqa: PLC0415

        return self._make_view_from_mapped_source(
            self.mapper.schema.s21_subpath, S21View
        )

    @functools.cached_property
    def sweep(self) -> SweepView:
        """I/Q sweep view with a frequency coordinate."""
        from .views import SweepView  # noqa: PLC0415

        return self._make_view_from_mapped_source(
            self.mapper.schema.sweep_subpath, SweepView
        )

    @functools.cached_property
    def multi_sweep(self) -> MultiSweepView:
        """Multi-channel I/Q sweep view with frequency coordinates."""
        from .views import MultiSweepView  # noqa: PLC0415

        return MultiSweepView(self.sweep.data_source, self.sweep.mapper)

    @functools.cached_property
    def timestream(self) -> TimestreamView:
        """I/Q timestream view with a shared time coordinate."""
        from .views import TimestreamView  # noqa: PLC0415

        return self._make_view_from_mapped_source(
            self.mapper.schema.timestream_subpath, TimestreamView
        )

    @functools.cached_property
    def multi_timestream(self) -> MultiTimestreamView:
        """Multi-channel I/Q timestream view with time coordinates."""
        from .views import MultiTimestreamView  # noqa: PLC0415

        return MultiTimestreamView(self.timestream.data_source, self.timestream.mapper)

    @property
    def d21(self) -> D21View:
        """D21 namespace view with its producing analysis and derivative arrays."""
        from ..analysis.d21 import D21View  # noqa: PLC0415

        return D21View(cast("xr.DataTree", self.data_source))

    @property
    def psd(self) -> PSDView:
        """PSD namespace view with its producing analysis and spectral statistics."""
        from ..analysis.psd import PSDView  # noqa: PLC0415

        return PSDView(cast("xr.DataTree", self.data_source))
