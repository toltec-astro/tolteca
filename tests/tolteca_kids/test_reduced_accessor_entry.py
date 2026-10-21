"""Reduced sweep and timestream view entry points."""

import numpy as np
import xarray as xr

from tolteca_kidsproc.reduction import reduce_sweep_samples

SWEEP_SUBPATH = "sweep"


class TestReducedSweepAccessorEntry:
    """Test dt.toltec_kids.sweep entry point."""

    def _make_sweep_dt(self) -> xr.DataTree:

        n_chans = 10
        n_sweeps = 5
        n_sps = 4  # samples per sweep step
        n_times = n_sweeps * n_sps

        f_lo_center = 150e6
        f_sweep = np.linspace(-2e6, 2e6, n_sweeps)
        f_lo = np.repeat(f_sweep + f_lo_center, n_sps)

        rng = np.random.default_rng(0)
        I = rng.standard_normal((n_times, n_chans))  # noqa: E741 -- Physical I/Q pair.
        Q = rng.standard_normal((n_times, n_chans))

        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (["time", "iqlen"], I),
                "Data.Toltec.Qs": (["time", "iqlen"], Q),
                "Data.Toltec.LoFreq": (["time"], f_lo),
            },
            attrs={"Header.Toltec.LoCenterFreq": f_lo_center},
        )
        reduced = reduce_sweep_samples(
            ds["Data.Toltec.Is"],
            ds["Data.Toltec.Qs"],
            sample_dim="time",
            sample_slices=[slice(i * n_sps, (i + 1) * n_sps) for i in range(n_sweeps)],
        )
        return xr.DataTree.from_dict({"sweep": reduced})

    def test_accessor_registered_on_datatree(self):
        """A reduced sweep tree exposes the processing accessor."""
        dt = self._make_sweep_dt()
        assert hasattr(dt, "toltec_kids")

    def test_reduced_sweep_returns_view(self):
        """dt.toltec_kids.sweep returns a ReducedSweepView."""
        from tolteca_kids import ReducedSweepView

        dt = self._make_sweep_dt()
        view = dt.toltec_kids.sweep
        assert isinstance(view, ReducedSweepView)

    def test_reduced_sweep_view_has_data(self):
        """View resolves the sweep child node and exposes I/Q data."""
        dt = self._make_sweep_dt()
        view = dt.toltec_kids.sweep
        assert view.I is not None
        assert view.Q is not None

    def test_reduced_sweep_view_matches_child_directly(self):
        """View data matches accessing the child node directly."""
        dt = self._make_sweep_dt()
        view = dt.toltec_kids.sweep
        child_ds = dt.children[SWEEP_SUBPATH].dataset

        np.testing.assert_array_equal(
            view.I.values,
            child_ds["I"].values,
        )

    def test_reduced_sweep_from_raw_dataset(self):
        """toltec_kids accessor on a plain Dataset returns a view (no child data)."""
        from tolteca_kids import ReducedSweepView

        # Raw dataset has no reduced child — view is constructed but I is None
        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (["time", "iqlen"], np.zeros((20, 4))),
                "Data.Toltec.Qs": (["time", "iqlen"], np.zeros((20, 4))),
            }
        )
        view = ReducedSweepView(ds)
        assert isinstance(view, ReducedSweepView)
        # No reduced data present → I should be None
        assert view.I is None


class TestReducedTimestreamAccessorEntry:
    """Test dt.toltec_kids.timestream entry point."""

    def _make_ts_dt(self) -> xr.DataTree:

        n_samples = 2000
        fsmp = 200.0
        rng = np.random.default_rng(1)

        ds = xr.Dataset(
            {
                "I": (["time"], rng.standard_normal(n_samples)),
                "Q": (["time"], rng.standard_normal(n_samples)),
            }
        )
        ds["time"] = xr.DataArray(np.arange(n_samples) / fsmp, dims="time").u.set("s")
        ds.attrs["f_smp"] = fsmp
        from tolteca_kidsproc.analysis import PSDAnalysis

        return PSDAnalysis(nperseg=256)(ds.kids.timestream)

    def test_accessor_registered_on_datatree(self):
        """A PSD analysis tree exposes the processing accessor."""
        dt = self._make_ts_dt()
        assert hasattr(dt, "toltec_kids")

    def test_reduced_timestream_returns_view(self):
        """dt.toltec_kids.timestream returns a TimestreamView."""
        from tolteca_kids import TimestreamView

        dt = self._make_ts_dt()
        view = dt.toltec_kids.timestream
        assert isinstance(view, TimestreamView)

    def test_reduced_timestream_view_has_psd(self):
        """View resolves the timestream child node and exposes PSD data."""
        dt = self._make_ts_dt()
        view = dt.toltec_kids.timestream
        assert view.f_psd is not None
        assert view.I_psd is not None
        assert view.Q_psd is not None

    def test_reduced_timestream_view_matches_child_directly(self):
        """View f_psd matches accessing the child node directly."""
        dt = self._make_ts_dt()
        view = dt.toltec_kids.timestream
        spectra = dt.kids.psd

        np.testing.assert_array_equal(
            view.f_psd.values,
            spectra.f_psd.values,
        )

    def test_native_data_source_on_processing_tree(self):
        """Processing accessor exposes its mapped native dataset."""
        dt = self._make_ts_dt()
        acc = dt.toltec_kids
        xr.testing.assert_identical(acc.data_source, dt.dataset)
        assert acc.tree is dt
