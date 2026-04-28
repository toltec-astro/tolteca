"""Tests for KIDs data accessor and views."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from astropy import units as u

from tolteca_kidsproc.accessors.dataset import make_kids_dataset
from tolteca_kidsproc.accessors.kids import (
    KidsAccessor,
    KidsMapper,
    KidsSchema,
)
from tolteca_kidsproc.accessors.views import (
    MultiSweepView,
    MultiTimestreamView,
    S21View,
    SweepView,
    TimestreamView,
)

# Fixtures
# ========


@pytest.fixture
def rng():
    """Provide numpy random number generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sweep_dataset_1d():
    """Create a 1-D sweep dataset (single channel)."""
    n_tones = 101
    freq = np.linspace(1.0e9, 1.1e9, n_tones) << u.Hz

    # Create Lorentzian resonance
    f0 = 1.05e9
    Q = 1e5
    s21 = 1 / (1 + 1j * Q * (freq.value - f0) / f0)

    return make_kids_dataset(s21.real, s21.imag, frequency=freq)


@pytest.fixture
def sweep_dataset_2d():
    """Create a 2-D sweep dataset (multi-channel)."""
    n_channels = 10
    n_tones = 101

    # Create frequency grid
    f_chans = np.linspace(1.0e9, 1.1e9, n_channels)
    f_sweep = np.linspace(-5e6, 5e6, n_tones)
    frequency = (f_sweep[np.newaxis, :] + f_chans[:, np.newaxis]) << u.Hz

    # Create synthetic Lorentzian resonances for each channel
    i_data = np.zeros((n_channels, n_tones))
    q_data = np.zeros((n_channels, n_tones))
    for i in range(n_channels):
        f0 = f_chans[i]
        Q = 1e5 + i * 1e4  # Vary Q factor per channel
        s21 = 1 / (1 + 1j * Q * (frequency.value[i, :] - f0) / f0)
        i_data[i, :] = s21.real
        q_data[i, :] = s21.imag

    return make_kids_dataset(i_data, q_data, frequency=frequency)


@pytest.fixture
def timestream_dataset_1d():
    """Create a 1-D timestream dataset (single channel)."""
    n_samples = 1000
    time = (np.arange(n_samples) * 0.001) << u.s  # 1 ms sampling

    # Create sinusoidal signal with small deterministic noise
    noise_scale = 0.01
    # Use deterministic pseudo-noise based on sample index
    noise_i = noise_scale * np.sin(time.value * 123.4) * np.cos(time.value * 456.7)
    noise_q = noise_scale * np.sin(time.value * 234.5) * np.cos(time.value * 567.8)

    i_data = np.cos(2 * np.pi * 10 * time.value) + noise_i
    q_data = np.sin(2 * np.pi * 10 * time.value) + noise_q

    return make_kids_dataset(i_data, q_data, time=time)


@pytest.fixture
def timestream_dataset_2d():
    """Create a 2-D timestream dataset (multi-channel)."""
    n_channels = 5
    n_samples = 1000
    time = np.arange(n_samples) * 0.001
    # Broadcast time to 2D to match data shape
    time_2d = (np.ones((n_channels, 1)) * time) << u.s

    # Create multi-frequency deterministic timestreams
    i_data = np.zeros((n_channels, n_samples))
    q_data = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        freq = 10 + i * 5  # Different frequency per channel
        phase = i * np.pi / n_channels  # Different phase per channel
        i_data[i, :] = np.cos(2 * np.pi * freq * time + phase) * (1 + 0.1 * i)
        q_data[i, :] = np.sin(2 * np.pi * freq * time + phase) * (1 + 0.1 * i)

    return make_kids_dataset(i_data, q_data, time=time_2d)


# KidsSchema Tests
# ================


class TestKidsSchema:
    """Test KidsSchema field definitions."""

    def test_schema_fields(self):
        """Test schema has expected fields."""
        schema = KidsSchema()
        assert hasattr(schema, "I")
        assert hasattr(schema, "Q")
        assert hasattr(schema, "frequency")
        assert hasattr(schema, "time")

    def test_field_names(self):
        """Test field mapping names."""
        schema = KidsSchema()
        assert schema.I.names == ("I",)
        assert schema.Q.names == ("Q",)
        assert schema.frequency.names == ("frequency",)
        assert schema.time.names == ("time",)


# KidsMapper Tests
# ================


class TestKidsMapper:
    """Test KidsMapper validation methods."""

    def test_mapper_creation(self, sweep_dataset_1d):
        """Test mapper can be created from dataset."""
        mapper = KidsMapper.from_data_source(sweep_dataset_1d)
        assert isinstance(mapper, KidsMapper)
        assert mapper.schema.I in mapper
        assert mapper.schema.Q in mapper
        assert mapper.schema.frequency in mapper

    def test_validate_has_field_success(self, sweep_dataset_1d):
        """Test validate_has_field succeeds for existing field."""
        mapper = KidsMapper.from_data_source(sweep_dataset_1d)
        mapper.validate_has_field(mapper.schema.I)  # Should not raise

    def test_validate_has_field_missing(self):
        """Test validate_has_field fails for missing field."""
        ds = xr.Dataset({"I": (["tone"], np.zeros(10))})
        mapper = KidsMapper.from_data_source(ds)

        with pytest.raises(ValueError, match="Missing required field 'Q'"):
            mapper.validate_has_field(mapper.schema.Q)

    def test_validate_ndim_1d(self, sweep_dataset_1d):
        """Test validate_ndim for 1-D data."""
        mapper = KidsMapper.from_data_source(sweep_dataset_1d)
        mapper.validate_ndim(sweep_dataset_1d, mapper.schema.I, 1)

    def test_validate_ndim_2d(self, sweep_dataset_2d):
        """Test validate_ndim for 2-D data."""
        mapper = KidsMapper.from_data_source(sweep_dataset_2d)
        mapper.validate_ndim(sweep_dataset_2d, mapper.schema.I, 2)

    def test_validate_ndim_mismatch(self, sweep_dataset_1d):
        """Test validate_ndim fails for wrong dimensionality."""
        mapper = KidsMapper.from_data_source(sweep_dataset_1d)

        with pytest.raises(ValueError, match=r"Field 'I' must be 2-D, got 1-D"):
            mapper.validate_ndim(sweep_dataset_1d, mapper.schema.I, 2)


# SweepView Tests
# ===============


class TestSweepView:
    """Test single-channel sweep view."""

    def test_sweep_view_creation(self, sweep_dataset_1d):
        """Test SweepView can be created from 1-D sweep data."""
        view = SweepView(sweep_dataset_1d)
        assert isinstance(view, SweepView)

    def test_sweep_view_I_Q(self, sweep_dataset_1d):
        """Test I and Q properties."""
        view = sweep_dataset_1d.kids.sweep

        assert isinstance(view.I, xr.DataArray)
        assert isinstance(view.Q, xr.DataArray)
        assert view.I.shape == (101,)
        assert view.Q.shape == (101,)

    def test_sweep_view_S21(self, sweep_dataset_1d):
        """Test S21 complex property."""
        view = sweep_dataset_1d.kids.sweep
        s21 = view.S21

        assert isinstance(s21, xr.DataArray)
        assert s21.dtype == np.complex128
        assert s21.shape == (101,)

        # Verify S21 = I + 1j*Q
        np.testing.assert_allclose(s21.real, view.I.values)
        np.testing.assert_allclose(s21.imag, view.Q.values)

    def test_sweep_view_aS21(self, sweep_dataset_1d):
        """Test absolute value |S21|."""
        view = sweep_dataset_1d.kids.sweep
        as21 = view.aS21

        assert isinstance(as21, xr.DataArray)
        assert as21.dtype == np.float64
        assert np.all(as21.values >= 0)

        # Verify |S21| = sqrt(I^2 + Q^2)
        expected = np.sqrt(view.I.values**2 + view.Q.values**2)
        np.testing.assert_allclose(as21.values, expected)

    def test_sweep_view_aS21_db(self, sweep_dataset_1d):
        """Test amplitude in dB."""
        view = sweep_dataset_1d.kids.sweep
        as21_db = view.aS21_db

        assert isinstance(as21_db, xr.DataArray)

        # Verify dB = 20*log10(|S21|)
        expected = 20 * np.log10(view.aS21.values)
        np.testing.assert_allclose(as21_db.values, expected)

    def test_sweep_view_frequency(self, sweep_dataset_1d):
        """Test frequency coordinate access."""
        view = sweep_dataset_1d.kids.sweep
        freq = view.frequency

        assert isinstance(freq, xr.DataArray)
        assert freq.shape == (101,)
        assert freq.min() == pytest.approx(1.0e9)
        assert freq.max() == pytest.approx(1.1e9)

    def test_sweep_view_validation_missing_frequency(self):
        """Test validation fails without frequency coordinate."""
        ds = xr.Dataset(
            {
                "I": (["sweep"], np.zeros(10)),
                "Q": (["sweep"], np.zeros(10)),
            },
        )

        with pytest.raises(ValueError, match="Missing required field 'frequency'"):
            _ = ds.kids.sweep


# MultiSweepView Tests
# ====================


class TestMultiSweepView:
    """Test multi-channel sweep view."""

    def test_multi_sweep_view_creation(self, sweep_dataset_2d):
        """Test MultiSweepView can be created from 2-D sweep data."""
        view = MultiSweepView(sweep_dataset_2d)
        assert isinstance(view, MultiSweepView)

    def test_multi_sweep_view_I_Q(self, sweep_dataset_2d):
        """Test I and Q properties for multi-channel."""
        view = sweep_dataset_2d.kids.multi_sweep

        assert view.I.shape == (10, 101)
        assert view.Q.shape == (10, 101)

    def test_multi_sweep_view_S21(self, sweep_dataset_2d):
        """Test S21 for multi-channel."""
        view = sweep_dataset_2d.kids.multi_sweep
        s21 = view.S21

        assert s21.shape == (10, 101)
        assert s21.dtype == np.complex128

    def test_multi_sweep_view_n_chans(self, sweep_dataset_2d):
        """Test channel count property."""
        view = sweep_dataset_2d.kids.multi_sweep
        assert view.n_chans == 10

    def test_multi_sweep_view_frequency(self, sweep_dataset_2d):
        """Test frequency coordinate for multi-channel."""
        view = sweep_dataset_2d.kids.multi_sweep
        freq = view.frequency

        assert freq.shape == (10, 101)

    def test_multi_sweep_view_validation_wrong_dims(self, sweep_dataset_1d):
        """Test validation fails for 1-D data."""
        with pytest.raises(ValueError, match=r"Field 'I' must be 2-D, got 1-D"):
            _ = sweep_dataset_1d.kids.multi_sweep


# TimestreamView Tests
# ====================


class TestTimestreamView:
    """Test single-channel timestream view."""

    def test_timestream_view_creation(self, timestream_dataset_1d):
        """Test TimestreamView can be created from 1-D timestream data."""
        view = TimestreamView(timestream_dataset_1d)
        assert isinstance(view, TimestreamView)

    def test_timestream_view_I_Q(self, timestream_dataset_1d):
        """Test I and Q properties."""
        view = timestream_dataset_1d.kids.timestream

        assert view.I.shape == (1000,)
        assert view.Q.shape == (1000,)

    def test_timestream_view_S21(self, timestream_dataset_1d):
        """Test S21 complex property."""
        view = timestream_dataset_1d.kids.timestream
        s21 = view.S21

        assert s21.shape == (1000,)
        assert s21.dtype == np.complex128

    def test_timestream_view_time(self, timestream_dataset_1d):
        """Test time coordinate access."""
        view = timestream_dataset_1d.kids.timestream
        time = view.time

        assert isinstance(time, xr.DataArray)
        assert time.shape == (1000,)

    def test_timestream_view_aS21(self, timestream_dataset_1d):
        """Test absolute |S21| on timestream data (inherited from S21View)."""
        view = timestream_dataset_1d.kids.timestream
        as21 = view.aS21

        assert isinstance(as21, xr.DataArray)
        assert as21.dtype == np.float64
        assert np.all(as21.values >= 0)
        np.testing.assert_allclose(
            as21.values,
            np.sqrt(view.I.values**2 + view.Q.values**2),
        )

    def test_timestream_view_aS21_db(self, timestream_dataset_1d):
        """Test amplitude in dB on timestream data (inherited from S21View)."""
        view = timestream_dataset_1d.kids.timestream
        as21_db = view.aS21_db

        assert isinstance(as21_db, xr.DataArray)
        np.testing.assert_allclose(
            as21_db.values,
            20 * np.log10(view.aS21.values),
        )

    def test_timestream_view_validation_missing_time(self):
        """Test validation fails without time coordinate."""
        ds = xr.Dataset(
            {
                "I": (["sample"], np.zeros(100)),
                "Q": (["sample"], np.zeros(100)),
            },
        )

        with pytest.raises(ValueError, match="Missing required field 'time'"):
            _ = ds.kids.timestream


# MultiTimestreamView Tests
# ==========================


class TestMultiTimestreamView:
    """Test multi-channel timestream view."""

    def test_multi_timestream_view_creation(self, timestream_dataset_2d):
        """Test MultiTimestreamView can be created from 2-D timestream data."""
        view = MultiTimestreamView(timestream_dataset_2d)
        assert isinstance(view, MultiTimestreamView)

    def test_multi_timestream_view_I_Q(self, timestream_dataset_2d):
        """Test I and Q properties for multi-channel."""
        view = timestream_dataset_2d.kids.multi_timestream

        assert view.I.shape == (5, 1000)
        assert view.Q.shape == (5, 1000)

    def test_multi_timestream_view_n_chans(self, timestream_dataset_2d):
        """Test channel count property."""
        view = timestream_dataset_2d.kids.multi_timestream
        assert view.n_chans == 5

    def test_multi_timestream_view_S21(self, timestream_dataset_2d):
        """Test S21 for multi-channel timestream (inherited from S21View)."""
        view = timestream_dataset_2d.kids.multi_timestream
        s21 = view.S21

        assert s21.shape == (5, 1000)
        assert s21.dtype == np.complex128

    def test_multi_timestream_view_validation_wrong_dims(self, timestream_dataset_1d):
        """Test validation fails for 1-D data."""
        with pytest.raises(ValueError, match=r"Field 'I' must be 2-D, got 1-D"):
            _ = timestream_dataset_1d.kids.multi_timestream


# KidsAccessor Tests
# ==================


class TestKidsAccessor:
    """Test KidsAccessor integration."""

    def test_accessor_registration(self, sweep_dataset_1d):
        """Test accessor is registered on xarray Dataset."""
        assert hasattr(sweep_dataset_1d, "kids")
        assert isinstance(sweep_dataset_1d.kids, KidsAccessor)

    def test_accessor_mapper(self, sweep_dataset_1d):
        """Test accessor provides mapper access."""
        accessor = sweep_dataset_1d.kids
        assert hasattr(accessor, "mapper")
        assert isinstance(accessor.mapper, KidsMapper)

    def test_accessor_sweep_cached(self, sweep_dataset_1d):
        """Test sweep view is cached."""
        accessor = sweep_dataset_1d.kids
        view1 = accessor.sweep
        view2 = accessor.sweep
        assert view1 is view2  # Same instance

    def test_accessor_multi_sweep_cached(self, sweep_dataset_2d):
        """Test multi_sweep view is cached."""
        accessor = sweep_dataset_2d.kids
        view1 = accessor.multi_sweep
        view2 = accessor.multi_sweep
        assert view1 is view2

    def test_accessor_timestream_cached(self, timestream_dataset_1d):
        """Test timestream view is cached."""
        accessor = timestream_dataset_1d.kids
        view1 = accessor.timestream
        view2 = accessor.timestream
        assert view1 is view2

    def test_accessor_multi_timestream_cached(self, timestream_dataset_2d):
        """Test multi_timestream view is cached."""
        accessor = timestream_dataset_2d.kids
        view1 = accessor.multi_timestream
        view2 = accessor.multi_timestream
        assert view1 is view2

    def test_accessor_s21_cached(self, sweep_dataset_1d):
        """Test s21 view is cached (same instance on repeated access)."""
        accessor = sweep_dataset_1d.kids
        view1 = accessor.s21
        view2 = accessor.s21
        assert view1 is view2

    def test_accessor_s21_returns_s21view(self, sweep_dataset_1d):
        """Test s21 property returns S21View with correct properties."""
        view = sweep_dataset_1d.kids.s21
        assert isinstance(view, S21View)
        assert isinstance(view.S21, xr.DataArray)
        assert isinstance(view.aS21, xr.DataArray)

    def test_accessor_d21_view_error_without_analysis(self, sweep_dataset_1d):
        """Test D21 view raises ValueError when no D21 analysis in dataset."""
        with pytest.raises(ValueError, match="No D21 analysis found"):
            _ = sweep_dataset_1d.kids.d21


# Edge Cases and Error Handling
# ==============================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataset(self):
        """Test accessor on empty dataset."""
        ds = xr.Dataset()
        accessor = ds.kids
        assert isinstance(accessor, KidsAccessor)

        with pytest.raises(ValueError):
            _ = accessor.sweep

    def test_missing_Q_field(self):
        """Test dataset missing Q field."""
        ds = xr.Dataset(
            {
                "I": (["sweep"], np.zeros(10)),
            },
            coords={
                "frequency": (
                    ["sweep"],
                    np.linspace(1e9, 1.1e9, 10),
                    {"units": "Hz"},
                ),
            },
        )

        with pytest.raises(ValueError, match="Missing required field 'Q'"):
            _ = ds.kids.sweep

    def test_inconsistent_IQ_shapes(self):
        """Test I and Q with different shapes."""
        ds = xr.Dataset(
            {
                "I": (["sweep"], np.zeros(10)),
                "Q": (["sample"], np.zeros(20)),
            },
            coords={
                "frequency": (
                    ["sweep"],
                    np.linspace(1e9, 1.1e9, 10),
                    {"units": "Hz"},
                ),
            },
        )

        # Mapper can be created but accessing view properties will fail
        # because Q doesn't have compatible dimensions with frequency
        accessor = ds.kids
        assert isinstance(accessor, KidsAccessor)

    def test_view_inheritance_chain(self, sweep_dataset_2d):
        """Test MultiSweepView inheritance relationships."""
        view = sweep_dataset_2d.kids.multi_sweep

        # MultiSweepView inherits from SweepView which inherits from S21View
        assert isinstance(view, MultiSweepView)
        assert isinstance(view, SweepView)
        assert isinstance(view, S21View)

    def test_timestream_view_inheritance_chain(self, timestream_dataset_2d):
        """Test MultiTimestreamView inheritance relationships."""
        view = timestream_dataset_2d.kids.multi_timestream

        # MultiTimestreamView inherits from TimestreamView which inherits from S21View
        assert isinstance(view, MultiTimestreamView)
        assert isinstance(view, TimestreamView)
        assert isinstance(view, S21View)

    def test_all_views_have_S21_properties(
        self,
        sweep_dataset_1d,
        sweep_dataset_2d,
        timestream_dataset_1d,
        timestream_dataset_2d,
    ):
        """Test all view types provide S21 properties through inheritance."""
        views = [
            sweep_dataset_1d.kids.sweep,
            sweep_dataset_2d.kids.multi_sweep,
            timestream_dataset_1d.kids.timestream,
            timestream_dataset_2d.kids.multi_timestream,
        ]

        for view in views:
            assert hasattr(view, "I")
            assert hasattr(view, "Q")
            assert hasattr(view, "S21")
            assert hasattr(view, "aS21")
            assert hasattr(view, "aS21_db")


# Integration Tests
# =================


class TestIntegration:
    """Integration tests with realistic workflows."""

    def test_sweep_analysis_workflow(self, sweep_dataset_1d):
        """Test typical sweep analysis workflow."""
        # Access sweep data
        sweep = sweep_dataset_1d.kids.sweep

        # Get complex S21
        s21 = sweep.S21

        # Find resonance (minimum |S21|)
        as21 = sweep.aS21
        min_idx = int(np.argmin(as21.data))

        # Get resonance frequency
        f_res = sweep.frequency.isel(sweep=min_idx).item()

        assert 1.0e9 <= f_res <= 1.1e9

    def test_multi_channel_workflow(self, sweep_dataset_2d):
        """Test multi-channel analysis workflow."""
        multi = sweep_dataset_2d.kids.multi_sweep

        # Verify channel count
        assert multi.n_chans == 10

        # Compute average |S21| across all channels
        avg_as21 = multi.aS21.mean(dim="chan")
        assert avg_as21.shape == (101,)

    def test_property_computation_lazy(self, sweep_dataset_1d):
        """Test that properties compute correctly (not testing lazy eval here)."""
        sweep = sweep_dataset_1d.kids.sweep

        # Access multiple derived properties
        s21 = sweep.S21
        as21 = sweep.aS21
        as21_db = sweep.aS21_db

        # Verify they're all DataArrays
        assert all(isinstance(x, xr.DataArray) for x in [s21, as21, as21_db])

    def test_netcdf_round_trip_sweep(self, sweep_dataset_1d, tmp_path):
        """Test dataset can be saved to and loaded from NetCDF with accessor."""
        # Save to NetCDF
        nc_path = tmp_path / "test_sweep.nc"
        sweep_dataset_1d.to_netcdf(nc_path)

        # Load back
        loaded = xr.open_dataset(nc_path)

        # Verify accessor still works
        assert hasattr(loaded, "kids")
        sweep = loaded.kids.sweep

        # Verify units were preserved
        assert loaded.coords["frequency"].attrs["units"] == "Hz"
        assert sweep.frequency.u.unit_str == "Hz"

        # Verify views still work
        assert isinstance(sweep.S21, xr.DataArray)
        assert sweep.S21.shape == (101,)

        # Verify data integrity
        np.testing.assert_allclose(sweep.I.values, sweep_dataset_1d["I"].values)
        np.testing.assert_allclose(sweep.Q.values, sweep_dataset_1d["Q"].values)

        loaded.close()

    def test_netcdf_round_trip_timestream(self, timestream_dataset_1d, tmp_path):
        """Test timestream dataset can be saved to and loaded from NetCDF."""
        # Save to NetCDF
        nc_path = tmp_path / "test_timestream.nc"
        timestream_dataset_1d.to_netcdf(nc_path)

        # Load back
        loaded = xr.open_dataset(nc_path)

        # Verify accessor still works
        timestream = loaded.kids.timestream

        # Verify units were preserved
        assert loaded.coords["time"].attrs["units"] == "s"
        assert timestream.time.u.unit_str == "s"

        # Verify S21 computation
        s21 = timestream.S21
        assert s21.dtype == np.complex128
        assert s21.shape == (1000,)

        loaded.close()
