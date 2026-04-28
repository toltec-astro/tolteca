"""Tests for dataset utilities module."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from astropy import units as u

from tolteca_kidsproc.accessors.dataset import make_kids_dataset, open_datatree
from tolteca_kidsproc.accessors.kids import KidsMapper


@pytest.fixture
def rng():
    """Provide numpy random number generator for reproducible tests."""
    return np.random.default_rng(seed=42)


class TestMakeKidsDataset:
    """Tests for make_kids_dataset() function."""

    def test_1d_frequency_only(self):
        """Test creating 1-D sweep dataset with frequency."""
        freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
        s21 = 1 / (1 + 1j * (freq.value - 1.05e9) / 1e6)

        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq)

        assert ds.sizes == {"sweep": 101}
        assert "frequency" in ds.coords
        assert "I" in ds.data_vars
        assert "Q" in ds.data_vars
        assert ds.frequency.attrs["units"] == "Hz"
        np.testing.assert_array_equal(ds.frequency.values, freq.value)

    def test_2d_frequency_only(self, rng):
        """Test creating 2-D sweep dataset with frequency."""
        freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
        freq_2d = (np.ones((5, 1)) * freq.value) << u.Hz
        s21 = rng.standard_normal((5, 101)) + 1j * rng.standard_normal((5, 101))

        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq_2d)

        assert ds.sizes == {"chan": 5, "sweep": 101}
        assert "frequency" in ds.coords
        assert ds.frequency.dims == ("chan", "sweep")

    def test_1d_time_only(self, rng):
        """Test creating 1-D timestream dataset with time."""
        time = np.linspace(0, 10, 1000) << u.s
        s21 = rng.standard_normal(1000) + 1j * rng.standard_normal(1000)

        ds = make_kids_dataset(s21.real, s21.imag, time=time)

        assert ds.sizes == {"time": 1000}
        assert "time" in ds.coords
        assert "I" in ds.data_vars
        assert "Q" in ds.data_vars
        assert ds.time.attrs["units"] == "s"
        np.testing.assert_array_equal(ds.time.values, time.value)

    def test_2d_time_only(self, rng):
        """Test creating 2-D timestream dataset with time."""
        time = np.linspace(0, 10, 1000) << u.s
        time_2d = (np.ones((10, 1)) * time.value) << u.s
        s21 = rng.standard_normal((10, 1000)) + 1j * rng.standard_normal((10, 1000))

        ds = make_kids_dataset(s21.real, s21.imag, time=time_2d)

        assert ds.sizes == {"chan": 10, "time": 1000}
        assert "time" in ds.coords
        assert ds.time.dims == ("chan", "time")

    def test_1d_both_frequency_and_time(self, rng):
        """Test creating 1-D dataset with both frequency and time."""
        freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
        time = np.linspace(0, 1, 101) << u.s
        s21 = rng.standard_normal(101) + 1j * rng.standard_normal(101)

        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq, time=time)

        # When both provided, time sets dims first, frequency uses those dims
        assert ds.sizes == {"time": 101}
        assert "frequency" in ds.coords
        assert "time" in ds.coords
        assert ds.frequency.dims == ("time",)
        assert ds.time.dims == ("time",)

    def test_2d_both_frequency_and_time(self, rng):
        """Test creating 2-D dataset with both frequency and time."""
        freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
        freq_2d = (np.ones((5, 1)) * freq.value) << u.Hz
        time = np.linspace(0, 1, 101) << u.s
        time_2d = (np.ones((5, 1)) * time.value) << u.s
        s21 = rng.standard_normal((5, 101)) + 1j * rng.standard_normal((5, 101))

        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq_2d, time=time_2d)

        assert ds.sizes == {"chan": 5, "time": 101}
        assert "frequency" in ds.coords
        assert "time" in ds.coords
        assert ds.frequency.dims == ("chan", "time")
        assert ds.time.dims == ("chan", "time")

    def test_1d_neither_frequency_nor_time(self):
        """Test creating 1-D S21-only dataset (no frequency or time)."""
        s21 = np.array([1 + 2j, 3 + 4j, 5 + 6j])

        ds = make_kids_dataset(s21.real, s21.imag)  # pyright: ignore[reportAttributeAccessIssue]

        assert ds.sizes == {"sample": 3}
        assert "frequency" not in ds.coords
        assert "time" not in ds.coords
        assert "I" in ds.data_vars
        assert "Q" in ds.data_vars

    def test_2d_neither_frequency_nor_time(self, rng):
        """Test creating 2-D S21-only dataset (no frequency or time)."""
        s21 = rng.standard_normal((5, 100)) + 1j * rng.standard_normal((5, 100))

        ds = make_kids_dataset(s21.real, s21.imag)

        assert ds.sizes == {"chan": 5, "sample": 100}
        assert "frequency" not in ds.coords
        assert "time" not in ds.coords

    def test_custom_mapper(self, rng):
        """Test using custom mapper for field names."""
        mapper = KidsMapper.from_defaults()
        freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
        s21 = rng.standard_normal(101) + 1j * rng.standard_normal(101)

        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq, mapper=mapper)

        schema = mapper.schema
        i_key = mapper.get_name(schema.I)
        q_key = mapper.get_name(schema.Q)
        f_key = mapper.get_name(schema.frequency)

        assert i_key in ds.data_vars
        assert q_key in ds.data_vars
        assert f_key in ds.coords

    def test_frequency_units_conversion(self, rng):
        """Test frequency with different units."""
        freq_mhz = np.linspace(1000, 1100, 101) << u.MHz
        s21 = rng.standard_normal(101) + 1j * rng.standard_normal(101)

        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq_mhz)

        assert ds.frequency.attrs["units"] == "MHz"
        np.testing.assert_array_equal(ds.frequency.values, freq_mhz.value)

    def test_time_units_conversion(self, rng):
        """Test time with different units."""
        time_ms = np.linspace(0, 10000, 1000) << u.ms
        s21 = rng.standard_normal(1000) + 1j * rng.standard_normal(1000)

        ds = make_kids_dataset(s21.real, s21.imag, time=time_ms)

        assert ds.time.attrs["units"] == "ms"
        np.testing.assert_array_equal(ds.time.values, time_ms.value)

    def test_mismatched_i_q_shape(self, rng):
        """Test error when I and Q have different shapes."""
        i_data = rng.standard_normal(100)
        q_data = rng.standard_normal(50)

        with pytest.raises(ValueError, match="I and Q data must have same shape"):
            make_kids_dataset(i_data, q_data)

    def test_mismatched_frequency_shape(self, rng):
        """Test error when frequency shape doesn't match I/Q."""
        freq = np.linspace(1e9, 1.1e9, 50) << u.Hz
        s21 = rng.standard_normal(100) + 1j * rng.standard_normal(100)

        with pytest.raises(ValueError, match="Frequency must match I/Q shape"):
            make_kids_dataset(s21.real, s21.imag, frequency=freq)

    def test_mismatched_time_shape(self, rng):
        """Test error when time shape doesn't match I/Q."""
        time = np.linspace(0, 10, 50) << u.s
        s21 = rng.standard_normal(100) + 1j * rng.standard_normal(100)

        with pytest.raises(ValueError, match="Frequency must match I/Q shape"):
            make_kids_dataset(s21.real, s21.imag, time=time)

    def test_invalid_frequency_units(self, rng):
        """Test error when frequency has non-frequency units."""
        freq = np.linspace(0, 10, 100) << u.s  # Wrong unit type

        s21 = rng.standard_normal(100) + 1j * rng.standard_normal(100)

        with pytest.raises(ValueError, match="frequency must have frequency units"):
            make_kids_dataset(s21.real, s21.imag, frequency=freq)

    def test_invalid_time_units(self, rng):
        """Test error when time has non-time units."""
        time = np.linspace(1e9, 1.1e9, 100) << u.Hz  # Wrong unit type
        s21 = rng.standard_normal(100) + 1j * rng.standard_normal(100)

        with pytest.raises(ValueError, match="time must have time units"):
            make_kids_dataset(s21.real, s21.imag, time=time)

    def test_3d_data_error(self, rng):
        """Test error when data is 3-D."""
        s21 = rng.standard_normal((5, 10, 100)) + 1j * rng.standard_normal((5, 10, 100))

        with pytest.raises(ValueError, match="Data must be 1-D or 2-D, got 3-D"):
            make_kids_dataset(s21.real, s21.imag)

    def test_returns_kids_dataset_type(self, rng):
        """Test that function returns KidsDataset type for IDE support."""
        freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
        s21 = rng.standard_normal(101) + 1j * rng.standard_normal(101)

        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq)

        # Runtime: should be xarray Dataset
        assert isinstance(ds, xr.Dataset)
        # Type checking would verify KidsDataset type
        assert hasattr(ds, "kids")

    def test_accessor_available(self, rng):
        """Test that .kids accessor is available on created dataset."""
        freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
        s21 = rng.standard_normal(101) + 1j * rng.standard_normal(101)

        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq)

        # Check accessor is registered and accessible
        assert hasattr(ds, "kids")
        assert hasattr(ds.kids, "sweep")


class TestOpenDataTree:
    """Tests for open_datatree() function."""

    def test_open_datatree_wrapper(self, tmp_path, rng):
        """Test that open_datatree wraps xr.open_datatree."""
        # Create a test dataset file
        freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
        s21 = rng.standard_normal(101) + 1j * rng.standard_normal(101)
        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq)

        filepath = tmp_path / "test.nc"
        ds.to_netcdf(filepath)

        # Open with our wrapper
        loaded_dt = open_datatree(filepath)

        assert isinstance(loaded_dt, xr.DataTree)
        assert "I" in loaded_dt.ds.data_vars
        assert "Q" in loaded_dt.ds.data_vars
        assert "frequency" in loaded_dt.ds.coords

    def test_open_datatree_accessor_available(self, tmp_path, rng):
        """Test that .kids accessor is available on opened datatree."""
        # Create and save a dataset
        freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
        s21 = rng.standard_normal(101) + 1j * rng.standard_normal(101)
        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq)

        filepath = tmp_path / "test.nc"
        ds.to_netcdf(filepath)

        # Open and verify accessor
        loaded_dt = open_datatree(filepath)

        assert hasattr(loaded_dt, "kids")
        assert hasattr(loaded_dt.kids, "sweep")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_data(self):
        """Test handling of empty arrays."""
        i_data = np.array([])
        q_data = np.array([])

        ds = make_kids_dataset(i_data, q_data)

        assert ds.sizes == {"sample": 0}

    def test_single_point_sweep(self):
        """Test single-point sweep."""
        freq = np.array([1e9]) << u.Hz
        s21 = np.array([1 + 1j])

        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq)  # pyright: ignore[reportAttributeAccessIssue]

        assert ds.sizes == {"sweep": 1}

    def test_single_channel_2d(self, rng):
        """Test 2-D data with single channel."""
        freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
        freq_2d = freq.reshape(1, -1)
        s21 = rng.standard_normal((1, 101)) + 1j * rng.standard_normal((1, 101))

        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq_2d)  # pyright: ignore[reportArgumentType]

        assert ds.sizes == {"chan": 1, "sweep": 101}

    def test_very_long_timestream(self, rng):
        """Test large timestream data."""
        n_samples = 100000
        time = np.linspace(0, 100, n_samples) << u.s
        s21 = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)

        ds = make_kids_dataset(s21.real, s21.imag, time=time)

        assert ds.sizes == {"time": n_samples}

    def test_many_channels(self, rng):
        """Test multi-channel with many channels."""
        n_chans = 1000
        n_samples = 100
        freq = np.linspace(1e9, 1.1e9, n_samples) << u.Hz
        freq_2d = (np.ones((n_chans, 1)) * freq.value) << u.Hz
        s21 = rng.standard_normal((n_chans, n_samples)) + 1j * rng.standard_normal(
            (n_chans, n_samples),
        )

        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq_2d)

        assert ds.sizes == {"chan": n_chans, "sweep": n_samples}


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_round_trip_netcdf_frequency(self, tmp_path, rng):
        """Test creating, saving, and loading sweep dataset."""
        freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
        s21 = rng.standard_normal(101) + 1j * rng.standard_normal(101)

        # Create dataset
        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq)

        # Save to file
        filepath = tmp_path / "sweep.nc"
        ds.to_netcdf(filepath)

        # Load back
        loaded = open_datatree(filepath)

        # Verify
        assert loaded.ds.sizes == ds.sizes
        assert list(loaded.ds.coords.keys()) == list(ds.coords.keys())
        np.testing.assert_array_equal(loaded.ds.I.values, ds.I.values)
        np.testing.assert_array_equal(loaded.ds.Q.values, ds.Q.values)

    def test_round_trip_netcdf_time(self, tmp_path, rng):
        """Test creating, saving, and loading timestream dataset."""
        time = np.linspace(0, 10, 1000) << u.s
        s21 = rng.standard_normal(1000) + 1j * rng.standard_normal(1000)

        # Create dataset
        ds = make_kids_dataset(s21.real, s21.imag, time=time)

        # Save to file
        filepath = tmp_path / "timestream.nc"
        ds.to_netcdf(filepath)

        # Load back
        loaded = open_datatree(filepath)

        # Verify
        assert loaded.ds.sizes == ds.sizes
        assert "time" in loaded.ds.coords
        np.testing.assert_array_equal(loaded.ds.time.values, ds.time.values)

    def test_round_trip_netcdf_both(self, tmp_path, rng):
        """Test creating, saving, and loading dataset with both coords."""
        freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
        time = np.linspace(0, 1, 101) << u.s
        s21 = rng.standard_normal(101) + 1j * rng.standard_normal(101)

        # Create dataset
        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq, time=time)

        # Save to file
        filepath = tmp_path / "both.nc"
        ds.to_netcdf(filepath)

        # Load back
        loaded = open_datatree(filepath)

        # Verify both coordinates preserved
        assert "frequency" in loaded.ds.coords
        assert "time" in loaded.ds.coords
        np.testing.assert_array_equal(loaded.ds.frequency.values, ds.frequency.values)
        np.testing.assert_array_equal(loaded.ds.time.values, ds.time.values)

    def test_accessor_workflow_frequency(self):
        """Test complete workflow with frequency dataset."""
        freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
        s21 = 1 / (1 + 1j * (freq.value - 1.05e9) / 1e6)

        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq)

        # Access sweep view
        sweep = ds.kids.sweep

        # Verify properties
        assert sweep.S21.shape == (101,)
        assert sweep.frequency.shape == (101,)
        np.testing.assert_array_almost_equal(sweep.I.values, s21.real)
        np.testing.assert_array_almost_equal(sweep.Q.values, s21.imag)

    def test_accessor_workflow_time(self, rng):
        """Test complete workflow with time dataset."""
        time = np.linspace(0, 10, 1000) << u.s
        s21 = rng.standard_normal(1000) + 1j * rng.standard_normal(1000)

        ds = make_kids_dataset(s21.real, s21.imag, time=time)

        # Access timestream view
        ts = ds.kids.timestream

        # Verify properties
        assert ts.S21.shape == (1000,)
        assert ts.time.shape == (1000,)

    def test_accessor_workflow_s21_only(self, rng):
        """Test complete workflow with S21-only dataset."""
        s21 = rng.standard_normal(100) + 1j * rng.standard_normal(100)

        ds = make_kids_dataset(s21.real, s21.imag)

        # Access S21 view
        s21_view = ds.kids.s21

        # Verify properties
        assert s21_view.S21.shape == (100,)
        assert s21_view.I.shape == (100,)
        assert s21_view.Q.shape == (100,)
