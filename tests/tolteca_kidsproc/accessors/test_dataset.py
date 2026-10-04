"""Tests for dataset utilities module."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr
from astropy import units as u
from tollan.accessor import NameMapping

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
        assert ds.kids.timestream.f_smp.to_value(u.Hz) == pytest.approx(99.9)

    def test_2d_time_only(self, rng):
        """Test creating 2-D timestream dataset with time."""
        time = np.linspace(0, 10, 1000) << u.s
        s21 = rng.standard_normal((10, 1000)) + 1j * rng.standard_normal((10, 1000))

        ds = make_kids_dataset(s21.real, s21.imag, time=time, f_smp=99.9 << u.Hz)

        assert ds.sizes == {"chan": 10, "time": 1000}
        assert "time" in ds.coords
        assert ds.time.dims == ("time",)
        np.testing.assert_array_equal(ds.time.values, time.value)
        assert ds.kids.multi_timestream.time.size == 1000
        assert ds.isel(chan=0).time.identical(ds.time)

    def test_multi_channel_time_must_be_shared(self):
        """Per-channel time arrays are not a shared sample axis."""
        signal = np.zeros((3, 16))
        time = np.broadcast_to(np.arange(16), signal.shape) * u.s
        with pytest.raises(ValueError, match="Time must be 1-D"):
            make_kids_dataset(signal, signal, time=time)

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
        s21 = rng.standard_normal((5, 101)) + 1j * rng.standard_normal((5, 101))

        ds = make_kids_dataset(s21.real, s21.imag, frequency=freq_2d, time=time)

        assert ds.sizes == {"chan": 5, "time": 101}
        assert "frequency" in ds.coords
        assert "time" in ds.coords
        assert ds.frequency.dims == ("chan", "time")
        assert ds.time.dims == ("time",)

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

    @pytest.mark.parametrize("provided", ["time", "rate", "both"])
    def test_sampling_coordinates_use_mapper_and_units(self, monkeypatch, provided):
        schema = replace(
            KidsMapper.schema,
            time=NameMapping.var("clock"),
            f_smp=NameMapping.attr("sample_rate"),
        )
        monkeypatch.setattr(KidsMapper, "schema", schema)
        sampling = {}
        if provided in ("time", "both"):
            sampling["time"] = (5000 + np.arange(32) * 10) << u.ms
        if provided in ("rate", "both"):
            sampling["f_smp"] = 0.1 << u.kHz
        source = make_kids_dataset(np.ones(32), np.zeros(32), **sampling)
        assert source.attrs["sample_rate"] == pytest.approx(100)
        assert "f_smp" not in source.attrs
        view = source.kids.timestream
        assert view.f_smp.to_value(u.Hz) == pytest.approx(100)
        assert view.time.name == "clock"
        offset = 0 if provided == "rate" else 5
        np.testing.assert_allclose(view.time.u.to(u.s), offset + np.arange(32) / 100)
        assert view.time.u.unit == (u.s if provided == "rate" else u.ms)

    def test_rate_constructs_shared_time_for_multiple_channels(self):
        signal = np.ones((3, 32))
        source = make_kids_dataset(signal, signal, f_smp=128 << u.Hz)
        assert source.I.dims == ("chan", "time")
        assert source.kids.multi_timestream.time.dims == ("time",)
        np.testing.assert_array_equal(source.time, np.arange(32) / 128)

    @pytest.mark.parametrize("recorded", [False, True])
    @pytest.mark.parametrize(
        ("times", "message"),
        [
            ([0, 0.01, 0.03], "uniformly spaced"),
            ([0, 0.01, 0.025], "uniformly spaced"),
            ([0.02, 0.01, 0], "strictly increasing"),
            ([0, 0.01, 0.01], "strictly increasing"),
            ([0, np.nan, 0.02], "finite"),
        ],
        ids=["missing_sample", "irregular", "decreasing", "duplicate", "nonfinite"],
    )
    def test_invalid_sampling_rejected_at_construction(self, times, message, recorded):
        signal = np.ones(3)
        rate = 100 << u.Hz if recorded else None
        with pytest.raises(ValueError, match=message):
            make_kids_dataset(signal, signal, time=times << u.s, f_smp=rate)

    def test_time_and_rate_must_agree(self):
        signal = np.ones(32)
        with pytest.raises(ValueError, match="agree with f_smp"):
            make_kids_dataset(
                signal, signal, time=np.arange(32) / 100 << u.s, f_smp=50 << u.Hz
            )

    @pytest.mark.parametrize("rate", [0, -100, np.nan, np.inf])
    def test_invalid_sampling_rate(self, rate):
        signal = np.ones(32)
        with pytest.raises(ValueError, match="finite and positive"):
            make_kids_dataset(signal, signal, f_smp=rate << u.Hz)

    def test_sampling_rate_requires_frequency_units(self):
        signal = np.ones(32)
        with pytest.raises(u.UnitConversionError):
            make_kids_dataset(signal, signal, f_smp=100 << u.s)

    def test_single_sample_requires_explicit_rate(self):
        signal = np.ones(1)
        time = np.array([5]) << u.s
        with pytest.raises(ValueError, match="at least two"):
            make_kids_dataset(signal, signal, time=time)
        source = make_kids_dataset(signal, signal, time=time, f_smp=100 << u.Hz)
        assert source.kids.timestream.f_smp == 100 << u.Hz
        np.testing.assert_array_equal(source.time, [5])

    @pytest.mark.parametrize("kind", ["sweep", "timestream"])
    def test_custom_channel_dimension(self, monkeypatch, kind):
        schema = replace(KidsMapper.schema, chan_dim=NameMapping.dim("detector"))
        monkeypatch.setattr(KidsMapper, "schema", schema)
        signal = np.ones((3, 16))
        coordinate = (
            {"frequency": np.broadcast_to(np.arange(16) + 1e9, signal.shape) * u.Hz}
            if kind == "sweep"
            else {"time": np.arange(16) * u.s, "f_smp": 1 << u.Hz}
        )
        source = make_kids_dataset(
            signal, signal, mapper=KidsMapper.from_defaults(), **coordinate
        )
        assert source.I.dims[0] == "detector"
        assert source.sizes["detector"] == 3
        assert "chan" not in source.dims
        for data in (source, source.transpose()):
            view = getattr(data.kids, f"multi_{kind}")
            assert view.n_chans == 3
            assert view.I.data is data.I.data

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

        with pytest.raises(ValueError, match="Time must be 1-D with length matching"):
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
        with open_datatree(filepath) as loaded_dt:
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
        with open_datatree(filepath) as loaded_dt:
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
        with open_datatree(filepath) as loaded:
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
        with open_datatree(filepath) as loaded:
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
        with open_datatree(filepath) as loaded:
            # Verify both coordinates preserved
            assert "frequency" in loaded.ds.coords
            assert "time" in loaded.ds.coords
            np.testing.assert_array_equal(
                loaded.ds.frequency.values, ds.frequency.values
            )
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

        ds = make_kids_dataset(s21.real, s21.imag, time=time, f_smp=99.9 << u.Hz)

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
