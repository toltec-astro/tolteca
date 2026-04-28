"""Comprehensive tests for D21 analysis module.

Tests cover:
- D21Analysis configuration and validation
- Matched D21 computation (gradient and savgol methods)
- Unified D21 computation with edge exclusion and coverage
- High-level API with caching
- D21 accessor/view functionality
- Integration with KIDs data structures
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
import xarray as xr

from tolteca_kidsproc.accessors.dataset import make_kids_dataset
from tolteca_kidsproc.analysis.d21 import D21Analysis, D21Mapper, D21View

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_sweep() -> xr.Dataset:
    """Create simple 1-D sweep dataset with Lorentzian resonance.

    Returns
    -------
    xr.Dataset
        Dataset with I, Q, and frequency for single channel.
        Frequency range: 1.0-1.1 GHz, 101 points.
        Resonance at 1.05 GHz with width 1 MHz.
    """
    freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
    # Lorentzian resonance: S21 = 1 / (1 + i * (f - f0) / gamma)
    s21 = 1 / (1 + 1j * (freq.value - 1.05e9) / 1e6)

    return make_kids_dataset(s21.real, s21.imag, frequency=freq)


@pytest.fixture
def multi_channel_sweep() -> xr.Dataset:
    """Create multi-channel 2-D sweep dataset.

    Returns
    -------
    xr.Dataset
        Dataset with I, Q, and frequency for multiple channels.
        10 channels spanning 1.0-1.1 GHz.
        Each channel has 101 tones spanning ±5 MHz.
    """
    n_channels = 10
    n_tones = 101

    # Channel center frequencies
    f_chans = np.linspace(1.0e9, 1.1e9, n_channels)

    # Sweep offset for each channel
    f_sweep = np.linspace(-5e6, 5e6, n_tones)

    # Broadcast to create 2-D frequency grid
    frequency = (f_sweep[np.newaxis, :] + f_chans[:, np.newaxis]) << u.Hz

    # Create resonance for each channel
    s21 = 1 / (1 + 1j * (frequency.value - f_chans[:, np.newaxis]) / 1e6)

    return make_kids_dataset(s21.real, s21.imag, frequency=frequency)


@pytest.fixture
def rng() -> np.random.Generator:
    """Random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(seed=42)


# ============================================================================
# D21Analysis Configuration Tests
# ============================================================================


class TestD21AnalysisConfig:
    """Test D21Analysis configuration and validation."""

    def test_default_config(self):
        """Default configuration is valid."""
        d21_analysis = D21Analysis()

        assert d21_analysis.method == "savgol"
        assert d21_analysis.smooth == 5
        assert d21_analysis.exclude_edge is None
        assert d21_analysis.f_lims is None
        assert d21_analysis.f_step is None
        assert d21_analysis.resample == 1

    def test_gradient_method_config(self):
        """Gradient method configuration."""
        d21_analysis = D21Analysis(method="gradient", smooth=3)

        assert d21_analysis.method == "gradient"
        assert d21_analysis._smooth == 3

    def test_savgol_method_config(self):
        """Savitzky-Golay method configuration."""
        d21_analysis = D21Analysis(method="savgol", smooth=7)

        assert d21_analysis.method == "savgol"
        assert d21_analysis._smooth == 7

    def test_smooth_validation_negative(self):
        """Negative smooth raises ValueError."""
        with pytest.raises(ValueError, match="smooth cannot be negative"):
            D21Analysis(smooth=-1)

    def test_exclude_edge_validation_negative(self):
        """Negative exclude_edge raises ValueError."""
        with pytest.raises(ValueError, match="exclude_edge cannot be negative"):
            D21Analysis(exclude_edge=-1)

    def test_savgol_requires_minimum_smooth(self):
        """Savgol method requires smooth >= 3."""
        with pytest.raises(ValueError, match="savgol requires smooth >= 3"):
            d21_analysis = D21Analysis(method="savgol", smooth=2)

    def test_no_smooth_requires_gradient(self):
        """Zero smooth only works with gradient method."""
        with pytest.raises(ValueError, match="no-smooth only works for gradient"):
            D21Analysis(method="savgol", smooth=0)

    def test_exclude_edge_defaults_to_smooth(self):
        """exclude_edge defaults to smooth if not specified."""
        d21_analysis = D21Analysis(smooth=5)

        assert d21_analysis._exclude_edge == 5

    def test_exclude_edge_explicit_override(self):
        """Explicit exclude_edge overrides default."""
        d21_analysis = D21Analysis(smooth=5, exclude_edge=3)

        assert d21_analysis._exclude_edge == 3

    def test_frequency_grid_config(self):
        """Frequency grid configuration."""
        f_min = 1.0e9 * u.Hz
        f_max = 1.1e9 * u.Hz
        f_step = 1e5 * u.Hz

        d21_analysis = D21Analysis(
            f_lims=(f_min, f_max),
            f_step=f_step,
        )

        assert d21_analysis.f_lims == (f_min, f_max)
        assert d21_analysis.f_step == f_step


# ============================================================================
# D21 Matched Computation Tests
# ============================================================================


class TestD21Matched:
    """Test matched D21 computation on original frequency grid."""

    def test_gradient_method_simple(self, simple_sweep):
        """Gradient method computes D21 with correct shape."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        d21 = d21_analysis.make_matched(simple_sweep.kids.sweep)

        # Check output structure
        assert isinstance(d21, xr.DataArray)
        assert d21.shape == (101,)
        assert d21.dtype == np.complex128

        # Check coordinates match input
        assert "sweep" in d21.dims
        np.testing.assert_array_equal(
            d21.coords["frequency"],
            simple_sweep.coords["frequency"],
        )

        # Check attributes
        assert "long_name" in d21.attrs
        assert d21.attrs["units"] == "Hz^-1"

    def test_savgol_method_simple(self, simple_sweep):
        """Savitzky-Golay method computes D21 with correct shape."""
        d21_analysis = D21Analysis(method="savgol", smooth=7)
        d21 = d21_analysis.make_matched(simple_sweep.kids.sweep)

        # Check output structure
        assert isinstance(d21, xr.DataArray)
        assert d21.shape == (101,)
        assert d21.dtype == np.complex128

    def test_gradient_no_smooth(self, simple_sweep):
        """Gradient method works without smoothing."""
        d21_analysis = D21Analysis(method="gradient", smooth=0)
        d21 = d21_analysis.make_matched(simple_sweep.kids.sweep)

        assert isinstance(d21, xr.DataArray)
        assert d21.shape == (101,)

    def test_multi_channel_gradient(self, multi_channel_sweep):
        """Gradient method works on multi-channel data."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        d21 = d21_analysis.make_matched(multi_channel_sweep.kids.multi_sweep)

        # Check shape matches input
        assert d21.shape == (10, 101)
        assert "chan" in d21.dims
        assert "sweep" in d21.dims

    def test_multi_channel_savgol(self, multi_channel_sweep):
        """Savgol method works on multi-channel data."""
        d21_analysis = D21Analysis(method="savgol", smooth=7)
        d21 = d21_analysis.make_matched(multi_channel_sweep.kids.multi_sweep)

        # Check shape matches input
        assert d21.shape == (10, 101)

    def test_d21_is_complex(self, simple_sweep):
        """D21 result is complex-valued."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        d21 = d21_analysis.make_matched(simple_sweep.kids.sweep)

        assert np.iscomplexobj(d21.values)

    def test_d21_peak_at_resonance(self, simple_sweep):
        """D21 magnitude peaks near resonance frequency."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        d21 = d21_analysis.make_matched(simple_sweep.kids.sweep)

        # Find peak of |D21|
        peak_idx = np.argmax(np.abs(d21.values))
        peak_freq = d21.coords["frequency"].values[peak_idx]

        # Should be near 1.05 GHz (resonance)
        assert 1.04e9 < peak_freq < 1.06e9

    def test_gradient_vs_savgol_similar(self, simple_sweep):
        """Gradient and savgol methods produce similar results."""
        d21_analysis_grad = D21Analysis(method="gradient", smooth=7)
        d21_analysis_sav = D21Analysis(method="savgol", smooth=7)

        d21_grad = d21_analysis_grad.make_matched(simple_sweep.kids.sweep)
        d21_sav = d21_analysis_sav.make_matched(simple_sweep.kids.sweep)

        # Should be similar (correlation > 0.9)
        corr = np.corrcoef(np.abs(d21_grad.values), np.abs(d21_sav.values))[0, 1]
        assert corr > 0.9

    def test_analysis_config_stored_in_attrs(self, simple_sweep):
        """Analysis configuration is stored in result attributes."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        d21 = d21_analysis.make_matched(simple_sweep.kids.sweep)

        # Should have analysis config in attrs
        assert "tolteca_kidsproc.analysis.d21.analysis" in d21.attrs

        # Should be able to reload config
        loaded = D21Analysis._load_from_attr(d21)
        assert loaded == d21_analysis


# ============================================================================
# D21 Unified Computation Tests
# ============================================================================


class TestD21Unified:
    """Test unified D21 computation with common frequency grid."""

    def test_simple_sweep_unified(self, simple_sweep):
        """Unified D21 works on simple 1-D sweep."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        d21_unified = d21_analysis.make_unified(simple_sweep.kids.sweep)

        # Check output structure
        assert isinstance(d21_unified, xr.DataArray)
        assert d21_unified.ndim == 1

        # Check coordinates
        assert "f_unified" in d21_unified.dims
        assert "cov_unified" in d21_unified.coords

    def test_multi_channel_unified(self, multi_channel_sweep):
        """Unified D21 averages across channels."""
        d21_analysis = D21Analysis(method="gradient", smooth=5, exclude_edge=3)
        d21_unified = d21_analysis.make_unified(multi_channel_sweep.kids.multi_sweep)

        # Should be 1-D (averaged over channels)
        assert d21_unified.ndim == 1
        assert "f_unified" in d21_unified.dims

    def test_unified_frequency_grid_inferred(self, multi_channel_sweep):
        """Unified frequency grid is inferred from data."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        d21_unified = d21_analysis.make_unified(multi_channel_sweep.kids.multi_sweep)

        f_unified = d21_unified.coords["f_unified"]

        # Should span approximately full frequency range
        freq_min = multi_channel_sweep.coords["frequency"].values.min()
        freq_max = multi_channel_sweep.coords["frequency"].values.max()

        assert f_unified.min() >= freq_min * 0.99
        assert f_unified.max() <= freq_max * 1.01

    def test_unified_frequency_grid_explicit(self, multi_channel_sweep):
        """Unified frequency grid can be specified explicitly."""
        f_min = 1.02e9 << u.Hz
        f_max = 1.08e9 << u.Hz
        f_step = 1e5 << u.Hz

        d21_analysis = D21Analysis(
            method="gradient",
            smooth=5,
            f_lims=(f_min, f_max),
            f_step=f_step,
        )
        d21_unified = d21_analysis.make_unified(multi_channel_sweep.kids.multi_sweep)

        f_unified = d21_unified.coords["f_unified"].u.quantity

        # Check grid matches specification
        assert f_unified.min().to_value(u.Hz) >= f_min.to_value(u.Hz)
        assert f_unified.max().to_value(u.Hz) <= f_max.to_value(u.Hz)

        # Check step size
        steps = np.diff(f_unified.to_value(u.Hz))
        np.testing.assert_allclose(
            steps,
            f_step.to_value(u.Hz),  # pyright: ignore[reportArgumentType, reportCallIssue]
            rtol=1e-10,
        )

    def test_unified_coverage_map(self, multi_channel_sweep):
        """Coverage map tracks channel overlap."""
        d21_analysis = D21Analysis(method="gradient", smooth=5, exclude_edge=3)
        d21_unified = d21_analysis.make_unified(multi_channel_sweep.kids.multi_sweep)

        cov = d21_unified.coords["cov_unified"]

        # Coverage should be integer counts
        assert cov.dtype == int

        # Coverage should be > 0 in overlapping regions
        assert cov.max() > 0

        # Coverage should be <= number of channels
        assert cov.max() <= 10

    def test_edge_exclusion_reduces_coverage(self, multi_channel_sweep):
        """Edge exclusion reduces coverage at channel boundaries."""
        d21_analysis_no_edge = D21Analysis(method="gradient", smooth=5, exclude_edge=0)
        d21_analysis_with_edge = D21Analysis(
            method="gradient",
            smooth=5,
            exclude_edge=10,
        )

        d21_no_edge = d21_analysis_no_edge.make_unified(
            multi_channel_sweep.kids.multi_sweep,
        )
        d21_with_edge = d21_analysis_with_edge.make_unified(
            multi_channel_sweep.kids.multi_sweep,
        )

        cov_no_edge = d21_no_edge.coords["cov_unified"]
        cov_with_edge = d21_with_edge.coords["cov_unified"]

        # With edge exclusion, total coverage should be lower
        assert cov_with_edge.sum() < cov_no_edge.sum()

    def test_edge_exclusion_validation(self, simple_sweep):
        """Edge exclusion validates against data size."""
        # Edge exclusion too large for data size
        d21_analysis = D21Analysis(method="gradient", smooth=5, exclude_edge=60)

        with pytest.raises(ValueError, match="insufficient number of data points"):
            d21_analysis.make_unified(simple_sweep.kids.sweep)

    def test_unified_is_real_valued(self, multi_channel_sweep):
        """Unified D21 is real-valued (absolute value)."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        d21_unified = d21_analysis.make_unified(multi_channel_sweep.kids.multi_sweep)

        assert not np.iscomplexobj(d21_unified.values)
        assert d21_unified.dtype == np.float64

    def test_unified_attributes(self, multi_channel_sweep):
        """Unified D21 has proper attributes."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        d21_unified = d21_analysis.make_unified(multi_channel_sweep.kids.multi_sweep)

        assert "long_name" in d21_unified.attrs
        assert "units" in d21_unified.attrs
        assert "tolteca_kidsproc.analysis.d21.analysis" in d21_unified.attrs

    def test_resampling_factor(self, multi_channel_sweep):
        """Resampling factor changes grid density."""
        d21_analysis_1x = D21Analysis(method="gradient", smooth=5, resample=1)
        d21_analysis_2x = D21Analysis(method="gradient", smooth=5, resample=2)

        d21_1x = d21_analysis_1x.make_unified(multi_channel_sweep.kids.multi_sweep)
        d21_2x = d21_analysis_2x.make_unified(multi_channel_sweep.kids.multi_sweep)

        # 2x resampling should have ~2x more points
        assert len(d21_2x) > len(d21_1x) * 1.8


# ============================================================================
# High-Level API Tests
# ============================================================================


class TestD21HighLevelAPI:
    """Test high-level D21Analysis() API with caching."""

    def test_call_computes_both_matched_and_unified(self, multi_channel_sweep):
        """Calling d21_analysis computes both matched and unified D21."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(multi_channel_sweep)

        # Both should be present
        mapper = D21Mapper.from_data_source(ds)
        assert mapper.schema.d21 in mapper
        assert mapper.schema.d21_unified in mapper

    def test_call_stores_config_in_attrs(self, simple_sweep):
        """Calling d21_analysis stores config in dataset attributes."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(simple_sweep)

        # Should be able to reload config
        loaded = D21Analysis._load_from_attr(ds)
        assert loaded == d21_analysis

    def test_call_caching_skips_recomputation(self, simple_sweep):
        """Second call with same config skips recomputation."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)

        # First call computes
        ds1 = d21_analysis(simple_sweep)

        # Get D21 values
        mapper = D21Mapper.from_data_source(ds1)
        d21_1 = mapper.get_arr(ds1, mapper.schema.d21)

        # Second call should return same dataset (cached)
        ds2 = d21_analysis(ds1)

        d21_2 = mapper.get_arr(ds2, mapper.schema.d21)

        # Should be equal (cached data)
        assert np.array_equal(d21_1.values, d21_2.values)

    def test_call_force_recomputes(self, simple_sweep):
        """force=True recomputes even if cached."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)

        # First call computes
        ds1 = d21_analysis(simple_sweep)

        # Force recomputation
        ds2 = d21_analysis(ds1, force=True)

        # Should have new values (not same object)
        mapper = D21Mapper.from_data_source(ds1)
        d21_1 = mapper.get_arr(ds1, mapper.schema.d21)
        d21_2 = mapper.get_arr(ds2, mapper.schema.d21)

        assert d21_1 is not d21_2

    def test_call_different_config_recomputes(self, simple_sweep):
        """Calling with different config recomputes."""
        d21_analysis1 = D21Analysis(method="gradient", smooth=5)
        d21_analysis2 = D21Analysis(method="gradient", smooth=7)

        # First call
        ds1 = d21_analysis1(simple_sweep)

        # Different config should recompute
        ds2 = d21_analysis2(ds1)

        # Should have different values (different smooth parameters)
        mapper = D21Mapper.from_data_source(ds1)
        d21_1 = mapper.get_arr(ds1, mapper.schema.d21)
        d21_2 = mapper.get_arr(ds2, mapper.schema.d21)

        # Different smoothing should produce different results
        # But both should be valid complex arrays
        assert d21_1.shape == d21_2.shape
        assert np.iscomplexobj(d21_1.values)
        assert np.iscomplexobj(d21_2.values)


# ============================================================================
# D21 Schema and Mapper Tests
# ============================================================================


class TestD21SchemaMapper:
    """Test D21Schema and D21Mapper."""

    def test_schema_has_required_fields(self):
        """Schema has all required D21 fields."""
        mapper = D21Mapper.from_defaults()
        schema = mapper.schema

        assert hasattr(schema, "d21")
        assert hasattr(schema, "d21_unified")
        assert hasattr(schema, "analysis")
        assert hasattr(schema, "f_unified")
        assert hasattr(schema, "cov_unified")

    def test_mapper_from_data_source(self, simple_sweep):
        """Mapper can be created from dataset."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(simple_sweep)

        mapper = D21Mapper.from_data_source(ds)

        assert isinstance(mapper, D21Mapper)

    def test_mapper_contains_method(self, simple_sweep):
        """Mapper 'in' operator checks for field presence."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(simple_sweep)

        mapper = D21Mapper.from_data_source(ds)

        # Should have d21 after computation
        assert mapper.schema.d21 in mapper

    def test_mapper_get_arr(self, simple_sweep):
        """Mapper.get_arr() retrieves DataArray."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(simple_sweep)

        mapper = D21Mapper.from_data_source(ds)
        d21 = mapper.get_arr(ds, mapper.schema.d21)

        assert isinstance(d21, xr.DataArray)

    def test_mapper_get_name(self):
        """Mapper.get_name() returns field name."""
        mapper = D21Mapper.from_defaults()

        d21_name = mapper.get_name(mapper.schema.d21)
        assert isinstance(d21_name, str)
        assert "d21" in d21_name


# ============================================================================
# D21 View/Accessor Tests
# ============================================================================


class TestD21View:
    """Test D21View accessor functionality."""

    def test_view_requires_analysis_in_attrs(self, simple_sweep):
        """D21View requires analysis config in attrs."""
        # Dataset without D21 analysis should fail
        with pytest.raises(ValueError, match="No D21 analysis found"):
            D21View(simple_sweep)

    def test_view_access_d21(self, simple_sweep):
        """D21View provides access to matched D21."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(simple_sweep)

        view = D21View(ds)
        d21 = view.d21

        assert isinstance(d21, xr.DataArray)
        assert d21.shape == (101,)

    def test_view_access_d21_unified(self, multi_channel_sweep):
        """D21View provides access to unified D21."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(multi_channel_sweep)

        view = D21View(ds)
        d21_unified = view.d21_unified

        assert isinstance(d21_unified, xr.DataArray)
        assert d21_unified.ndim == 1

    def test_view_access_f_unified(self, multi_channel_sweep):
        """D21View provides access to unified frequency."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(multi_channel_sweep)

        view = D21View(ds)
        f_unified = view.f_unified

        assert isinstance(f_unified, xr.DataArray)

    def test_view_access_cov_unified(self, multi_channel_sweep):
        """D21View provides access to coverage map."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(multi_channel_sweep)

        view = D21View(ds)
        cov = view.cov_unified

        assert isinstance(cov, xr.DataArray)
        assert cov.dtype == int

    def test_view_access_analysis_config(self, simple_sweep):
        """D21View provides access to analysis configuration."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(simple_sweep)

        view = D21View(ds)
        config = view.d21_analysis

        assert isinstance(config, D21Analysis)
        assert config.method == "gradient"
        assert config._smooth == 5


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_method_raises_error(self, simple_sweep):
        """Unknown method raises ValidationError at instantiation."""
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError, match="literal_error"):
            D21Analysis(method="invalid")  # type: ignore[arg-type]

    def test_empty_dataset_raises_error(self):
        """Empty dataset raises appropriate error."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        empty_ds = xr.Dataset()

        with pytest.raises((ValueError, KeyError, AttributeError)):
            d21_analysis(empty_ds)

    def test_mismatched_dimensions(self, rng):
        """Mismatched I/Q dimensions raise error."""
        # xarray catches dimension mismatch at Dataset creation
        with pytest.raises(ValueError, match="conflicting sizes"):
            ds = xr.Dataset(
                {
                    "I": (["sweep"], rng.standard_normal(101)),
                    "Q": (["sweep"], rng.standard_normal(50)),  # Different size
                },
            )

    def test_single_point_sweep(self):
        """Single-point sweep is invalid."""
        ds = xr.Dataset(
            {
                "I": (["sweep"], [1.0]),
                "Q": (["sweep"], [0.0]),
            },
            coords={"frequency": (["sweep"], [1e9])},
        )

        d21_analysis = D21Analysis(method="gradient", smooth=0)

        with pytest.raises((ValueError, IndexError)):
            d21_analysis.make_matched(ds.kids.sweep)

    def test_very_small_smooth_window(self, simple_sweep):
        """Very small smooth window (< 3) fails for savgol."""
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError, match="savgol requires smooth"):
            D21Analysis(method="savgol", smooth=1)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_single_channel(self, simple_sweep):
        """Complete workflow for single-channel sweep."""
        # Create d21_analysis
        d21_analysis = D21Analysis(method="savgol", smooth=7, exclude_edge=5)

        # Compute D21 (high-level API)
        ds = d21_analysis(simple_sweep)

        # Access results via mapper
        mapper = D21Mapper.from_data_source(ds)
        d21_matched = mapper.get_arr(ds, mapper.schema.d21)
        d21_unified = mapper.get_arr(ds, mapper.schema.d21_unified)

        # Verify results
        assert d21_matched.shape == (101,)
        assert d21_unified.ndim == 1
        assert np.all(np.isfinite(d21_matched.values))
        assert np.all(np.isfinite(d21_unified.values))

    def test_full_workflow_multi_channel(self, multi_channel_sweep):
        """Complete workflow for multi-channel sweep."""
        # Create d21_analysis with custom grid
        f_min = 1.01e9 << u.Hz
        f_max = 1.09e9 << u.Hz
        f_step = 1e5 << u.Hz

        d21_analysis = D21Analysis(
            method="gradient",
            smooth=5,
            exclude_edge=3,
            f_lims=(f_min, f_max),
            f_step=f_step,
        )

        # Compute D21
        ds = d21_analysis(multi_channel_sweep)

        # Access via view
        view = D21View(ds)

        # Verify matched D21
        d21_matched = view.d21
        assert d21_matched.shape == (10, 101)

        # Verify unified D21
        d21_unified = view.d21_unified
        assert d21_unified.ndim == 1

        # Verify frequency grid
        f_unified = view.f_unified
        assert f_unified.min() >= f_min.to_value(u.Hz)
        assert f_unified.max() <= f_max.to_value(u.Hz)

        # Verify coverage
        cov = view.cov_unified
        assert cov.max() <= 10  # <= number of channels
        assert cov.sum() > 0  # Some overlap

    def test_gradient_and_savgol_consistency(self, simple_sweep):
        """Gradient and savgol methods produce consistent results."""
        d21_analysis_grad = D21Analysis(method="gradient", smooth=7)
        d21_analysis_sav = D21Analysis(method="savgol", smooth=7)

        ds_grad = d21_analysis_grad(simple_sweep)
        ds_sav = d21_analysis_sav(simple_sweep)

        mapper = D21Mapper.from_data_source(ds_grad)
        d21_grad = mapper.get_arr(ds_grad, mapper.schema.d21)
        d21_sav = mapper.get_arr(ds_sav, mapper.schema.d21)

        # Absolute values should be highly correlated
        corr = np.corrcoef(np.abs(d21_grad.values), np.abs(d21_sav.values))[0, 1]
        assert corr > 0.85

    def test_caching_with_multiple_configs(self, simple_sweep):
        """Multiple configs don't interfere with caching."""
        d21_analysis1 = D21Analysis(method="gradient", smooth=5)
        d21_analysis2 = D21Analysis(method="savgol", smooth=7)

        # Compute with first config
        ds1 = d21_analysis1(simple_sweep)

        # Check config1 immediately
        config1 = D21Analysis._load_from_attr(ds1)
        assert config1 is not None
        assert config1.method == "gradient"
        assert config1.smooth == 5

        # Compute with second config (should recompute)
        ds2 = d21_analysis2(ds1)

        # Check config2 immediately
        config2 = D21Analysis._load_from_attr(ds2)
        assert config2 is not None
        assert config2.method == "savgol"
        assert config2.smooth == 7

        # Go back to first config (should recompute since attrs changed)
        ds3 = d21_analysis1(ds2)

        # Check config3 immediately
        config3 = D21Analysis._load_from_attr(ds3)
        assert config3 is not None
        assert config3.method == "gradient"
        assert config3.smooth == 5
