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

from dataclasses import replace

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from tollan.accessor import SubpathMapping

from tolteca_kidsproc.accessors.dataset import make_kids_dataset
from tolteca_kidsproc.accessors.kids import KidsMapper
from tolteca_kidsproc.accessors.views import SweepView
from tolteca_kidsproc.analysis.d21 import D21Analysis, D21Mapper, D21View


@pytest.mark.parametrize("method", ["gradient", "savgol"])
@pytest.mark.parametrize("n_chans", [None, 3])
def test_numerical_derivative_without_model_or_view(method, n_chans):
    frequency = np.linspace(0.0, 1.0, 101)
    if n_chans is not None:
        frequency = frequency + np.arange(n_chans)[:, None]
    signal = (1 + 2j) * frequency**2
    frequency.setflags(write=False)
    signal.setflags(write=False)
    actual = D21Analysis.compute_matched(
        signal, frequency, method=method, smooth=0 if method == "gradient" else 5
    )
    expected = 2 * (1 + 2j) * frequency
    np.testing.assert_allclose(actual[..., 1:-1], expected[..., 1:-1], atol=1e-12)
    assert not np.shares_memory(actual, signal)


def test_numerical_unified_coverage_and_uncovered_bins():
    frequency = np.array([[0.0, 1, 2, 3, 4], [2.0, 3, 4, 5, 6]])
    derivative = np.array([[2j] * 5, [4j] * 5])
    frequency.setflags(write=False)
    derivative.setflags(write=False)
    grid, magnitude, coverage = D21Analysis.compute_unified(
        derivative, frequency, f_lims=(-1.0, 8.0), f_step=1.0, exclude_edge=1
    )
    np.testing.assert_array_equal(grid, np.arange(-1.0, 8.0))
    np.testing.assert_array_equal(magnitude, [0, 0, 2, 2, 3, 4, 4, 0, 0])
    np.testing.assert_array_equal(coverage, [0, 0, 1, 1, 2, 1, 1, 0, 0])


@pytest.mark.parametrize(
    "kind", ["dataset", "empty_dataset", "array", "tree", "timestream"]
)
def test_rejects_non_sweep_inputs_before_reuse(simple_sweep, kind):
    analysis = D21Analysis()
    output = analysis(simple_sweep.kids.sweep)
    before = output.copy(deep=False)
    timestream = make_kids_dataset(
        np.ones(16), np.zeros(16), time=np.arange(16) * u.s, f_smp=1 << u.Hz
    ).kids.timestream
    invalid = {
        "dataset": simple_sweep,
        "empty_dataset": xr.Dataset(),
        "array": simple_sweep.I,
        "tree": output,
        "timestream": timestream,
    }[kind]
    with pytest.raises(TypeError, match="requires a sweep view"):
        analysis(invalid, output=output)  # ty: ignore[invalid-argument-type] -- Runtime rejection.
    xr.testing.assert_identical(output, before)


@pytest.mark.parametrize("destination", ["dataset", "parent_without_output"])
def test_rejects_invalid_destination(simple_sweep, destination):
    """Result destinations must be trees and parent selection needs a tree."""
    analysis = D21Analysis()
    if destination == "dataset":
        with pytest.raises(TypeError, match="output must be a DataTree"):
            analysis(simple_sweep.kids.sweep, output=xr.Dataset())  # ty: ignore[invalid-argument-type] -- Runtime rejection.
    else:
        with pytest.raises(TypeError, match="requires an output tree"):
            analysis(simple_sweep.kids.sweep, parent_namespace="scan")


def test_input_view_preserves_custom_field_mapping(simple_sweep):
    analysis = D21Analysis(method="gradient", smooth=5)
    expected = analysis(simple_sweep.kids.sweep).kids.d21
    names = {"I": "in_phase", "Q": "quadrature", "frequency": "probe_frequency"}
    source = simple_sweep.rename(names)
    original = KidsMapper.from_data_source(simple_sweep)
    mapper = KidsMapper(
        {
            field: replace(resolved, name=names.get(resolved.name, resolved.name))
            for field, resolved in original.mapped_fields.items()
        }
    )
    view = SweepView(source, mapper=mapper)
    result = analysis(view)
    assert view.mapper is mapper
    xr.testing.assert_identical(result["_input"].to_dataset(), source)
    xr.testing.assert_allclose(
        result.kids.d21.d21, expected.d21.rename({"frequency": "probe_frequency"})
    )
    xr.testing.assert_allclose(result.kids.d21.d21_unified, expected.d21_unified)


def test_schema_paths_control_tree_analysis(monkeypatch, multi_channel_sweep):
    schema = replace(
        D21Mapper.schema,
        _namespace=SubpathMapping.path("diagnostics/derivative"),
        d21_subpath=SubpathMapping.path("arrays"),
    )
    monkeypatch.setattr(D21Mapper, "schema", schema)
    tree = xr.DataTree.from_dict({"measurement/arrays": multi_channel_sweep})
    analysis = D21Analysis(method="gradient", smooth=5)
    result = analysis(
        tree["measurement/arrays"].kids.sweep,
        output=tree,
        parent_namespace="measurement",
    )
    scope = result["measurement"]
    assert isinstance(scope, xr.DataTree)
    namespace = scope["diagnostics/derivative"]
    assert isinstance(namespace, xr.DataTree)
    assert "d21" not in scope.children
    assert analysis(scope["arrays"].kids.sweep, output=scope) is scope
    assert result is tree
    expected = analysis(multi_channel_sweep.copy(deep=False).kids.sweep).kids.d21
    for view in (
        scope.kids.d21,
        D21View.from_namespace(namespace),
        D21View.from_namespace(scope.kids.d21.data_source),
        scope.copy(deep=False).kids.d21,
    ):
        xr.testing.assert_equal(view.d21, expected.d21)
        xr.testing.assert_equal(view.d21_unified, expected.d21_unified)
        assert view.d21_analysis == analysis


@pytest.mark.parametrize("force", [False, True])
def test_output_layout_is_checked_before_derivative_computation(
    monkeypatch, simple_sweep, force
):
    analysis = D21Analysis()
    tree = analysis(simple_sweep.kids.sweep)
    before = tree.copy(deep=False)
    monkeypatch.setattr(
        D21Mapper,
        "schema",
        replace(
            D21Mapper.schema, d21_unified_subpath=SubpathMapping.path("d21/unified")
        ),
    )
    monkeypatch.setattr(
        D21Analysis,
        "make_matched",
        lambda *args, **kwargs: pytest.fail("Layout must be checked before D21"),
    )
    with pytest.raises(ValueError, match="output subpaths must be separate"):
        analysis(simple_sweep.kids.sweep, output=tree, force=force)
    xr.testing.assert_identical(tree, before)


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

    @pytest.mark.parametrize("smooth", [1, 2])
    def test_savgol_requires_minimum_smooth(self, smooth):
        """Savgol method requires smooth >= 3."""
        with pytest.raises(ValueError, match="savgol requires smooth >= 3"):
            D21Analysis(method="savgol", smooth=smooth)

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

    @pytest.mark.parametrize("method", ["gradient", "savgol"])
    @pytest.mark.parametrize("fixture_name", ["simple_sweep", "multi_channel_sweep"])
    def test_matched_output_contract(self, request, method, fixture_name):
        """Both estimators retain signal coordinates, shape, and derivative units."""
        source = request.getfixturevalue(fixture_name)
        derivative = D21Analysis(method=method, smooth=5).make_matched(
            source.kids.sweep
        )
        assert derivative.dims == source.I.dims
        assert derivative.shape == source.I.shape
        assert np.iscomplexobj(derivative.values)
        xr.testing.assert_identical(derivative.frequency, source.frequency)
        assert derivative.attrs["units"] == "Hz^-1"

    @pytest.mark.parametrize("method", ["gradient", "savgol"])
    def test_derivative_of_quadratic_in_physical_units(self, method):
        """A known complex polynomial fixes derivative sign, scale, and Hz units."""
        x = np.linspace(-2, 2, 101)
        frequency = (1000 + x) * u.MHz
        signal = 2 * x**2 + 3j * x
        source = make_kids_dataset(signal.real, signal.imag, frequency=frequency)
        analysis = D21Analysis(method=method, smooth=0 if method == "gradient" else 7)
        derivative = analysis.make_matched(source.kids.sweep)
        expected = (4 * x + 3j) / 1e6
        # The gradient uses one-sided differences at the two endpoints.
        np.testing.assert_allclose(derivative[1:-1], expected[1:-1], rtol=1e-10)

    def test_d21_peak_at_resonance(self, simple_sweep):
        """D21 magnitude peaks near resonance frequency."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        d21 = d21_analysis.make_matched(simple_sweep.kids.sweep)

        # Find peak of |D21|
        peak_idx = np.argmax(np.abs(d21.values))
        peak_freq = d21.coords["frequency"].values[peak_idx]

        # Should be near 1.05 GHz (resonance)
        assert 1.04e9 < peak_freq < 1.06e9

    def test_analysis_config_stored_in_attrs(self, simple_sweep):
        """Analysis configuration is stored in result attributes."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        d21 = d21_analysis.make_matched(simple_sweep.kids.sweep)

        # Should have analysis config in attrs
        assert "tolteca_kidsproc.analysis.d21.analysis" in d21.attrs

        # Should be able to reload config
        loaded = D21Analysis._load_from_attr(d21, mapper=D21Mapper.from_defaults())
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
        ds = d21_analysis(multi_channel_sweep.kids.sweep)

        # Both should be present
        mapper = D21Mapper.from_namespace(ds.kids.d21.data_source)
        assert mapper.schema.d21_subpath in mapper
        assert mapper.schema.d21_unified_subpath in mapper

    def test_call_stores_config_in_attrs(self, simple_sweep):
        """Calling d21_analysis stores config in dataset attributes."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(simple_sweep.kids.sweep)

        # Should be able to reload config
        loaded = D21Analysis._load_from_attr(
            ds.kids.d21.data_source,
            mapper=D21Mapper.from_namespace(ds.kids.d21.data_source),
        )
        assert loaded == d21_analysis

    @pytest.mark.parametrize("change", ["none", "input", "configuration"])
    def test_result_reuse_and_recomputation(self, simple_sweep, monkeypatch, change):
        """Reuse skips computation; force and changed settings compute new results."""
        calls = 0
        make_matched = D21Analysis.make_matched

        def counted(analysis, *args, **kwargs):
            nonlocal calls
            calls += 1
            return make_matched(analysis, *args, **kwargs)

        monkeypatch.setattr(D21Analysis, "make_matched", counted)
        analysis = D21Analysis(method="gradient", smooth=5)
        result = analysis(simple_sweep.kids.sweep)
        before = result.kids.d21
        input_data = simple_sweep.copy(deep=True)
        if change == "input":
            input_data["I"] *= 2
            input_data["Q"] *= 2
        elif change == "configuration":
            analysis = D21Analysis(method="gradient", smooth=7)

        returned = analysis(
            input_data.kids.sweep, output=result, force=change == "input"
        )
        after = result.kids.d21
        assert returned is result
        assert calls == (1 if change == "none" else 2)
        assert after.d21_analysis == analysis
        if change == "none":
            assert after.data_source is before.data_source
        else:
            assert after.data_source is not before.data_source
            if change == "input":
                np.testing.assert_allclose(after.d21, 2 * before.d21)
            else:
                assert not np.array_equal(after.d21.values, before.d21.values)


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
        ds = d21_analysis(simple_sweep.kids.sweep)

        mapper = D21Mapper.from_namespace(ds.kids.d21.data_source)

        assert isinstance(mapper, D21Mapper)

    def test_mapper_contains_method(self, simple_sweep):
        """Mapper 'in' operator checks for field presence."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(simple_sweep.kids.sweep)

        mapper = D21Mapper.from_namespace(ds.kids.d21.data_source)

        # Should have d21 after computation
        assert mapper.schema.d21_subpath in mapper

    def test_mapper_get_var(self, simple_sweep):
        """Mapper.get_arr() retrieves DataArray."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(simple_sweep.kids.sweep)

        d21 = ds.kids.d21.d21

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
        with pytest.raises(KeyError):
            D21View(simple_sweep)
        result = D21Analysis()(simple_sweep.kids.sweep)
        result["tolteca_kidsproc.analysis.d21"].attrs.clear()
        with pytest.raises(
            ValueError, match="Missing producing analysis configuration"
        ):
            D21View(result)

    def test_view_access_d21(self, simple_sweep):
        """D21View provides access to matched D21."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(simple_sweep.kids.sweep)

        view = D21View(ds)
        d21 = view.d21

        assert isinstance(d21, xr.DataArray)
        assert d21.shape == (101,)

    def test_view_access_d21_unified(self, multi_channel_sweep):
        """D21View provides access to unified D21."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(multi_channel_sweep.kids.sweep)

        view = D21View(ds)
        d21_unified = view.d21_unified

        assert isinstance(d21_unified, xr.DataArray)
        assert d21_unified.ndim == 1

    def test_view_access_f_unified(self, multi_channel_sweep):
        """D21View provides access to unified frequency."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(multi_channel_sweep.kids.sweep)

        view = D21View(ds)
        f_unified = view.f_unified

        assert isinstance(f_unified, xr.DataArray)

    def test_view_access_cov_unified(self, multi_channel_sweep):
        """D21View provides access to coverage map."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(multi_channel_sweep.kids.sweep)

        view = D21View(ds)
        cov = view.cov_unified

        assert isinstance(cov, xr.DataArray)
        assert cov.dtype == int

    def test_view_access_analysis_config(self, simple_sweep):
        """D21View provides access to analysis configuration."""
        d21_analysis = D21Analysis(method="gradient", smooth=5)
        ds = d21_analysis(simple_sweep.kids.sweep)

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
            D21Analysis.model_validate({"method": "invalid"})

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
        ds = d21_analysis(simple_sweep.kids.sweep)

        # Access results via mapper
        d21_matched = ds.kids.d21.d21
        d21_unified = ds.kids.d21.d21_unified

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
        ds = d21_analysis(multi_channel_sweep.kids.sweep)

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

    def test_caching_with_multiple_configs(self, simple_sweep):
        """Multiple configs don't interfere with caching."""
        d21_analysis1 = D21Analysis(method="gradient", smooth=5)
        d21_analysis2 = D21Analysis(method="savgol", smooth=7)

        # Compute with first config
        ds1 = d21_analysis1(simple_sweep.kids.sweep)

        # Check config1 immediately
        config1 = D21Analysis._load_from_attr(
            ds1.kids.d21.data_source,
            mapper=D21Mapper.from_namespace(ds1.kids.d21.data_source),
        )
        assert config1 is not None
        assert config1.method == "gradient"
        assert config1.smooth == 5

        # Compute with second config (should recompute)
        ds2 = d21_analysis2(ds1["_input"].kids.sweep, output=ds1)

        # Check config2 immediately
        config2 = D21Analysis._load_from_attr(
            ds2.kids.d21.data_source,
            mapper=D21Mapper.from_namespace(ds2.kids.d21.data_source),
        )
        assert config2 is not None
        assert config2.method == "savgol"
        assert config2.smooth == 7

        # Go back to first config (should recompute since attrs changed)
        ds3 = d21_analysis1(ds2["_input"].kids.sweep, output=ds2)

        # Check config3 immediately
        config3 = D21Analysis._load_from_attr(
            ds3.kids.d21.data_source,
            mapper=D21Mapper.from_namespace(ds3.kids.d21.data_source),
        )
        assert config3 is not None
        assert config3.method == "gradient"
        assert config3.smooth == 5


def test_tree_boundary_preserves_science_and_replaces_frequency_grid(
    simple_sweep, tmp_path
):
    original = simple_sweep.copy(deep=True)
    first = D21Analysis(method="gradient", smooth=5, resample=1)(
        simple_sweep.kids.sweep
    )
    analysis = D21Analysis(method="gradient", smooth=5, resample=2)
    size_first = first.kids.d21.f_unified.size
    updated = analysis(first["_input"].kids.sweep, output=first, force=True)
    view = updated.kids.d21
    xr.testing.assert_identical(simple_sweep, original)
    xr.testing.assert_equal(
        view.d21, analysis.make_matched(simple_sweep.kids.sweep).rename(view.d21.name)
    )
    xr.testing.assert_equal(
        view.d21_unified,
        analysis.make_unified(simple_sweep.kids.sweep).rename(view.d21_unified.name),
    )
    assert view.f_unified.size == 2 * size_first
    assert set(updated.children) == {"_input", "tolteca_kidsproc.analysis.d21"}
    assert "kids_node_type" not in updated["tolteca_kidsproc.analysis.d21"].attrs
    target = tmp_path / "derivative.zarr"
    updated.to_zarr(target)
    with xr.open_datatree(target, engine="zarr") as reopened:
        xr.testing.assert_identical(reopened.kids.d21.d21, view.d21)
        assert reopened.kids.d21.d21_analysis == analysis


def test_namespace_owns_config_and_two_independent_outputs(simple_sweep, tmp_path):
    analysis = D21Analysis(method="gradient", smooth=5)
    tree = analysis(simple_sweep.kids.sweep)
    view = tree.kids.d21
    namespace = view.data_source
    assert namespace is tree["tolteca_kidsproc.analysis.d21"]
    assert not namespace.dataset.variables
    assert set(namespace.children) == {"d21", "d21_unified"}
    key = view.mapper.get_name(view.mapper.schema.analysis)
    assert key in namespace.attrs
    for leaf in namespace.children.values():
        assert key not in leaf.attrs
        assert all(key not in var.attrs for var in leaf.dataset.variables.values())
    assert view.d21_analysis is view.d21_analysis
    assert view.d21_analysis == analysis
    assert view.d21_analysis is not analysis
    target = tmp_path / "d21-namespace.zarr"
    namespace.to_zarr(target)
    with xr.open_datatree(target, engine="zarr") as reopened:
        restored = D21View.from_namespace(reopened)
        assert restored.data_source is reopened
        assert restored.d21_analysis == analysis
        xr.testing.assert_identical(restored.d21, view.d21)
        xr.testing.assert_identical(restored.d21_unified, view.d21_unified)


def test_namespace_replacement_preserves_old_view_configuration(simple_sweep):
    first = D21Analysis(method="gradient", smooth=5, resample=1)
    tree = first(simple_sweep.kids.sweep)
    before = tree.kids.d21
    after_analysis = D21Analysis(method="gradient", smooth=7, resample=2)
    assert after_analysis(tree["_input"].kids.sweep, output=tree) is tree
    after = tree.kids.d21
    assert before.d21_analysis == first
    assert after.d21_analysis == after_analysis
    assert before.data_source is not after.data_source
    assert after.d21_unified.size == 2 * before.d21_unified.size
    np.testing.assert_allclose(before.d21, first.make_matched(simple_sweep.kids.sweep))
    np.testing.assert_allclose(
        after.d21, after_analysis.make_matched(simple_sweep.kids.sweep)
    )


def test_output_candidates_are_preserved_when_recomputing(simple_sweep, monkeypatch):
    from tollan.accessor import NameMapping

    analysis = D21Analysis()
    result = analysis(simple_sweep.kids.sweep)
    namespace = result.kids.d21.data_source
    schema = D21Mapper.schema
    namespace["d21"].dataset = (
        namespace["d21"].to_dataset().rename({schema.d21.names[0]: "derivative"})
    )
    namespace["d21_unified"].dataset = (
        namespace["d21_unified"]
        .to_dataset()
        .rename(
            {
                schema.d21_unified.names[0]: "magnitude",
                "f_unified": "frequency_grid",
                "cov_unified": "coverage",
            }
        )
    )
    monkeypatch.setattr(
        D21Mapper,
        "schema",
        replace(
            schema,
            d21=schema.d21_subpath / NameMapping.var(("absent", "derivative")),
            d21_unified=schema.d21_unified_subpath
            / NameMapping.var(("absent", "magnitude")),
            f_unified=schema.d21_unified_subpath
            / NameMapping.var(("absent_frequency", "frequency_grid")),
            cov_unified=schema.d21_unified_subpath
            / NameMapping.var(("absent_coverage", "coverage")),
        ),
    )
    for view in (
        D21View(result),
        D21View.from_namespace(namespace),
        D21View.from_namespace(namespace.copy(deep=False)),
    ):
        assert view.d21.name == "derivative"
        assert view.d21_unified.name == "magnitude"
        assert view.f_unified.name == "frequency_grid"
        assert view.cov_unified.name == "coverage"
    before = D21View(result)
    assert analysis(simple_sweep.kids.sweep, output=result, force=True) is result
    after = D21View(result)
    xr.testing.assert_identical(after.d21, before.d21)
    xr.testing.assert_identical(after.d21_unified, before.d21_unified)


@pytest.mark.parametrize("method", ["gradient", "savgol"])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("explicit_grid", [False, True])
def test_dask_frequency_and_signal_match_eager(
    multi_channel_sweep, method, multi, explicit_grid, tmp_path
):
    """Configured standalone arrays and saved workflow results accept lazy inputs."""
    source = (
        multi_channel_sweep if multi else multi_channel_sweep.isel(chan=0, drop=True)
    )
    source = source.copy(deep=False)
    source["frequency"] = source.frequency.u.to("MHz")
    lazy = source.chunk({dim: max(1, size // 3) for dim, size in source.sizes.items()})
    settings = {"method": method, "smooth": 5}
    if explicit_grid:
        frequencies = source.frequency.u.to("Hz").values
        settings.update(
            f_lims=(frequencies.min() * u.Hz, frequencies.max() * u.Hz),
            f_step=0.1 * u.MHz,
        )
    analysis = D21Analysis.model_validate(settings)
    eager = analysis(source.kids.sweep)
    matched = analysis.make_matched(lazy.kids.sweep)
    unified = analysis.make_unified(lazy.kids.sweep)
    np.testing.assert_array_equal(matched, eager.kids.d21.d21)
    np.testing.assert_array_equal(unified, eager.kids.d21.d21_unified)
    result = analysis(lazy.kids.sweep)
    for name in ("d21", "d21_unified", "f_unified", "cov_unified"):
        xr.testing.assert_identical(
            getattr(result.kids.d21, name), getattr(eager.kids.d21, name)
        )
        assert isinstance(getattr(result.kids.d21, name).data, np.ndarray)
    assert lazy.I.chunks is not None
    target = tmp_path / "d21.zarr"
    result.to_zarr(target, zarr_format=2)
    with xr.open_datatree(target, engine="zarr", chunks={}) as saved:
        xr.testing.assert_equal(saved.kids.d21.d21, eager.kids.d21.d21)
        xr.testing.assert_equal(saved.kids.d21.d21_unified, eager.kids.d21.d21_unified)
        detached = saved.kids.d21.data_source.copy(deep=False)
        xr.testing.assert_equal(
            D21View.from_namespace(detached).cov_unified, eager.kids.d21.cov_unified
        )
