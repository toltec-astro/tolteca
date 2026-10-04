"""Tests for DataTree support in the KIDs accessor.

Tests cover:
- open_datatree() utility function
- kids accessor availability on DataTree
- Sweep data access via DataTree (parity with Dataset)
- DataTree save/load round-trip for sweep data
- D21Analysis high-level API: ds.kids.d21 accessor after analyzer(ds.kids.sweep)
- D21View on a detached analysis namespace
- D21View auto-resolution to hierarchical child node
- Metadata attributes preserved through DataTree save/load
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
from astropy import units as u

from tolteca_kidsproc.accessors.dataset import (
    make_kids_dataset,
    open_datatree,
)
from tolteca_kidsproc.analysis.d21 import D21Analysis, D21Mapper, D21View

if TYPE_CHECKING:
    from pathlib import Path


# ============================================================================
# Shared fixture helpers
# ============================================================================


def _make_lorentzian_sweep() -> xr.Dataset:
    """Create a 1-D sweep dataset with a Lorentzian resonance at 1.05 GHz.

    Returns a simple synthetic dataset with 101 frequency points spanning
    1.0-1.1 GHz and a Lorentzian resonance centred at 1.05 GHz.
    """
    freq = np.linspace(1e9, 1.1e9, 101) << u.Hz
    s21 = 1 / (1 + 1j * (freq.value - 1.05e9) / 1e6)
    return make_kids_dataset(s21.real, s21.imag, frequency=freq)


# ============================================================================
# TestDataTreeOpenAndAccess
# ============================================================================


class TestDataTreeOpenAndAccess:
    """Test DataTree creation, opening, and basic accessor availability."""

    def test_open_datatree_returns_datatree(self, tmp_path: Path) -> None:
        """open_datatree returns a DataTree with the kids accessor."""
        ds = _make_lorentzian_sweep()
        dt = xr.DataTree(ds)
        filepath = tmp_path / "test.nc"
        dt.to_netcdf(filepath)

        result = open_datatree(filepath)

        assert isinstance(result, xr.DataTree)
        assert hasattr(result, "kids")

    def test_datatree_sweep_accessor_available(self) -> None:
        """DataTree kids accessor exposes the sweep view."""
        ds = _make_lorentzian_sweep()
        dt = xr.DataTree(ds)

        assert dt.kids.sweep.S21 is not None
        assert dt.kids.sweep.frequency is not None

    def test_datatree_and_dataset_sweep_parity(self) -> None:
        """DataTree and Dataset sweep views return identical S21 values."""
        ds = _make_lorentzian_sweep()
        dt = xr.DataTree(ds)

        np.testing.assert_array_equal(
            dt.kids.sweep.S21.values,
            ds.kids.sweep.S21.values,
        )

    def test_datatree_save_load_sweep(self, tmp_path: Path) -> None:
        """Sweep data survives a DataTree save/load round-trip via NetCDF."""
        ds = _make_lorentzian_sweep()
        dt = xr.DataTree(ds)
        filepath = tmp_path / "sweep.nc"
        dt.to_netcdf(filepath)

        dt_loaded = open_datatree(filepath)

        assert isinstance(dt_loaded, xr.DataTree)
        assert dt_loaded.kids.sweep.S21 is not None
        np.testing.assert_allclose(
            dt_loaded.kids.sweep.S21.values,
            ds.kids.sweep.S21.values,
        )


# ============================================================================
# TestDataTreeD21Integration
# ============================================================================


class TestDataTreeD21Integration:
    """Test D21 analysis integration via the DataTree + kids.d21 accessor."""

    def test_d21_call_api_produces_accessible_results(self) -> None:
        """D21Analysis.__call__ stores results accessible via ds.kids.d21."""
        ds = _make_lorentzian_sweep()
        analyzer = D21Analysis(method="gradient", smooth=5)
        ds_with_d21 = analyzer(ds.kids.sweep)

        # High-level accessor API
        assert ds_with_d21.kids.d21.d21 is not None
        assert ds_with_d21.kids.d21.d21_unified is not None

    def test_d21_view_config_matches_analyzer(self) -> None:
        """D21View.d21_analysis reflects the analyzer configuration."""
        ds = _make_lorentzian_sweep()
        analyzer = D21Analysis(method="gradient", smooth=5)
        ds_with_d21 = analyzer(ds.kids.sweep)

        config = ds_with_d21.kids.d21.d21_analysis

        assert config.method == "gradient"
        assert config.smooth == 5

    def test_d21_view_on_detached_namespace(self) -> None:
        """D21View represents the supplied detached analysis namespace."""
        ds = _make_lorentzian_sweep()
        ds_with_d21 = D21Analysis(method="gradient", smooth=5)(ds.kids.sweep)
        # A detached namespace retains its configuration and both output leaves.
        dt = ds_with_d21.kids.d21.data_source.copy(deep=False)

        mapper = D21Mapper.from_namespace(dt)
        assert mapper.namespace_path == "."
        assert D21View.from_namespace(dt).d21 is not None
        assert D21View.from_namespace(dt).d21_unified is not None

    def test_d21_view_on_hierarchical_datatree_resolves_child(self) -> None:
        """D21View auto-resolves to the child node in a hierarchical DataTree."""
        ds = _make_lorentzian_sweep()
        analyzer = D21Analysis(method="gradient", smooth=5)
        ds_with_d21 = analyzer(ds.kids.sweep)

        mapper = D21Mapper.from_data_source(ds_with_d21)
        result_node = mapper.get_namespace(ds_with_d21)
        assert D21View.from_namespace(result_node).d21 is not None
        assert D21View.from_namespace(result_node).d21_unified is not None
        assert ds_with_d21["_input"].kids.sweep.S21 is not None

    def test_d21_view_raises_when_no_analysis_present(self) -> None:
        """D21View raises KeyError when its namespace is absent."""
        ds = _make_lorentzian_sweep()
        dt = xr.DataTree(ds)

        with pytest.raises(KeyError, match="tolteca_kidsproc.analysis.d21"):
            _ = dt.kids.d21.d21


# ============================================================================
# TestDataTreeMetadata
# ============================================================================


class TestDataTreeMetadata:
    """Test metadata and structure preservation through DataTree save/load."""

    def test_dataset_attrs_preserved_after_save_load(self, tmp_path: Path) -> None:
        """Dataset attributes survive a DataTree NetCDF save/load round-trip."""
        ds = _make_lorentzian_sweep()
        ds.attrs["obs_num"] = 42
        ds.attrs["array"] = "a1100"

        dt = xr.DataTree(ds)
        filepath = tmp_path / "metadata.nc"
        dt.to_netcdf(filepath)
        dt_loaded = open_datatree(filepath)

        assert dt_loaded.ds.attrs["obs_num"] == 42
        assert dt_loaded.ds.attrs["array"] == "a1100"

    def test_hierarchical_child_structure_survives_save_load(
        self,
        tmp_path: Path,
    ) -> None:
        """Hierarchical DataTree child node structure is preserved through save/load."""
        ds = _make_lorentzian_sweep()
        analyzer = D21Analysis(method="gradient", smooth=5)
        dt = analyzer(ds.kids.sweep)
        mapper = D21Mapper.from_data_source(dt)
        result_path = mapper.namespace_path
        data_path = "_input"
        expected = dt.kids.d21.d21_unified
        # Export the real-valued output; configuration stays on its namespace.
        del dt[result_path][mapper.get_name(mapper.schema.d21_subpath)]
        dt[
            f"{result_path}/{mapper.get_name(mapper.schema.d21_unified_subpath)}"
        ].dataset = expected.to_dataset()
        filepath = tmp_path / "hierarchical.nc"
        dt.to_netcdf(filepath)
        with open_datatree(filepath) as loaded:
            assert set(loaded.children) == {result_path, data_path}
            assert loaded[data_path].kids.sweep.S21 is not None
            xr.testing.assert_equal(loaded.kids.d21.d21_unified, expected)
            assert loaded.kids.d21.d21_analysis == analyzer
