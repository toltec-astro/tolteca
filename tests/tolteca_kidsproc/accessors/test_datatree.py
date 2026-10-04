"""Tests for DataTree support in the KIDs accessor.

Tests cover:
- open_datatree() utility function
- kids accessor availability on DataTree
- Sweep data access via DataTree (parity with Dataset)
- DataTree save/load round-trip for sweep data
- Analysis hierarchy and producing configuration preserved by NetCDF
- Metadata attributes preserved through DataTree save/load
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from astropy import units as u

from tolteca_kidsproc.accessors.dataset import (
    make_kids_dataset,
    open_datatree,
)
from tolteca_kidsproc.analysis.d21 import D21Analysis, D21Mapper

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

        with open_datatree(filepath) as result:
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

        with open_datatree(filepath) as dt_loaded:
            assert isinstance(dt_loaded, xr.DataTree)
            assert dt_loaded.kids.sweep.S21 is not None
            np.testing.assert_allclose(
                dt_loaded.kids.sweep.S21.values,
                ds.kids.sweep.S21.values,
            )


# ============================================================================
# Analysis scope/view contracts live in analysis/test_d21.py and test_scope_views.py
# ============================================================================


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
        with open_datatree(filepath) as dt_loaded:
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
        namespace = mapper.get_namespace(dt)
        matched_path = mapper.get_name(mapper.schema.d21_subpath)
        assert isinstance(matched_path, str)
        del namespace[matched_path]
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
