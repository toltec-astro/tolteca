"""Diagnostic step uncertainty must not alter sweep selection or fitting."""

import json

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError
from tollan.accessor.xarray import ensure_datatree

from tolteca_kids import (
    ModelFit,
    ModelFitConfig,
    SweepBitMask,
    SweepCheck,
    SweepCheckConfig,
    SweepCheckView,
    SweepReduce,
)
from tolteca_kids.sweep_check import _step_unc_flags
from tolteca_kidsproc.analysis.model_fit import KidsModelAnalysis


def test_uncertainty_compares_scatter_not_resonance_depth():
    signal = np.ones((4, 9), dtype=complex) * (1 + 1j)
    signal[:, 4] *= 1e-5  # A deep resonance alone must not flag a step.
    unc_i = np.ones(signal.shape) * 0.02
    unc_q = np.ones(signal.shape) * 0.04
    unc_i[0, 2] = 0.08
    unc_q[0, 6] = 0.16
    unc_i[1] = 0
    unc_q[1] = 0  # Constant zero scatter has no statistical outliers.
    unc_i[2] = np.nan  # Q must retain its independent reference.
    unc_q[3, 0] = np.inf
    signal[3, 1] = np.nan
    flags, ref_i, ref_q, std_i, std_q = _step_unc_flags(
        signal, unc_i, unc_q, limits=(0, np.inf), n_sigma=5
    )
    expected = np.zeros(signal.shape, dtype=int)
    expected[0, [2, 6]] = int(SweepBitMask.unc_outlier)
    expected[2, :] = int(SweepBitMask.unc_nonfinite)
    expected[3, :2] = int(SweepBitMask.unc_nonfinite)
    np.testing.assert_array_equal(flags, expected)
    np.testing.assert_array_equal(ref_i[[0, 3]], [0.02, 0.02])
    np.testing.assert_array_equal(ref_q[[0, 2, 3]], [0.04, 0.04, 0.04])
    assert np.isnan(ref_i[2]) and np.isnan(std_i[2])
    assert ref_i[1] == ref_q[1] == std_i[1] == std_q[1] == 0
    assert np.isfinite(ref_q).all()


@pytest.mark.parametrize("n_sigma", [0, -1, np.nan, np.inf])
def test_uncertainty_threshold_is_positive_and_finite(n_sigma):
    with pytest.raises(ValidationError):
        SweepCheckConfig(step_unc_outlier_n_sigma_min=n_sigma)


@pytest.mark.parametrize("limits", [(-1, 1), (0, -1), (2, 1), (np.nan, 1), (0, np.nan)])
def test_absolute_limits_are_nonnegative_and_ordered(limits):
    with pytest.raises(ValidationError):
        SweepCheckConfig(step_unc_min=limits[0], step_unc_max=limits[1])


def test_absolute_limits_and_statistical_outliers_are_independent():
    signal = np.ones((2, 101), dtype=complex)
    # Equal excursions are unusual in a stable channel but ordinary in a broad one.
    unc_i = np.stack((np.linspace(0.019, 0.021, 101), np.linspace(0.001, 0.039, 101)))
    unc_i[:, 0] = 0.035
    unc_i[:, 1] = 0.005
    unc_q = np.full_like(unc_i, 0.02)
    flags, *refs = _step_unc_flags(signal, unc_i, unc_q, limits=(0, np.inf), n_sigma=5)
    assert flags[0, 0] & int(SweepBitMask.unc_outlier)
    assert flags[0, 1] & int(SweepBitMask.unc_outlier)
    assert not np.any(flags[1])
    limited, *limited_refs = _step_unc_flags(
        signal, unc_i, unc_q, limits=(0.01, 0.03), n_sigma=5
    )
    assert np.all(limited[:, 0] & int(SweepBitMask.unc_high))
    assert np.all(limited[:, 1] & int(SweepBitMask.unc_low))
    np.testing.assert_array_equal(
        flags & int(SweepBitMask.unc_outlier), limited & int(SweepBitMask.unc_outlier)
    )
    for original, bounded in zip(refs, limited_refs, strict=True):
        np.testing.assert_array_equal(original, bounded)


def test_real_flags_are_diagnostic_persisted_and_do_not_change_fit(
    target_sweep_path, tmp_path
):
    with xr.open_dataset(target_sweep_path) as raw:
        tree = ensure_datatree(raw.toltec_kids.select_channels([8, 9, 10]).load())
    SweepReduce()(tree)
    block = tree.toltec_kids.blocks[0]
    original = block.sweep.data_source.copy(deep=True)
    # Diagnose a known finite scatter increase without changing mean IQ.
    block.sweep.unc_Q.values[0, 20] = np.median(block.sweep.unc_Q.values[0]) * 8
    measured = block.sweep.data_source.copy(deep=True)
    configs = {"normal": 15.0, "loose": 1e6}
    checks = {}
    fits = {}
    sweep = block.sweep
    frequency = sweep.frequency.values
    signal = sweep.I.values + 1j * sweep.Q.values
    plan = xr.Dataset(
        {
            "group_id": ("row", ["g"]),
            "chan": ("row", [8]),
            "resonator_id": ("row", ["r"]),
            "sample_start": ("row", [0]),
            "sample_stop": ("row", [signal.shape[1]]),
            "fr_initial": ("row", [frequency[0, np.argmin(abs(signal[0]))]]),
            "r_initial": ("row", [1 / 20000]),
        }
    )
    for alias, n_sigma in configs.items():
        SweepCheck(
            alias=alias,
            chunk_size=5,
            bad_chan_bits=0,
            step_unc_outlier_n_sigma_min=n_sigma,
        )(block.tree)
        checks[alias] = SweepCheck.view(block.tree, alias=alias)
        ModelFit(
            ModelFitConfig(
                alias=alias,
                check_alias=alias,
                analysis=KidsModelAnalysis(max_nfev=40),
            ),
            plan=plan,
        )(block.tree)
        fits[alias] = ModelFit.view(block.tree, alias=alias)
    normal, loose = checks["normal"], checks["loose"]
    assert normal.bitmask[0, 20] & int(SweepBitMask.unc_outlier)
    unc_bits = int(
        SweepBitMask.unc_low
        | SweepBitMask.unc_high
        | SweepBitMask.unc_outlier
        | SweepBitMask.unc_nonfinite
    )
    assert not np.any(loose.bitmask & unc_bits)
    np.testing.assert_array_equal(
        (normal.bitmask & int(SweepBitMask.spike)) != 0, normal.mask_spike
    )
    assert not np.any(normal.bitmask_chan & unc_bits)
    assert not np.any(normal.bitmask_chunk & unc_bits)
    for group in (
        "samples",
        "channels",
        "chunks",
        "d21",
        "d21_chunks",
        "sweep",
        "readout_power",
    ):
        left, right = (
            v.data_source[group].to_dataset(inherit=False) for v in (normal, loose)
        )
        if group == "samples":
            np.testing.assert_array_equal(
                left.bitmask.values & ~unc_bits, right.bitmask.values & ~unc_bits
            )
            left, right = (v.drop_vars("bitmask") for v in (left, right))
        elif group == "channels":
            # Clipping thresholds can change the diagnostic center and spread.
            refs = [
                "chan_unc_I_ref",
                "chan_unc_Q_ref",
                "chan_unc_I_std",
                "chan_unc_Q_std",
            ]
            left, right = (v.drop_vars(refs) for v in (left, right))
        xr.testing.assert_identical(left, right)
    for name in ("samples", "fit", "chan_prop"):
        xr.testing.assert_identical(
            getattr(fits["normal"], name), getattr(fits["loose"], name)
        )
    xr.testing.assert_identical(block.sweep.data_source, measured)
    xr.testing.assert_identical(block.sweep.I, original.I)
    xr.testing.assert_identical(block.sweep.Q, original.Q)

    _assert_check_roundtrip(normal, sweep, tmp_path / "check.zarr")

    # Reinvocation populates the diagnostics on an earlier saved check, even
    # when its effective configuration otherwise matches the current defaults.
    namespace = block.tree["sweep_check_normal"]
    expected_flags = normal.bitmask.copy()
    namespace["samples/bitmask"].values[:] &= ~unc_bits
    config_name = normal.mapper.get_name(normal.mapper.schema.config)
    earlier_config = json.loads(namespace.attrs[config_name])
    earlier_config.pop("step_unc_outlier_n_sigma_min")
    namespace.attrs[config_name] = json.dumps(earlier_config)
    SweepCheck(alias="normal", chunk_size=5, bad_chan_bits=0)(block.tree)
    restored = SweepCheck.view(block.tree, alias="normal")
    np.testing.assert_array_equal(restored.bitmask, expected_flags)


def _assert_check_roundtrip(normal, sweep, destination):
    normal.data_source.to_zarr(destination, zarr_format=2, consolidated=True)
    with xr.open_datatree(destination, engine="zarr", consolidated=True) as saved:
        restored = SweepCheckView(saved)
        np.testing.assert_array_equal(restored.bitmask, normal.bitmask)
        assert restored.bitmask.dtype == normal.bitmask.dtype
        assert restored.config.step_unc_outlier_n_sigma_min == 15
        assert restored.config.step_unc_min == 0
        assert restored.config.step_unc_max == np.inf
        assert restored.chan_unc_I_ref.unit == (
            sweep.unc_I.u.unit or u.dimensionless_unscaled
        )
        np.testing.assert_array_equal(restored.chan_unc_I_ref, normal.chan_unc_I_ref)
        np.testing.assert_array_equal(restored.chan_unc_Q_ref, normal.chan_unc_Q_ref)
        np.testing.assert_array_equal(restored.chan_unc_I_std, normal.chan_unc_I_std)
        np.testing.assert_array_equal(restored.chan_unc_Q_std, normal.chan_unc_Q_std)
        flags = saved["samples/bitmask"]
        assert flags.attrs["flag_masks"] == [int(flag) for flag in SweepBitMask]
        assert flags.attrs["flag_meanings"] == " ".join(
            flag.name for flag in SweepBitMask
        )
        assert flags.attrs["diagnostic_flag_masks"] == [
            int(SweepBitMask.unc_nonfinite),
            int(SweepBitMask.unc_low),
            int(SweepBitMask.unc_high),
            int(SweepBitMask.unc_outlier),
        ]
