"""The report's bias/noise axes must describe the same fixed-drive response."""

import json
import runpy
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from tolteca_kidsproc._nonlinear import nonlinear_detuning


def test_working_zone_predicts_calibration_error_and_not_noise():
    """A finite signal reproduces the signed transfer error behind the plot."""
    module = runpy.run_path(
        str(
            Path(__file__).parents[2]
            / "examples/tune_model_comparison/network_report.py"
        )
    )
    r, a, xd = 2e-5, 0.5, -1e-5
    b = 2 * a * r**3
    x = xd - b / (r**2 + xd**2)
    result = module["sensitivity"](r, xd, b)
    step = 1e-10
    y, _ = nonlinear_detuning(np.array([x - step, x + step]) / (2 * r), a)
    driven = 2 * r * y
    for epsilon in (-0.1, 0.1):
        recovered = driven - b * (1 + epsilon) / (r**2 + driven**2)
        bias = np.diff(recovered).item() / (2 * step) - 1
        np.testing.assert_allclose(abs(bias), result["bias_10pct"], rtol=1e-7)
    assert result["readout_noise_ratio"] != result["bias_10pct"]


def test_fixed_drive_recovers_intrinsic_detuning_when_loss_changes():
    """The fixed-b hypothesis recovers x_opt despite a changing loss."""
    r_ref, a_ref = 2e-5, 0.5
    b = 2 * a_ref * r_ref**3
    r = r_ref * np.linspace(0.95, 1.05, 21)
    x = np.linspace(-3e-5, -2e-5, len(r))
    a = b / (2 * r**3)
    y, _ = nonlinear_detuning(x / (2 * r), a)
    driven = 2 * r * y
    # Construct actual normalized IQ, then use the declared singlet recovery.
    coupling = 3e-6 + 1e-6j
    gain = 1.1 - 0.3j
    iq = gain * (1 - coupling / (r + 1j * driven))
    w = coupling / (1 - iq / gain)
    recovered_x = w.imag - b / (w.real**2 + w.imag**2)
    np.testing.assert_allclose(w.real, r, rtol=1e-12)
    np.testing.assert_allclose(recovered_x, x, rtol=1e-12)
    np.testing.assert_allclose(a / a_ref, (r_ref / r) ** 3)
    assert not np.allclose(driven, x, rtol=0, atol=1e-7)


def test_drive_calibration_error_couples_loss_into_returned_signal():
    """The loss term must not be hidden by the pure-frequency bias scenario."""
    r, a, xd = 2e-5, 0.5, -1e-5
    b = 2 * a * r**3
    d = r**2 + xd**2
    x = xd - b / d
    slope = 1 + 2 * b * xd / d**2
    mixing = 2 * b * r / d**2
    step, epsilon = 1e-10, 0.1
    losses = r + np.array([-step, step])
    y, _ = nonlinear_detuning(x / (2 * losses), b / (2 * losses**3))
    driven = 2 * losses * y
    recovered = driven - b * (1 + epsilon) / (losses**2 + driven**2)
    measured_derivative = np.diff(recovered).item() / (2 * step)
    np.testing.assert_allclose(measured_derivative, epsilon * mixing / slope, rtol=1e-7)

    # A common loading perturbation changes both optical detuning and loss.
    # This synthetic beta checks the transfer formula, not a measured ratio.
    beta = 0.04
    optical = x + np.array([-step, step])
    losses = r + beta * np.array([-step, step])
    y, _ = nonlinear_detuning(optical / (2 * losses), b / (2 * losses**3))
    effective = 2 * losses * y
    recovered = effective - b * (1 + epsilon) / (losses**2 + effective**2)
    transfer = 1 + epsilon * ((slope - 1) + mixing * beta) / slope
    np.testing.assert_allclose(
        np.diff(recovered).item() / (2 * step), transfer, rtol=1e-7
    )


def test_frequency_spectrum_preserves_frequency_variance():
    """The exported density integrates to Hz², not fractional-detuning power."""
    module = runpy.run_path(
        str(
            Path(__file__).parents[2]
            / "examples/tune_model_comparison/network_quality.py"
        )
    )
    rate, amplitude, tone = 128.0, 12.0, 8.0
    times = np.arange(4096) / rate
    values = 5e8 + amplitude * np.sin(2 * np.pi * tone * times)
    frequency, power = module["spectrum"](values, rate)
    assert frequency[np.argmax(power)] == tone
    variance = power.sum() * (frequency[1] - frequency[0])
    np.testing.assert_allclose(variance, amplitude**2 / 2, rtol=1e-8)
    _, double_power = module["spectrum"](2 * values, rate)
    np.testing.assert_allclose(double_power, 4 * power)


def test_zero_nonlinearity_stays_linear_when_loss_changes():
    """Changing loss cannot create a nonlinear correction from a zero-b fit."""
    from tolteca_kidsproc.joint_timestream import JointTimestreamSolver

    # Exercise the maintained solver, not a second implementation of its equation.
    solver = JointTimestreamSolver(
        [500e6],
        [500e6],
        [2e-5],
        [1 + 0j],
        [[3e-6 + 1e-6j]],
        np.eye(2),
        channels=[0],
        resonator_ids=[0],
        b=[0.0],
    )
    states = np.zeros((21, 2))
    states[:, 0] = np.linspace(-0.1, 0.1, 21)
    states[:, 1] = np.linspace(-0.05, 0.05, 21)
    iq, _ = solver._response(states)
    fr, r, flags, *_ = solver(iq[:, :1], iq[:, 1:])
    assert not np.any(flags)
    np.testing.assert_allclose(r[:, 0], 2e-5 * (1 + states[:, 0]), rtol=1e-10)
    np.testing.assert_allclose(fr[:, 0], 500e6 * (1 + 4e-5 * states[:, 1]), rtol=1e-12)
    assert np.array_equal(solver.b, [0.0])


def test_projection_preserves_assumed_signal_and_uses_only_training_half():
    """Held-out fluctuations cannot tune the reported projection weights."""
    module = runpy.run_path(
        str(
            Path(__file__).parents[2]
            / "examples/tune_model_comparison/network_response.py"
        )
    )
    rng = np.random.default_rng(145609)
    rate, count, beta = 128.0, 8192, 0.03
    common, extra = rng.normal(size=(2, count))
    x, r = 1e-7 * common, 1e-7 * (0.6 * common + 0.4 * extra)
    first, _ = module["assess"](x, r, rate, beta, [0.02, 0.04])
    changed = x.copy()
    changed[count // 2 :] *= 10
    second, _ = module["assess"](changed, r, rate, beta, [0.02, 0.04])
    assert first["weight_x"] == second["weight_x"]
    assert first["weight_r"] == second["weight_r"]
    np.testing.assert_allclose(first["assumed_signal_transfer"], 1.0)
    assert first["projected_to_x_asd"] < 1
    assert second["projected_asd_5_20_hz"] > first["projected_asd_5_20_hz"]
    # A different physical response direction has a different signal transfer.
    assert first["transfer_beta16"] != first["transfer_beta84"]


def test_parity_reports_different_failure_masks():
    """Numerical agreement among survivors must not conceal dropped samples."""
    module = runpy.run_path(
        str(
            Path(__file__).parents[2]
            / "examples/tune_model_comparison/network_coverage.py"
        )
    )
    result = module["same_arrays"]([1.0, np.nan], [1.0, 2.0], atol=1e-12)
    assert not result["same_finite_mask"]
    assert not result["within_tolerance"]
    assert result["max_abs_difference"] == 0
    # Hz-valued frequencies must not gain an implicit relative tolerance.
    large = module["same_arrays"]([5e8], [5e8 + 0.1], atol=0.02)
    assert not large["within_tolerance"]


def test_stub_transport_checks_samples_after_python_prefix(tmp_path):
    """A late transport corruption must fail even when the prefix agrees."""
    module = runpy.run_path(
        str(
            Path(__file__).parents[2]
            / "examples/tune_model_comparison/network_coverage.py"
        )
    )
    report = Table(meta={"joint_models": {"groups": [{"group_id": "3"}]}})
    for mode in ("ecsv", "stub"):
        for suffix, fields in (("", ("r", "x")), ("group-3", ("fr", "r"))):
            folder = tmp_path / "native" / mode / suffix
            folder.mkdir(parents=True, exist_ok=True)
            np.zeros(300, dtype="uint16").tofile(folder / "flags.bin")
            for field in fields:
                np.ones(300).tofile(folder / f"{field}.bin")
    check = module["compare_transports"]
    module["require_parity"](check(tmp_path, report))
    corrupted = np.ones(300)
    corrupted[-1] = np.nan
    corrupted.tofile(tmp_path / "native/stub/group-3/r.bin")
    with pytest.raises(ValueError, match="same_finite_mask"):
        module["require_parity"](check(tmp_path, report))


def test_saved_parity_uses_exact_model_and_acquired_probe_order(
    timestream_path, monkeypatch
):
    """Reusing states must not compare a different model or reordered I/Q."""
    import copy

    import xarray as xr
    from tollan.accessor.xarray import ensure_datatree

    from tolteca_datamodels.toltec import open_toltec_kids
    from tolteca_kids import BlockCreate
    from tolteca_kids.timestream_reduce import TimestreamReduce

    module = runpy.run_path(
        str(
            Path(__file__).parents[2]
            / "examples/tune_model_comparison/network_coverage.py"
        )
    )
    read = module["saved_python_results"]
    with open_toltec_kids(timestream_path) as acquisition:
        raw = acquisition.isel(time=slice(0, 4))
        i, q = (raw[name].to_numpy() for name in ("Data.Toltec.Is", "Data.Toltec.Qs"))
        # The saved channel order differs from acquisition order; group identity
        # also differs from its ordinal tree node name.
        tree = BlockCreate()(ensure_datatree(raw.toltec_kids.select_channels([5, 0])))
        block = tree.toltec_kids.blocks[0].tree
        TimestreamReduce(psd=None)(block)
        model = {"group_id": "physical-group", "fr": [5e8], "channels": [5, 0]}
        block["solved_groups"] = xr.DataTree(
            dataset=xr.Dataset(attrs={"iq_normalization": 4.0})
        )
        block["solved_groups/groups/0"] = xr.DataTree(
            dataset=xr.Dataset(
                {
                    "fr": (("time", "resonator"), np.full((4, 1), 5e8)),
                    "r": (("time", "resonator"), np.full((4, 1), 2e-5)),
                    "flags": ("time", np.zeros(4, dtype="uint16")),
                },
                attrs={"model": model},
            )
        )
        monkeypatch.setitem(
            read.__globals__, "open_kids_product", lambda _: nullcontext(tree)
        )
        report = Table(meta={"joint_models": {"groups": [model]}})
        actual = read("saved/index.yaml", report, i, q, 4.0)
        assert list(actual) == ["physical-group"]
        np.testing.assert_array_equal(actual["physical-group"][0], np.full((4, 1), 5e8))
        with pytest.raises(ValueError, match="normalization"):
            read("saved/index.yaml", report, i, q, 2.0)
        changed = copy.deepcopy(report)
        changed.meta["joint_models"]["groups"][0]["fr"][0] += 1
        with pytest.raises(ValueError, match="different fitted group"):
            read("saved/index.yaml", changed, i, q, 4.0)
        wrong = i.copy()
        wrong[0, 5] += 1
        with pytest.raises(AssertionError):
            read("saved/index.yaml", report, wrong, q, 4.0)


def test_coverage_preserves_joint_components_and_missing_selection(tmp_path):
    """A numeric group ID survives YAML while unselected components stay counted."""
    module = runpy.run_path(
        str(
            Path(__file__).parents[2]
            / "examples/tune_model_comparison/network_report.py"
        )
    )
    pd.DataFrame(
        {"group_id": [1, 1, 2], "resonator_id": [0, 1, 2], "multiplicity": [2, 2, 1]}
    ).to_csv(tmp_path / "components.csv", index=False)
    pd.DataFrame({"chan": [0, 1], "group_id": [1, 1]}).to_csv(
        tmp_path / "channels.csv", index=False
    )
    (tmp_path / "request.json").write_text(json.dumps({"samples": 2, "channels": 2}))
    native = tmp_path / "native/stub"
    group = native / "group-1"
    group.mkdir(parents=True)
    (native / "summary.yaml").write_text("joint: [{group_id: 1}]\n")
    np.full((2, 2), 8, dtype="uint16").tofile(native / "flags.bin")
    np.array([0, 128], dtype="uint16").tofile(group / "flags.bin")
    result = module["component_coverage"](tmp_path, tmp_path / "components.csv")
    assert result.native_fraction.tolist() == [0.5, 0.5, 0]
    assert result.native_route.tolist() == ["joint", "joint", "unselected"]
