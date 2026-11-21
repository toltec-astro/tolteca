"""Fresh nonlinear Tune fits through the common Python/native model report."""

import json
import os
import subprocess
from typing import Any, cast

import numpy as np
import pytest
from tollan.accessor.xarray import ensure_datatree
from tollan.utils.yaml import yaml_load

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids.block_create import BlockCreate
from tolteca_kids.model_calibration import calibration_report
from tolteca_kids.model_fit import ModelFit, ModelFitConfig
from tolteca_kids.model_joint import joint_solver
from tolteca_kids.sweep_reduce import SweepReduce
from tolteca_kidsproc._nonlinear import nonlinear_detuning
from tolteca_kidsproc.analysis import KidsModelAnalysis
from tolteca_kidsproc.model_flags import ModelFitFlag
from tolteca_reduce.stubs import _pack_models

from .test_python_model_stubs import _pair_plan


def _trajectory(solver):
    states = np.column_stack(
        [
            np.linspace(0, 0.003, 16),
            np.linspace(-0.01, 0.01, 16),
            np.linspace(0.001, 0.004, 16),
            np.linspace(0.005, -0.005, 16),
        ]
    )
    loss = solver.r * (1 + states[:, ::2])
    fr = solver.fr * (1 + 2 * solver.r * states[:, 1::2])
    y, _ = nonlinear_detuning(
        (solver.probes[None, :, None] / fr[:, None, :] - 1) / (2 * loss[:, None, :]),
        solver.b / (2 * loss[:, None, :] ** 3),
    )
    iq = solver.gain * (
        1 - np.sum(solver.coupling / (loss[:, None, :] * (1 + 2j * y)), axis=-1)
    )
    return states, fr, loss, iq


def _check_rejected_group_metadata(group):
    with pytest.raises(ValueError, match="gain family"):
        joint_solver(group | {"gain_family": "unsupported"})
    with pytest.raises(ValueError, match="strengths"):
        joint_solver(group | {"a_nl": [0.1]})


@pytest.mark.parametrize("family", ["affine", "delay"])
def test_both_tune_blocks_nonlinear_report_native_parity(
    data_lmt_path, tmp_path, family
):
    executable = os.environ.get("KIDS_MODEL_REPORT_CLI")
    if not executable:
        pytest.skip("Set KIDS_MODEL_REPORT_CLI to the native model-report test driver")
    path = (
        data_lmt_path
        / "toltec/tcs/toltec0"
        / ("toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc")
    )
    summaries = []
    with open_toltec_kids(path) as raw:
        tree = BlockCreate()(ensure_datatree(raw))
        SweepReduce()(tree)
        for block in tree.toltec_kids.blocks:
            step = ModelFit(
                ModelFitConfig(
                    backend="python",
                    analysis=KidsModelAnalysis(
                        gain_family=family, nonlinear=True, max_nfev=500
                    ),
                ),
                plan=_pair_plan(block.sweep),
            )
            step(block.tree)
            report = calibration_report(
                ModelFit.view(tree, block=block.meta.block), block
            )
            report["flag"] = np.asarray(report["flag"], dtype=np.uint32)
            [group] = report.meta["joint_models"]["groups"]
            assert max(group["a_nl"]) > 0
            _check_rejected_group_metadata(group)
            solver = joint_solver(group)
            states, fr, loss, iq = _trajectory(solver)
            result = solver(iq.real, iq.imag)
            accepted = result[2] == 0
            assert accepted.all()
            np.testing.assert_allclose(
                result[0][accepted], fr[accepted], rtol=0, atol=0.02
            )
            np.testing.assert_allclose(
                result[1][accepted], loss[accepted], rtol=1e-6, atol=1e-12
            )
            assert np.isnan(result[0][~accepted]).all()
            tag = f"{family}-block{block.meta.block}"
            ecsv, stub = tmp_path / f"{tag}.ecsv", tmp_path / f"{tag}_stub.nc"
            report.write(ecsv, format="ascii.ecsv")
            _pack_models(raw, {block.meta.block: report}).to_netcdf(stub)
            request = {
                "ecsv": str(ecsv),
                "stub": str(stub),
                "block": block.meta.block,
                "I": iq.real.tolist(),
                "Q": iq.imag.tolist(),
                "normalization": 1.0,
                "states": states.tolist(),
            }
            config, output = tmp_path / f"{tag}.json", tmp_path / f"{tag}-native.yaml"
            config.write_text(json.dumps(request))
            subprocess.run(  # noqa: S603 -- Explicit local test executable from environment.
                [executable, str(config), str(output)], check=True, capture_output=False
            )
            actual = cast("dict[str, Any]", yaml_load(output))
            assert actual["flag_definitions"] == {
                flag.name: int(flag) for flag in ModelFitFlag
            }
            for mode in ("ecsv", "stub"):
                assert actual[mode]["flags"] == result[2].tolist()
                assert actual[mode]["routed_flags"] == result[2].tolist()
                np.testing.assert_equal(actual[mode]["routed_r"], actual[mode]["r"])
                np.testing.assert_equal(actual[mode]["routed_fr"], actual[mode]["fr"])
                assert actual[mode]["independent_masked"]
                expected_x = solver.probes[None, :, None] / result[0][:, None, :] - 1
                np.testing.assert_allclose(
                    actual[mode]["routed_x"],
                    expected_x.reshape(len(iq), -1),
                    rtol=0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    actual[mode]["fr"], result[0], rtol=0, atol=0.02
                )
                np.testing.assert_allclose(
                    actual[mode]["r"], result[1], rtol=1e-6, atol=1e-12
                )
                np.testing.assert_allclose(
                    actual[mode]["response"],
                    np.c_[iq.real, iq.imag],
                    rtol=2e-12,
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    actual[mode]["jacobian"],
                    solver._response(np.zeros(4))[1],
                    rtol=2e-10,
                    atol=1e-8,
                )
                assert actual[mode]["model_report"] == group
                assert actual[mode]["accumlen"] == report.meta["Header.Toltec.AccumLen"]
            np.testing.assert_equal(actual["ecsv"], actual["stub"])
            summaries.append(
                {
                    "block": block.meta.block,
                    "family": family,
                    "a_nl": group["a_nl"],
                    "fit_flag": group["fit_flag"],
                    "accepted_samples": int(accepted.sum()),
                    "sample_flags": result[2].tolist(),
                    "max_forward_difference": float(
                        np.max(
                            abs(
                                np.asarray(actual["stub"]["response"])
                                - np.c_[iq.real, iq.imag]
                            )
                        )
                    ),
                    "max_fr_difference_hz": float(
                        np.max(
                            abs(
                                np.asarray(actual["stub"]["fr"])[accepted]
                                - result[0][accepted]
                            )
                        )
                    )
                    if accepted.any()
                    else None,
                    "max_r_difference": float(
                        np.max(
                            abs(
                                np.asarray(actual["stub"]["r"])[accepted]
                                - result[1][accepted]
                            )
                        )
                    )
                    if accepted.any()
                    else None,
                }
            )
    (tmp_path / "parity-summary.json").write_text(json.dumps(summaries, indent=2))
    _check_measured_stream(report, data_lmt_path, tmp_path, family, executable)


def _check_measured_stream(report, data_lmt_path, tmp_path, family, executable):
    """Use the final Tune block on all measured samples of the following scan."""
    path = (
        data_lmt_path
        / "toltec/tcs/toltec0"
        / ("toltec0_145609_000_0002_2025_10_31_09_27_57.nc")
    )
    [group] = report.meta["joint_models"]["groups"]
    solver = joint_solver(group)
    with open_toltec_kids(path) as raw:
        arrays = []
        for key in "IQ":
            values = raw[f"Data.Toltec.{key}s"]
            arrays.append(values.isel({values.dims[1]: group["channels"]}).values)
        packets = raw["Data.Toltec.Ts"].values[:, 3].astype(np.uint32)
        reset = np.r_[False, np.diff(packets) != 1]
        assert not reset.any()
        normalization = (
            float(raw["Header.Toltec.AccumLen"].item())
            / report.meta["Header.Toltec.AccumLen"]
        )
        probes = raw["Header.Toltec.ToneFreq"].values[0, group["channels"]] + float(
            raw["Header.Toltec.LoCenterFreq"].item()
        )
        np.testing.assert_allclose(probes, solver.probes, rtol=0, atol=1e-6)
        stub = tmp_path / f"{family}-measured_stub.nc"
        _pack_models(raw, {0: report}).to_netcdf(stub)
    result = solver(*arrays, normalization=normalization, reset=reset)
    assert len(result[2]) == 7802
    assert (result[2] == 0).all()
    ecsv = tmp_path / f"{family}-block1.ecsv"
    config = tmp_path / f"{family}-measured.json"
    output = tmp_path / f"{family}-measured-native.yaml"
    config.write_text(
        json.dumps(
            {
                "ecsv": str(ecsv),
                "stub": str(stub),
                "block": 0,
                "I": arrays[0].tolist(),
                "Q": arrays[1].tolist(),
                "normalization": normalization,
                "states": [np.zeros(4).tolist()],
            }
        )
    )
    subprocess.run(  # noqa: S603 -- Explicit native test executable.
        [executable, str(config), str(output)], check=True, capture_output=True
    )
    actual = cast("dict[str, Any]", yaml_load(output))
    for mode in ("ecsv", "stub"):
        np.testing.assert_array_equal(actual[mode]["flags"], result[2])
        np.testing.assert_array_equal(actual[mode]["routed_flags"], result[2])
        np.testing.assert_allclose(actual[mode]["fr"], result[0], rtol=0, atol=0.02)
        np.testing.assert_allclose(actual[mode]["r"], result[1], rtol=1e-6, atol=1e-12)
        np.testing.assert_equal(actual[mode]["routed_fr"], actual[mode]["fr"])
    states = np.empty((len(result[0]), 4))
    states[:, ::2] = result[1] / solver.r - 1
    states[:, 1::2] = (result[0] / solver.fr - 1) / (2 * solver.r)
    np.savez(
        tmp_path / f"{family}-measured.npz",
        fr=result[0],
        r=result[1],
        flags=result[2],
        states=states,
    )
    summary = {
        "source": str(path),
        "stub": str(stub),
        "family": family,
        "samples": len(result[2]),
        "accepted": int((result[2] == 0).sum()),
        "reset_count": int(reset.sum()),
        "normalization": normalization,
        "state_step_max": solver.state_step_max,
        "first_state_distance": float(abs(states[0]).max()),
        "largest_step": float(abs(np.diff(states, axis=0)).max()),
        "max_fr_difference_hz": float(
            abs(np.asarray(actual["stub"]["fr"]) - result[0]).max()
        ),
        "max_r_difference": float(
            abs(np.asarray(actual["stub"]["r"]) - result[1]).max()
        ),
    }
    (tmp_path / f"{family}-measured-summary.json").write_text(
        json.dumps(summary, indent=2)
    )
