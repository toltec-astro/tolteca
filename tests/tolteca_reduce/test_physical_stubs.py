"""Atomic model replacement permits physical schemas without changing axes."""

import copy
import json

import numpy as np
import pytest
import xarray as xr
from astropy.table import Table

from tolteca_datamodels.raw_obs_context import open_acquisition
from tolteca_kids.model_calibration import PHYSICAL_SCHEMAS
from tolteca_kids.model_fit import _validate_report
from tolteca_kids.model_joint import validate_joint_models
from tolteca_reduce.stubs import _pack_models


@pytest.mark.parametrize("family", ["affine", "delay"])
def test_physical_stub_replaces_model_dimensions_only(data_lmt_path, tmp_path, family):
    target_sweep_path = (
        data_lmt_path / "toltec/tcs/toltec0/"
        "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc"
    )
    with xr.open_dataset(target_sweep_path, decode_times=False) as raw:
        n = raw["Header.Toltec.ToneFreq"].shape[-1]
        schema = PHYSICAL_SCHEMAS[family]
        report = Table(
            {
                "chan": np.arange(n, dtype=np.int64),
                "f_in": np.full(n, 500e6),
                "f_out": np.full(n, 500e6),
                "flag": np.zeros(n, dtype=np.int64),
            }
        )
        for column in schema["parameters"]:
            report[column["name"]] = np.ones(n)
            report[column["name"]].unit = column["unit"] or None
        for name in ("f_in", "f_out"):
            report[name].unit = "Hz"
        report.meta.update(
            schema="kids.fit/1", model=schema["name"], model_schema=schema
        )
        report.meta["Header.Toltec.AccumLen"] = raw["Header.Toltec.AccumLen"].item()
        payload = _joint_payload(family)
        report.meta["joint_models"] = payload
        _validate_report(report)
        report.write(tmp_path / "report.ecsv", format="ascii.ecsv")
        report = Table.read(tmp_path / "report.ecsv", format="ascii.ecsv")
        stub = _pack_models(raw, {0: report})
        stub.to_netcdf(tmp_path / "model.nc")
        old_shape = raw["Data.Toltec.Is"].shape
        with open_acquisition(
            target_sweep_path, stubs=[(tmp_path / "model.nc", "override")]
        ) as data:
            assert data["Data.Toltec.Is"].shape == old_shape
            np.testing.assert_array_equal(
                data["Header.Toltec.ToneFreq"], raw["Header.Toltec.ToneFreq"]
            )
            assert (
                json.loads(data["Header.Toltec.ModelParams"].attrs["joint_models_0"])
                == payload
            )
            assert (
                json.loads(data["Header.Toltec.ModelParams"].attrs["model_schema_0"])
                == schema
            )
        # Default keeps the entire recorded model group, including its dimensions.
        if "Header.Toltec.ModelParams" in raw:
            with open_acquisition(
                target_sweep_path, stubs=[(tmp_path / "model.nc", "default")]
            ) as data:
                xr.testing.assert_equal(
                    data["Header.Toltec.ModelParams"], raw["Header.Toltec.ModelParams"]
                )
        # An isolated replacement removes the previous group's payload atomically.
        report.meta.pop("joint_models")
        replacement = _pack_models(stub, {0: report})
        assert "joint_models_0" not in replacement["Header.Toltec.ModelParams"].attrs
        report["r"].unit = "Hz"
        with pytest.raises(ValueError, match="units"):
            _validate_report(report)


def _joint_payload(family="affine"):
    return {
        "schema": "kids.joint/1",
        "groups": [
            {
                "group_id": "pair",
                "gain_family": family,
                "channels": [0, 1],
                "resonator_ids": ["221", "222"],
                "probes": [500e6, 500e6],
                "fr": [500e6 - 1e4, 500e6 + 1e4],
                "r": [1e-5, 1e-5],
                "gain_re": [1.0, 1.0],
                "gain_im": [0.0, 0.0],
                "coupling_re": [[1e-5, 1e-5], [1e-5, 1e-5]],
                "coupling_im": [[0.0, 0.0], [0.0, 0.0]],
                "iq_covariance": np.eye(4).tolist(),
                "noise_source": "unit_iq_weights",
                "fit_flag": 1 << 17,
                "rank_rtol": 1e-8,
                "condition_max": None,
            }
        ],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("channels", [0, 99]),
        ("channels", [0, 0]),
        ("probes", [500e6, 500e6 + 1]),
        ("resonator_ids", ["same", "same"]),
        ("fit_flag", -1),
        ("iq_covariance", np.zeros((4, 4)).tolist()),
        ("coupling_re", [[1e-5], [1e-5]]),
    ],
)
def test_joint_calibration_rejects_invalid_binding_or_covariance(field, value):
    payload = _joint_payload()
    # Equal responses are valid calibration data; recovery reports rank failure.
    validate_joint_models(payload, [0, 1], [500e6, 500e6])
    invalid = copy.deepcopy(payload)
    invalid["groups"][0][field] = value
    with pytest.raises(ValueError):
        validate_joint_models(invalid, [0, 1], [500e6, 500e6])
