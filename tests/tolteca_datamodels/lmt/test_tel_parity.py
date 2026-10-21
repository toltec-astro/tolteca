"""Telescope reader parity against v2 and the LMTMC reference catalog."""

import csv
import json
import os
import subprocess
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from tolteca_datamodels.lmt import open_lmt_tel
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.lmt.types import LmtObsPgm


@pytest.mark.parametrize(
    "suffix",
    [
        "145605_00_0001",
        "145606_00_0001",
        "145608_00_0001",
        "145609_00_0001",
        "145609_00_0002",
        "145610_00_0002",
        "145611_00_0002",
    ],
)
@pytest.mark.parametrize("chunks", [None, {}])
def test_values_equal_v2(data_lmt_path, tmp_path, suffix, chunks):
    interpreter = os.environ.get("TOLTECA_V2_PYTHON")
    if not interpreter:
        pytest.skip("Set TOLTECA_V2_PYTHON to enable cross-environment parity tests")
    path = next((data_lmt_path / "tel").glob(f"*_{suffix}.nc"))
    reference = tmp_path / "v2.npz"
    subprocess.run(  # noqa: S603 -- explicit local comparison interpreter
        [
            interpreter,
            str(Path(__file__).with_name("v2_tel_reference.py")),
            str(path),
            str(reference),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    with np.load(reference) as expected, open_lmt_tel(path, chunks=chunks) as ds:
        tel = ds.lmt_tel
        meta = tel.meta
        old = json.loads(expected["metadata"].item())
        for name in (
            "obsnum",
            "subobsnum",
            "scannum",
            "obs_goal",
        ):
            assert getattr(meta, name) == old[name], name
        assert meta.instru == old["instru"] == "lmt"
        assert old["instru_component"] == "tcs" and meta.interface_kind == "tel"
        assert meta.master_code == old["master"]
        assert meta.master == old["master_name"]
        assert old["interface"] == "lmt" and meta.interface == "tel"
        assert meta.obs_pgm == old["mapping_type"].lower()
        assert ds.encoding["source"] == old["source"]
        assert isinstance(meta.filepath_info_orig, LmtFilepathInfo)
        assert meta.filepath_info_orig.ut.isoformat() == old["file_timestamp"]
        assert meta.filepath_info_orig.suffix == old["file_suffix"]
        assert meta.filepath_info_orig.ext == old["file_ext"].removeprefix(".")
        assert meta.t0.unix == old["t0"]
        assert meta.t0.format == old["t0_format"]
        assert meta.target.frame.name == old["target_frame"]
        assert meta.target_ref.frame.name == old["target_off_frame"]
        np.testing.assert_array_equal(
            [meta.target.ra.rad, meta.target.dec.rad], old["target"]
        )
        np.testing.assert_array_equal(
            [meta.target_ref.ra.rad, meta.target_ref.dec.rad], old["target_off"]
        )
        for name, name_v2 in (
            ("t", "time"),
            ("az", "az"),
            ("alt", "alt"),
            ("ra", "ra"),
            ("dec", "dec"),
            ("hold_flag", "holdflag"),
        ):
            array = getattr(tel, name)
            np.testing.assert_array_equal(
                array.to_numpy(), expected[name_v2], err_msg=name
            )
            assert array.dtype == expected[name_v2].dtype, name
            if name_v2 in old["units"]:
                assert array.u.unit == u.Unit(old["units"][name_v2])


def test_reference_lmtmc_rows(data_lmt_path):
    # The deployment points at the reference repository's data_lmt directory.
    fixtures = data_lmt_path.resolve().parent / "test_fixtures"
    rows = {}
    for name in ("lmtmc_slice.csv", "lmtmc_drive_fit.csv"):
        with (fixtures / name).open(newline="") as stream:
            rows.update(
                {Path(row["FileName"]).name: row for row in csv.DictReader(stream)}
            )
    for path in sorted((data_lmt_path / "tel").glob("tel*.nc")):
        row = rows[path.name]
        with open_lmt_tel(path) as ds:
            meta = ds.lmt_tel.meta
            assert isinstance(meta.filepath_info_orig, LmtFilepathInfo)
            assert meta.filepath_orig == row["FileName"]
            assert meta.interface == row["Instrument"]
            assert (meta.obsnum, meta.subobsnum, meta.scannum) == tuple(
                map(int, row["ObsNum"].split("."))
            )
            assert meta.obs_pgm is LmtObsPgm(row["ObsPgm"].lower())
            assert meta.obs_goal == row["ObsGoal"].lower()
            # CSV is rounded to hundredths; do not round the reader's timer.
            assert meta.t_int.to_value(u.s) == pytest.approx(
                float(row["IntegrationTime"]), abs=0.005
            )
            assert meta.master_code == 0
            assert meta.master == "tcs"
            assert meta.filepath_info_orig.master == meta.master
