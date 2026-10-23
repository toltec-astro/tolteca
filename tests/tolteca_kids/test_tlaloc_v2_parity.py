"""Real acquisitions reproduce the complete production-v2 Tlaloc handoff."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from tollan.config import RuntimeContext

from tolteca_kids.product import open_sweep_product
from tolteca_kids.runner import run_source
from tolteca_kids.tlaloc_output import TlalocOutput


@pytest.mark.parametrize(
    ("obsnum", "suffix", "block"),
    [
        (145480, "vnasweep", 0),
        (145485, "targsweep", 0),
        (145483, "tune", 0),
        (145483, "tune", 1),
        (145605, "vnasweep", 0),
        (145606, "targsweep", 0),
        (145609, "tune", 0),
        (145609, "tune", 1),
    ],
)
def test_full_v2_tlaloc_handoff(data_lmt_path, tmp_path, obsnum, suffix, block):
    interpreter = os.environ.get("TOLTECA_V2_PYTHON")
    if not interpreter:
        pytest.skip("Set TOLTECA_V2_PYTHON for production-v2 comparison")
    source = next(
        (data_lmt_path / "toltec/tcs/toltec0").glob(f"*{obsnum}*_{suffix}.nc")
    )
    expected_root = tmp_path / "v2"
    comparison = subprocess.run(  # noqa: S603 -- selected production interpreter
        [
            interpreter,
            str(Path(__file__).with_name("v2_tlaloc_reference.py")),
            str(source),
            str(expected_root),
            str(block),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    assert comparison.returncode == 0, comparison.stderr
    config = {
        "kids": {
            "sweep_check": {"enabled": True},
            "kids_find": {"enabled": True},
            "kids_find_plot": {"enabled": obsnum == 145485},
            "tone_prepare": {
                "enabled": True,
                "tone_phases": {"method": "random", "seed": 42},
            },
            "tlaloc_output": {"enabled": True, "path": tmp_path / "etc"},
            "output": {"enabled": True, "path": tmp_path / "products"},
        }
    }
    network = tmp_path / "etc/toltec0"
    network.mkdir(parents=True)
    attenuation = network / "atten_drive.dat"
    attenuation.write_text("8.25\n")
    product = run_source(source, RuntimeContext(config))
    assert attenuation.read_text() == "8.25\n"
    with open_sweep_product(product) as reopened:
        scope = reopened.toltec_kids.blocks[block].tree
        # Reopened output consumes saved values without repeating the reduction.
        replay = tmp_path / "replay"
        TlalocOutput(enabled=True, path=replay)(scope)
    expected = Table.read(expected_root / "toltec0/targ_freqs.dat", format="ascii.ecsv")
    actual = Table.read(replay / "toltec0/targ_freqs.dat", format="ascii.ecsv")
    assert actual.colnames == expected.colnames
    for name in (
        "Header.Toltec.ObsNum",
        "Header.Toltec.SubObsNum",
        "Header.Toltec.ScanNum",
        "Header.Toltec.RoachIndex",
        "Header.Toltec.LoCenterFreq",
    ):
        assert actual.meta[name] == expected.meta[name]
    for col in expected.colnames:
        np.testing.assert_array_equal(actual[col], expected[col], err_msg=col)
    for name in ["random_phases.dat", "default_targ_masks.dat"]:
        np.testing.assert_array_equal(
            np.loadtxt(replay / "toltec0" / name),
            np.loadtxt(expected_root / "toltec0" / name),
        )
    before = np.loadtxt(expected_root / "toltec0/default_targ_amps.dat")
    after = np.loadtxt(replay / "toltec0/default_targ_amps.dat")
    np.testing.assert_array_equal(after, before)
    for name in ["kids_find", "chan_prop"]:
        table_path = next(
            product.glob(f"artifacts/toltec0/blocks/{block}/**/{name}.ecsv")
        )
        result = Table.read(table_path, format="ascii.ecsv")
        reference = Table.read(expected_root / f"{name}.ecsv", format="ascii.ecsv")
        for col in reference.colnames:
            np.testing.assert_array_equal(result[col], reference[col], err_msg=col)
    if obsnum == 145485:
        assert {p.stem for p in product.glob("**/*kids_find_plot*/*.html")} == {
            "peaks",
            "peak_props",
            "d21_summary",
            "s21_summary",
            "det_summary",
            "matched",
            "matched_ref",
        }
    assert not (replay / "toltec0/atten_drive.dat").exists()
