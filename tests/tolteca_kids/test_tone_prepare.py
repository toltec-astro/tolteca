"""V2 tone ordering, ordinary publication, and unchanged-comb commit checks."""

from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from astropy.table import QTable, Table
from tollan.config import RuntimeContext

from tolteca_kids.product import open_sweep_product
from tolteca_kids.runner import run_source
from tolteca_kids.tlaloc_output import TlalocOutput, TlalocToneArtifact, TlalocTones
from tolteca_kids.tone_prepare import TonePrepareConfig, prepare_tone_table


def example_tones(*, fixed=False, placeholders=0):
    probes = QTable(
        {
            "chan": [10, 20, 30, 40],
            "f_chan": np.array([499e6, 498e6, 501e6, 502e6]) * u.Hz,
            "tone_amp": [0.1, 0.2, 0.3, 0.4],
            "tone_mask": [True] * 4,
        }
    )
    # Output count and order differ from the input; amplitudes are matched values.
    detected = QTable(
        {
            "f": np.array([498.1e6, 501.1e6]) * u.Hz,
            "Qr": [10000.0, 20000.0],
            "amp_tone": [0.2, 0.3],
            "d_phi": [1.0, 2.0] * u.deg,
            "bitmask": [0, 0],
        }
    )
    matched = QTable({"idx_chan": [10, 20, 30, 40], "d_phi": [80, 1, 2, 90] * u.deg})
    meta = SimpleNamespace(
        f_lo_center=500e6,
        roach=0,
        obsnum=17068,
        subobsnum=30,
        scannum=0,
        obs_type="tune" if fixed else "targ",
    )
    config = TonePrepareConfig(
        enabled=True,
        tone_phases={"method": "random", "seed": 42},
        placeholders={"n": placeholders},
    )
    return prepare_tone_table(detected, matched, probes, meta, config)


@pytest.mark.parametrize(
    ("fixed", "count", "frequencies", "amps"),
    [
        (False, 2, [1.1e6, -1.9e6], [0.3, 0.2]),
        (True, 4, [1.1e6, 2e6, -1.9e6, -1e6], [0.3, 0.4, 0.2, 0.1]),
    ],
)
def test_v2_shared_file_order_and_tune_count(tmp_path, fixed, count, frequencies, amps):
    data = example_tones(fixed=fixed)
    assert data.sizes["tone"] == count
    np.testing.assert_array_equal(data.f_centered, frequencies)
    np.testing.assert_array_equal(data.amp_tone, amps)
    files = TlalocToneArtifact().write(TlalocTones(data), path=tmp_path)
    assert len(files) == 4
    root = tmp_path / "toltec0"
    report = Table.read(root / "targ_freqs.dat", format="ascii.ecsv")
    assert report.colnames == [
        "f_centered",
        "f_out",
        "f_in",
        "flag",
        "fr",
        "Qr",
        "normI",
        "normQ",
        "fp",
        "slopeI",
        "slopeQ",
        "interceptI",
        "interceptQ",
        "Qc",
        "A",
    ]
    assert all(report[name].dtype == np.dtype("float64") for name in report.colnames)
    for name, column in [
        ("default_targ_amps.dat", "amp_tone"),
        ("random_phases.dat", "phase_tone"),
        ("default_targ_masks.dat", "mask_tone"),
    ]:
        actual = np.loadtxt(root / name)
        np.testing.assert_array_equal(actual, data[column])
    assert (root / "default_targ_masks.dat").read_text() == "1\n" * count


def test_v2_placeholders_keep_nan_models_and_zero_amplitudes(tmp_path):
    data = example_tones(placeholders=2)
    assert data.sizes["tone"] == 4
    masked = ~data.mask_tone.values
    np.testing.assert_array_equal(data.amp_tone.values[masked], [0.0, 0.0])
    assert np.isnan(data.Qr.values[masked]).all()
    TlalocToneArtifact().write(TlalocTones(data), path=tmp_path)
    assert np.loadtxt(tmp_path / "toltec0/default_targ_masks.dat").sum() == 2


@pytest.mark.parametrize("mismatch", ["order", "frequency", "count", "lo", "missing"])
def test_commit_rejects_wrong_reload_comb_before_replacement(tmp_path, mismatch):
    network = tmp_path / "toltec0"
    network.mkdir()
    frequencies = np.array([1e6, -1e6])
    if mismatch == "order":
        frequencies = frequencies[::-1]
    elif mismatch == "frequency":
        frequencies[0] += 488.28125
    elif mismatch == "count":
        frequencies = frequencies[:1]
    if mismatch != "missing":
        Table(
            {"f_centered": frequencies},
            meta={"f_lo_center": 501e6 if mismatch == "lo" else 500e6},
        ).write(network / "targ_freqs.dat", format="ascii.ecsv")
    for name in ["default_targ_amps.dat", "atten_drive.dat", "random_phases.dat"]:
        (network / name).write_text("old")
    tree = xr.DataTree.from_dict(
        {
            "tlaloc": xr.Dataset(
                {"amp_tone": ("chan", [0.5, 1.0]), "f_chan": ("chan", [501e6, 499e6])},
                attrs={"roach": 0, "atten_drive": 10.0, "f_lo_center": 500e6},
            )
        }
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        TlalocOutput(enabled=True, path=tmp_path)(tree)
    for name in ["default_targ_amps.dat", "atten_drive.dat", "random_phases.dat"]:
        assert (network / name).read_text() == "old"


def test_native_publication_and_replay_use_same_four_tone_files(
    drive_fit_calibration_sequence, tmp_path
):
    source = drive_fit_calibration_sequence[-1]
    config = {
        "kids": {
            "sweep_check": {"enabled": True},
            "kids_find": {"enabled": True},
            "tone_prepare": {
                "enabled": True,
                "tone_phases": {"method": "random", "seed": 42},
            },
            "output": {"enabled": True, "path": tmp_path / "products"},
            "tlaloc_output": {"enabled": True, "path": tmp_path / "etc"},
        }
    }
    path = run_source(source, RuntimeContext(config))
    with open_sweep_product(path) as tree:
        block = tree.toltec_kids.blocks[0]
        assert len(block.chan_axis_table) == 685
        assert block.tree["tlaloc"].sizes["tone"] == 551
        TlalocOutput(enabled=True, path=tmp_path / "replayed")(block.tree)
    for file in (tmp_path / "etc/toltec0").glob("*.dat"):
        assert (
            file.read_bytes()
            == (tmp_path / "replayed/toltec0" / file.name).read_bytes()
        )
        assert file.read_bytes() == next(path.rglob(file.name)).read_bytes()


def test_tune_installs_only_last_block_with_fixed_input_count(data_lmt_path, tmp_path):
    source = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    config = {
        "kids": {
            "sweep_check": {"enabled": True},
            "kids_find": {"enabled": True},
            "tone_prepare": {
                "enabled": True,
                "tone_phases": {"method": "random", "seed": 42},
            },
            "output": {"enabled": True, "path": tmp_path / "products"},
            "tlaloc_output": {"enabled": True, "path": tmp_path / "etc"},
        }
    }
    path = run_source(source, RuntimeContext(config))
    with open_sweep_product(path) as tree:
        blocks = list(tree.toltec_kids.blocks)
        assert len(blocks) == 2
        for block in blocks:
            assert block.tree["tlaloc"].sizes["tone"] == len(block.chan_axis_table)
        report = Table.read(
            tmp_path / "etc/toltec0/targ_freqs.dat", format="ascii.ecsv"
        )
        np.testing.assert_array_equal(
            report["f_centered"], blocks[-1].tree["tlaloc/f_centered"]
        )
        assert not np.array_equal(
            report["f_centered"], blocks[0].tree["tlaloc/f_centered"]
        )


def test_publication_failure_prevents_tone_installation(
    data_lmt_path, tmp_path, monkeypatch
):
    source = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145606*_targsweep.nc"))

    def fail_export(*args, **kwargs):
        msg = "simulated product export failure"
        raise OSError(msg)

    monkeypatch.setattr(TlalocToneArtifact, "write", fail_export)
    config = {
        "kids": {
            "sweep_check": {"enabled": True},
            "kids_find": {"enabled": True},
            "tone_prepare": {"enabled": True},
            "output": {"enabled": True, "path": tmp_path / "products"},
            "tlaloc_output": {"enabled": True, "path": tmp_path / "etc"},
        }
    }
    with pytest.raises(
        (OSError, RuntimeError), match="export failure|successful publication"
    ):
        run_source(source, RuntimeContext(config))
    assert not (tmp_path / "etc").exists()
