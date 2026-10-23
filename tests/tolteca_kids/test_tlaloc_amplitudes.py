"""Relative LUT normalization, ambiguous groups and immutable tone replay."""

from hashlib import sha256
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from astropy.table import QTable, Table
from tollan.config import RuntimeContext

from tolteca_kids.drive_fit.power import tone_power_fractions
from tolteca_kids.kids_find import KidsFind
from tolteca_kids.product import open_sweep_product
from tolteca_kids.runner import run_data
from tolteca_kids.tlaloc_output import (
    TlalocOutput,
    TlalocOutputConfig,
    fill_tone_amplitudes,
)
from tolteca_kids.tone_prepare import TonePrepare, TonePrepareConfig, prepare_tone_table


@pytest.fixture
def lut_root(tmp_path):
    network = tmp_path / "etc/toltec0"
    network.mkdir(parents=True)
    np.savetxt(
        network / "amps_correction_lut.csv",
        [[400e6, 0.2], [500e6, 0.8], [600e6, 0.4]],
        delimiter=",",
    )
    return network.parent


def probes(amps=(0.1, 0.2), frequencies=(400e6, 600e6), enabled=(True, True)):
    return QTable(
        {
            "chan": np.arange(len(amps)) * 10 + 10,
            "f_chan": np.asarray(frequencies) * u.Hz,
            "tone_amp": amps,
            "tone_mask": enabled,
        }
    )


def fill(  # noqa: PLR0913 -- abbreviated scientific input fixture
    amps, frequencies, inputs, cfg, *, ambiguous=None, unique=(10, 20)
):
    return fill_tone_amplitudes(
        amps,
        frequencies,
        inputs,
        fill_mask=np.zeros(len(amps), bool) if ambiguous is None else ambiguous,
        unique_probe_ids=unique,
        roach=0,
        config=cfg,
    )


def test_natural_spline_and_linear_extrapolation_with_known_scale(lut_root):
    frequency = [390e6, 400e6, 450e6, 500e6, 550e6, 600e6, 610e6]
    values, filled, info = fill(
        np.full(len(frequency), np.nan),
        frequency,
        probes(),
        TlalocOutputConfig(path=lut_root, tone_amps_fallback="lut"),
    )
    # Endpoint derivatives: 0.85 and -0.65 per 100 MHz. Both anchors
    # have input/LUT=0.5, including the voltage (20 log10) convention.
    np.testing.assert_allclose(
        values,
        np.array(
            [
                0.115,
                0.2,
                0.59375,
                0.8,
                0.69375,
                0.4,
                0.335,
            ]
        )
        * 0.5,
        atol=1e-15,
    )
    assert filled.all()
    assert info["tone_amps_lut_scale"] == pytest.approx(0.5)
    path = lut_root / "toltec0/amps_correction_lut.csv"
    assert info["tone_amps_lut_sha256"] == sha256(path.read_bytes()).hexdigest()


def test_scale_is_median_in_db_and_ignores_ambiguous_or_disabled_probes(lut_root):
    inputs = probes(
        amps=[0.1, 0.8, 0.9, 1.0],
        frequencies=[400e6, 500e6, 600e6, 550e6],
        enabled=[True, True, True, False],
    )
    _, _, info = fill(
        [np.nan],
        [450e6],
        inputs,
        TlalocOutputConfig(path=lut_root, tone_amps_fallback="lut"),
        unique=[10, 20, 40],
    )
    assert info["tone_amps_lut_anchor_count"] == 2
    assert info["tone_amps_lut_scale"] == pytest.approx(np.sqrt(0.5))


def test_lut_common_scale_cancels_and_finite_matches_are_retained(lut_root):
    cfg = TlalocOutputConfig(path=lut_root, tone_amps_fallback="lut")
    kwargs = {"ambiguous": [False, False, True]}
    before, _, _ = fill([0.11, 0, 0.9], [400e6, 500e6, 450e6], probes(), cfg, **kwargs)
    path = lut_root / "toltec0/amps_correction_lut.csv"
    lut = np.loadtxt(path, delimiter=",")
    lut[:, 1] *= 7.3
    np.savetxt(path, lut, delimiter=",")
    after, filled, _ = fill(
        [0.11, 0, 0.9], [400e6, 500e6, 450e6], probes(), cfg, **kwargs
    )
    np.testing.assert_allclose(after, before, rtol=2e-15)
    np.testing.assert_array_equal(after[:2], [0.11, 0])
    np.testing.assert_array_equal(filled, [False, False, True])


def test_median_uses_enabled_finite_inputs_and_preserves_zero():
    values, filled, _ = fill(
        [0.1, 0, np.nan, 0.9],
        [450e6] * 4,
        probes([0.2, 0.4, 0.9], [400e6, 500e6, 600e6], [True, True, False]),
        TlalocOutputConfig(tone_amps_fallback="median"),
        ambiguous=[False, False, False, True],
    )
    np.testing.assert_allclose(values, [0.1, 0, 0.3, 0.3])
    np.testing.assert_array_equal(filled, [False, False, True, True])


@pytest.mark.parametrize(("method", "expected"), [("match", 0.3), ("ones", 1.0)])
def test_baseline_and_unity_ignore_ambiguity_without_reading_lut(
    tmp_path, method, expected
):
    values, filled, info = fill(
        [0.3],
        [500e6],
        probes(),
        TlalocOutputConfig(
            path=tmp_path,
            tone_amps_method=method,
            tone_amps_fallback="lut" if method == "ones" else None,
        ),
        ambiguous=[True],
        unique=[],
    )
    assert values[0] == expected
    assert not filled.any() and "tone_amps_lut_sha256" not in info


@pytest.mark.parametrize(
    "case", ["missing", "unordered", "nan", "negative", "duplicate", "no_anchor"]
)
def test_invalid_lut_or_absent_anchors_fail_without_silent_fallback(lut_root, case):
    path = lut_root / "toltec0/amps_correction_lut.csv"
    lut = np.loadtxt(path, delimiter=",")
    if case == "missing":
        path.unlink()
    elif case == "unordered":
        np.savetxt(path, lut[::-1], delimiter=",")
    elif case == "nan":
        lut[1, 1] = np.nan
        np.savetxt(path, lut, delimiter=",")
    elif case == "negative":
        lut[1, 1] = -0.1
        np.savetxt(path, lut, delimiter=",")
    elif case == "duplicate":
        lut[1, 0] = lut[0, 0]
        np.savetxt(path, lut, delimiter=",")
    with pytest.raises((ValueError, FileNotFoundError)):
        fill(
            [np.nan],
            [450e6],
            probes(),
            TlalocOutputConfig(path=lut_root, tone_amps_fallback="lut"),
            unique=[] if case == "no_anchor" else [10, 20],
        )


def test_relative_power_survives_joint_normalization(lut_root):
    # A flat high-frequency LUT makes fallback amplitudes 4.0 before the joint
    # normalization. Both members of a three-node match group must use it.
    path = lut_root / "toltec0/amps_correction_lut.csv"
    np.savetxt(path, [[400e6, 0.1], [500e6, 0.8], [600e6, 0.8]], delimiter=",")
    p = probes([0.5, 0.1], [400e6, 500e6])
    detected = QTable(
        {
            "f": [400e6, 500e6, 600e6] * u.Hz,
            "Qr": [10000.0] * 3,
            "amp_tone": [0.5, 0.1, 0.1],
            "d_phi": [0, 1, 2] * u.deg,
            "bitmask": [0, 0, 0],
        }
    )
    query = QTable({"size": [2, 3, 3], "idx_chan": [10, 20, 20]})
    meta = SimpleNamespace(
        f_lo_center=450e6, roach=0, obsnum=1, subobsnum=0, scannum=0, obs_type="targ"
    )
    cfg = TlalocOutputConfig(path=lut_root, tone_amps_fallback="lut")
    result = prepare_tone_table(
        detected,
        QTable(),
        p,
        meta,
        TonePrepareConfig(),
        tlaloc_config=cfg,
        query_matches=query,
    )
    np.testing.assert_allclose(result.amp_tone, [1, 1, 0.125])
    np.testing.assert_array_equal(result.amp_tone_filled, [True, True, False])
    assert result.attrs["tone_amps_normalization_divisor"] == pytest.approx(4)
    np.testing.assert_allclose(
        result.tone_power_fraction, tone_power_fractions([4, 4, 0.5], result.f_centered)
    )


def test_reference_lut_publication_fills_whole_ambiguous_groups_and_replays(
    data_lmt_path, tmp_path
):
    network = tmp_path / "etc/toltec0"
    network.mkdir(parents=True)
    lut_path = network / "amps_correction_lut.csv"
    # Controlled test LUT. The observing LUT is an external calibration input.
    np.savetxt(lut_path, [[400e6, 0.5], [650e6, 0.5], [1000e6, 0.5]], delimiter=",")
    (network / "atten_drive.dat").write_text("8.25\n")
    source = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145485*_targsweep.nc"))
    runtime = RuntimeContext(
        {
            "kids": {
                "sweep_check": {"enabled": True},
                "kids_find": {"enabled": True},
                "tone_prepare": {
                    "enabled": True,
                    "tone_phases": {"method": "random", "seed": 42},
                },
                "tlaloc_output": {
                    "enabled": True,
                    "path": network.parent,
                    "tone_amps_fallback": "lut",
                },
                "output": {"enabled": True, "path": tmp_path / "products"},
            }
        }
    )
    with xr.open_dataset(source) as raw:
        tree = run_data(raw, source, runtime)
        block = tree.toltec_kids.blocks[0].tree
        matches = KidsFind.get_context(block).data.detected_matched
        ambiguous = set(matches["f_det"][matches["size"] > 2].to_value(u.Hz))
        prepared = block["tlaloc"].to_dataset().load()
        context = TonePrepare.get_context(block)
        restored = TonePrepare.load_context(TonePrepare.dump_context(context))
        restored.make_step()(block)
        xr.testing.assert_identical(prepared, block["tlaloc"].to_dataset())
    assert int(prepared.amp_tone_filled.sum()) == len(ambiguous) == 27
    assert set(prepared.f_out.values[prepared.amp_tone_filled.values]) == ambiguous
    np.testing.assert_array_equal(prepared.amp_tone_filled, prepared.amp_tone_ambiguous)
    assert (network / "atten_drive.dat").read_text() == "8.25\n"
    original = {p.name: p.read_bytes() for p in network.glob("*.dat")}
    lut_path.unlink()
    product = next(
        (tmp_path / "products").glob("scratch/*/dp_raw_obs_proc/*/array.zarr")
    ).parent
    with open_sweep_product(product) as saved:
        TlalocOutput(enabled=True, path=tmp_path / "replay")(
            saved.toltec_kids.blocks[0].tree
        )
    for path in (tmp_path / "replay/toltec0").glob("*.dat"):
        assert path.read_bytes() == original[path.name]
    assert not (tmp_path / "replay/toltec0/atten_drive.dat").exists()
    report = Table.read(network / "targ_freqs.dat", format="ascii.ecsv")
    assert len(report) == 634
