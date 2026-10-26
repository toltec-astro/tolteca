"""Offline regression checks using the separately retained production capture.

Set TOLTECA_DRIVE_FIT_CAPTURE to the capture root. No acquisition connection,
pickle loading or mutable production etc directory is used by these tests.
"""

import hashlib
import json
import os
from pathlib import Path

import netCDF4
import numpy as np
import pytest
from astropy.table import Table
from tollan.config import RuntimeContext

from tolteca_kids.drive_fit import DriveFit, DriveFitOutputStep, DriveFitStep
from tolteca_kids.drive_fit_commit import (
    DriveFitCommit,
    DriveFitCommitStep,
    compute_ampcor,
    open_drive_fit_commit_product,
)
from tolteca_kids.tlaloc_output import TlalocOutput


@pytest.fixture(scope="module")
def capture():
    location = os.environ.get("TOLTECA_DRIVE_FIT_CAPTURE")
    if location is None:
        pytest.skip("Set TOLTECA_DRIVE_FIT_CAPTURE to run production regressions")
    root = Path(location)
    records = json.loads((root / "inventory/retained-manifest.json").read_text())
    manifest = {row["local_path"]: row for row in records}
    checked = set()

    def read_path(pattern):
        [path] = sorted(root.glob(pattern))
        key = path.relative_to(root).as_posix()
        if key not in checked:
            record = manifest[key]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            assert digest == record["sha256"], key
            assert path.stat().st_size == record["bytes"], key
            checked.add(key)
        return path

    return read_path


def recorded_comb(path):
    with netCDF4.Dataset(path) as data:

        def column(name):
            value = np.asarray(data["Header.Toltec." + name][:])
            return value[0] if value.ndim == 2 else value

        return {
            "frequency": column("ToneFreq"),
            "amplitude": column("ToneAmp"),
            "mask": column("ToneMask"),
            "lo": column("LoCenterFreq").item(),
        }


@pytest.mark.parametrize(
    ("network", "observation", "expected_global"),
    [
        (4, 114111, 15),
        (5, 114111, 13.75),
        (0, 114200, 9.5),
        (2, 114200, 11),
        (7, 114200, 9.5),
        (8, 114200, 3.5),
        (9, 114200, 11.75),
        (11, 114200, 2.5),
    ],
)
def test_normalization_on_original_fit_selections(
    capture, network, observation, expected_global
):
    """Hold March selections fixed; check modern policy with an independent sum."""
    products = "hosts/clipy-mx/surviving-products/data1/dataprod_toltec/scratch"
    table = Table.read(
        capture(f"{products}/drive_atten_toltec{network}_{observation}_adrv.csv"),
        format="ascii.csv",
    )
    table.sort("tone_num")
    comb = recorded_comb(
        capture(
            f"historical/{observation}/data_lmt/toltec/tcs/toltec{network}/"
            f"toltec{network}_{observation}_001_*_targsweep.nc"
        )
    )
    # These recorded combs have no masks or DAC-bin collisions, so a plain sum
    # is an independent oracle for the shared fixed-total implementation.
    frequency, amplitude = comb["frequency"], comb["amplitude"]
    bins = np.sign(frequency) * np.floor(np.abs(frequency) / 488.28125 + 0.5)
    assert len(set(bins)) == len(amplitude)
    assert np.all(comb["mask"] == 1)
    fraction = amplitude**2 / np.sum(amplitude**2)
    attenuation = np.asarray(table["drive_atten"])
    flags = np.asarray(table["drive_atten_flag"])
    good = (flags == 0) & (attenuation < 30)
    accepted = np.maximum(attenuation[good], 0)
    reference = np.percentile(accepted, 10)
    filled = np.where(good, np.maximum(attenuation, 0), np.median(accepted))
    filled = np.maximum(filled, reference)
    requested = fraction * 10 ** (-filled / 10)
    exact = -10 * np.log10(np.sum(requested))
    expected_amp = np.sqrt(requested / requested.max())

    result = compute_ampcor(
        attenuation,
        flags,
        fraction,
        output_frequency=frequency,
        output_mask=comb["mask"],
        adrv_max=30,
        fill_percentile=50,
        reference_percentile=10,
        reference_step=0.25,
    )
    np.testing.assert_array_equal(result["adrv_filled"], filled)
    np.testing.assert_allclose(result["ampcor"], expected_amp, rtol=1e-14)
    assert result["adrv_exact"] == pytest.approx(exact, abs=1e-12)
    assert result["adrv_ref_total"] == expected_global
    delivered = result["ampcor"] ** 2 / np.sum(result["ampcor"] ** 2)
    error = 10 * np.log10(delivered / requested) - result["adrv_ref_total"]
    np.testing.assert_allclose(error, exact - expected_global, atol=1e-12)
    assert np.all((error >= 0) & (error < 0.25))


@pytest.mark.parametrize(
    ("network", "reference_obs", "counts"),
    [(4, 150275, (510, 513, 519)), (5, 149325, (474, 489, 487))],
)
def test_fresh_production_fit_commit_and_destination_contract(
    capture, tmp_path, network, reference_obs, counts
):
    """Use the logged reference series and reject the recorded mismatched reload.

    This is a fresh analytic fit. Missing January operational ECSV values are
    not inferred from this replay or used as a numerical baseline.
    """
    n_reference, n_target, n_loaded = counts
    raw = f"data_lmt/toltec/tcs/toltec{network}/toltec{network}"
    sources = [
        capture(f"{raw}_{reference_obs}_{subobs:03d}_*_targsweep.nc")
        for subobs in range(1, 8)
    ]
    target = capture(f"{raw}_150301_000_*_targsweep.nc")
    loaded = recorded_comb(capture(f"{raw}_150302_000_*_tune.nc"))
    comb = recorded_comb(target)
    assert len(comb["amplitude"]) == n_target
    assert len(loaded["amplitude"]) == n_loaded
    fit = DriveFit(
        RuntimeContext(
            {
                "drive_fit": {
                    "output": {
                        "enabled": True,
                        "path": tmp_path / "products",
                    }
                }
            }
        )
    )(sources)
    reference = DriveFitOutputStep.get_context(fit).data
    fit_data = DriveFitStep.view(fit[f"interfaces/toltec{network}"]).data
    assert fit_data.sizes["chan"] == n_reference
    assert fit_data.sizes["power"] == 7
    assert np.count_nonzero(fit_data.p_flag.values == 0) > 0
    if network == 5:
        failed = fit_data.sel(chan=173)
        assert np.all(failed.fit_flag.values == 1)
        assert np.all(
            np.char.find(failed.fit_error.values.astype(str), "at least four samples")
            >= 0
        )
        assert failed.p_flag.item() == 1
        assert np.isnan(failed.adrv_best.item())
    committed = DriveFitCommit(
        RuntimeContext(
            {
                "drive_fit_commit": {
                    "output": {
                        "enabled": True,
                        "path": tmp_path / "products",
                    }
                }
            }
        )
    )([target], reference=reference)
    tree = committed[f"interfaces/toltec{network}"]
    result = DriveFitCommitStep.view(tree).data
    path = DriveFitOutputStep.get_context(committed).data
    assert result.sizes["chan"] == n_target
    np.testing.assert_array_equal(result.f_chan, comb["frequency"] + comb["lo"])
    emitted = np.isfinite(result.power_error_db.values)
    error = result.power_error_db.values[emitted]
    assert np.all((error >= -1e-12) & (error < 0.25))
    # Exported artifacts must be usable after reopening, without fitting again.
    with open_drive_fit_commit_product(path) as reopened:
        destination = tmp_path / "etc" / f"toltec{network}"
        destination.mkdir(parents=True)
        for name in ("default_targ_amps.dat", "atten_drive.dat"):
            (destination / name).write_text("previous settings")
        Table(
            {"f_centered": loaded["frequency"]},
            meta={
                "f_lo_center": loaded["lo"],
            },
        ).write(destination / "targ_freqs.dat", format="ascii.ecsv")
        with pytest.raises(ValueError, match="tone count, frequencies or order"):
            TlalocOutput(enabled=True, path=destination.parent)(reopened)
        for name in ("default_targ_amps.dat", "atten_drive.dat"):
            assert (destination / name).read_text() == "previous settings"
        Table(
            {"f_centered": comb["frequency"]},
            meta={
                "f_lo_center": comb["lo"],
            },
        ).write(destination / "targ_freqs.dat", format="ascii.ecsv", overwrite=True)
        TlalocOutput(enabled=True, path=destination.parent)(reopened)
        np.testing.assert_array_equal(
            np.loadtxt(destination / "default_targ_amps.dat"),
            result.amp_best,
        )
        assert (
            float((destination / "atten_drive.dat").read_text())
            == (result.attrs["atten_drive_global"])
        )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "network": network,
                "reference_obs": reference_obs,
                "target_obs": 150301,
                "reference_channels": n_reference,
                "target_channels": n_target,
                "loaded_channels": n_loaded,
                "accepted_fits": int(np.count_nonzero(fit_data.p_flag.values == 0)),
                "accepted_commit": int(np.count_nonzero(~result.amp_flag.values)),
                "attenuation_db": result.attrs["atten_drive_global"],
                "power_error_db": [float(error.min()), float(error.max())],
                "fit_product": str(reference),
                "commit_product": str(path),
                "interpretation": (
                    "Fresh analytic fit, not recovered January selections"
                ),
            },
            indent=2,
        )
        + "\n"
    )
