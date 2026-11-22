"""Group identities, spectral validity and ordinary artifact persistence."""

import hashlib

import numpy as np
import pytest
import xarray as xr
from astropy.table import Table
from tollan.pipeline import ArtifactContext

from tests.tolteca_kids.test_model_fit_diagnostics import (
    fitted,  # noqa: F401 -- shared physical-fit fixture
)
from tests.tolteca_kidsproc.test_joint_continuation import case
from tolteca_kids.model_fit_quality import (
    ModelFitQualityConfig,
    model_fit_quality,
    residual_to_sweep_variation,
)
from tolteca_kids.model_joint import joint_models, joint_solver
from tolteca_kids.product import KidsDataArtifact
from tolteca_kids.timestream_diagnostics import (
    DiagnosticTables,
    _evidence,
    _signal_referred_ratio,
    diagnose_group,
)
from tolteca_kids.timestream_model import solve_fitted_groups
from tolteca_kids.timestream_plot import TimestreamPlotArtifacts


def solved_case():
    solver, _, _, _ = case(b=np.array([2e-15, 1e-15]))
    time = np.arange(512) / 128
    fr = solver.fr * (1 + np.sin(2 * np.pi * 8 * time[:, None]) * 1e-7)
    loss = solver.r * (1 + np.cos(2 * np.pi * 8 * time[:, None]) * 1e-4)
    group = {
        "group_id": "overlap",
        "channels": solver.channels.tolist(),
        "resonator_ids": solver.resonator_ids.tolist(),
        "probes": solver.probes.tolist(),
        "fr": solver.fr.tolist(),
        "r": solver.r.tolist(),
        "gain_family": "affine",
        "gain_model": {
            "f_g": solver.probes.tolist(),
            "H0_re": solver.gain.real.tolist(),
            "H0_im": solver.gain.imag.tolist(),
            "H1_re": [0.0] * 3,
            "H1_im": [0.0] * 3,
        },
        "coupling_re": solver.coupling.real.tolist(),
        "coupling_im": solver.coupling.imag.tolist(),
        "a_nl": (solver.b / (2 * solver.r**3)).tolist(),
        "nonlinear_branch": "unique",
        "iq_covariance": np.eye(6).tolist(),
        "noise_source": "unit_iq_weights",
        "fit_flag": 0,
        "rank_rtol": 1e-8,
        "condition_max": None,
    }
    # These are accepted states; the numerical solver itself is tested against
    # controlled I/Q in test_joint_continuation and test_joint_nonlinear.
    return xr.Dataset(
        {"fr": (("time", "resonator"), fr), "r": (("time", "resonator"), loss)},
        coords={
            "time": time,
            "sample": ("time", np.arange(512)),
            "resonator": solver.resonator_ids,
        },
        attrs={"model": group},
    )


def test_full_fit_routes_probe_groups_and_keeps_all_fitted_identities(fitted):  # noqa: F811 -- shared pytest fixture
    view = fitted.kids.model_fit
    node = view.mapper.get_node(view.data_source, view.mapper.schema.chan_prop_subpath)
    props = node.to_dataset()
    props["fp"] = ("chan", [500e6, 500.04e6, 500.08e6])
    node.dataset = props
    quality = model_fit_quality(view, ModelFitQualityConfig())
    [group] = joint_models(view, quality, include_single=True)["groups"]
    solver = joint_solver(group)
    fr = np.tile(solver.fr, (8, 1))
    loss = np.tile(solver.r, (8, 1))
    expected = solver.evaluate(fr, loss)
    signals = (
        expected[["I", "Q"]]
        .rename(sample="time")
        .assign_coords(time=np.arange(8) / 128, sample=("time", np.arange(8)))
    )
    tree = solve_fitted_groups(signals, view, quality, chunk_samples=3).compute()
    assert tree.sizes["component"] == 2
    assert tree["channels"].sizes["chan"] == 3  # includes unmatched acquired probe
    actual = tree["groups/0"].to_dataset(inherit=False)
    assert actual.resonator.values.tolist() == group["resonator_ids"]
    assert actual.chan.values.tolist() == group["channels"]
    np.testing.assert_allclose(actual.fr, fr, rtol=0, atol=0.02)
    np.testing.assert_allclose(actual.r, loss, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="Missing simultaneous"):
        solve_fitted_groups(signals.isel(chan=[0]), view, quality)


def test_diagnostics_roundtrip_keep_group_axes_and_signed_cross_terms(tmp_path):
    solved = solved_case()
    measured = joint_solver(solved.attrs["model"]).evaluate(
        solved.fr.values, solved.r.values
    )
    measured = (
        measured[["I", "Q"]].rename(sample="time").assign_coords(time=solved.time)
    )
    result, rows = diagnose_group(solved, f_smp=128, nperseg=128, observed=measured)
    np.testing.assert_allclose(
        result["coordinates/forward_relative_rms"], 0, atol=1e-14
    )
    assert [r["resonator_id"] for r in rows] == ["r7", "r2"]
    assert all(r["spectral_status"] == "complete" for r in rows)
    assert result["component_spectra/cross_real"].shape == (65, 2, 2)
    for name in ("0", "1"):
        spectra = result[f"spectra/{name}"].to_dataset()
        np.testing.assert_allclose(
            spectra.linear_predicted, spectra.linear_measured, atol=1e-25, rtol=1e-10
        )
        np.testing.assert_allclose(
            spectra.exact_reconstruction,
            spectra.cross_real.sel(coordinate="x_opt", coordinate2="x_opt"),
            atol=1e-25,
            rtol=1e-10,
        )
    result.to_zarr(tmp_path / "result.zarr", zarr_format=2)
    with xr.open_datatree(tmp_path / "result.zarr", engine="zarr") as saved:
        xr.testing.assert_allclose(
            saved["coordinates"].to_dataset(), result["coordinates"].to_dataset()
        )
        assert saved.attrs["model"] == solved.attrs["model"]


@pytest.mark.parametrize("issue", ["missing_sample", "gap", "short"])
def test_spectra_never_drop_samples_or_hide_incomplete_solutions(issue):
    solved = solved_case()
    grid = "uniform"
    if issue == "missing_sample":
        solved["fr"][10] = np.nan
    elif issue == "gap":
        grid = "gaps"
    else:
        solved = solved.isel(time=slice(0, 200))
    result, rows = diagnose_group(solved, f_smp=128, nperseg=128, sample_grid=grid)
    assert "spectra" not in result.children
    assert all(r["spectral_status"] != "complete" for r in rows)
    assert result["coordinates"].sizes["sample"] == solved.sizes["time"]
    if issue == "missing_sample":
        assert np.isnan(result["coordinates/x_opt"].isel(sample=10)).all()


def test_evidence_is_optional_and_pinned_to_exact_model(tmp_path):
    model = tmp_path / "model.ecsv"
    model.write_text("exact model\n")
    assert _evidence(None, model) == ({}, {})
    table = Table(
        {
            "group_id": ["g"],
            "resonator_id": ["d"],
            "cohort": ["dark_candidate"],
            "slope": [np.nan],
            "reference_frequency_hz": [500e6],
        }
    )
    path = tmp_path / "evidence.ecsv"
    table.write(path)
    with pytest.raises(ValueError, match="selected model"):
        _evidence(path, model)
    table.meta["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    table.write(path, overwrite=True)
    rows, provenance = _evidence(path, model)
    assert rows["g", "d"]["cohort"] == "dark_candidate"
    assert provenance["evidence_sha256"]


def test_plot_and_table_artifacts_reopen_without_solving(tmp_path):
    group, rows = diagnose_group(solved_case(), f_smp=128, nperseg=128)
    for row in rows:
        row.update(a_fit=0.2, cohort="missing_evidence")
    ledger = xr.Dataset({key: ("component", [r[key] for r in rows]) for key in rows[0]})
    tree = xr.DataTree(dataset=ledger)
    tree["groups/0"] = group
    figures = TimestreamPlotArtifacts()
    artifacts = figures(tree)
    receipts = figures.write(artifacts, path=tmp_path / "figures")
    assert {r.name for r in receipts} == {
        "timestream_browser",
        "plotly",
        "summary",
        "group_0",
    }
    tables = DiagnosticTables()
    context = ArtifactContext()
    context.add(tables.ledger, ledger)
    tables.write(context, path=tmp_path / "tables")
    saved = Table.read(tmp_path / "tables/detectors.ecsv")
    assert saved["cohort"].tolist() == ["missing_evidence"] * 2


def test_fit_policy_uses_sweep_variation_not_total_iq():
    samples = xr.Dataset(
        {
            "chan": ("sample", [0, 0, 1, 1]),
            "fit_selected": ("sample", [True] * 4),
            "I": ("sample", [99.0, 101.0, 999.0, 1001.0]),
            "Q": ("sample", [0.0] * 4),
            "I_residual": ("sample", [0.06] * 4),
            "Q_residual": ("sample", [0.0] * 4),
        }
    )
    assert residual_to_sweep_variation(samples) == pytest.approx(0.06)
    samples["I"] += 10000
    assert residual_to_sweep_variation(samples) == pytest.approx(0.06)
    samples["I"][:] = 1
    assert np.isinf(residual_to_sweep_variation(samples))


def test_zero_signal_transfer_does_not_claim_zero_referred_noise():
    """Cancellation of the loading response cannot imply perfect sensitivity."""
    row = {
        "cohort": "active_suitable",
        "fit_policy_pass": True,
        "spectral_status": "complete",
        "response_slope": 0.03,
        "D": 2,
        "mu": 0,
        "coordinate_asd_ratio": 2,
    }
    assert _signal_referred_ratio(row) == pytest.approx(1)
    row["mu"] = 1 / row["response_slope"]
    assert np.isnan(_signal_referred_ratio(row))
    row["D"] = 0
    assert np.isnan(_signal_referred_ratio(row))


def test_group_artifact_consolidates_once_and_reopens(tmp_path, monkeypatch):
    """Many independent groups share one final consolidated metadata tree."""
    import zarr

    from tolteca_kids import product

    calls = []
    consolidate = zarr.consolidate_metadata

    def record(*args, **kwargs):
        calls.append(args[0])
        return consolidate(*args, **kwargs)

    monkeypatch.setattr(zarr, "consolidate_metadata", record)
    monkeypatch.setattr(product, "consolidate_metadata", record)
    tree = xr.DataTree()
    for i in range(12):
        tree[f"groups/{i}"] = xr.DataTree(
            dataset=xr.Dataset({"r": ("time", np.arange(8) + i)})
        )
    [artifact] = KidsDataArtifact().write(tree, path=tmp_path)
    assert artifact.name == "Scientific data"
    assert len(calls) == 1
    with xr.open_datatree(
        tmp_path / "array.zarr", engine="zarr", consolidated=True
    ) as saved:
        xr.testing.assert_equal(saved, tree)
