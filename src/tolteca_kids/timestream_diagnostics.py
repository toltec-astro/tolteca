"""Post-solving evidence from the same fitted detector groups."""

from __future__ import annotations

import hashlib
from itertools import pairwise
from pathlib import Path

import numpy as np
import xarray as xr
from astropy.table import Table
from pydantic import ConfigDict, Field
from tollan.config.types import AbsAnyPath  # noqa: TC002 -- Pydantic field type
from tollan.pipeline import (
    ArtifactContext,
    ArtifactFile,
    ArtifactSpec,
    Step,
    StepConfig,
    StepContext,
)

from tolteca_kidsproc.timestream_diagnostics import coordinate_spectra, spectral_matrix

from .model_joint import joint_solver

_BAND_HZ = (5, 20)


class ResponseEvidence(ArtifactSpec[bytes]):
    """Exact optional response-evidence input retained with its diagnostics."""

    def write(self, data, *, path):
        """Write the original pinned evidence bytes without changing identities."""
        target = path / "response_evidence.ecsv"
        target.write_bytes(data)
        return [ArtifactFile(target, "ecsv", name="response_evidence")]


class NetworkPerformance(ArtifactSpec[Table]):
    """Counts and conditional noise summaries; bins do not change the model."""

    def __call__(self, ledger):
        """Summarize singlets and multiplets without summing correlated weights."""
        rows = []
        bounds = [0, 0.01, 0.2, 0.75, np.inf]
        for kind in ("singlet", "multiplet"):
            for lo, hi in pairwise(bounds):
                mask = (
                    (ledger.multiplicity == 1)
                    if kind == "singlet"
                    else (ledger.multiplicity > 1)
                )
                mask = mask & (ledger.a_fit >= lo) & (ledger.a_fit < hi)
                part = ledger.isel(component=np.flatnonzero(mask))
                ratios = part.signal_referred_asd_ratio.values
                qualified = ratios[np.isfinite(ratios)]
                rows.append(
                    {
                        "kind": kind,
                        "a_min": lo,
                        "a_max": hi,
                        "components": part.sizes["component"],
                        "numerical_complete": int(part.numerical_complete.sum()),
                        "spectral_complete": int(
                            (part.spectral_status == "complete").sum()
                        ),
                        "fit_policy_pass": int(part.fit_policy_pass.sum()),
                        "missing_evidence": int(
                            (part.cohort == "missing_evidence").sum()
                        ),
                        "response_comparisons": len(qualified),
                        "median_signal_referred_asd_ratio": float(np.median(qualified))
                        if len(qualified)
                        else np.nan,
                    }
                )
        return Table(
            rows=rows,
            meta={
                "bins": "diagnostic only; one continuous nonlinear model",
                "mapping_speed": "not inferred",
            },
        )

    def write(self, data, *, path):
        """Write the already prepared network matrix as an ECSV artifact."""
        target = path / "performance.ecsv"
        data.write(target, format="ascii.ecsv", overwrite=False)
        return [ArtifactFile(target, "ecsv", name="network_performance")]


class DetectorLedger(ArtifactSpec[xr.Dataset]):
    """Portable detector ledger, including evidence and policy decisions."""

    def write(self, data, *, path):
        """Write the prepared ledger as ECSV and return its artifact receipt."""
        path.mkdir(parents=True, exist_ok=True)
        table = Table({name: value.values for name, value in data.data_vars.items()})
        table.meta.update(data.attrs)
        target = path / "detectors.ecsv"
        table.write(target, format="ascii.ecsv", overwrite=False)
        return [ArtifactFile(target, "ecsv", name="detector_diagnostics")]


class DiagnosticTables(ArtifactSpec[ArtifactContext]):
    """Declared ledger, performance and optional external-evidence artifacts."""

    ledger: DetectorLedger = Field(default_factory=DetectorLedger)
    performance: NetworkPerformance = Field(default_factory=NetworkPerformance)
    evidence: ResponseEvidence = Field(default_factory=ResponseEvidence)

    def write(self, data, *, path):
        """Materialize prepared tables through their retained specifications."""
        path.mkdir(parents=True, exist_ok=True)
        return [
            receipt
            for item in data.values()
            for receipt in item.spec.write(item.data, path=path)
        ]


class TimestreamDiagnosticsConfig(StepConfig):
    """Spectral comparison settings; evidence never changes the fitted model.

    Parameters
    ----------
    nperseg : int, optional
        Welch window length, default 1024. Two windows are required.
    fit_residual_max : float, optional
        Residual / within-probe sweep-variation limit, default 0.05. This does
        not veto solving; total-IQ normalization is a separate saved metric.
    evidence : pathlib.Path or None, optional
        ECSV with group_id, resonator_id, cohort and slope (R_r/R_x), pinned by
        model_sha256 metadata. Omitted evidence remains explicitly unknown.
    """

    model_config = ConfigDict(extra="forbid")
    artifacts: DiagnosticTables = Field(default_factory=DiagnosticTables)
    nperseg: int = Field(default=1024, ge=2)
    fit_residual_max: float = Field(default=0.05, gt=0)
    evidence: AbsAnyPath | None = None


class TimestreamDiagnosticsContext(
    StepContext["TimestreamDiagnostics", TimestreamDiagnosticsConfig]
):
    """Saved spectral settings and retained detector-ledger artifact."""


def _sample_grid(block):
    """Check recorded packet/clock counters; do not infer uniformity from row order."""
    if "Data.Toltec.Ts" not in block.raw:
        return "unverified"
    ticks = block.raw["Data.Toltec.Ts"]
    names = [
        part.strip().split()[0]
        for part in ticks.attrs.get("long_name", "").split(",")
        if part.strip()
    ]
    if not {"PacketCount", "ClockCount"}.issubset(names):
        return "unverified"
    values = ticks.values
    packets = np.asarray(values[:, names.index("PacketCount")], dtype=np.uint32)
    clocks = np.asarray(values[:, names.index("ClockCount")], dtype=np.uint32)
    return (
        "uniform"
        if len(packets) > 1
        and np.all(np.diff(packets) == 1)
        and np.all(np.diff(clocks) == np.diff(clocks)[0])
        else "gaps"
    )


def diagnose_group(  # noqa: PLR0915 -- one group, coordinates and spectral checks
    solved, *, f_smp, nperseg=1024, sample_grid="uniform", observed=None
):
    """Evaluate solved states and measure spectra without refitting or gap filling.

    Parameters
    ----------
    solved : xarray.Dataset
        One group from solve_fitted_groups, including its complete model.
    f_smp : float
        Recorded sampling frequency in Hz.
    nperseg : int, optional
        Welch window length, default 1024.
    sample_grid : str, optional
        Acquisition counter assessment. Only 'uniform' permits spectra.
    observed : xarray.Dataset or None, optional
        Measured I/Q in calibration units, on the same time and chan coordinates.
        None omits the measured-versus-predicted check.

    Returns
    -------
    xarray.DataTree, list of dict
        Evaluated time coordinates and per-component spectral matrices, plus
        summary rows. Failed samples remain NaN at their original positions.
        No absolute optical sensitivity or independent mapping speed is inferred.
    """
    solved = solved.load()
    group = solved.attrs["model"]
    solver = joint_solver(group)
    fr, loss = solved.fr.values, solved.r.values
    valid = np.isfinite(fr).all(axis=1) & np.isfinite(loss).all(axis=1)
    out = xr.DataTree()
    records = []
    count = len(fr)
    complete = bool(valid.all())
    if valid.any():
        evaluated = solver.evaluate(fr[valid], loss[valid]).assign_coords(
            sample=np.flatnonzero(valid)
        )
        evaluated = evaluated.reindex(sample=np.arange(count))
        evaluated["time"] = ("sample", solved.time.values)
        evaluated["sample_index"] = ("sample", solved.sample.values)
        if observed is not None:
            selected = observed.sel(chan=solver.channels).transpose("time", "chan")
            if not np.array_equal(selected.time, solved.time):
                msg = "Observed I/Q samples differ from the solved sample grid"
                raise ValueError(msg)
            for name in ("I", "Q"):
                evaluated[f"{name}_observed"] = (
                    ("sample", "chan"),
                    selected[name].values,
                )
                evaluated[f"{name}_residual"] = (
                    evaluated[name] - evaluated[f"{name}_observed"]
                )
            residual = evaluated.I_residual.values + 1j * evaluated.Q_residual.values
            evaluated["forward_relative_rms"] = (
                "chan",
                np.sqrt(np.mean(abs(residual[valid] / solver.gain) ** 2, axis=0)),
            )
        reference = solver.evaluate(
            np.median(fr[valid], axis=0, keepdims=True),
            np.median(loss[valid], axis=0, keepdims=True),
        )
        out["coordinates"] = xr.DataTree(dataset=evaluated)
    selected_series = []
    for j, rid in enumerate(group["resonator_ids"]):
        k = int(np.argmin(abs(solver.probes - solver.fr[j])))
        row = {
            "group_id": group["group_id"],
            "resonator_id": rid,
            "probe": int(solver.channels[k]),
            "probe_frequency_hz": float(solver.probes[k]),
            "valid_fraction": float(valid.mean()),
            "numerical_complete": complete,
            "multiplicity": len(solver.fr),
            "spectral_status": "incomplete_solution",
            "b": float(solver.b[j]),
            "coordinate_asd_ratio": np.nan,
            "D": np.nan,
            "mu": np.nan,
        }
        records.append(row)
        if not complete:
            continue
        row["spectral_status"] = (
            "unverified_sample_grid"
            if sample_grid != "uniform"
            else "insufficient_samples"
        )
        if sample_grid != "uniform" or count < 2 * nperseg:
            continue
        pair = evaluated.isel(chan=k, resonator=j)
        local = reference.isel(chan=k, resonator=j)
        D, mu = float(local.D.item()), float(local.mu.item())
        if (
            not np.isfinite([pair.x_prime, pair.x_opt, pair.r]).all()
            or not np.isfinite([D, mu]).all()
        ):
            row["spectral_status"] = "unsupported_model_state"
            continue
        spectra = coordinate_spectra(
            pair.x_prime, pair.x_opt, pair.r, f_smp, D=D, mu=mu, nperseg=nperseg
        )
        spectra.attrs.update(
            probe=row["probe"],
            resonator_id=rid,
            reference="median solved fr,r",
            b=row["b"],
        )
        out[f"spectra/{j}"] = xr.DataTree(dataset=spectra)
        band = (spectra.f_psd >= _BAND_HZ[0]) & (spectra.f_psd <= _BAND_HZ[1])
        power = spectra.cross_real.values.diagonal(axis1=1, axis2=2)
        mean = power[band].mean(axis=0) if band.any() else np.full(4, np.nan)
        row.update(
            spectral_status="complete",
            D=D,
            mu=mu,
            coordinate_asd_ratio=float(np.sqrt(mean[1] / mean[0]))
            if mean[0] > 0
            else np.nan,
        )
        selected_series.append(pair.x_opt.values)
    if len(selected_series) == len(solver.fr) and len(solver.fr) > 1:
        frequency, cross = spectral_matrix(
            np.column_stack(selected_series), f_smp, nperseg=nperseg
        )
        out["component_spectra"] = xr.DataTree(
            dataset=xr.Dataset(
                {
                    "cross_real": (("f_psd", "resonator", "resonator2"), cross.real),
                    "cross_imag": (("f_psd", "resonator", "resonator2"), cross.imag),
                },
                coords={
                    "f_psd": frequency,
                    "resonator": solver.resonator_ids,
                    "resonator2": solver.resonator_ids,
                },
            )
        )
    out.attrs["model"] = group
    return out, records


def _evidence(path, model_source):
    if path is None:
        return {}, {}
    table = Table.read(path, format="ascii.ecsv")
    if (
        table.meta.get("model_sha256")
        != hashlib.sha256(Path(model_source).read_bytes()).hexdigest()
    ):
        msg = "Response evidence does not identify the selected model report"
        raise ValueError(msg)
    required = {"group_id", "resonator_id", "cohort", "slope"}
    if not required.issubset(table.colnames):
        msg = f"Response evidence requires {sorted(required)}"
        raise ValueError(msg)
    rows = {(str(row["group_id"]), str(row["resonator_id"])): row for row in table}
    if len(rows) != len(table):
        msg = "Duplicate response-evidence component"
        raise ValueError(msg)
    if "reference_frequency_hz" not in table.colnames or any(
        np.isfinite(row["slope"])
        and (
            not np.isfinite(row["reference_frequency_hz"])
            or row["reference_frequency_hz"] <= 0
        )
        for row in table
    ):
        msg = "Response slopes require their positive reference frequency in Hz"
        raise ValueError(msg)
    return rows, {
        "evidence_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "evidence_metadata": table.meta,
    }


class TimestreamDiagnostics(
    Step[TimestreamDiagnosticsConfig, TimestreamDiagnosticsContext]
):
    """Measure saved-model signal coordinates and network performance after solving."""

    @classmethod
    def run(cls, data, context):
        """Retain the full ledger and group spectra for ordinary product output."""
        if "solved_groups" not in data.children:
            msg = "TimestreamDiagnostics requires solved Python model groups"
            raise ValueError(msg)
        solved, config = data["solved_groups"], context.config
        evidence, meta = _evidence(config.evidence, solved.attrs["calibration_source"])
        grid = _sample_grid(data.toltec_kids.block)
        f_smp = data.toltec_kids.block.meta.f_smp.to_value("Hz")
        ledger = solved.to_dataset(inherit=False).to_dataframe().to_dict("records")
        lookup = {(row["group_id"], row["resonator_id"]): row for row in ledger}
        if not evidence.keys() <= lookup.keys():
            msg = "Response evidence names components absent from the fitted model"
            raise ValueError(msg)
        result = xr.DataTree()
        # Each group is computed once; persistence uses these accepted arrays.
        solved.load()
        observed = (
            data.toltec_kids.block.timestream.data_source[["I", "Q"]]
            / solved.attrs["iq_normalization"]
        )
        for name, group in solved["groups"].children.items():
            diagnostics, records = diagnose_group(
                group.to_dataset(inherit=False),
                f_smp=f_smp,
                nperseg=config.nperseg,
                sample_grid=grid,
                observed=observed,
            )
            result[f"groups/{name}"] = diagnostics
            for row in records:
                lookup[(row["group_id"], row["resonator_id"])].update(row)
        for row in ledger:
            row.setdefault("numerical_complete", False)
            row.setdefault("spectral_status", "no_selected_probe")
            row.setdefault(
                "multiplicity", sum(r["group_id"] == row["group_id"] for r in ledger)
            )
            row["fit_policy_pass"] = (
                row["residual_to_sweep_variation"] <= config.fit_residual_max
            )
            item = evidence.get((row["group_id"], row["resonator_id"]))
            row["cohort"] = "missing_evidence" if item is None else str(item["cohort"])
            row["response_slope"] = (
                np.nan
                if item is None
                else float(item["slope"])
                * float(item["reference_frequency_hz"])
                / row.get("probe_frequency_hz", np.nan)
            )
            row["signal_referred_asd_ratio"] = np.nan
            if (
                row["cohort"] == "active_suitable"
                and row["fit_policy_pass"]
                and row["spectral_status"] == "complete"
                and np.isfinite(row["response_slope"])
            ):
                transfer = (1 - row["mu"] * row["response_slope"]) / row["D"]
                row["signal_referred_asd_ratio"] = row["coordinate_asd_ratio"] * abs(
                    transfer
                )
        keys = set().union(*(row.keys() for row in ledger))
        dataset = xr.Dataset(
            {
                key: ("component", [row.get(key, np.nan) for row in ledger])
                for key in sorted(keys)
            },
            attrs={
                "contract": "kids.timestream_diagnostics/1",
                "sample_grid": grid,
                "band_hz": [5, 20],
                "fit_residual_max": config.fit_residual_max,
                "optical_assumption": (
                    "external response slope, same loading assumed; "
                    "no absolute sensitivity"
                ),
                **meta,
            },
        )
        result.dataset = dataset
        data["timestream_diagnostics"] = result
        context.artifacts = ArtifactContext()
        context.artifacts.add(config.artifacts.ledger, dataset)
        performance = config.artifacts.performance
        context.artifacts.add(performance, performance(dataset))
        if config.evidence is not None:
            context.artifacts.add(
                config.artifacts.evidence, config.evidence.read_bytes()
            )
