"""Tlaloc tone and acquired-channel model files shared by KIDs workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from astropy.table import Table
from filelock import FileLock
from pydantic import ConfigDict, Field, model_validator
from tollan.config.types import AbsAnyPath  # noqa: TC002 -- Pydantic field.
from tollan.pipeline import (
    ArtifactFile,
    ArtifactSpec,
    Step,
    StepConfig,
    StepContext,
    StepFailure,
    StepFailureKind,
)

if TYPE_CHECKING:
    from .blocks import ToltecBlockView

# This is a next-comb proposal, not an acquired-channel model.
TLALOC_REPORT_COLUMNS = ("f_centered", "f_out", "flag")


@dataclass(frozen=True)
class TlalocKidsModel:
    """Acquired-channel model handoff, with zero rows when no fit is supplied.

    Fitted rows retain the recorded channel order and model descriptor. An
    empty handoff retains acquisition identity without claiming a model.
    """

    data: xr.Dataset

    def __post_init__(self):
        from .model_fit import _validate_report  # noqa: PLC0415 -- model/output cycle

        table = self.to_table()
        if len(table):
            _validate_report(table)
        else:
            from .model_fit import _validate_report_columns  # noqa: PLC0415

            required = {"chan", "f_in", "f_out", "flag"}
            if (
                table.meta.get("schema") != "kids.fit/1"
                or set(table.colnames) != required
                or "model_schema" in table.meta
                or "model" in table.meta
            ):
                msg = (
                    "Empty handoff requires kids.fit/1 channel columns without a model"
                )
                raise ValueError(msg)
            _validate_report_columns(table, required, {})
        for key in (
            "Header.Toltec.ObsNum",
            "Header.Toltec.SubObsNum",
            "Header.Toltec.ScanNum",
            "Header.Toltec.RoachIndex",
            "block",
        ):
            value = table.meta.get(key)
            if type(value) is not int or value < 0:
                msg = f"Model handoff requires nonnegative integer {key}"
                raise ValueError(msg)
        accumlen = table.meta.get("Header.Toltec.AccumLen")
        if (
            not isinstance(accumlen, (int, float))
            or not np.isfinite(accumlen)
            or accumlen <= 0
        ):
            msg = "Model handoff requires positive acquisition accumulation length"
            raise ValueError(msg)

    @classmethod
    def empty(cls, block: ToltecBlockView) -> TlalocKidsModel:
        """Create a zero-row handoff for an acquired block without fitted models.

        Parameters
        ----------
        block : ToltecBlockView
            Acquired sweep block with recorded identity and IQ accumulation
            length. Keep its raw source open unless normalization is retained
            in the reduced sweep.

        Returns
        -------
        TlalocKidsModel
            Empty channel columns and acquisition metadata, without a model
            descriptor or computed fit result.
        """
        from .model_calibration import calibration_accumlen  # noqa: PLC0415

        meta = block.meta
        table = Table(
            names=["chan", "f_in", "f_out", "flag"],
            dtype=["int64", "float64", "float64", "int64"],
        )
        for name in ("f_in", "f_out"):
            table[name].unit = "Hz"
        table.meta = {
            "schema": "kids.fit/1",
            "block": meta.block,
            "Header.Toltec.ObsNum": meta.obsnum,
            "Header.Toltec.SubObsNum": meta.subobsnum,
            "Header.Toltec.ScanNum": meta.scannum,
            "Header.Toltec.RoachIndex": meta.roach,
            "Header.Toltec.AccumLen": calibration_accumlen(block),
        }
        return cls.from_table(table)

    def validate_block(self, block: ToltecBlockView) -> None:
        """Check acquisition identity and, for fitted rows, the acquired probes.

        Parameters
        ----------
        block : ToltecBlockView
            Source sweep block, accessible until validation returns.

        Returns
        -------
        None
            The handoff matches the block and its IQ normalization.

        Raises
        ------
        ValueError
            The acquisition, block, normalization or fitted channel order differs.
        """
        from .model_calibration import calibration_accumlen  # noqa: PLC0415
        from .model_fit import _validate_binding  # noqa: PLC0415

        meta = block.meta
        expected = {
            "Header.Toltec.ObsNum": meta.obsnum,
            "Header.Toltec.SubObsNum": meta.subobsnum,
            "Header.Toltec.ScanNum": meta.scannum,
            "Header.Toltec.RoachIndex": meta.roach,
            "Header.Toltec.AccumLen": calibration_accumlen(block),
            "block": meta.block,
        }
        if any(self.data.attrs.get(key) != value for key, value in expected.items()):
            msg = "Model handoff differs from the acquired block"
            raise ValueError(msg)
        if self.data.sizes["chan"]:
            _validate_binding(self.to_table(), block)

    @classmethod
    def from_table(cls, table: Table) -> TlalocKidsModel:
        """Retain a validated fit report with its column units and metadata."""
        data = xr.Dataset(
            {name: ("chan", np.asarray(table[name])) for name in table.colnames},
            attrs=dict(table.meta),
        )
        for name in table.colnames:
            if table[name].unit is not None:
                data[name].attrs["unit"] = str(table[name].unit)
        return cls(data)

    @classmethod
    def read(cls, network: Path) -> TlalocKidsModel:
        """Read the acquired-channel handoff from a staged network directory."""
        return cls.from_table(
            Table.read(network / "kids_model.ecsv", format="ascii.ecsv")
        )

    def to_table(self) -> Table:
        """Return the fit table with its schema and acquisition provenance."""
        table = Table({name: self.data[name].values for name in self.data.variables})
        table.meta = dict(self.data.attrs)
        for name in table.colnames:
            table[name].unit = self.data[name].attrs.get("unit")
        return table


class TlalocKidsModelArtifact(ArtifactSpec[TlalocKidsModel]):
    """Acquired-channel model report for the Tlaloc header handoff."""

    def write(self, data: TlalocKidsModel, *, path: Path) -> list[ArtifactFile]:
        """Write kids_model.ecsv below toltecN without changing model rows.

        Parameters
        ----------
        data : TlalocKidsModel
            Validated acquired-channel fit or empty acquisition handoff.
        path : pathlib.Path
            Output root. An existing report is rejected.

        Returns
        -------
        list of ArtifactFile
            One acquired-channel ECSV report, named ``kids_model``.
        """
        table = data.to_table()
        destination = path / f"toltec{table.meta['Header.Toltec.RoachIndex']}"
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "kids_model.ecsv"
        table.write(target, format="ascii.ecsv")
        return [ArtifactFile(target, "ecsv", name="kids_model")]


@dataclass(frozen=True)
class TlalocTones:
    """Prepared next-tone proposal in its final shared file order.

    Parameters
    ----------
    data : xarray.Dataset
        One-dimensional frequency, flag, target power, phase and mask columns.
        Metadata identifies the acquisition,
        ROACH and LO frequency. Target powers are in dB relative to a full-scale
        DAC sinusoid after drive attenuation. Values must
        remain available until export.
    """

    data: xr.Dataset

    @classmethod
    def read(cls, network: Path) -> TlalocTones:
        """Read staged frequencies, power targets, phases and placeholder masks."""
        report = Table.read(network / "targ_freqs.dat", format="ascii.ecsv")
        if tuple(report.colnames) != TLALOC_REPORT_COLUMNS:
            msg = "Unexpected Tlaloc tone report layout"
            raise ValueError(msg)
        ds = xr.Dataset(
            {name: ("tone", np.asarray(report[name])) for name in report.colnames},
            attrs=dict(report.meta),
        )
        ds.attrs.update(
            kind="tones",
            report_columns=report.colnames,
            roach=report.meta["Header.Toltec.RoachIndex"],
            f_lo_center=report.meta["Header.Toltec.LoCenterFreq"],
        )
        for name in report.colnames:
            if report[name].unit is not None:
                ds[name].attrs["unit"] = str(report[name].unit)
        for name, filename in [
            ("power_target_db", "targ_power.dat"),
            ("phase_tone", "random_phases.dat"),
            ("mask_tone", "default_targ_masks.dat"),
        ]:
            if not (network / filename).read_text().strip():
                msg = f"Empty Tlaloc tone file: {filename}"
                raise ValueError(msg)
            values = np.loadtxt(network / filename, ndmin=1)
            if name == "mask_tone" and not np.isin(values, [0, 1]).all():
                msg = "Invalid Tlaloc mask values"
                raise ValueError(msg)
            ds[name] = ("tone", values.astype(bool) if name == "mask_tone" else values)
        return cls(ds)

    def __post_init__(self):
        required = {
            "f_centered",
            "f_out",
            "flag",
            "power_target_db",
            "phase_tone",
            "mask_tone",
        }
        if not required.issubset(self.data.data_vars):
            msg = "Incomplete Tlaloc tone table"
            raise ValueError(msg)
        if (
            not 0 <= self.roach <= 12  # noqa: PLR2004 -- network range.
            or not 0 < self.data.sizes.get("tone", 0) <= 1000  # noqa: PLR2004 -- firmware limit.
        ):
            msg = "Invalid Tlaloc network or tone count"
            raise ValueError(msg)
        for name, column in self.data.data_vars.items():
            if column.dims != ("tone",):
                msg = f"Tlaloc column {name} must have only the tone dimension"
                raise ValueError(msg)
        for name in ("f_centered", "f_out", "phase_tone"):
            if not np.isfinite(self.data[name]).all():
                msg = f"Nonfinite Tlaloc tone property: {name}"
                raise ValueError(msg)
        mask = self.data.mask_tone.values
        targets = self.data.power_target_db.values
        if not np.isin(mask, [0, 1]).all() or not np.any(mask):
            msg = "Tlaloc requires binary masks and at least one enabled tone"
            raise ValueError(msg)
        if not np.isfinite(targets[np.asarray(mask, bool)]).all():
            msg = "Every enabled Tlaloc tone requires a finite power target"
            raise ValueError(msg)
        if not np.isnan(targets[~np.asarray(mask, bool)]).all():
            msg = "Placeholder Tlaloc tones require NaN power targets"
            raise ValueError(msg)

    @property
    def roach(self) -> int:
        """Readout network index."""
        return int(self.data.attrs["roach"])


class TlalocToneArtifact(ArtifactSpec[TlalocTones]):
    """Aligned Tlaloc frequencies, power targets, phases and placeholder masks."""

    def write(self, data: TlalocTones, *, path: Path) -> list[ArtifactFile]:
        """Export the per-tone power handoff in its shared channel order.

        Parameters
        ----------
        data : TlalocTones
            Prepared table. No independent sorting or amplitude scaling occurs.
        path : pathlib.Path
            New export root, containing toltecN. Existing files are rejected.

        Returns
        -------
        list of ArtifactFile
            ECSV frequency report and aligned headerless power, phase and mask files.
        """
        ds = data.data
        destination = path / f"toltec{data.roach}"
        files = {
            "targ_freqs.dat": None,
            "targ_power.dat": "power_target_db",
            "random_phases.dat": "phase_tone",
            "default_targ_masks.dat": "mask_tone",
        }
        if any((destination / name).exists() for name in files):
            msg = f"Tlaloc tone files already exist in {destination}"
            raise FileExistsError(msg)
        report = Table({name: ds[name].values for name in ds.attrs["report_columns"]})
        for name in report.colnames:
            report[name].unit = ds[name].attrs.get("unit")
        report.meta = {
            key: value
            for key, value in ds.attrs.items()
            if key.startswith("Header.Toltec.")
            or key
            in {
                "schema",
                "roach",
                "f_lo_center",
                "block",
            }
        }
        destination.mkdir(parents=True, exist_ok=True)
        receipts = []
        for name, column in files.items():
            target = destination / name
            fmt = "ascii.ecsv" if column is None else "ascii.no_header"
            table = (
                report
                if column is None
                else Table(
                    {
                        column: ds[column].values.astype(int)
                        if column == "mask_tone"
                        else ds[column].values
                    }
                )
            )
            table.write(target, format=fmt)
            receipts.append(ArtifactFile(target, "ecsv" if column is None else "ascii"))
        return receipts


class TlalocOutputConfig(StepConfig):
    """Install prepared settings in a Tlaloc etc directory.

    Parameters
    ----------
    enabled : bool, optional
        Enable file installation in a workflow, default False.
    path : pathlib.Path or None, optional
        Tlaloc etc root containing toltecN directories. Required when enabled.
        Prepared tones replace frequency, target power, phase and mask files.
        Each tone handoff installs kids_model.ecsv in acquired-channel order;
        its count can differ from the next-tone count. Without fitting, the
        file has zero rows and retains the acquisition identity. An invalid
        handoff fails before replacing files.
        Writing files does not upload tones or apply hardware attenuation.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    path: AbsAnyPath | None = None

    @model_validator(mode="after")
    def _require_path(self):
        if self.enabled and self.path is None:
            msg = "Enabled TlalocOutput requires an explicit path"
            raise ValueError(msg)
        return self


class TlalocOutputContext(StepContext["TlalocOutput", TlalocOutputConfig]):
    """Installed settings, written-file receipts and execution status."""

    data: TlalocTones | None = Field(default=None, exclude=True)
    files: list[ArtifactFile] = Field(default_factory=list)


class TlalocOutput(Step[TlalocOutputConfig, TlalocOutputContext]):
    """Install next-tone files and acquired-channel models.

    This step can consume live or reopened KIDs products.
    It validates the payload before replacing destination files under a
    per-network writer lock. Each file replacement is atomic; the set is not
    a filesystem transaction.
    """

    @classmethod
    def run(cls, data: xr.DataTree, context: TlalocOutputContext) -> None:
        """Install prepared files without recomputing or reordering tones.

        Parameters
        ----------
        data : xarray.DataTree
            Interface tree with a tlaloc dataset containing prepared settings.
        context : TlalocOutputContext
            Destination settings and resulting payload/file receipts.

        Returns
        -------
        None
            Missing settings record a failure. Filesystem errors are raised and
            recorded by the step lifecycle.
        """
        if "tlaloc" not in data.children:
            context.failures.append(
                StepFailure(
                    kind=StepFailureKind.missing_input,
                    message="Tlaloc output requires prepared tone settings.",
                )
            )
            return
        if context.config.path is None:
            msg = "TlalocOutput requires an explicit path"
            raise ValueError(msg)
        from tempfile import TemporaryDirectory  # noqa: PLC0415 -- output boundary

        prepared = data["tlaloc"].to_dataset()
        settings = TlalocTones(prepared.load())
        root = context.config.path
        destination = root / f"toltec{settings.roach}"
        destination.mkdir(parents=True, exist_ok=True)
        channels = _prepared_model(data)
        with (
            FileLock(destination / ".tlaloc-output.lock"),
            TemporaryDirectory(prefix=".tlaloc-output-", dir=root) as staging,
        ):
            files = TlalocToneArtifact().write(settings, path=Path(staging))
            files.extend(TlalocKidsModelArtifact().write(channels, path=Path(staging)))
            # Replace the tone report after its companions.
            files.sort(key=lambda receipt: receipt.path.name == "targ_freqs.dat")
            for receipt in files:
                target = destination / receipt.path.name
                receipt.path.replace(target)
                context.files.append(ArtifactFile(target, receipt.format))
        context.data = settings


def _prepared_model(data: xr.DataTree) -> TlalocKidsModel:
    """Validate model presence and acquisition identity before installation."""
    prepared = data["tlaloc"].to_dataset()
    if "tlaloc_kids_model" in data.children:
        channels = TlalocKidsModel(data["tlaloc_kids_model"].to_dataset())
        identity = (
            "Header.Toltec.ObsNum",
            "Header.Toltec.SubObsNum",
            "Header.Toltec.ScanNum",
            "Header.Toltec.RoachIndex",
            "block",
        )
        if any(channels.data.attrs[key] != prepared.attrs[key] for key in identity):
            msg = "Channel model and tone proposal identify different blocks"
            raise ValueError(msg)
        return channels
    return TlalocKidsModel.empty(data.toltec_kids.block)
