"""Commit artifacts: scientific diagnostics, table and Tlaloc settings files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astropy.table import QTable
from pydantic import Field
from tollan.pipeline import ArtifactContext, ArtifactFile, ArtifactSpec

from tolteca_kids.drive_fit.artifacts import DriveFitDataArtifact
from tolteca_kids.tlaloc_output import TlalocSettings, TlalocSettingsArtifact

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

    from .step import DriveFitCommitResult


class DriveFitCommitTableArtifact(ArtifactSpec[QTable]):
    """Target-channel commit table in ECSV format."""

    def write(self, data: QTable, *, path: Path) -> list[ArtifactFile]:
        """Write the prepared commit table.

        Parameters
        ----------
        data : astropy.table.QTable
            Loaded target settings and match diagnostics.
        path : pathlib.Path
            Destination directory for adrv_commit.ecsv.

        Returns
        -------
        list of ArtifactFile
            Written table receipt.
        """
        path.mkdir(parents=True, exist_ok=True)
        target = path / "adrv_commit.ecsv"
        data.write(target, format="ascii.ecsv")
        return [ArtifactFile(target, "ecsv")]


class DriveFitCommitArtifacts(ArtifactSpec[ArtifactContext]):
    """Scientific commit, summary table and instrument-settings artifacts."""

    arrays: DriveFitDataArtifact = Field(default_factory=DriveFitDataArtifact)
    table: DriveFitCommitTableArtifact = Field(
        default_factory=DriveFitCommitTableArtifact
    )
    settings: TlalocSettingsArtifact = Field(default_factory=TlalocSettingsArtifact)

    def __call__(
        self, tree: xr.DataTree, result: DriveFitCommitResult
    ) -> ArtifactContext:
        """Prepare outputs from a completed commit without repeating matching.

        Parameters
        ----------
        tree : xarray.DataTree
            Scientific interface tree, retained until materialization completes.
        result : DriveFitCommitResult
            Computed target settings.

        Returns
        -------
        ArtifactContext
            Native tree and loaded tables paired with their specifications.
        """
        table = result.table
        context = ArtifactContext()
        context.add(self.arrays, tree)
        context.add(self.table, table)
        context.add(
            self.settings, TlalocSettings.from_data(tree["tlaloc"].to_dataset())
        )
        return context

    def write(self, data: ArtifactContext, *, path: Path) -> list[ArtifactFile]:
        """Materialize retained commit outputs.

        Parameters
        ----------
        data : ArtifactContext
            Previously prepared commit artifacts.
        path : pathlib.Path
            Caller-owned destination directory.

        Returns
        -------
        list of ArtifactFile
            Receipts for every scientific, tabular and settings artifact.
        """
        return [
            receipt
            for artifact in data.values()
            for receipt in artifact.spec.write(
                artifact.data,
                path=path / "settings"
                if isinstance(artifact.spec, TlalocSettingsArtifact)
                else path,
            )
        ]
