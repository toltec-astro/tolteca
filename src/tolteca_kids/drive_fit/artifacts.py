"""Scientific drive-fit artifacts and their materialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr
from pydantic import Field
from tollan.pipeline import (
    PIPELINE_CONTEXT_KEY,
    ArtifactContext,
    ArtifactFile,
    ArtifactSpec,
)

if TYPE_CHECKING:
    from pathlib import Path

    from astropy.table import QTable

    from .result import DriveFitResult


class DriveFitDataArtifact(ArtifactSpec[xr.DataTree]):
    """Scientific arrays, input identities and saved step execution records."""

    def write(self, data: xr.DataTree, *, path: Path) -> list[ArtifactFile]:
        """Write an interface tree and its execution records as Zarr.

        Parameters
        ----------
        data : xarray.DataTree
            Working interface with fitted arrays. Required resources stay open
            until writing finishes; the input tree is not modified.
        path : pathlib.Path
            Destination directory for ``result.zarr``.

        Returns
        -------
        list of ArtifactFile
            The closed Zarr store.
        """
        saved = data.copy(deep=False)
        saved.name = None
        for node in saved.subtree:
            records = node.attrs.pop(PIPELINE_CONTEXT_KEY, {})
            if records:
                node["execution"] = xr.DataTree(
                    dataset=xr.Dataset(
                        attrs={
                            key: ctx.step_cls.dump_context(ctx)
                            for key, ctx in records.items()
                        }
                    )
                )
            node.dataset = node.dataset.drop_encoding()
        target = path / "result.zarr"
        saved.to_zarr(target, mode="w-", zarr_format=2, consolidated=True)
        with xr.open_datatree(target, engine="zarr", consolidated=True) as reopened:
            xr.testing.assert_identical(saved, reopened.load())
        return [ArtifactFile(target, "zarr")]


class DriveFitTableArtifact(ArtifactSpec["QTable"]):
    """Per-channel drive selections in an Astropy ECSV table."""

    def __call__(self, result: DriveFitResult) -> QTable:
        """Select the channel summary from a fitted result.

        Parameters
        ----------
        result : DriveFitResult
            Completed result with accessible arrays.

        Returns
        -------
        astropy.table.QTable
            Loaded summary with units and fit provenance.
        """
        return result.table

    def write(self, data: QTable, *, path: Path) -> list[ArtifactFile]:
        """Write a retained channel summary.

        Parameters
        ----------
        data : astropy.table.QTable
            Prepared drive selections.
        path : pathlib.Path
            Destination directory for ``adrv.ecsv``.

        Returns
        -------
        list of ArtifactFile
            Written table.
        """
        path.mkdir(parents=True, exist_ok=True)
        target = path / "adrv.ecsv"
        data.write(target, format="ascii.ecsv")
        return [ArtifactFile(target, "ecsv")]


class DriveFitArtifacts(ArtifactSpec[ArtifactContext]):
    """Scientific tree and channel-summary artifacts of a fit invocation."""

    arrays: DriveFitDataArtifact = Field(default_factory=DriveFitDataArtifact)
    table: DriveFitTableArtifact = Field(default_factory=DriveFitTableArtifact)

    def __call__(self, tree: xr.DataTree, result: DriveFitResult) -> ArtifactContext:
        """Retain the scientific tree and prepare its channel table.

        Parameters
        ----------
        tree : xarray.DataTree
            Working interface tree, retained by reference until publication.
        result : DriveFitResult
            Fitted arrays installed in that tree.

        Returns
        -------
        ArtifactContext
            Typed scientific and tabular artifacts.
        """
        artifacts = ArtifactContext()
        artifacts.add(self.arrays, tree)
        artifacts.add(self.table, self.table(result))
        return artifacts

    def write(self, data: ArtifactContext, *, path: Path) -> list[ArtifactFile]:
        """Materialize retained scientific artifacts in one directory.

        Parameters
        ----------
        data : ArtifactContext
            Previously prepared artifacts.
        path : pathlib.Path
            Caller-owned output directory.

        Returns
        -------
        list of ArtifactFile
            Scientific store and summary table receipts.
        """
        return [
            receipt
            for artifact in data.values()
            for receipt in artifact.spec.write(artifact.data, path=path)
        ]
