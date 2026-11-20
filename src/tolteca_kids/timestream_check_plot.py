"""Timestream-check artifacts for one block and their pipeline execution record.

Use ``TimestreamCheckPlot()(block)`` to retain diagnostic figures in the step's
``context.artifacts``. ``context.config.artifacts`` owns construction, export and
local-display settings. Data-product output collects these retained artifacts.
For a single figure without a step, use the Viz classes in ``timestream_check_viz``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ConfigDict, Field
from tollan.config.types import AbsAnyPath  # noqa: TC002 -- Pydantic field
from tollan.pipeline import (
    ArtifactFile,
    Step,
    StepConfig,
    StepContext,
    StepFailure,
    StepFailureKind,
)

from .timestream_check import TimestreamCheck, TimestreamCheckView
from .timestream_check_viz import TimestreamCheckArtifacts

if TYPE_CHECKING:
    import xarray as xr


class TimestreamCheckPlotConfig(StepConfig):
    """Figure configuration and optional immediate export/display.

    Parameters
    ----------
    artifacts : TimestreamCheckArtifacts, optional
        Per-figure construction and shared export/display settings.
    check_alias : str or None, optional
        Check result to plot. None selects the primary check.
    save_path : pathlib.Path or None, optional
        Immediate export directory. None defers writing to data-product output.
        Supply a separate directory per block when exporting several blocks.
    show : bool, optional
        Display the produced figures locally, by default False.
    """

    model_config = ConfigDict(extra="forbid")

    artifacts: TimestreamCheckArtifacts = Field(
        default_factory=TimestreamCheckArtifacts
    )
    check_alias: str | None = None
    save_path: AbsAnyPath | None = None
    show: bool = False


class TimestreamCheckPlotContext(
    StepContext["TimestreamCheckPlot", TimestreamCheckPlotConfig]
):
    """Block-local figures in artifacts and any immediately exported files."""

    files: list[ArtifactFile] = Field(default_factory=list)


class TimestreamCheckPlot(Step[TimestreamCheckPlotConfig, TimestreamCheckPlotContext]):
    """Construct diagnostics from one block's saved timestream check.

    The context retains typed figures independently of their later publication.
    Missing checks record a missing-input failure. Use the KIDs runner to visit
    all blocks in an acquisition.
    """

    @classmethod
    def run(cls, data: xr.DataTree, context: TimestreamCheckPlotContext) -> None:
        """Create the configured artifacts for one checked block.

        Parameters
        ----------
        data : xarray.DataTree
            Single block containing the requested timestream-check subtree.
        context : TimestreamCheckPlotContext
            Configuration and destination for live artifacts and export receipts.

        Returns
        -------
        None
            Artifacts are retained on the context. Optional immediate export and
            local display use those same objects.
        """
        cfg = context.config
        path = TimestreamCheck.result_path(alias=cfg.check_alias)
        if path not in data.children:
            context.failures.append(
                StepFailure(
                    kind=StepFailureKind.missing_input,
                    message="Timestream-check plotting requires a checked block.",
                    details={"path": data.path},
                )
            )
            return
        context.artifacts = cfg.artifacts(TimestreamCheckView(data[path]))
        if cfg.save_path is not None:
            context.files = cfg.artifacts.write(context.artifacts, path=cfg.save_path)
        if cfg.show:
            cfg.artifacts.show(context.artifacts)
