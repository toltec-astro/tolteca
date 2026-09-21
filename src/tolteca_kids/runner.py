"""Catalog preparation and resource-scoped execution of KIDs pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.utils.log import logger, timeit

from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.filestore import resolve_sources
from tolteca_datamodels.toltec.kids.source import identify_source

from .core import Kids
from .dataprod_output import DataProdOutput
from .noop import NoOp

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import pandas as pd
    from tollan.config import RuntimeContext

    from tolteca_datamodels.toltec.file import SourceInfoModel

__all__ = ["build_input_catalog", "run_source"]


def build_input_catalog(
    sources: Iterable[str | Path], *, data_root: Path | None = None
) -> pd.DataFrame:
    """Discover files and read scalar headers for configuration and selection.

    Files are closed before returning. The catalog contains only metadata;
    I/Q arrays are not read. Unsupported file kinds retain discovery metadata
    so callers can inspect or filter them before attempting execution.
    """
    catalog = resolve_sources(sources, data_root=data_root)
    logger.debug("Discovered {} distinct input files", len(catalog))
    records = []
    for record in catalog.toltec_file.to_info_list():
        info = record
        if info.filepath.suffix == ".nc" and (
            info.interface is None or info.roach is not None
        ):
            with xr.open_dataset(info.filepath) as data:
                info = identify_source(data, info)
        logger.debug("Input metadata: {}", info)
        records.append(info)
    return ToltecFileAccessor.from_info_list(records)


@timeit
def run_source(
    info: SourceInfoModel, runtime: RuntimeContext, *, dry_run: bool = False
) -> Path | None:
    """Run one source with its conditional config, closing resources on exit.

    Return the published product path, or ``None`` when output is disabled.
    Use the configured steps directly for interactive in-memory results.
    With ``dry_run=True``, validate configuration and execute only a no-op step.
    Scalar headers are read; signal arrays and product writers are not used.
    """
    logger.info("Processing source {} (dry_run={})", info.filepath, dry_run)
    with xr.open_dataset(info.filepath) as raw:
        info = identify_source(raw, info)
        context_input = info.model_dump(
            mode="json",
            context={"flat": True},
            exclude={"filepath_info", "filepath_info_orig"},
        )
        logger.debug("Resolving configuration with input context: {}", context_input)
        kids = Kids(runtime.with_config(validation_context=context_input))
        logger.opt(lazy=True).debug(
            "Effective KIDs configuration: {}", kids.config.model_dump
        )
        pipeline = kids.noop_pipeline if dry_run else kids.pipeline_for(info.data_kind)
        logger.debug(
            "Selected steps: {}", [step.context_key for step in pipeline.steps]
        )
        # Wrapping only: source arrays and metadata are not transformed.
        working = ensure_datatree(raw)
        working["tolteca_kids"] = xr.DataTree()
        working["tolteca_kids"].attrs["input"] = context_input
        pipeline(working)
        if dry_run:
            logger.info(
                "No-op completed for {}: {}",
                info.filepath,
                NoOp.get_context(working).completed,
            )
            return None
        if DataProdOutput.has_context(working):
            return DataProdOutput.get_context(working).product_path
    logger.debug("Closed source {} without product output", info.filepath)
    return None
