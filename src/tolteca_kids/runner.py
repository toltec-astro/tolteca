"""Catalog preparation and resource-scoped execution of KIDs pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.utils.log import logger, timeit

from tolteca_datamodels.filestore import FileAccess
from tolteca_datamodels.toltec.kids.source import identify_source

from .core import Kids
from .dataprod_output import DataProdOutput
from .noop import NoOp

if TYPE_CHECKING:
    from pathlib import Path

    from tollan.config import RuntimeContext

    from tolteca_datamodels.file import SourceInfoModel

__all__ = ["create_kids", "run_source"]


def create_kids(runtime: RuntimeContext, *, context_input: dict) -> Kids:
    """Create a per-input KIDs handler without opening source files.

    ``context_input`` supplies metadata for conditional configuration. The
    shared product root is the default output path; explicit stage settings
    take precedence. The supplied runtime is not modified.
    """
    files = FileAccess(runtime).config
    defaults = None
    if files.dataprod_toltec_path is not None:
        defaults = Kids.prepare_runtime_config_data(
            {"output": {"path": files.dataprod_toltec_path}}
        )
    return Kids(
        runtime.with_config(defaults=defaults, validation_context=context_input)
    )


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
        kids = create_kids(runtime, context_input=context_input)
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
        working["tolteca_kids"].attrs["input"] = info.model_dump(
            mode="json", round_trip=True
        )
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
