"""KIDs pipeline execution on open data or file sources."""

from __future__ import annotations

from contextlib import ExitStack, closing, nullcontext
from sys import exception
from typing import TYPE_CHECKING, cast

import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.pipeline import ArtifactContext, StepFailure
from tollan.utils.fileloc import FileLoc, fileloc
from tollan.utils.log import logger, timeit

from tolteca_datamodels.data_prod import (
    DataProdStore,
    DataProdTagBusyError,
    WorkflowRun,
)
from tolteca_datamodels.toltec.file import guess_info_from_sources
from tolteca_datamodels.toltec.types import ToltecDataKind

from .context import _prepare_context_inputs, _raw_index_sources, _raw_source
from .core import Kids
from .dataprod_output import DataProdOutput
from .model_fit import ModelFit
from .noop import NoOp
from .product import _step_records
from .timestream_solve import TimestreamSolve
from .tone_prep import TonePrep

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from pathlib import Path
    from typing import Any

    import pandas as pd
    from tollan.config import RuntimeContext

    from .context import _ContextInputs

__all__ = [
    "create_kids",
    "publish_quicklook",
    "run_data",
    "run_input",
    "run_inputs",
    "run_source",
    "run_sources",
]


def _input_table(table: pd.DataFrame) -> pd.DataFrame:
    """Select input information without live handles or private reader state."""
    return table[
        [
            name
            for name in table
            if not (
                isinstance(name, str)
                and (name.startswith("_") or name in {"io_obj", "accessor_obj"})
            )
        ]
    ]


def create_kids(runtime: RuntimeContext, *, context_input: dict) -> Kids:
    """Create a per-input KIDs handler without opening source files.

    Parameters
    ----------
    runtime : tollan.config.RuntimeContext
        Runtime supplying KIDs processing and output settings. Not modified.
    context_input : dict
        Input information used to evaluate conditional configuration.

    Returns
    -------
    Kids
        Handler configured for this input.
    """
    return Kids(runtime.with_config(validation_context=context_input))


def _run_blocks(
    data: xr.DataTree, kids: Kids, context_inputs: _ContextInputs | None = None
) -> None:
    """Prepare acquisition blocks and run each block's configured pipeline."""
    kids.block_identify(data)
    kids.block_create(data)
    for block in data.toltec_kids.blocks:
        if kids.config.timestream_solve.enabled:
            if context_inputs is None:
                msg = "Use run_sources or run_inputs with the calibration inventory"
                raise ValueError(msg)
            calibration = context_inputs.calibration(
                block.meta.uid_raw_obs, block.meta.interface, block.meta.block
            )
            target = calibration.data_items[0]
            model = target.model
            index = context_inputs.model_indexes[model.product]
            artifact = model.resolve(
                index,
                raw_obs=calibration.raw_obs,
                interface=target.interface,
                block=target.block,
            )
            TimestreamSolve(
                kids.config.timestream_solve,
                source=data["tolteca_kids"].attrs["source"],
                sweep_source=_raw_source(
                    context_inputs.raw_indexes[calibration.raw_obs.uid],
                    target.interface,
                ),
                model_source=fileloc(index.get_source_uri(artifact.source)).path,
                calibration=calibration,
                model_index=index,
                stubs=context_inputs.stub_sources(
                    block.meta.uid_raw_obs, block.meta.interface
                ),
            )(block.tree)
        if kids.config.model_fit is None or not block.data_kind & ToltecDataKind.sweep:
            kids.pipeline_for(block.data_kind)(block.tree)
        else:
            kids.make_sweep_pipeline(
                model_fit=ModelFit(
                    kids.config.model_fit,
                    source=data["tolteca_kids"].attrs["source"],
                )
            )(block.tree)
            if kids.config.model_fit.enabled and (
                not ModelFit.has_context(block.tree, alias=kids.config.model_fit.alias)
                or not ModelFit.get_context(
                    block.tree, alias=kids.config.model_fit.alias
                ).completed
            ):
                msg = f"Model fitting did not complete for block {block.meta.block}"
                raise RuntimeError(msg)


def _install_tones(working, kids, *, publication_tree=None):
    """Install the latest prepared block after configured publication succeeds."""
    if not kids.config.tlaloc_output.enabled:
        return
    if kids.config.output.enabled:
        output = DataProdOutput.get_context(
            working if publication_tree is None else publication_tree
        )
        if not output.completed or (
            kids.config.output.registration and not output.registered
        ):
            msg = "Tlaloc installation requires successful publication and registration"
            raise RuntimeError(msg)
    blocks = [
        block for block in working.toltec_kids.blocks if "sweep" in block.tree.children
    ]
    if not blocks:
        msg = "Tlaloc installation requires a sweep block"
        raise ValueError(msg)
    block = blocks[-1].tree
    if not TonePrep.get_context(block).completed:
        msg = "Tlaloc installation requires completed tone preparation"
        raise ValueError(msg)
    kids.tlaloc_output(block)


def run_data(
    data: xr.Dataset | xr.DataTree,
    source: str | Path | FileLoc,
    runtime: RuntimeContext,
    *,
    dry_run: bool = False,
    source_loc: str | Path | FileLoc | None = None,
) -> xr.DataTree:
    """Run a KIDs pipeline on open data and return its working tree.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataTree
        Open acquisition data. A tree may contain native data at the root or
        in the acquisition ``raw`` leaf. The caller owns the data and must keep
        it open for lazy access.
    source : str, pathlib.Path or FileLoc
        Source location recorded with the results; not opened here.
    runtime : tollan.config.RuntimeContext
        Runtime used to configure the pipeline from the input's header metadata.
    dry_run : bool, optional
        Whether to run only the no-op step, without reduction or output.
        Default is False.
    source_loc : str, pathlib.Path or FileLoc, optional
        Filename identification location, if different from ``source``.

    Returns
    -------
    xarray.DataTree
        Working tree containing results and execution contexts. A Dataset is
        wrapped without copying its arrays; a supplied tree is updated in place.
        All identified blocks are processed. Processing contexts belong to their
        blocks; preparation and publication contexts belong to the acquisition.
        Unrelated nodes are preserved.

    Raises
    ------
    ValueError
        If no usable input is selected, or input validation fails with
        ``kids.input.raise_on_errors=True``.
    """
    metadata_source = (
        data.toltec_kids.data_source if isinstance(data, xr.DataTree) else data
    )
    table = guess_info_from_sources(
        [
            {
                "source": metadata_source,
                "source_loc": source if source_loc is None else source_loc,
            }
        ],
        on_error="record",
    )
    with table.toltec_file.open(on_error="record") as opened:
        table.toltec_file.read_metadata(on_error="record")
        selected = Kids(runtime).input.select(opened)
        if selected.empty:
            msg = "No inputs selected"
            raise ValueError(msg)
        # Live handles belong to the open-table scope, not configuration or steps.
        row = cast("Any", next(_input_table(selected).itertuples(index=False)))
        # Pipeline provenance records the location supplied by the caller.
        return run_input(data, row._replace(source=source), runtime, dry_run=dry_run)


def run_input(  # noqa: PLR0913 -- open source, runtime and caller-owned execution policy
    data: xr.Dataset | xr.DataTree,
    row: Any,
    runtime: RuntimeContext,
    *,
    dry_run: bool = False,
    defer_output: bool = False,
    context_inputs: _ContextInputs | None = None,
) -> xr.DataTree:
    """Run a pipeline with an opened input and its prepared table row.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataTree
        Open input. The caller owns its lifetime.
    row : namedtuple
        Selected, header-populated row from ``itertuples(index=False)``, excluding
        live ``io_obj`` and ``accessor_obj`` handles and private columns.
        Supplies source, source_loc, data_kind_toltec and the per-input
        configuration context. Must describe ``data``.
    runtime : tollan.config.RuntimeContext
        Runtime supplying processing configuration.
    dry_run : bool, optional
        Run the no-op pipeline only, by default False.
    defer_output : bool, optional
        Leave publication to the caller after processing. Default False.
        `run_inputs` uses this to publish selected interfaces together.
    context_inputs : object or None, optional
        Internally prepared calibration inputs supplied by run_inputs. Direct
        calibrated calls should use run_sources to prepare the full inventory.

    Returns
    -------
    xarray.DataTree
        Working tree with results and execution contexts. Input metadata is
        consumed as supplied, without opening files or reading headers again.
        Batch selection must already have been applied with ``Kids.input.select``.
    """
    source = row.source
    source_loc = row.source_loc
    logger.info("Processing source {} (dry_run={})", source, dry_run)
    context_input = row._asdict()
    logger.debug("Resolving configuration with input context: {}", context_input)
    kids = create_kids(runtime, context_input=context_input)
    logger.opt(lazy=True).debug(
        "Effective KIDs configuration: {}", kids.config.model_dump
    )
    working = ensure_datatree(data)
    if "tolteca_kids" not in working:
        working["tolteca_kids"] = xr.DataTree()
    working["tolteca_kids"].attrs["source"] = fileloc(source).url
    working["tolteca_kids"].attrs["source_loc"] = (
        None if source_loc is None else fileloc(source_loc).url
    )
    # Retain the prepared configuration context for live inspection.
    working["tolteca_kids"].attrs["input_row"] = context_input
    working["tolteca_kids"].attrs["config"] = kids.config.model_dump(
        mode="json", round_trip=True
    )
    if dry_run:
        kids.noop_pipeline(working)
    elif defer_output or not kids.config.output.enabled:
        _run_blocks(working, kids, context_inputs)
    else:
        config = kids.config.output
        registration = kids.registration
        run = WorkflowRun.create(
            uid=kids.config.run_uid,
            meta=kids.config.run_meta,
            runtime_info=runtime.runtime_info,
        )
        with ExitStack() as destinations, run:
            _retain_run(destinations, config, run, registration)
            try:
                output = DataProdOutput(config, registration=registration)
                with output.prepare(row, run=run):
                    _run_blocks(working, kids, context_inputs)
                    output._artifacts = _quicklook(
                        {working.toltec_kids.meta.interface: working},
                        kids.config.quicklook,
                    )
                    output(working)
                if _has_step_failure(working):
                    run.fail()
                _install_tones(working, kids)
            except DataProdTagBusyError as exc:
                run.fail()
                _record_tag_busy(working, config, exc)
    if not dry_run and not defer_output and not kids.config.output.enabled:
        _install_tones(working, kids)
    if dry_run:
        logger.info(
            "No-op completed for {}: {}",
            source,
            NoOp.get_context(working).completed,
        )
    return working


@timeit
def run_source(
    source: str | Path | FileLoc,
    runtime: RuntimeContext,
    *,
    dry_run: bool = False,
    source_loc: str | Path | FileLoc | None = None,
) -> Path | None:
    """Run one local file and close it on completion or failure.

    Parameters
    ----------
    source : str, pathlib.Path or FileLoc
        Local input file. Remote resources must be staged before calling.
    runtime : tollan.config.RuntimeContext
        Runtime used to configure the pipeline.
    dry_run : bool, optional
        Whether to run only the no-op step. Default is False.
    source_loc : str, pathlib.Path or FileLoc, optional
        Filename identification location, if different from ``source``.

    Returns
    -------
    pathlib.Path or None
        Pinned run directory, or None when no product was published
        (including disabled output, recorded publication failure, or ``dry_run=True``).
        Use `run_data` for caller-owned data and access to the working tree.

    Raises
    ------
    ValueError
        If the source is remote, no usable input is selected, or input validation
        fails with ``kids.input.raise_on_errors=True``.
    """
    location = fileloc(source)
    if location.is_remote():
        msg = "A remote source has no local filepath; stage it before opening"
        raise ValueError(msg)
    table = guess_info_from_sources(
        [{"source": location, "source_loc": source_loc}], on_error="record"
    )
    with table.toltec_file.open(on_error="record") as opened:
        table.toltec_file.read_metadata(on_error="record")
        selected = Kids(runtime).input.select(opened)
        if selected.empty:
            msg = "No inputs selected"
            raise ValueError(msg)
        row = next(_input_table(selected).itertuples(index=False))
        working = run_input(selected.io_obj.iloc[0], row, runtime, dry_run=dry_run)
        if dry_run:
            return None
        if DataProdOutput.has_context(working):
            return DataProdOutput.get_context(working).data
    logger.debug("Closed source {} without product output", location.path)
    return None


def run_sources(
    sources: Iterable[str | Path | FileLoc],
    runtime: RuntimeContext,
    *,
    dry_run: bool = False,
) -> Generator[tuple[str, xr.DataTree]]:
    """Open files and process the selected acquisitions.

    Parameters
    ----------
    sources : iterable of str, pathlib.Path or FileLoc
        Input files, prepared by the caller's discovery operation.
    runtime : tollan.config.RuntimeContext
        KIDs input selection, processing and output settings.
    dry_run : bool, optional
        Run only the no-op pipeline, by default False.

    Yields
    ------
    tuple of str and xarray.DataTree
        Acquisition UID and working tree, as returned by `run_inputs`.
        Opening and processing begin on iteration. Files remain open until
        iteration finishes or the generator is closed. Consume lazy results
        within that lifetime; use `run_inputs` with caller-owned open inputs
        when results must remain accessible afterwards.

    Raises
    ------
    ValueError
        No usable inputs are selected, or KIDs input validation fails.
    """
    config = Kids(runtime).config.context
    if config is not None:
        sources = list(
            dict.fromkeys(
                [fileloc(source).url for source in sources] + _raw_index_sources(config)
            )
        )
    table = guess_info_from_sources(sources, on_error="record")
    with table.toltec_file.open(on_error="record") as opened:
        table.toltec_file.read_metadata(on_error="record")
        selected = Kids(runtime).input.select(opened)
        if selected.empty:
            msg = "No inputs selected"
            raise ValueError(msg)
        yield from run_inputs(selected, runtime, dry_run=dry_run, available=opened)


def publish_quicklook(
    sources: Iterable[str | Path | FileLoc], runtime: RuntimeContext
) -> list[Path]:
    """Publish configured KIDs feedback from files.

    Parameters
    ----------
    sources : iterable of str, pathlib.Path or FileLoc
        Closed acquisition files to process. Sources may cover several
        acquisitions and interfaces.
    runtime : tollan.config.RuntimeContext
        KIDs settings with output and quicklook enabled. Input selection and
        block processing follow the same configuration as the KIDs CLI.

    Returns
    -------
    list of pathlib.Path
        Namespace descriptors and run/version-group indexes, in registration.items
        order. Files are closed before returning. No database registration is
        performed when ``kids.output.registration`` is False; pass these paths to
        the product-index ingestion workflow to register separately.

    Raises
    ------
    ValueError
        Output or quicklook is disabled, or no inputs are selected.
    RuntimeError
        A publication step reports failure. Acquisitions already published
        remain available through their output indexes.
    """
    config = Kids(runtime).config
    if not (config.output.enabled and config.quicklook.enabled):
        msg = "Enable kids.output and kids.quicklook for this workflow"
        raise ValueError(msg)
    references: list[Path] = []
    with closing(run_sources(sources, runtime)) as results:
        for _, working in results:
            output = DataProdOutput.get_context(working)
            if not output.completed or output.data is None:
                msg = f"Publication failed: {output.failures}"
                raise RuntimeError(msg)
            store = DataProdStore(output.config.path, namespace=output.config.namespace)
            for path in (
                output.data / "index.yaml",
                output.data.parent / "index.yaml",
                store.namespace_path / "namespace.yaml",
            ):
                if path not in references:
                    references.append(path)
    return [p for p in references if p.name != "namespace.yaml"] + [
        p for p in references if p.name == "namespace.yaml"
    ]


def run_inputs(
    selected: pd.DataFrame,
    runtime: RuntimeContext,
    *,
    dry_run: bool = False,
    available: pd.DataFrame | None = None,
) -> Generator[tuple[str, xr.DataTree]]:
    """Process selected open inputs and publish one product per acquisition.

    Parameters
    ----------
    selected : pandas.DataFrame
        Selected, header-populated input table including live ``io_obj`` handles.
        The caller owns these handles and keeps them open during iteration.
    runtime : tollan.config.RuntimeContext
        Per-input processing and output configuration.
    dry_run : bool, optional
        Run a no-op for each selected input without grouping or publication.
        Default False.
    available : pandas.DataFrame or None, optional
        Full opened collection before processing selection, used to prepare raw
        context inventories. None uses selected plus configured raw indexes.
        The caller retains ownership of all open handles.

    Yields
    ------
    tuple of str and xarray.DataTree
        Acquisition UID and working tree. Multiple interfaces occupy children
        ``interfaces/<interface>``; output context belongs to the collection.
        With ``dry_run=True``, yield one tree per input in table order; acquisition
        UIDs may repeat.

    Raises
    ------
    ValueError
        A processing acquisition has duplicate interfaces or conflicting output
        settings.
    """
    if dry_run:
        records_noop = _input_table(selected).itertuples(index=False)
        for data, row in zip(selected.io_obj, records_noop, strict=True):
            yield str(row.uid_raw_obs), run_input(data, row, runtime, dry_run=True)
        return
    context_config = Kids(runtime).config.context
    context_inputs = (
        _prepare_context_inputs(
            selected if available is None else available, selected, context_config
        )
        if context_config is not None
        else None
    )
    invocation = _batch_invocation(selected, runtime, context_inputs=context_inputs)
    with ExitStack() as destinations, invocation:
        retained = set()
        for uid, rows in selected.groupby("uid_raw_obs", sort=False):
            records: list[Any] = list(_input_table(rows).itertuples(index=False))
            configs = [
                create_kids(runtime, context_input=row._asdict()).config
                for row in records
            ]
            config = _shared_config(
                [item.output for item in configs], uid=uid, name="output"
            )
            registration = create_kids(
                runtime, context_input=records[0]._asdict()
            ).registration
            quicklook = _shared_config(
                [item.quicklook for item in configs], uid=uid, name="quicklook"
            )
            if rows.interface.duplicated().any():
                msg = f"Duplicate interfaces for acquisition {uid}"
                raise ValueError(msg)
            key = (str(config.path), config.namespace, config.registration)
            if config.enabled and key not in retained:
                _retain_run(destinations, config, invocation, registration)
                retained.add(key)
            output = DataProdOutput(config, registration=registration)
            scope = (
                output.prepare(records[0], run=invocation)
                if config.enabled
                else nullcontext()
            )
            try:
                with scope:
                    trees = {
                        row.interface: run_input(
                            data,
                            row,
                            runtime,
                            defer_output=True,
                            context_inputs=context_inputs,
                        )
                        for data, row in zip(rows.io_obj, records, strict=True)
                    }
                    working = _combine_interfaces(trees)
                    if config.enabled:
                        output._artifacts = _quicklook(trees, quicklook)
                        output(working)
                for row in records:
                    _install_tones(
                        trees[row.interface],
                        create_kids(runtime, context_input=row._asdict()),
                        publication_tree=working,
                    )
            except DataProdTagBusyError as exc:
                invocation.fail()
                working = xr.DataTree()
                _record_tag_busy(working, config, exc)
            if _has_step_failure(working):
                invocation.fail()
            yield str(uid), working


def _has_step_failure(working):
    """Whether an enabled scientific or output step reported failure."""
    return any(
        not context.completed
        for _, records in _step_records(working)
        for context in records.values()
        if context.config.enabled
    )


def _combine_interfaces(trees):
    """Return the working acquisition with each selected interface."""
    if len(trees) == 1:
        return next(iter(trees.values()))
    working = xr.DataTree()
    for interface, tree in trees.items():
        working[f"interfaces/{interface}"] = tree
    return working


def _batch_invocation(selected, runtime, *, context_inputs=None):
    """Create one execution identity and capture selected workflow settings."""
    records: list[Any] = list(_input_table(selected).itertuples(index=False))
    processing = {
        str(row.uid_raw_obs_file): {
            "source": fileloc(row.source).url,
            "config": create_kids(
                runtime, context_input=row._asdict()
            ).config.model_dump(mode="json", round_trip=True),
        }
        for row in records
    }
    context = (
        {"inputs": processing}
        if context_inputs is None
        else {
            "inputs": context_inputs.inputs,
            "input_indexes": context_inputs.snapshot_indexes(),
            "processing": processing,
        }
    )
    return WorkflowRun.create(
        uid=Kids(runtime).config.run_uid,
        meta=Kids(runtime).config.run_meta,
        runtime_info=runtime.runtime_info,
        context=context,
    )


def _retain_run(destinations, config, run, registration):
    """Deliver the final execution snapshot after the runner closes its scope."""
    store = DataProdStore(config.path, namespace=config.namespace)

    def save():
        previous = exception()
        try:
            store.write_run(run)
            if registration is not None:
                registration(
                    [store.namespace_path / "namespace.yaml"], run_uids=[run.uid]
                )
        except Exception as exc:
            if previous is None:
                raise
            previous.add_note(f"Final run persistence also failed: {exc}")

    destinations.callback(save)


def _record_tag_busy(working, config, exc):
    """Record early contention without invoking any processing step."""
    context = DataProdOutput(config).create_context(working)
    context.failures.append(
        StepFailure(
            message=str(exc),
            details={"code": "tag_busy", "tag": exc.tag, "path": str(exc.path)},
        )
    )


def _shared_config(values, *, uid, name):
    """Require one acquisition-wide setting across its participating inputs."""
    if any(value != values[0] for value in values[1:]):
        msg = f"Conflicting {name} settings for acquisition {uid}"
        raise ValueError(msg)
    return values[0]


def _quicklook(trees, spec):
    """Retain configured acquisition feedback for the publication step."""
    from .quicklook import select_quicklook_inputs  # noqa: PLC0415

    artifacts = ArtifactContext()
    if spec.enabled:
        artifacts.add(spec, spec(select_quicklook_inputs(trees.values())))
    return artifacts
