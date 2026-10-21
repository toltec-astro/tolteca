"""KIDs pipeline execution on open data or file sources."""

from __future__ import annotations

from contextlib import closing, contextmanager, nullcontext
from sys import exception
from typing import TYPE_CHECKING
from uuid import uuid4

import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.pipeline import ArtifactContext, StepFailure
from tollan.utils.fileloc import FileLoc, fileloc
from tollan.utils.log import logger, timeit

from tolteca_datamodels.data_prod_run import prepare_run_request
from tolteca_datamodels.data_prod_store import DataProdStore, DataProdTagBusyError
from tolteca_datamodels.toltec.file import guess_info_from_sources

from .core import Kids
from .dataprod_output import DataProdOutput
from .noop import NoOp
from .product import make_kids_product_index
from .tone_prepare import TonePrepare

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator
    from pathlib import Path
    from typing import Any

    import pandas as pd
    from tollan.config import RuntimeContext

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


def _run_blocks(data: xr.DataTree, kids: Kids) -> None:
    """Prepare acquisition blocks and run each block's configured pipeline."""
    kids.block_identify(data)
    kids.block_create(data)
    for block in data.toltec_kids.blocks:
        kids.pipeline_for(block.data_kind)(block.tree)


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
    if not TonePrepare.get_context(block).completed:
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
    table = guess_info_from_sources(
        [{"source": source, "source_loc": source_loc}], on_error="record"
    )
    metadata_source = (
        data.toltec_kids.data_source if isinstance(data, xr.DataTree) else data
    )
    with table.toltec_file.open(
        io_objs=[metadata_source], update_from_metadata=True, on_error="record"
    ) as opened:
        selected = Kids(runtime).input.select(opened)
        if selected.empty:
            msg = "No inputs selected"
            raise ValueError(msg)
        # Live handles belong to the open-table scope, not configuration or steps.
        row = next(_input_table(selected).itertuples(index=False))
        return run_input(data, row, runtime, dry_run=dry_run)


def run_input(
    data: xr.Dataset | xr.DataTree,
    row: Any,
    runtime: RuntimeContext,
    *,
    dry_run: bool = False,
    defer_output: bool = False,
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
        _run_blocks(working, kids)
    else:
        config = _with_run_request(kids.config.output, uuid4())
        registration = kids.registration
        scope = _publication(config, row, runtime)
        try:
            with _finalize_registration(
                scope,
                registration,
                DataProdStore(config.path, namespace=config.namespace).namespace_path,
            ) as pending:
                _run_blocks(working, kids)
                DataProdOutput(
                    config,
                    publication=pending,
                    registration=registration,
                    artifacts=_quicklook(
                        {working.toltec_kids.meta.interface: working},
                        kids.config.quicklook,
                    ),
                )(working)
        except DataProdTagBusyError as exc:
            _record_tag_busy(working, config, exc)
    if not dry_run and not defer_output:
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
    with table.toltec_file.open(update_from_metadata=True, on_error="record") as opened:
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
    table = guess_info_from_sources(sources, on_error="record")
    with table.toltec_file.open(update_from_metadata=True, on_error="record") as opened:
        selected = Kids(runtime).input.select(opened)
        if selected.empty:
            msg = "No inputs selected"
            raise ValueError(msg)
        yield from run_inputs(selected, runtime, dry_run=dry_run)


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
        Namespace descriptors and run/version-group indexes, in registration
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
) -> Iterator[tuple[str, xr.DataTree]]:
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
    invocation_scope = _batch_invocation(selected, runtime)
    with invocation_scope as invocation:
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
            scope = (
                _publication(config, records[0], runtime, invocation=invocation)
                if config.enabled
                else nullcontext()
            )
            try:
                with scope as pending:
                    trees = {
                        row.interface: run_input(data, row, runtime, defer_output=True)
                        for data, row in zip(rows.io_obj, records, strict=True)
                    }
                    if len(trees) == 1:
                        working = next(iter(trees.values()))
                    else:
                        working = xr.DataTree()
                        for interface, tree in trees.items():
                            working[f"interfaces/{interface}"] = tree
                    if config.enabled:
                        DataProdOutput(
                            config,
                            publication=pending,
                            registration=registration,
                            artifacts=_quicklook(trees, quicklook),
                        )(working)
                for row in records:
                    _install_tones(
                        trees[row.interface],
                        create_kids(runtime, context_input=row._asdict()),
                        publication_tree=working,
                    )
            except DataProdTagBusyError as exc:
                invocation._request(
                    f"{uid}_proc", "dp_raw_obs_proc", str(uid), config.tag
                )
                invocation._record(f"{uid}_proc", "failed", str(exc))
                working = xr.DataTree()
                _record_tag_busy(working, config, exc)
            yield str(uid), working


def _batch_invocation(selected, runtime):
    """Prepare one namespace reservation from the selected effective settings."""
    records_all: list[Any] = list(_input_table(selected).itertuples(index=False))
    handlers = [
        create_kids(runtime, context_input=row._asdict()) for row in records_all
    ]
    enabled = [kids.config.output for kids in handlers if kids.config.output.enabled]
    invocation_scope = nullcontext()
    if enabled:
        output = enabled[0]
        if any(
            (c.path, c.namespace, c.run) != (output.path, output.namespace, output.run)
            for c in enabled[1:]
        ):
            msg = "One invocation requires one output store, namespace and run request"
            raise ValueError(msg)
        store = DataProdStore(output.path, namespace=output.namespace)
        invocation_scope = store.invocation(
            run=output.run,
            runtime_info=runtime.runtime_info,
            context={
                "inputs": {
                    str(row.uid_raw_obs_file): {
                        "source": fileloc(row.source).url,
                        "config": create_kids(
                            runtime, context_input=row._asdict()
                        ).config.model_dump(mode="json", round_trip=True),
                    }
                    for row in records_all
                }
            },
        )
        registration = next(
            (
                kids.registration
                for kids in handlers
                if kids.config.output.enabled and kids.config.output.registration
            ),
            None,
        )
        invocation_scope = _finalize_registration(
            invocation_scope, registration, store.namespace_path
        )
    return invocation_scope


@contextmanager
def _finalize_registration(scope, registration, namespace_path):
    """Register the final run record after the store closes its invocation."""
    active = None
    try:
        with scope as active:
            yield active
    finally:
        if active is not None and registration is not None:
            previous = exception()
            try:
                registration(
                    [namespace_path / "namespace.yaml"], run_uids=[active.run.uid]
                )
            except Exception as exc:
                if previous is None:
                    raise
                previous.add_note(f"Final run registration also failed: {exc}")


def _with_run_request(config, uid):
    """Prepare the invocation UID and naming defaults."""
    return config.model_copy(update={"run": prepare_run_request(config.run, uid)})


def _publication(config, row, runtime, *, invocation=None):
    """Acquire the output tag and reserve a run before acquisition processing."""
    index = make_kids_product_index(
        row.master, int(row.obsnum), int(row.subobsnum), int(row.scannum)
    )
    return DataProdStore(config.path, namespace=config.namespace).publication(
        index,
        primary_uid=row.uid_raw_obs,
        run=config.run if invocation is None else None,
        invocation=invocation,
        tag=config.tag,
        runtime_info=runtime.runtime_info,
    )


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
    artifacts = ArtifactContext()
    if spec.enabled:
        artifacts.add(spec, spec(trees))
    return artifacts
