"""Command-line entry point for configured KIDs reduction."""

import json
import re
from pathlib import Path  # noqa: TC003 -- Typer inspects these types at runtime
from typing import Annotated, Any, cast

import typer
from pydantic_core import to_jsonable_python
from tollan.config import RuntimeContext
from tollan.utils.cli import split_cli_args
from tollan.utils.fileloc import fileloc
from tollan.utils.log import logger
from tollan.utils.typer import TollanTyper

from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.input_discovery import InputDiscovery
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec

from .core import Kids
from .dataprod_output import DataProdOutput
from .runner import _input_table, create_kids, run_inputs

app = TollanTyper(
    name=f"tolteca_{Kids.subconfig_key}",
    help="Process configured inputs, exact file locations, or observation selectors.",
    no_args_is_help=True,
    subcommand_metavar="",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "allow_interspersed_args": True,
    },
)


def _discovery_inputs(sources: list[str], data_lmt_path: Path | None) -> list[dict]:
    """Translate positional shorthand into discovery entries."""
    entries: list[dict] = []
    for source in sources:
        try:
            spec = ObsSpec.parse(source)
        except ValueError:
            if entries and "sources" in entries[-1]:
                entries[-1]["sources"].append(source)
            else:
                entries.append({"sources": [source]})
            continue
        if data_lmt_path is None:
            msg = f"An observation selector requires --data_lmt_path: {source}"
            raise typer.BadParameter(msg)
        entries.append({"data_lmt_path": str(data_lmt_path), "obs_spec": str(spec)})
    if not entries and data_lmt_path is not None:
        entries.append({"data_lmt_path": str(data_lmt_path)})
    return entries


@app.callback(invoke_without_command=True)
@RuntimeContext.add_cli_options()
def main(  # noqa: C901, PLR0913 -- CLI options and inspection branches
    ctx: typer.Context,
    sources: Annotated[
        list[str] | None,
        typer.Argument(
            help="Exact locations or ObsSpec; replaces input_discovery.inputs."
        ),
    ] = None,
    *,
    data_lmt_path: Annotated[
        Path | None,
        typer.Option(help="Acquisition tree for positional ObsSpec (default *)."),
    ] = None,
    dataprod_toltec_path: Annotated[
        Path | None,
        typer.Option(help="Default kids.output.path; does not enable output."),
    ] = None,
    select: Annotated[
        str | None,
        typer.Option(
            help="Override kids.input.select, applied after opening candidates."
        ),
    ] = None,
    show_inputs: Annotated[
        bool,
        typer.Option(help="Print the selected, header-populated input table and exit."),
    ] = False,
    show_config: Annotated[
        bool, typer.Option(help="Print effective config for each input and exit.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Prepare inputs and config, then run a no-op pipeline.")
    ] = False,
) -> None:
    """Process configured inputs or positional shorthand.

    Parameters
    ----------
    ctx : typer.Context
        CLI invocation and runtime configuration options.
    sources : list of str, optional
        Exact file locations or observation selectors, replacing configured
        discovery entries. Put positional inputs after ``--``. Locations are
        literal; configure ``{glob: ...}`` entries to expand patterns.
    data_lmt_path : pathlib.Path, optional
        Acquisition root for positional selectors. With no positional inputs,
        supplies a wildcard discovery entry only if none are configured.
    dataprod_toltec_path : pathlib.Path, optional
        Default product root. Explicit ``kids.output.path`` takes precedence;
        supplying a root does not enable output.
    select : str, optional
        Query overriding ``kids.input.select``, applied after opening candidates
        and reading their metadata. If omitted, use the configured selection.
    show_inputs : bool, optional
        Print the selected table and exit, including an empty selection.
        Default is False.
    show_config : bool, optional
        Print effective configuration for each selected input and exit.
        Default is False.
    dry_run : bool, optional
        Run the no-op pipeline without reduction or product output.
        Default is False.

    Notes
    -----
    Dotted ``--kids.*`` and ``--input_discovery.*`` overrides take precedence
    over shorthand options. Inspection flags exit before per-file execution;
    ``show_inputs`` takes precedence over ``show_config``.
    """
    keys = (Kids.subconfig_key, InputDiscovery.subconfig_key, "db", "data_prod")
    cli_args, sources = split_cli_args(
        rf"(?:{'|'.join(map(re.escape, keys))})\..+", [*(sources or []), *ctx.args]
    )
    sources = [source for source in sources if source != "--"]
    unknown = [source for source in sources if source.startswith("-")]
    if unknown:
        msg = f"Unknown options: {', '.join(unknown)}"
        raise typer.BadParameter(msg)
    shorthand = []
    if sources:
        shorthand += [
            f"--{InputDiscovery.subconfig_key}.inputs",
            json.dumps(_discovery_inputs(sources, data_lmt_path)),
        ]
    if select is not None:
        shorthand += [f"--{Kids.subconfig_key}.input.select", json.dumps(select)]
    runtime = RuntimeContext.from_cli_ctx(ctx, cli_args=[*shorthand, *cli_args])
    if dataprod_toltec_path is not None:
        runtime = runtime.with_config(
            defaults=Kids.prepare_runtime_config_data(
                {"output": {"path": str(dataprod_toltec_path)}}
            )
        )
    if not sources and data_lmt_path is not None:
        runtime = runtime.with_config(
            defaults={
                InputDiscovery.subconfig_key: {
                    "inputs": _discovery_inputs([], data_lmt_path)
                }
            }
        )
    discovery = InputDiscovery(runtime)
    if not discovery.config.inputs:
        msg = "Provide input_discovery configuration or positional inputs"
        raise typer.BadParameter(msg)
    candidates = discovery.discover(
        data_models=DataModelSet.from_select(
            selection=["toltec"],
        ),
        on_error="record",
    )
    with candidates.toltec_file.open(
        update_from_metadata=True, on_error="record"
    ) as opened:
        selected = Kids(runtime).input.select(opened)
        logger.info("Selected {} input(s)", len(selected))
        if show_inputs:
            serialized = _input_table(selected).assign(
                **{
                    name: selected[name].map(
                        lambda source: fileloc(cast("Any", source)).url
                    )
                    for name in ("source", "source_loc")
                }
            )
            typer.echo(
                serialized.to_json(
                    orient="records",
                    indent=2,
                    date_format="iso",
                    default_handler=to_jsonable_python,
                )
            )
            return
        if selected.empty:
            msg = "No inputs selected"
            raise typer.BadParameter(msg)
        if show_config:
            for row in _input_table(selected).itertuples(index=False):
                row = cast("Any", row)
                kids = create_kids(runtime, context_input=row._asdict())
                effective = {
                    InputDiscovery.subconfig_key: discovery.config.model_dump(
                        mode="json"
                    ),
                    Kids.subconfig_key: kids.config.model_dump(mode="json"),
                }
                typer.echo(
                    f"{fileloc(row.source).path}\n{json.dumps(effective, indent=2)}"
                )
            return
        _run_selected_inputs(selected, runtime, dry_run=dry_run)


def _run_selected_inputs(selected, runtime, *, dry_run):
    """Run opened inputs, report publication outcomes and set the exit status."""
    publication_failed = False
    for uid, working in run_inputs(selected, runtime, dry_run=dry_run):
        if dry_run:
            source = fileloc(working["tolteca_kids"].attrs["source"]).path
            typer.echo(f"{source}: no-op completed")
            continue
        output = (
            DataProdOutput.get_context(working)
            if DataProdOutput.has_context(working)
            else None
        )
        if output is not None and not output.completed:
            publication_failed = True
            messages = "; ".join(failure.message for failure in output.failures)
            typer.echo(f"{uid}: output failed: {messages}", err=True)
            continue
        product = None if output is None else output.data
        typer.echo(
            str(product) if product is not None else f"{uid}: completed without output"
        )

    if publication_failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
