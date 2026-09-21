"""Command-line entry point for configured KIDs reduction."""

import json
import re
from pathlib import Path  # noqa: TC003 -- Typer inspects these types at runtime
from typing import Annotated

import typer
from tollan.config import RuntimeContext
from tollan.utils.cli import split_cli_args
from tollan.utils.log import logger
from tollan.utils.typer import TollanTyper

from tolteca_datamodels.filestore import FileAccess

from .core import Kids
from .runner import build_input_catalog, run_source

app = TollanTyper(
    name=f"tolteca_{Kids.subconfig_key}",
    help="Process TolTEC KIDs files or observation selectors.",
    no_args_is_help=True,
    subcommand_metavar="",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "allow_interspersed_args": True,
    },
)


@app.callback(invoke_without_command=True)
@RuntimeContext.add_cli_options()
def main(  # noqa: PLR0913 -- CLI options form the public command signature
    ctx: typer.Context,
    sources: Annotated[
        list[str] | None,
        typer.Argument(help="Files, quoted globs, or observation selectors."),
    ] = None,
    *,
    data_lmt_path: Annotated[
        Path | None, typer.Option(help="Raw data tree used for observation selectors.")
    ] = None,
    dataprod_toltec_path: Annotated[
        Path | None, typer.Option(help="Data product root; does not enable output.")
    ] = None,
    select: Annotated[
        str | None, typer.Option(help="Override files.select, a pandas catalog query.")
    ] = None,
    show_inputs: Annotated[
        bool, typer.Option(help="Print the selected metadata catalog and exit.")
    ] = False,
    show_config: Annotated[
        bool, typer.Option(help="Print effective config for each input and exit.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Resolve inputs and config, then run a no-op pipeline.")
    ] = False,
) -> None:
    """Process inputs with dotted --kids.* overrides before -- and inputs after it."""
    key = Kids.subconfig_key
    cli_args, sources = split_cli_args(
        rf"{re.escape(key)}\..+", [*(sources or []), *ctx.args]
    )
    sources = [source for source in sources if source != "--"]
    if not sources:
        msg = "Provide at least one input file or observation selector"
        raise typer.BadParameter(msg)
    unknown = [source for source in sources if source.startswith("-")]
    if unknown:
        msg = f"Unknown options: {', '.join(unknown)}"
        raise typer.BadParameter(msg)

    # Only these explicit options populate shared discovery settings. User
    # dotted options were already restricted to the KIDs namespace above.
    options = {
        "data_lmt_path": data_lmt_path,
        "dataprod_toltec_path": dataprod_toltec_path,
        "select": select,
    }
    cli_args = [
        arg
        for name, value in options.items()
        if value is not None
        for arg in (f"--{FileAccess.subconfig_key}.{name}", str(value))
    ] + cli_args
    runtime = RuntimeContext.from_cli_ctx(ctx, cli_args=cli_args)
    inputs = FileAccess(runtime).config
    if inputs.dataprod_toltec_path is not None:
        runtime.update_default_config(
            {key: {"output": {"path": inputs.dataprod_toltec_path}}}
        )
    logger.debug("Input selection settings: {}", inputs)
    catalog = build_input_catalog(sources, data_root=inputs.data_lmt_path)
    if inputs.select is not None:
        n_inputs = len(catalog)
        catalog = catalog.toltec_file.query(inputs.select)
        logger.debug(
            "Catalog selection {!r}: {} -> {} inputs",
            inputs.select,
            n_inputs,
            len(catalog),
        )
    logger.info("Selected {} input(s)", len(catalog))
    if show_inputs:
        typer.echo(catalog.to_json(orient="records", indent=2, date_format="iso"))
        return
    if catalog.empty:
        msg = "No inputs selected"
        raise typer.BadParameter(msg)
    for info in catalog.toltec_file.to_info_list():
        if show_config:
            kids = Kids(
                runtime.with_config(
                    validation_context=info.model_dump(
                        mode="json",
                        context={"flat": True},
                        exclude={"filepath_info", "filepath_info_orig"},
                    )
                )
            )
            effective = {
                FileAccess.subconfig_key: inputs.model_dump(mode="json"),
                key: kids.config.model_dump(mode="json"),
            }
            typer.echo(f"{info.filepath}\n{json.dumps(effective, indent=2)}")
        else:
            product = run_source(info, runtime, dry_run=dry_run)
            status = "no-op completed" if dry_run else "completed without output"
            typer.echo(
                str(product) if product is not None else f"{info.filepath}: {status}"
            )


if __name__ == "__main__":
    app()
