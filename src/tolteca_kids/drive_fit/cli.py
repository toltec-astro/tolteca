"""Runtime-backed command-line entry points for drive fitting and commit."""

from typing import Annotated

import typer
from tollan.config import RuntimeContext
from tollan.pipeline import get_pipeline_contexts
from tollan.utils.cli import split_cli_args
from tollan.utils.typer import TollanTyper

from .core import DriveFit
from .output import DriveFitOutputStep
from .runner import run


def _make_app(*, key, handler, runner, help_text):
    app = TollanTyper(
        name=f"tolteca_{key}",
        help=help_text,
        no_args_is_help=True,
        context_settings={
            "allow_extra_args": True,
            "ignore_unknown_options": True,
            "allow_interspersed_args": True,
        },
    )

    @app.callback(invoke_without_command=True)
    @RuntimeContext.add_cli_options()
    def main(
        ctx: typer.Context,
        sources: Annotated[
            list[str] | None,
            typer.Argument(
                help="Exact input locations; replaces input_discovery.inputs."
            ),
        ] = None,
        *,
        show_config: Annotated[
            bool, typer.Option(help="Show workflow settings without reading inputs.")
        ] = False,
    ) -> None:
        """Run the configured workflow using exact locations or input discovery."""
        cli_args, sources = split_cli_args(
            rf"(?:{key}|input_discovery)\..+",
            [*(sources or []), *ctx.args],
        )
        sources = [source for source in sources if source != "--"]
        if any(source.startswith("-") for source in sources):
            msg = "Unknown workflow command-line option"
            raise typer.BadParameter(msg)
        runtime = RuntimeContext.from_cli_ctx(ctx, cli_args=cli_args)
        if sources:
            runtime = runtime.with_config(
                overrides={
                    "input_discovery": {"inputs": [{"sources": sources}]},
                }
            )
        if show_config:
            typer.echo(handler(runtime).config.model_dump_json(indent=2))
            return
        for working in runner(runtime):
            if DriveFitOutputStep.has_context(working):
                typer.echo(str(DriveFitOutputStep.get_context(working).data))
            for interface, tree in working["interfaces"].children.items():
                for context in get_pipeline_contexts(tree).values():
                    if context.completed and hasattr(context.data, "n_success"):
                        result = context.data
                        n_channels = result.data.sizes["chan"]
                        typer.echo(
                            f"{interface}: {result.n_success}/{n_channels} "
                            "accepted channels"
                        )

    return app


app = _make_app(
    key="drive_fit",
    handler=DriveFit,
    runner=run,
    help_text="Fit TolTEC attenuation sequences and publish their scientific products.",
)
