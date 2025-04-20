"""Console script for tolteca."""

from pathlib import Path
from typing import Annotated

import typer
from tollan.utils.cli.typer import create_cli

from .. import _version
from . import check

app = create_cli(
    version=_version.__version__,
)


@app.callback()
def _main(
    ctx: typer.Context,
    config: Annotated[
        None | Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            help="Path to the config file.",
        ),
    ] = None,
):
    """TolTEC data analysis all-in-one."""
    from ..core import RuntimeContext

    ctx.obj = RuntimeContext(config)


app.add_typer(check.app, name="check")

if __name__ == "__main__":
    app()
