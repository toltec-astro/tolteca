import typer
from tollan.utils.fmt import pformat_yaml

app = typer.Typer(help="Check health.", no_args_is_help=True)


@app.command()
def config(ctx: typer.Context):
    """Print config."""
    rc = ctx.obj
    typer.echo(pformat_yaml(rc.config.model_dump()))
