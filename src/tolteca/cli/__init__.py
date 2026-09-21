"""Console script for tolteca."""

from rich.console import Console
from tollan.utils.typer import create_cli

app = create_cli(help="TolTECA data processing tools.")
console = Console()


if __name__ == "__main__":
    app()
