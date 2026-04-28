"""Console script for tolteca."""

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for tolteca."""
    console.print("This is tolteca CLI.")


if __name__ == "__main__":
    app()
