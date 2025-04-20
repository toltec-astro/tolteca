from typer.testing import CliRunner

import tolteca_config
from tolteca_config.cli import app

runner = CliRunner()


def test_main():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert tolteca_config.__version__ in result.stdout

    result = runner.invoke(app, ["-l", "DEBUG", "version"])
    assert result.exit_code == 0


def test_config():
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 0
    assert "Check health" in result.stdout

    result = runner.invoke(app, ["check", "config"])
    assert result.exit_code == 0
    assert f"version: {tolteca_config.__version__}" in result.stdout
