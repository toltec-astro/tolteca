"""Direct and Deploy-launched KIDs input/configuration equivalence."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tolteca_kids.cli import app


@pytest.mark.parametrize("mode", ["--show_inputs", "--show_config", "--dry_run"])
def test_reference_inputs_direct_and_deploy(data_lmt_path, tmp_path, mode):
    pytest.importorskip("tolteca_deploy")
    kids = Path(sys.executable).with_name("tolteca_kids")
    deploy = Path(sys.executable).with_name("tolteca_deploy")
    if not kids.exists() or not deploy.exists():
        pytest.skip("Install the editable Kids and Deploy entrypoints")
    root = data_lmt_path.resolve()
    sources = sorted((root / "toltec/tcs/toltec0").glob("*.nc"))
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in sources}
    workdir = tmp_path / "working directory"
    workdir.mkdir()
    config = workdir / "recipe.yaml"
    config.write_text("""
- source:
    files:
      select: roach == 0
    kids:
      sweep_reduce: {block: 1}
- enable_if: file_suffix == 'tune' & atten_drive == 7.25
  source:
    kids:
      sweep_reduce: {block: 2}
""")
    products = workdir / "products"
    # Kids consumes explicit paths, not inherited deployment variables.
    environment = {
        **os.environ,
        "TOLTECA_DATA_LMT_PATH": "/missing/inherited",
        "TOLTECA_DATAPROD_TOLTEC_PATH": "/missing/inherited-products",
        "NO_COLOR": "1",
    }
    args = [
        "--config_paths",
        "recipe.yaml",
        mode,
        "--data_lmt_path",
        str(root),
        "--dataprod_toltec_path",
        str(products),
        "--kids.sweep_reduce.force",
        "true",
        "--",
        "tcs-{145605,145606,145609}/0",
    ]
    direct = subprocess.run(  # noqa: S603 -- installed entrypoints and test-owned argv
        [str(kids), *args],
        cwd=workdir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    launched = subprocess.run(  # noqa: S603 -- installed entrypoints and test-owned argv
        [str(deploy), "run", "--", str(kids), *args],
        cwd=workdir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert direct.returncode == 0, direct.stderr
    assert launched.returncode == direct.returncode, launched.stderr
    assert launched.stdout == direct.stdout
    if mode == "--show_inputs":
        assert {row["data_kind"] for row in json.loads(direct.stdout)} == {
            "VnaSweep",
            "TargetSweep",
            "Tune",
            "RawTimeStream",
        }
    elif mode == "--show_config":
        assert direct.stdout.count('"force": true') == 4
        assert direct.stdout.count('"block": 2') == 1
        assert direct.stdout.count('"block": 1') == 3
        assert str(products) in direct.stdout
    else:
        assert direct.stdout.count("no-op completed") == 4
    assert not products.exists()
    assert before == {
        path: (path.stat().st_size, path.stat().st_mtime_ns) for path in sources
    }


@pytest.mark.parametrize("failure", ["selection", "config", "option"])
def test_deploy_preserves_kids_failure(timestream_path, failure):
    pytest.importorskip("tolteca_deploy")
    kids = Path(sys.executable).with_name("tolteca_kids")
    deploy = Path(sys.executable).with_name("tolteca_deploy")
    if not kids.exists() or not deploy.exists():
        pytest.skip("Install the editable Kids and Deploy entrypoints")
    source = timestream_path
    args = {
        "selection": ["--select", "roach == 7"],
        "config": ["--kids.sweep_reduce.block", "invalid"],
        "option": ["--dry_rnu"],
    }[failure]
    command = [str(kids), "--dry_run", *args, "--", str(source)]
    direct = subprocess.run(  # noqa: S603 -- installed entrypoints and test-owned argv
        command, capture_output=True, text=True, check=False, timeout=60
    )
    launched = subprocess.run(  # noqa: S603 -- installed entrypoints and test-owned argv
        [str(deploy), "run", "--", *command],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert direct.returncode != 0
    assert launched.returncode == direct.returncode
    assert "no-op completed" not in direct.stdout + launched.stdout
    expected = {
        "selection": "No inputs selected",
        "config": "Input should be a valid integer",
        "option": "Unknown options",
    }[failure]
    assert expected in direct.stdout + direct.stderr
    assert expected in launched.stdout + launched.stderr


@pytest.mark.parametrize("tier", ["yaml", "option"])
def test_path_and_selection_precedence(tmp_path, timestream_path, monkeypatch, tier):
    source = timestream_path
    paths = {}
    for name in ("yaml", "option"):
        paths[name] = tmp_path / name
        paths[name].mkdir()
    monkeypatch.setenv("TOLTECA_DATA_LMT_PATH", "/missing/inherited")
    monkeypatch.delenv("TOLTECA_DATAPROD_TOLTEC_PATH", raising=False)
    config = tmp_path / "settings.yaml"
    config.write_text(
        f"files:\n  data_lmt_path: {paths['yaml']}\n  select: roach == 0\n"
    )
    args = ["--config_paths", str(config)]
    if tier == "option":
        args += ["--data_lmt_path", str(paths["option"]), "--select", "roach < 7"]
    result = CliRunner().invoke(app, [*args, "--show_config", "--", str(source)])
    assert result.exit_code == 0, result.output
    config = json.loads(result.stdout.split("\n", 1)[1])
    assert config["files"]["data_lmt_path"] == str(paths[tier])
    assert config["files"]["select"] == (
        "roach == 0" if tier == "yaml" else "roach < 7"
    )
    assert "data_lmt_path" not in config["kids"]
    assert os.environ["TOLTECA_DATA_LMT_PATH"] == "/missing/inherited"


def test_environment_does_not_supply_kids_paths(tmp_path, timestream_path, monkeypatch):
    source = timestream_path
    monkeypatch.setenv("TOLTECA_DATA_LMT_PATH", str(tmp_path))
    monkeypatch.setenv("TOLTECA_DATAPROD_TOLTEC_PATH", str(tmp_path / "products"))
    env = tmp_path / "settings.env"
    env.write_text(f"TOLTECA_DATA_LMT_PATH={tmp_path}\n")
    result = CliRunner().invoke(
        app, ["--env_paths", str(env), "--show_config", "--", str(source)]
    )
    assert result.exit_code == 0, result.output
    config = json.loads(result.stdout.split("\n", 1)[1])
    assert config["files"]["data_lmt_path"] is None
    assert config["files"]["dataprod_toltec_path"] is None


@pytest.mark.parametrize("output_path", ["shared", "recipe", "cli"])
def test_product_root_defaults_do_not_override_output_recipe(
    tmp_path, timestream_path, output_path
):
    source = timestream_path
    products = tmp_path / "shared"
    config = tmp_path / "settings.yaml"
    config.write_text(
        "kids:\n  output:\n    path: " + str(tmp_path / "recipe") + "\n"
        if output_path != "shared"
        else "{}\n"
    )
    args = [
        "--config_paths",
        str(config),
        "--dataprod_toltec_path",
        str(products),
        "--show_config",
    ]
    if output_path == "cli":
        args += ["--kids.output.path", str(tmp_path / "cli")]
    result = CliRunner().invoke(app, [*args, "--", str(source)])
    assert result.exit_code == 0, result.output
    effective = json.loads(result.stdout.split("\n", 1)[1])
    assert effective["files"]["dataprod_toltec_path"] == str(products)
    assert effective["kids"]["output"]["path"] == str(tmp_path / output_path)
    assert effective["kids"]["output"]["enabled"] is False
    for name in ("shared", "recipe", "cli"):
        assert not (tmp_path / name).exists()


def test_symlink_input_is_processed_once(tmp_path, timestream_path):
    source = timestream_path
    alias = tmp_path / "alias.nc"
    alias.symlink_to(source)
    result = CliRunner().invoke(app, ["--dry_run", "--", str(source), str(alias)])
    assert result.exit_code == 0, result.output
    assert result.stdout.count("no-op completed") == 1


@pytest.mark.parametrize(
    "option", ["--kids.", "--web.debug", "--kidss.debug", "--files.select"]
)
def test_only_nonempty_kids_overrides_reach_runtime(monkeypatch, option):
    from tolteca_kids import cli

    def unexpected(*args, **kwargs):
        pytest.fail("An invalid override reached runtime loading")

    monkeypatch.setattr(cli.RuntimeContext, "from_cli_ctx", unexpected)
    result = CliRunner().invoke(app, ["--dry_run", option, "true", "--", "input.nc"])
    assert result.exit_code == 2
    assert "Unknown options" in result.output
