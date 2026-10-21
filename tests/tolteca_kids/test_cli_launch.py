"""Direct and Deploy-launched KIDs input/configuration equivalence."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tolteca_datamodels.input_discovery import InputDiscovery
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kids.cli import app


@pytest.mark.parametrize(
    ("selection", "enriched"),
    [("[]", False), ('["toltec"]', True), ('["dummy_instru"]', False)],
)
def test_cli_discovery_data_model_selection(
    data_lmt_path, selection, enriched, monkeypatch
):
    discovery_tables = []
    discover = InputDiscovery.discover

    def trace_discovery(self, **kwargs):
        table = discover(self, **kwargs)
        discovery_tables.append(table.copy())
        return table

    monkeypatch.setattr(InputDiscovery, "discover", trace_discovery)
    result = CliRunner().invoke(
        app,
        [
            "--data_lmt_path",
            str(data_lmt_path),
            "--show_inputs",
            "--input_discovery.data_model_select",
            selection,
            "--",
            "145609/tel",
        ],
    )
    assert result.exit_code == 0, result.exception
    (discovered,) = discovery_tables
    assert len(discovered) == (2 if enriched else 0)
    assert ("data_kind_toltec" in discovered) is enriched
    # Pipeline preparation reads headers for the discovered candidates.
    rows = json.loads(result.stdout)
    assert len(rows) == len(discovered)
    assert all(row["data_kind_toltec"] == ToltecDataKind.lmt_tel.value for row in rows)


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
@pytest.mark.parametrize("select_good", [False, True])
@pytest.mark.parametrize("raise_on_errors", [False, True])
def test_cli_selects_before_error_policy(
    data_lmt_path, tmp_path, mode, select_good, raise_on_errors
):
    good = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    missing = tmp_path / "missing.nc"
    recipe = tmp_path / "inputs.yaml"
    recipe.write_text(
        json.dumps(
            {
                "input_discovery": {
                    "inputs": [
                        {
                            "sources": [str(missing), str(good)],
                            "mode": mode,
                        }
                    ]
                },
                "kids": {
                    "input": {
                        "select": "obsnum == 145609" if select_good else None,
                        "raise_on_errors": raise_on_errors,
                    }
                },
            }
        )
    )
    result = CliRunner().invoke(app, ["--config_paths", str(recipe), "--show_inputs"])
    if raise_on_errors and not select_good:
        assert result.exit_code != 0
        assert "unusable selected input" in str(result.exception)
        assert "missing.nc" in str(result.exception)
    else:
        assert result.exit_code == 0, result.exception
        rows = json.loads(result.stdout)
        assert [row["obsnum"] for row in rows] == [145609]


@pytest.mark.parametrize("action", ["--show_inputs", "--dry_run"])
@pytest.mark.parametrize("invalid_date", [False, True])
def test_cli_recovers_filename_diagnostics_and_dispatches(
    data_lmt_path, tmp_path, action, invalid_date
):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    alias = tmp_path / (
        path.name.replace("2025-10-31", "2025-99-31")
        if invalid_date
        else "downloaded.nc"
    )
    alias.symlink_to(path)
    result = CliRunner().invoke(
        app,
        [
            action,
            "--kids.input.raise_on_errors",
            "true",
            "--",
            str(alias),
        ],
    )
    assert result.exit_code == 0, result.exception
    if action == "--dry_run":
        assert result.stdout.count("no-op completed") == 1
    else:
        (row,) = json.loads(result.stdout)
        (diagnostic,) = [
            event for event in row["source_trace"] if event["stage"] == "filepath"
        ]
        assert diagnostic["stage"] == "filepath"
        if invalid_date:
            assert diagnostic["exception_type"] == "ValueError"
            assert "does not match" in diagnostic["message"]
        else:
            assert diagnostic["exception_type"] is None
            assert diagnostic["message"] == "No accepted filename pattern matched"


@pytest.mark.parametrize(
    ("selector", "query", "count"),
    [
        ("145609", "obsnum >= 0", 4),
        ("145609/*", "interface == 'tel'", 2),
        # Five telescope files, including 145608, and four TolTEC files.
        ("tcs-(145605,145610)/{tel,0}", "obsnum >= 0", 9),
        ("tcs-145609/tel", "obs_goal == 'pointing'", 2),
        ("tcs-(145605,145610)/{tel,0}", "bits_any_set(data_kind_toltec, raw_sweep)", 3),
        ("tcs-(145605,145610)[-1]/tel", "obsnum >= 0", 2),
    ],
)
def test_mixed_reference_noop(data_lmt_path, selector, query, count):
    result = CliRunner().invoke(
        app,
        [
            "--data_lmt_path",
            str(data_lmt_path),
            "--select",
            query,
            "--dry_run",
            "--",
            selector,
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.count("no-op completed") == count


@pytest.mark.parametrize("action", ["--show_inputs", "--dry_run"])
def test_toltec_subtree_and_telescope_source_recipe(data_lmt_path, tmp_path, action):
    config = tmp_path / "subtree.yaml"
    config.write_text(
        json.dumps(
            {
                "input_discovery": {
                    "inputs": [
                        {
                            "data_toltec_path": str(data_lmt_path / "toltec"),
                            "obs_spec": "145609/{tel,0}",
                        },
                        {
                            "sources": [
                                str(
                                    data_lmt_path
                                    / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
                                )
                            ]
                        },
                    ]
                },
                "kids": {"input": {"select": "master_code == 0"}},
            }
        )
    )
    result = CliRunner().invoke(app, ["--config_paths", str(config), action])
    assert result.exit_code == 0, result.output
    if action == "--dry_run":
        assert result.stdout.count("no-op completed") == 3
    else:
        rows = json.loads(result.stdout)
        assert len(rows) == 3
        assert [row["interface"] for row in rows] == ["toltec0", "toltec0", "tel"]
        assert all(row["filepath_orig"] for row in rows)
        assert rows[-1]["obs_goal"] == "pointing"


@pytest.mark.parametrize("action", ["--show_inputs", "--dry_run"])
@pytest.mark.parametrize(
    ("selection", "expected_interfaces"),
    [
        ("toltec", {"tel", "toltec0"}),
        ({"excludes": ["toltec"]}, set()),
        (["lmt", "toltec"], {"tel", "toltec0"}),
    ],
)
def test_data_model_selection_recipe(
    data_lmt_path, tmp_path, action, selection, expected_interfaces
):
    config = tmp_path / "instruments.yaml"
    config.write_text(
        json.dumps(
            {
                "input_discovery": {
                    "inputs": [
                        {
                            "data_lmt_path": str(data_lmt_path),
                            "data_model_select": selection,
                            "obs_spec": "145609/{tel,0}",
                        },
                    ],
                },
                "kids": {"input": {"select": "obsnum == 145609"}},
            }
        )
    )
    result = CliRunner().invoke(app, ["--config_paths", str(config), action])
    expected_count = 2 * len(expected_interfaces)
    if action == "--dry_run" and not expected_count:
        assert result.exit_code == 2
        assert "No inputs selected" in result.output
        return
    assert result.exit_code == 0, result.output
    if action == "--dry_run":
        assert result.stdout.count("no-op completed") == expected_count
    else:
        rows = json.loads(result.stdout)
        assert len(rows) == expected_count
        assert {row["interface"] for row in rows} == expected_interfaces


def test_data_model_selection_cli_override(data_lmt_path, tmp_path):
    config = tmp_path / "inputs.yaml"
    config.write_text(
        json.dumps(
            {
                "input_discovery": {
                    "inputs": [
                        {
                            "data_lmt_path": str(data_lmt_path),
                            "data_model_select": [],
                            "obs_spec": "145609/{tel,0}",
                        }
                    ]
                }
            }
        )
    )
    result = CliRunner().invoke(
        app,
        [
            "--config_paths",
            str(config),
            "--input_discovery.inputs.0.data_model_select",
            "toltec",
            "--show_inputs",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert len(rows) == 4
    assert {row["interface"] for row in rows} == {"tel", "toltec0"}


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
    kids:
      input:
        select: roach == 0
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
        rows = json.loads(direct.stdout)
        assert all("data_kind_str" not in row for row in rows)
        assert {row["data_kind_toltec"] for row in rows} == {
            ToltecDataKind.vnasweep.value,
            ToltecDataKind.targsweep.value,
            ToltecDataKind.tune.value,
            ToltecDataKind.raw_timestream.value,
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


def test_positional_obs_spec_direct_and_deploy(data_lmt_path, tmp_path):
    pytest.importorskip("tolteca_deploy")
    kids = Path(sys.executable).with_name("tolteca_kids")
    deploy = Path(sys.executable).with_name("tolteca_deploy")
    if not kids.exists() or not deploy.exists():
        pytest.skip("Install the editable Kids and Deploy entrypoints")
    args = [
        str(kids),
        "--data_lmt_path",
        str(data_lmt_path),
        "--show_inputs",
        "--",
        "tcs-(145605,145610)[-1]/(0,1)",
    ]
    results = [
        subprocess.run(  # noqa: S603 -- installed entrypoints and test-owned argv
            command,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        for command in (args, [str(deploy), "run", "--", *args])
    ]
    for result in results:
        assert result.returncode == 0, result.stderr
    assert results[0].stdout == results[1].stdout
    rows = json.loads(results[0].stdout)
    assert [row["obsnum"] for row in rows] == [145609, 145609]
    assert [row["scannum"] for row in rows] == [1, 2]
    assert [row["roach"] for row in rows] == [0, 0]


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


@pytest.mark.parametrize("tier", ["yaml", "option", "dotted"])
def test_path_and_selection_precedence(tmp_path, timestream_path, monkeypatch, tier):
    monkeypatch.setenv("TOLTECA_DATA_LMT_PATH", "/missing/inherited")
    config = tmp_path / "settings.yaml"
    config.write_text(
        "input_discovery:\n  inputs:\n    - sources: [" + str(timestream_path) + "]\n"
        "kids:\n  input:\n    select: roach == 0\n"
    )
    args = ["--config_paths", str(config), "--show_config"]
    if tier != "yaml":
        args += ["--select", "roach < 7"]
    if tier == "dotted":
        args += ["--kids.input.select", "roach <= 6"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.exception
    effective = json.loads(result.stdout.split("\n", 1)[1])
    assert (
        effective["kids"]["input"]["select"]
        == ({"yaml": "roach == 0", "option": "roach < 7", "dotted": "roach <= 6"}[tier])
    )
    assert "input" not in effective
    assert "data_lmt_path" not in effective["kids"]
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
    assert "files" not in config
    assert config["kids"]["output"]["path"] == str(Path.cwd())
    assert config["kids"]["output"]["enabled"] is False


@pytest.mark.parametrize("configured", [False, True])
def test_discovery_dotted_override(tmp_path, timestream_path, configured):
    config = tmp_path / "settings.yaml"
    config.write_text(
        "input_discovery:\n  inputs:\n    - sources: ["
        + str(timestream_path)
        + "]\n      mode: guess\n"
    )
    args = ["--config_paths", str(config)] if configured else []
    args += ["--input_discovery.inputs.[0].mode", "read", "--show_config"]
    if not configured:
        args += ["--", str(timestream_path)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.exception
    effective = json.loads(result.stdout.split("\n", 1)[1])
    assert effective["input_discovery"]["inputs"][0]["mode"] == "read"


@pytest.mark.parametrize("output_path", ["launcher", "recipe", "cli"])
def test_product_root_defaults_do_not_override_output_recipe(
    tmp_path, timestream_path, output_path
):
    source = timestream_path
    products = tmp_path / "launcher"
    config = tmp_path / "settings.yaml"
    config.write_text(
        "kids:\n  output:\n    path: " + str(tmp_path / "recipe") + "\n"
        if output_path != "launcher"
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
    assert "files" not in effective
    assert effective["kids"]["output"]["path"] == str(tmp_path / output_path)
    assert effective["kids"]["output"]["enabled"] is False
    for name in ("launcher", "recipe", "cli"):
        assert not (tmp_path / name).exists()


def test_symlink_input_is_processed_once(tmp_path, timestream_path):
    source = timestream_path
    alias = tmp_path / "alias.nc"
    alias.symlink_to(source)
    result = CliRunner().invoke(app, ["--dry_run", "--", str(source), str(alias)])
    assert result.exit_code == 0, result.output
    assert result.stdout.count("no-op completed") == 1


@pytest.mark.parametrize(
    "option",
    ["--kids.", "--web.debug", "--kidss.debug", "--files.select", "--input.select"],
)
def test_only_nonempty_kids_overrides_reach_runtime(monkeypatch, option):
    from tolteca_kids import cli

    def unexpected(*args, **kwargs):
        pytest.fail("An invalid override reached runtime loading")

    monkeypatch.setattr(cli.RuntimeContext, "from_cli_ctx", unexpected)
    result = CliRunner().invoke(app, ["--dry_run", option, "true", "--", "input.nc"])
    assert result.exit_code == 2
    assert "Unknown options" in result.output


def test_positional_inputs_replace_configured_discovery(tmp_path, timestream_path):
    config = tmp_path / "settings.yaml"
    config.write_text("input_discovery:\n  inputs:\n    - sources: [missing.nc]\n")
    result = CliRunner().invoke(
        app,
        [
            "--config_paths",
            str(config),
            "--dry_run",
            "--",
            str(timestream_path),
        ],
    )
    assert result.exit_code == 0, result.exception
    assert result.stdout.count("no-op completed") == 1


@pytest.mark.parametrize("action", ["--show_inputs", "--dry_run"])
def test_configured_glob_direct_and_deploy(tmp_path, target_sweep_path, action):
    pytest.importorskip("tolteca_deploy")
    kids = Path(sys.executable).with_name("tolteca_kids")
    deploy = Path(sys.executable).with_name("tolteca_deploy")
    if not kids.exists() or not deploy.exists():
        pytest.skip("Install the editable Kids and Deploy entrypoints")
    (tmp_path / "downloaded.nc").symlink_to(target_sweep_path)
    config = tmp_path / "recipe.yaml"
    config.write_text(
        'input_discovery:\n  inputs:\n    - sources: [{glob: "*.nc"}]\n'
        "kids:\n  input:\n    select: bits_any_set(data_kind_toltec, raw_sweep)\n"
    )
    command = [str(kids), "--config_paths", str(config), action]
    results = [
        subprocess.run(  # noqa: S603 -- installed tools, test-owned argv
            args,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        for args in (command, [str(deploy), "run", "--", *command])
    ]
    for result in results:
        assert result.returncode == 0, result.stderr
    assert results[0].stdout == results[1].stdout


@pytest.mark.parametrize("option", ["--table_mode", "--no_glob_prefilter"])
def test_obsolete_discovery_options_are_rejected(option, timestream_path):
    result = CliRunner().invoke(app, [option, "--dry_run", "--", str(timestream_path)])
    assert result.exit_code == 2
    assert "Unknown options" in result.output


@pytest.mark.parametrize("action", ["--show_inputs", "--dry_run"])
def test_empty_glob_discovery_reaches_cli_selection(tmp_path, action):
    config = tmp_path / "recipe.yaml"
    config.write_text(
        "input_discovery:\n  inputs:\n    - sources: [{glob: '"
        + str(tmp_path / "missing*.nc")
        + "'}]\n"
    )
    result = CliRunner().invoke(app, ["--config_paths", str(config), action])
    if action == "--show_inputs":
        assert result.exit_code == 0, result.exception
        assert json.loads(result.stdout) == []
    else:
        assert result.exit_code == 2, result.exception
        assert "No inputs selected" in result.output


def test_cli_preparation_reuses_handle_during_dispatch(timestream_path, monkeypatch):
    import xarray as xr
    from xarray.backends.netCDF4_ import NetCDF4DataStore

    opened = []
    closed = []
    original_open = xr.open_dataset
    original_close = NetCDF4DataStore.close

    def open_data(path, **kwargs):
        opened.append(path)
        return original_open(path, **kwargs)

    def close(store, **kwargs):
        original_close(store, **kwargs)
        closed.append(store)

    monkeypatch.setattr(xr, "open_dataset", open_data)
    monkeypatch.setattr(NetCDF4DataStore, "close", close)
    result = CliRunner().invoke(app, ["--dry_run", "--", str(timestream_path)])
    assert result.exit_code == 0, result.exception
    assert opened == [timestream_path]
    assert len(closed) == 1


def test_launcher_root_does_not_replace_configured_inputs(tmp_path, timestream_path):
    config = tmp_path / "settings.yaml"
    config.write_text(
        "input_discovery:\n  inputs:\n    - sources: [" + str(timestream_path) + "]\n"
    )
    result = CliRunner().invoke(
        app,
        [
            "--data_lmt_path",
            str(tmp_path / "unused-root"),
            "--config_paths",
            str(config),
            "--dry_run",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert result.stdout.count("no-op completed") == 1
