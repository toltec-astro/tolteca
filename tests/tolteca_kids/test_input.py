"""Pipeline-owned selection of shared, header-populated inputs."""

import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError
from tollan.config import RuntimeContext
from tollan.utils.log import logger

from tolteca_datamodels.core.file import source_trace_issues
from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.input_discovery import InputDiscovery
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_kids.config import InputConfig
from tolteca_kids.core import Kids
from tolteca_kids.input import Input
from tolteca_kids.runner import run_data, run_source

_DATA_MODELS = DataModelSet.from_select(selection=["toltec"])


@pytest.mark.parametrize("reverse", [False, True])
def test_input_step_context_and_index_query(data_lmt_path, reverse):
    runtime = RuntimeContext(
        {
            "input_discovery": {
                "inputs": [{"data_lmt_path": data_lmt_path, "obs_spec": "145609/tel"}]
            },
            "kids": {"input": {"select": 'index == "last"'}},
        }
    )
    with (
        InputDiscovery(runtime)
        .discover(data_models=_DATA_MODELS)
        .toltec_file.open() as collection
    ):
        opened = collection.iloc[::-1] if reverse else collection
        opened.index = ["first", "last"]
        expected_scan = opened.scannum.iloc[1]
        invocation = opened.copy(deep=False)
        result, context = Kids(runtime).input(invocation, return_context=True)
        assert result is invocation
        assert Input.get_context(invocation) is context
        assert context.completed
        assert context.data is not None
        assert context.data.scannum.tolist() == [expected_scan]
        assert list(context.data.index) == ["last"]
        assert context.data.io_obj.iloc[0] is opened.io_obj.iloc[1]
        assert not context.data.attrs and not opened.attrs
        selected = Kids(runtime).input.select(opened)
        assert selected.scannum.tolist() == [expected_scan]


def test_selection_uses_headers_not_discovery_guesses(data_lmt_path, tmp_path):
    original = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    alias = tmp_path / "tel_toltec_2025-10-31_145609_00_0001.nc"
    alias.symlink_to(original)
    runtime = RuntimeContext(
        {
            "input_discovery": {"inputs": [{"sources": [alias], "mode": "guess"}]},
            "kids": {
                "input": {
                    "select": (
                        'contains(data_models, "toltec") & scannum == 2 '
                        '& obs_goal == "pointing"'
                    )
                }
            },
        }
    )
    table = InputDiscovery(runtime).discover(data_models=_DATA_MODELS)
    assert table.scannum.tolist() == [1]
    with table.toltec_file.open() as opened:
        opened.toltec_file.update_from_source_metadata(inplace=True)
        selected = Kids(runtime).input.select(opened)
        assert selected.scannum.tolist() == [2]
        assert selected.file_scannum.tolist() == [1]
        assert isinstance(selected.io_obj.iloc[0], xr.Dataset)
        assert not opened.attrs
    assert table.scannum.tolist() == [1]
    assert "io_obj" not in table


def test_two_pipelines_select_independently_from_same_open_collection(
    data_lmt_path, monkeypatch
):
    runtime = RuntimeContext(
        {
            "input_discovery": {
                "inputs": [{"data_lmt_path": data_lmt_path, "obs_spec": "145609/tel"}]
            }
        }
    )
    table = InputDiscovery(runtime).discover(data_models=_DATA_MODELS)
    first = Kids(
        runtime.with_config(
            overrides={"kids": {"input": {"select": "scannum == scannum.min()"}}}
        )
    )
    last = Kids(
        runtime.with_config(
            overrides={"kids": {"input": {"select": "scannum == scannum.max()"}}}
        )
    )
    with table.toltec_file.open() as opened:
        opened.toltec_file.update_from_source_metadata(inplace=True)
        opened.index = ["duplicate", "duplicate"]
        handles = dict(zip(opened.scannum, opened.io_obj, strict=True))

        def unexpected(*args, **kwargs):
            pytest.fail("Pipeline selection opened, closed or read files")

        with monkeypatch.context() as patch:
            patch.setattr(xr, "open_dataset", unexpected)
            patch.setattr(xr.Dataset, "close", unexpected)
            patch.setattr(
                type(opened.toltec_file), "update_from_source_metadata", unexpected
            )
            a = first.input.select(opened)
            b = last.input.select(opened)
        assert a.scannum.tolist() == [1]
        assert b.scannum.tolist() == [2]
        assert list(a.index) == list(b.index) == ["duplicate"]
        assert a.io_obj.iloc[0] is handles[1]
        assert b.io_obj.iloc[0] is handles[2]
        assert len(opened) == 2 and not opened.attrs
    assert len(table) == 2 and not table.attrs


def test_remote_candidates_must_be_staged():
    runtime = RuntimeContext(
        {
            "input_discovery": {
                "inputs": [
                    {"sources": ["https://example.org/data.nc"], "mode": "guess"}
                ]
            }
        }
    )
    table = InputDiscovery(runtime).discover(data_models=_DATA_MODELS)
    with pytest.raises(ValueError, match="opened, header-populated"):
        Kids(runtime).input.select(table)
    with (
        pytest.raises(ValueError, match="staged locally before opening"),
        table.toltec_file.open(on_error="raise"),
    ):
        pytest.fail("Unstaged data passed input preparation")


@pytest.mark.parametrize("raise_on_errors", [False, True])
def test_unrecognized_headers_cannot_supply_authoritative_metadata(
    tmp_path, raise_on_errors
):
    path = tmp_path / "empty.nc"
    xr.Dataset().to_netcdf(path)
    runtime = RuntimeContext(
        {
            "input_discovery": {"inputs": [{"sources": [path], "mode": "guess"}]},
            "kids": {"input": {"raise_on_errors": raise_on_errors}},
        }
    )
    table = InputDiscovery(runtime).discover(data_models=_DATA_MODELS)
    with table.toltec_file.open(on_error="record") as opened:
        assert opened.io_obj.iloc[0] is not None
        assert opened.accessor_obj.iloc[0] is None
        assert (
            opened.source_trace.map(source_trace_issues).iloc[0][-1].stage == "identify"
        )
        if raise_on_errors:
            with pytest.raises(ValueError, match="recognized acquisition metadata"):
                Kids(runtime).input.select(opened)
        else:
            assert Kids(runtime).input.select(opened).empty


@pytest.mark.parametrize("raise_on_errors", [False, True])
@pytest.mark.parametrize("populated", [False, True])
def test_recorded_metadata_failure_cannot_pass_as_filename_guess(
    data_lmt_path, raise_on_errors, populated
):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    runtime = RuntimeContext({"kids": {"input": {"raise_on_errors": raise_on_errors}}})
    with xr.open_dataset(path) as source:
        corrupt = source.drop_vars("Header.Source.Ra")
        table = ToltecFileAccessor.from_sources(
            [
                {"source": source, "source_loc": path},
                {"source": source, "source_loc": path},
            ],
            mode="read" if populated else "guess",
        )
        # A later failed read must not pass using guesses or previously read values.
        table["source"] = pd.Series([corrupt, source], dtype=object).array
        with table.toltec_file.open(on_error="record") as opened:
            opened.toltec_file.update_from_source_metadata(inplace=True, force=True)
            opened.index = [8, 8]
            before = opened.copy()
            assert opened.io_obj.notna().all()
            assert opened.data_kind_toltec.notna().all()
            if raise_on_errors:
                with pytest.raises(ValueError, match="Header.Source.Ra"):
                    Kids(runtime).input.select(opened)
            else:
                selected = Kids(runtime).input.select(opened)
                assert selected.index.tolist() == [8]
                assert selected.io_obj.iloc[0] is source
            pd.testing.assert_frame_equal(opened, before)


@pytest.mark.parametrize("select", ['tag == "usable"', "obsnum < 0"])
def test_strict_validation_only_checks_selected_rows(data_lmt_path, tmp_path, select):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with xr.open_dataset(path) as source:
        corrupt = source.drop_vars("Header.Source.Ra")
        # The empty object deliberately represents an unsupported resource.
        table = ToltecFileAccessor.from_sources(
            [
                {"source": source, "source_loc": path},
                {"source": corrupt, "source_loc": path},
                {"source": xr.Dataset(), "source_loc": "unsupported.nc"},
                tmp_path / "missing.nc",
            ]
        ).assign(tag=["usable", "corrupt", "unsupported", "missing"])
        with table.toltec_file.open(on_error="record") as opened:
            before = opened.copy()
            runtime = RuntimeContext(
                {
                    "kids": {
                        "input": {
                            "select": select,
                            "raise_on_errors": True,
                        }
                    }
                }
            )
            selected = Kids(runtime).input.select(opened)
            assert selected.tag.tolist() == (
                ["usable"] if select.startswith("tag") else []
            )
            pd.testing.assert_frame_equal(opened, before)


@pytest.mark.parametrize(
    "source_loc",
    [
        "renamed.nc",
        "tel_toltec_2025-99-31_145609_00_0002.nc",
    ],
)
def test_filepath_diagnostic_does_not_reject_header_recovered_input(
    data_lmt_path, source_loc
):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    runtime = RuntimeContext({"kids": {"input": {"raise_on_errors": True}}})
    with xr.open_dataset(path) as source:
        table = ToltecFileAccessor.from_sources(
            [{"source": source, "source_loc": source_loc}], on_error="record"
        )
        with table.toltec_file.open(on_error="record") as opened:
            opened.toltec_file.update_from_source_metadata(inplace=True)
            selected = Kids(runtime).input.select(opened)
            assert selected.obsnum.tolist() == [145609]
            assert selected.file_obsnum.isna().all()
            assert [
                item.stage
                for item in selected.source_trace.map(source_trace_issues).iloc[0]
            ] == ["filepath"]


@pytest.mark.parametrize("raise_on_errors", [False, True])
def test_selected_open_failure_policy_and_debug_logs(tmp_path, raise_on_errors):
    table = ToltecFileAccessor.from_sources([tmp_path / "missing.nc"])
    runtime = RuntimeContext({"kids": {"input": {"raise_on_errors": raise_on_errors}}})
    with table.toltec_file.open(on_error="record") as opened:
        before = opened.copy()
        messages = []
        sink = logger.add(messages.append, level="DEBUG")
        try:
            if raise_on_errors:
                with pytest.raises(ValueError, match="unusable selected input"):
                    Kids(runtime).input.select(opened)
            else:
                assert Kids(runtime).input.select(opened).empty
        finally:
            logger.remove(sink)
        text = "\n".join(messages)
        assert "missing.nc" in text and "FileNotFoundError" in text and "open:" in text
        assert f"policy={'raise' if raise_on_errors else 'skip'}" in text
        pd.testing.assert_frame_equal(opened, before)


def test_disabled_query_still_applies_error_policy(tmp_path):
    table = ToltecFileAccessor.from_sources([tmp_path / "missing.nc"])
    runtime = RuntimeContext(
        {
            "kids": {
                "input": {
                    "enabled": False,
                    "select": "undefined_name == 1",
                    "raise_on_errors": True,
                }
            }
        }
    )
    with (
        table.toltec_file.open(on_error="record") as opened,
        pytest.raises(ValueError, match="unusable selected input"),
    ):
        Kids(runtime).input.select(opened)


@pytest.mark.parametrize("failure", ["query", "consumer"])
@pytest.mark.parametrize("raise_on_errors", [False, True])
def test_open_scope_closes_all_files_on_failure(
    data_lmt_path, monkeypatch, failure, raise_on_errors
):
    from xarray.backends.netCDF4_ import NetCDF4DataStore

    runtime = RuntimeContext(
        {
            "input_discovery": {
                "inputs": [{"data_lmt_path": data_lmt_path, "obs_spec": "145609/tel"}]
            },
            "kids": {
                "input": {
                    "select": "undefined_column == 1" if failure == "query" else None,
                    "raise_on_errors": raise_on_errors,
                }
            },
        }
    )
    table = InputDiscovery(runtime).discover(data_models=_DATA_MODELS)
    closed = []
    original_close = NetCDF4DataStore.close

    def close(store, **kwargs):
        original_close(store, **kwargs)
        closed.append(store)

    monkeypatch.setattr(NetCDF4DataStore, "close", close)
    with (  # noqa: PT012 -- both failures must unwind the open-file scope
        pytest.raises((ValueError, NameError)),
        table.toltec_file.open() as opened,
    ):
        Kids(runtime).input.select(opened)
        msg = "consumer failed"
        raise ValueError(msg)
    assert len(closed) == len(table) == 2
    assert "io_obj" not in table


@pytest.mark.parametrize(("enabled", "expected"), [(True, 0), (False, 1)])
def test_empty_or_disabled_selection(timestream_path, enabled, expected):
    runtime = RuntimeContext(
        {
            "input_discovery": {"inputs": [{"sources": [timestream_path]}]},
            "kids": {"input": {"select": "obsnum < 0", "enabled": enabled}},
        }
    )
    with (
        InputDiscovery(runtime)
        .discover(data_models=_DATA_MODELS)
        .toltec_file.open() as opened
    ):
        selected = Kids(runtime).input.select(opened)
        assert len(selected) == expected
        assert len(opened) == 1


def test_standalone_runner_applies_pipeline_input_selection(timestream_path):
    runtime = RuntimeContext({"kids": {"input": {"select": "obsnum < 0"}}})
    with xr.open_dataset(timestream_path) as raw:
        with pytest.raises(ValueError, match="No inputs selected"):
            run_data(raw, timestream_path, runtime, dry_run=True)
        assert raw["Header.Toltec.ObsNum"].item() == 145609


@pytest.mark.parametrize("raise_on_errors", [False, True])
def test_standalone_file_runner_uses_input_error_policy(tmp_path, raise_on_errors):
    runtime = RuntimeContext({"kids": {"input": {"raise_on_errors": raise_on_errors}}})
    message = "unusable selected input" if raise_on_errors else "No inputs selected"
    with pytest.raises(ValueError, match=message):
        run_source(tmp_path / "missing.nc", runtime, dry_run=True)


def test_kids_input_config_snapshot_and_typo_validation():
    assert InputConfig().raise_on_errors is False
    runtime = RuntimeContext({"kids": {"input": {"select": "roach < 7"}}})
    kids = Kids(runtime)
    runtime.update_override_config({"kids": {"input": {"select": "roach >= 7"}}})
    assert kids.input.config.select == "roach < 7"
    assert kids.recreate().input.config.select == "roach >= 7"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        InputConfig.model_validate({"unknown_setting": "roach == 0"})
