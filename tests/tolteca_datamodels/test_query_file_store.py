"""Store queries share the direct and configured discovery workflow."""

import pandas as pd
import pytest
from tollan.config import RuntimeContext

from tolteca_datamodels.core.file import FileAccessorView
from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.input_discovery import InputDiscovery, query_file_store
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_reference_query_matches_building_blocks_and_configured_discovery(
    data_lmt_path, mode
):
    # Configured paths resolve symlinks; compare against the same store root.
    store = LmtFileStore(data_lmt_path.resolve())
    spec = ObsSpec.parse("145609/{tel,0}")
    models = DataModelSet.from_select(["toltec"])
    search = store.partial(store_types=models.file_store_types)
    table = FileAccessorView.from_sources(
        search.glob_obs_spec(spec),
        mode=mode,
        on_error="raise",
        accessor_types=models.file_accessor_types,
    ).table
    expected = (
        FileAccessor(table)
        .partial(accessor_types=models.file_accessor_types)
        .query_obs_spec(spec)
    )
    actual = query_file_store(
        store, spec, mode=mode, data_models=models, on_error="raise"
    )
    assert len(actual) == 4
    if mode == "read":
        assert actual.t0_mjd.notna().all()
    else:
        assert "t0_mjd" not in actual
    assert "io_obj" not in actual
    assert "accessor_obj" not in actual
    pd.testing.assert_frame_equal(actual, expected)
    configured = InputDiscovery(
        RuntimeContext(
            {
                "input_discovery": {
                    "inputs": [
                        {
                            "data_lmt_path": data_lmt_path,
                            "obs_spec": str(spec),
                            "mode": mode,
                        }
                    ]
                }
            }
        )
    ).discover(data_models=models, on_error="raise")
    pd.testing.assert_frame_equal(configured, actual.reset_index(drop=True))


def test_query_preserves_bound_store_restrictions(data_lmt_path):
    store = LmtFileStore(data_lmt_path)
    models = DataModelSet.from_select(["toltec"])
    table = query_file_store(store.tel, "145609", data_models=models)
    assert set(table.interface) == {"tel"}
    assert len(table) == 2
    assert len(query_file_store(store, "145609/{tel,0}", data_models=models)) == 4


def test_positional_selection_is_applied_after_combining_store_locations(data_lmt_path):
    store = LmtFileStore(data_lmt_path)
    models = DataModelSet.from_select(["toltec"])
    table = query_file_store(store, "145609/[-1]/{tel,0}", data_models=models)
    assert set(table.interface) == {"tel", "toltec0"}
    assert table.scannum.tolist() == [2, 2]


def test_explicit_empty_models_do_not_search_or_open(data_lmt_path, monkeypatch):
    from pathlib import Path

    def unexpected(*args, **kwargs):
        pytest.fail("An empty model set must not search or open files")

    monkeypatch.setattr(Path, "glob", unexpected)
    monkeypatch.setattr(FileAccessorView, "open", unexpected)
    table = query_file_store(
        LmtFileStore(data_lmt_path),
        "*",
        mode="read",
        data_models=DataModelSet.from_select([]),
    )
    assert table.empty
    assert {"source", "source_loc"} <= set(table)


def test_default_uses_available_models_and_unmatched_store_returns_table(data_lmt_path):
    store = LmtFileStore(data_lmt_path)
    default = query_file_store(store, "145609/tel")
    explicit = query_file_store(
        store, "145609/tel", data_models=DataModelSet.from_select()
    )
    pd.testing.assert_frame_equal(default, explicit)
    assert query_file_store(store, "0/tel").empty


def test_query_records_failed_reads_or_raises_as_requested(tmp_path):
    tel = tmp_path / "tel"
    tel.mkdir()
    (tel / "tel_toltec_2025-10-31_145609_00_0002.nc").write_bytes(b"broken netCDF")
    store = LmtFileStore(tmp_path)
    models = DataModelSet.from_select(["toltec"])
    table = query_file_store(store, "145609/tel", mode="read", data_models=models)
    assert table.obsnum.tolist() == [145609]
    assert table.source_trace.iloc[0].has_error
    with pytest.raises((OSError, ValueError)):
        query_file_store(
            store, "145609/tel", mode="read", data_models=models, on_error="raise"
        )
