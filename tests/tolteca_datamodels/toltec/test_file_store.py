"""Observation discovery in the recorded LMT and TolTEC directory layouts."""

from pathlib import Path

import pytest
import xarray as xr
from tollan.config import RuntimeContext
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.input_discovery import InputDiscovery
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.toltec.file_store import ToltecFileStore
from tolteca_datamodels.toltec.types import ToltecDataKind


def discover(entries):
    return InputDiscovery(
        RuntimeContext({"input_discovery": {"inputs": entries}})
    ).discover(
        data_models=DataModelSet.from_select(
            selection=["toltec"],
        )
    )


def test_candidates_include_both_scans_before_positional_selection(data_lmt_path):
    store = LmtFileStore(data_lmt_path)
    spec = "145609/[-1]/tel"
    scans = [
        store.tel.path / f"tel_toltec_2025-10-31_145609_00_000{n}.nc" for n in (1, 2)
    ]
    assert set(store.glob_obs_spec(spec)) == set(scans)
    catalog = discover([{"data_lmt_path": data_lmt_path, "obs_spec": spec}])
    assert [location.path for location in catalog.source_loc] == [scans[1].resolve()]


def test_store_discovery_can_be_empty(tmp_path):
    store = LmtFileStore(tmp_path / "missing")
    assert store.glob_obs_spec("145609/tel") == []
    assert store.glob_obs_spec("145609/0") == []


def test_toltec_candidates_keep_glob_order(data_lmt_path, monkeypatch):
    store = ToltecFileStore(data_lmt_path / "toltec")
    paths = sorted(store.path.glob("*/*/*145609*.nc"), reverse=True)
    assert len(paths) == 2

    def glob(path, pattern):
        assert path == store.path
        return iter(paths)

    monkeypatch.setattr(Path, "glob", glob)
    assert store.glob_obs_spec("145609/0") == paths


@pytest.mark.parametrize(
    "selector", ["145609/(1,5)", "145609/toltec(1,5)", "145609/(1,5)[-1]"]
)
def test_bounds_narrow_globs_without_applying_positions(
    tmp_path, monkeypatch, selector
):
    patterns = []

    def glob(path, pattern):
        patterns.append(str(path.relative_to(tmp_path) / pattern))
        return iter(())

    monkeypatch.setattr(Path, "glob", glob)
    assert ToltecFileStore(tmp_path).glob_obs_spec(selector) == []
    assert patterns == [f"*/toltec{i}/toltec{i}_145609_*.nc" for i in range(1, 5)]


@pytest.mark.parametrize("selector", ["145609/bar(1,5)", "145609/13", "145609/tel"])
def test_no_matching_declared_interfaces_does_not_probe_disk(
    tmp_path, monkeypatch, selector
):
    def unexpected(*args, **kwargs):
        pytest.fail("A selector with no declared interface matches reached glob")

    monkeypatch.setattr(Path, "glob", unexpected)
    assert ToltecFileStore(tmp_path).glob_obs_spec(selector) == []


@pytest.mark.parametrize(
    "selector",
    [
        "tcs-145609/(0,5)[-1]",
        "tcs-145609/toltec(0,5)[-1]",
        "tcs-145609/toltec()[-1]",
        "tcs-145609/toltec(,)[-1]",
    ],
)
def test_positions_use_observed_not_declared_interfaces(data_lmt_path, selector):
    # Only ROACH 0 has reference files here, not the largest eligible ROACH.
    table = discover([{"data_lmt_path": data_lmt_path, "obs_spec": selector}])
    assert table.interface.tolist() == ["toltec0", "toltec0"]
    assert (
        table.toltec_file.query_obs_spec(selector).index.tolist()
        == table.index.tolist()
    )


@pytest.mark.parametrize("selector", ["{tel,0}[-1]", "{tel,toltec0}[-1]", "*[-1]"])
def test_unordered_positions_keep_telescope_and_readout_files(data_lmt_path, selector):
    table = discover(
        [{"data_lmt_path": data_lmt_path, "obs_spec": f"tcs-145609/{selector}"}]
    )
    assert table.interface.tolist() == ["tel", "tel", "toltec0", "toltec0"]


@pytest.mark.parametrize(
    ("spec", "names"),
    [
        (
            "145609/tel",
            [
                "tel_toltec_2025-10-31_145609_00_0001.nc",
                "tel_toltec_2025-10-31_145609_00_0002.nc",
            ],
        ),
        ("145609/2/tel", ["tel_toltec_2025-10-31_145609_00_0002.nc"]),
        ("tcs-145609/2/tel", ["tel_toltec_2025-10-31_145609_00_0002.nc"]),
        (
            "{145605,145609}/tel",
            [
                "tel_toltec_2025-10-31_145605_00_0001.nc",
                "tel_toltec_2025-10-31_145609_00_0001.nc",
                "tel_toltec_2025-10-31_145609_00_0002.nc",
            ],
        ),
        (
            "(145605,145610)/tel",
            [
                "tel_toltec_2025-10-31_145605_00_0001.nc",
                "tel_toltec_2025-10-31_145606_00_0001.nc",
                "tel_toltec_2025-10-31_145608_00_0001.nc",
                "tel_toltec_2025-10-31_145609_00_0001.nc",
                "tel_toltec_2025-10-31_145609_00_0002.nc",
            ],
        ),
        (
            "(145605,145610)[::2]/tel",
            [
                "tel_toltec_2025-10-31_145605_00_0001.nc",
                "tel_toltec_2025-10-31_145608_00_0001.nc",
            ],
        ),
        (
            "[-1]/tel",
            [
                "tel_toltec_2025-10-31_145611_00_0002.nc",
            ],
        ),
        ("145609/[-1]/tel", ["tel_toltec_2025-10-31_145609_00_0002.nc"]),
        ("[-1]/2/tel", ["tel_toltec_2025-10-31_145611_00_0002.nc"]),
    ],
)
def test_telescope_selectors_use_recorded_layout_without_opening(
    data_lmt_path, monkeypatch, spec, names
):
    expected = [data_lmt_path.resolve() / "tel" / name for name in names]
    assert all(path.is_file() for path in expected)

    def unexpected(*args, **kwargs):
        pytest.fail("Filename discovery opened a source")

    monkeypatch.setattr(Path, "open", unexpected)
    monkeypatch.setattr(xr, "open_dataset", unexpected)
    catalog = discover([{"data_lmt_path": data_lmt_path, "obs_spec": spec}])
    assert {
        fileloc(r.source).path for r in list(catalog.itertuples(index=False))
    } == set(expected)
    assert catalog["interface"].tolist() == ["tel"] * len(expected)
    assert catalog["master"].tolist() == ["tcs"] * len(expected)
    assert catalog["data_kind_toltec"].tolist() == [ToltecDataKind.lmt_tel.value] * len(
        expected
    )


def test_explicit_roach_selectors_keep_toltec_scope(data_lmt_path):
    names = [
        "toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc",
        "toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
    ]
    expected = [
        str(data_lmt_path.resolve() / "toltec/tcs/toltec0" / name) for name in names
    ]
    for spec in ("145609/0", "145609/toltec0", "tcs-145609/0"):
        catalog = discover([{"data_lmt_path": data_lmt_path, "obs_spec": spec}])
        assert {
            str(fileloc(r.source).path) for r in list(catalog.itertuples(index=False))
        } == set(expected)
        assert catalog["interface"].tolist() == ["toltec0", "toltec0"]


@pytest.mark.parametrize("selector", ["145609", "145609/*", "tcs-145609"])
def test_unqualified_selector_includes_telescope_and_toltec(data_lmt_path, selector):
    expected = {
        "tel/tel_toltec_2025-10-31_145609_00_0001.nc",
        "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        "toltec/tcs/toltec0/toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc",
        "toltec/tcs/toltec0/toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
    }
    table = discover([{"data_lmt_path": data_lmt_path, "obs_spec": selector}])
    assert len(table) == 4
    assert {
        str(loc.path.relative_to(data_lmt_path.resolve())) for loc in table.source_loc
    } == expected
    assert set(table.interface) == {"tel", "toltec0"}


@pytest.mark.parametrize(
    "selector", ["145609/{tel,0}", "{ics,tcs}-145609/{toltec0,tel}"]
)
def test_mixed_interface_set_discovers_both_trees(data_lmt_path, selector):
    expected = discover(
        [
            {"data_lmt_path": data_lmt_path, "obs_spec": spec}
            for spec in ("145609/tel", "145609/0")
        ]
    )
    result = discover([{"data_lmt_path": data_lmt_path, "obs_spec": selector}])
    assert len(result) == len(expected) == 4
    assert {str(v.url) for v in result.source_loc} == {
        str(v.url) for v in expected.source_loc
    }
    assert set(result["interface"]) == {"tel", "toltec0"}


def test_mixed_interface_set_retains_tel_without_requested_roach(data_lmt_path):
    # These reference observations contain ROACH 0, not ROACH 1.
    expected = discover([{"data_lmt_path": data_lmt_path, "obs_spec": "145609/tel"}])
    mixed = discover([{"data_lmt_path": data_lmt_path, "obs_spec": "145609/{tel,1}"}])
    assert len(mixed) == 2
    assert mixed.equals(expected)


def test_telescope_selector_has_no_matches_for_non_tcs_master(data_lmt_path):
    assert discover(
        [{"data_lmt_path": data_lmt_path, "obs_spec": "ics-145609/tel"}]
    ).empty


def test_telescope_selector_and_explicit_path_deduplicate(data_lmt_path, tmp_path):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    alias = tmp_path / path.name
    alias.symlink_to(path)
    catalog = discover(
        [
            {"sources": [alias]},
            {"data_lmt_path": data_lmt_path, "obs_spec": "145609/2/tel"},
        ]
    )
    assert [fileloc(r.source).path for r in list(catalog.itertuples(index=False))] == [
        alias
    ]
