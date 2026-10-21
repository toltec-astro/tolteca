"""Shared discovery, catalog selection, and kind-based dispatch tests."""

import json
from typing import cast

import pandas as pd
import pytest
from tollan.config import RuntimeContext
from tollan.utils.fileloc import fileloc
from typer.testing import CliRunner

from tolteca_datamodels.input_discovery import InputDiscovery
from tolteca_datamodels.toltec.file import (
    get_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kids import SweepReduce
from tolteca_kids.cli import app
from tolteca_kids.core import Kids
from tolteca_kids.timestream_reduce import TimestreamReduce


@pytest.mark.parametrize(
    ("filename", "kind", "obsnum", "master"),
    [
        (
            "toltec0_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
            ToltecDataKind.vnasweep,
            145605,
            None,
        ),
        (
            "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc",
            ToltecDataKind.targsweep,
            145606,
            None,
        ),
        (
            "toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc",
            ToltecDataKind.tune,
            145609,
            None,
        ),
        (
            "toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
            ToltecDataKind.raw_timestream,
            145609,
            None,
        ),
        (
            "tel_toltec_2025-10-31_145609_00_0002.nc",
            ToltecDataKind.lmt_tel,
            145609,
            "tcs",
        ),
    ],
)
def test_filename_kinds_and_table_context(filename, kind, obsnum, master):
    # Basenames recorded in the reference ToltecDB/LMTMC slices.
    info = get_info_from_source(filename)
    assert info is not None
    assert info.data_kind_toltec == kind.value
    assert info.obsnum == obsnum
    assert pd.isna(info.master) if master is None else info.master == master
    assert info.source is filename
    assert isinstance(info.file_ut, pd.Timestamp)
    assert info._asdict()["data_kind_toltec"] == kind.value


def test_unknown_stays_partial_and_master_comes_from_layout(tmp_path):
    assert pd.isna(get_info_from_source("renamed.nc").obsnum)
    catalog = guess_info_from_sources(
        [
            tmp_path / "renamed.nc",
            tmp_path / "toltec/tcs/toltec0/"
            "toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc",
        ]
    )
    unknown, known = list(catalog.itertuples(index=False))
    assert pd.isna(unknown.obsnum)
    assert unknown.data_kind_toltec == ToltecDataKind.unknown.value
    assert known.master == "tcs"
    assert list(
        catalog.toltec_file.query("roach == 0 & obsnum == 145609").itertuples(
            index=False
        )
    ) == [known]


@pytest.mark.parametrize(
    "spec",
    [
        "145609",
        "145609-0",
        "145609-0-2",
        "145609/0",
        "145609/toltec0",
        "145609-0-2-0",
        "tcs-145609/0",
    ],
)
def test_observation_selectors(tmp_path, spec):
    path = (
        tmp_path / "toltec/tcs/toltec0/toltec0_145609_000_0002_2025_10_31_09_27_57.nc"
    )
    path.parent.mkdir(parents=True)
    path.touch()
    catalog = InputDiscovery(
        RuntimeContext(
            {
                "input_discovery": {
                    "inputs": [
                        {"data_lmt_path": tmp_path, "obs_spec": spec},
                    ]
                }
            }
        )
    ).discover()
    assert (
        fileloc(cast("str", next(catalog.itertuples(index=False)).source)).path == path
    )


def test_paths_globs_deduplicate_and_unmatched_are_empty(tmp_path):
    path = tmp_path / "unknown.nc"
    path.touch()
    catalog = InputDiscovery(
        RuntimeContext(
            {
                "input_discovery": {
                    "inputs": [
                        {
                            "sources": [path, {"glob": str(tmp_path / "*.nc")}],
                            "mode": "guess",
                        },
                    ]
                }
            }
        )
    ).discover()
    assert len(catalog) == 1
    assert (
        InputDiscovery(
            RuntimeContext(
                {
                    "input_discovery": {
                        "inputs": [
                            {"sources": [{"glob": str(tmp_path / "*.fits")}]},
                        ]
                    }
                }
            )
        )
        .discover()
        .empty
    )
    assert (
        list(InputDiscovery(RuntimeContext({})).discover().itertuples(index=False))
        == []
    )


@pytest.mark.parametrize(
    "expression",
    [
        "data_kind_toltec == targsweep.value",
        "bits_any_set(data_kind_toltec, targsweep)",
    ],
)
def test_cli_selects_renamed_file_by_header_identity(
    target_sweep_path, tmp_path, expression
):
    path = tmp_path / "downloaded.nc"
    path.symlink_to(target_sweep_path)
    original = f"/data_lmt/toltec/tcs/toltec0/{target_sweep_path.name}"
    result = CliRunner().invoke(
        app,
        [
            "--show_inputs",
            "--select",
            f"roach == 0 & ({expression})",
            "--",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.exception
    [row] = json.loads(result.stdout)
    assert row["source"] == path.as_uri()
    assert row["filepath_orig"] == original
    assert row["filepath_info"] is None
    assert row["file_prefix"] is None
    assert "metadata" not in row


@pytest.mark.parametrize(
    "kind",
    [
        ToltecDataKind.solved_timestream,
        ToltecDataKind.reduced_sweep,
        ToltecDataKind.unknown,
    ],
)
def test_unsupported_kind_never_enters_sweep_pipeline(kind, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported input invoked reduction")

    monkeypatch.setattr(SweepReduce, "__call__", unexpected)
    with pytest.raises(NotImplementedError, match=kind.name):
        Kids(RuntimeContext()).pipeline_for(kind)


def test_cli_selects_timestream_without_reduction(timestream_path, monkeypatch):
    path = timestream_path

    def unexpected(*args, **kwargs):
        pytest.fail("metadata-only command invoked reduction")

    monkeypatch.setattr(SweepReduce, "__call__", unexpected)
    monkeypatch.setattr(TimestreamReduce, "run", unexpected)
    result = CliRunner().invoke(
        app,
        [
            str(path),
            "--show_inputs",
            "--select",
            "roach == 0 & data_kind_toltec == raw_timestream.value",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert (
        json.loads(result.stdout)[0]["data_kind_toltec"]
        == ToltecDataKind.raw_timestream.value
    )
