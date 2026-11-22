"""Full-network accounting separates evidence gaps from readout outcomes."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def recipe():
    return runpy.run_path(
        str(
            Path(__file__).parents[2]
            / "examples/tune_model_comparison/network_census.py"
        )
    )


def test_unselected_is_not_an_attempted_solving_failure(recipe):
    row = {
        "native_route": "unselected",
        "psd_status": "no_complete_valid_estimate",
        "readout_suitable": False,
    }
    assert recipe["readout_state"](row) == "unselected"
    assert recipe["readout_state"](row | {"native_route": "joint"}) == "invalid_samples"
    assert (
        recipe["readout_state"](
            row | {"native_route": "joint", "psd_status": "measured"}
        )
        == "quality_screen"
    )


def test_missing_response_law_does_not_imply_bad_readout(recipe):
    row = {
        "association": "supported",
        "dark_evidence": False,
        "beta_skydip": np.nan,
        "native_route": "joint",
        "psd_status": "measured",
        "readout_suitable": True,
    }
    assert recipe["evidence_state"](row) == "loading_only"
    assert recipe["readout_state"](row) == "pass"
    assert (
        recipe["evidence_state"](row | {"association": "unresolved"})
        == "transfer_missing"
    )


def test_census_partitions_every_component_without_counting_probes(recipe):
    table = pd.DataFrame(
        [
            {
                "family": "Affine",
                "kind": "Multiplet",
                "evidence_state": "transfer_missing",
                "readout_state": "pass",
            },
            {
                "family": "Affine",
                "kind": "Multiplet",
                "evidence_state": "transfer_missing",
                "readout_state": "invalid_samples",
            },
            {
                "family": "Affine",
                "kind": "Singlet",
                "evidence_state": "individual_law",
                "readout_state": "quality_screen",
            },
        ]
    )
    census, cross = recipe["census_tables"](table)
    assert census.Fitted.sum() == cross.Total.sum() == len(table)
    assert census[list(recipe["READOUT"].values())].to_numpy().sum() == len(table)
    assert cross[list(recipe["READOUT"].values())].to_numpy().sum() == len(table)


def test_source_multiplet_exclusion_is_scope_not_fit_failure(recipe):
    cohort = [
        {
            "key": "group1",
            "traces": [1, 2],
            "reasons": ["multiple linked traces in connected group"],
        },
        {"key": "group2", "traces": [3], "reasons": [], "eligible": True},
    ]
    table = recipe["source_cohort"](cohort)
    assert table.traces.sum() == 3
    assert table.iloc[0].state == "Excluded by isolated-response-study scope"
    assert table.iloc[1].state == "Individual response-law contributor"
