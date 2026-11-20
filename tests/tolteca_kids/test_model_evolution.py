"""Only explicit, unambiguous correspondences establish model history."""

import numpy as np
import xarray as xr
from astropy.table import Table

from tolteca_kids.model_evolution import ModelEvolutionViz, model_evolution_rows


def test_blends_and_many_to_one_correspondences_remain_unresolved():
    fit = xr.Dataset(
        {
            k: ("row", v)
            for k, v in {
                "resonator_id": ["0", "1", "2", "3", "4"],
                "group_id": ["g0", "blend", "blend", "g3", "g4"],
                "fr": [1e8, 2e8, 3e8, 4e8, 5e8],
                "Qr": [10000.0] * 5,
                "a_nl": [0.0] * 5,
            }.items()
        }
    )
    finding = Table({"idx_det": [0, 1, 2, 3, 4], "idx_chan": [8, 2, 2, 99, 7]})
    # Channel tables carry projected idx_det too. These cannot resolve blends.
    probes = Table({"idx_chan": [99, 8, 7, 2], "idx_det": [3, 0, 4, 1]})
    # Permuted channels; two different probes claim reference tone 4.
    edges = Table({"idx_query": [0, 1, 2, 3], "idx_ref": [4, 6, 4, 9]})
    rows = model_evolution_rows(fit, finding, probes, edges)
    assert [
        (r["candidate"], r["reference"]) for r in rows if r["status"] == "linked"
    ] == [("0", 6)]
    assert (
        rows[1]["reason"] == rows[2]["reason"] == "ambiguous candidate/channel relation"
    )
    assert rows[1]["reference_alternatives"] == rows[2]["reference_alternatives"] == [9]
    assert rows[3]["reference_alternatives"] == rows[4]["reference_alternatives"] == [4]
    assert (
        rows[3]["reason"]
        == rows[4]["reason"]
        == "missing or ambiguous tone correspondence"
    )
    rows[0]["input"] = 0
    second = dict(rows[0], input=2, fr=rows[0]["fr"] + 2000)
    fig = ModelEvolutionViz()([*rows, second], ["first", "unresolved", "later"])
    np.testing.assert_allclose(fig.data[0].z[0], [0, np.nan, 2], equal_nan=True)


def test_reference_candidates_do_not_require_single_channel_projection():
    fit = xr.Dataset(
        {
            k: ("row", v)
            for k, v in {
                "resonator_id": ["0", "0", "1"],
                "group_id": ["g", "g", "g"],
                "fr": [1e8, 1e8, 1.1e8],
                "Qr": [10000.0] * 3,
                "a_nl": [0.0] * 3,
            }.items()
        }
    )
    finding = Table({"idx_det": [0, 0, 1], "idx_chan": [8, 2, 2]})
    reference = Table({"idx_det": [1, 0]})
    rows = model_evolution_rows(
        fit, finding, reference, None, reference=True, candidate_input=True
    )
    assert [r["reference"] for r in rows] == [1, 0]
    # A non-reference VNA uses the full candidate-to-reference matching edges.
    pairs = Table({"idx_query": [0, 1], "idx_ref": [9, 4]})
    matched = model_evolution_rows(fit, finding, reference, pairs, candidate_input=True)
    assert [r["reference"] for r in matched] == [4, 9]
    # A repeated candidate in different fits is ambiguous even on the reference.
    fit.group_id.values[1] = "h"
    rows = model_evolution_rows(
        fit, finding, reference, None, reference=True, candidate_input=True
    )
    assert rows[0]["reference"] is None
    assert rows[0]["reason"] == "candidate appears in multiple fits"
