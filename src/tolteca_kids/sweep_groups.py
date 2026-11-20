"""Bind per-block segment plans to saved cross-sweep finding correspondences."""

import numpy as np
import xarray as xr

from tolteca_datamodels.data_prod import DataProdRef
from tolteca_kidsproc.segmentation import group_sweep_plans

from .kids_find import KidsFind
from .model_fit import ModelFit
from .product import open_kids_product
from .sweep_check import SweepCheck


def prepare_sweep_groups(matched, indexes):
    """Construct local joint-analysis problems for one exact matched interface.

    Parameters
    ----------
    matched : ToneMatchView
        Open matching result containing exact processed-input bindings and
        original ``idx_det`` finding identifiers. Ambiguous edges are retained.
    indexes : mapping of DataProdRef to DataProdIndex
        Exact processed products named by the matching result.

    Returns
    -------
    tolteca_kidsproc.segmentation.CrossSweepPlan
        Complete local segment plans and their cross-sweep connectivity. Saved
        automatic ModelFit plans are reused. Otherwise the same ModelFit planner
        constructs segments from saved findings and quality masks, without
        fitting. Exact inputs and finding-to-matching rows remain in plan attrs.

    Raises
    ------
    ValueError
        Product, acquisition or finding bindings disagree, a supplied ModelFit
        plan lacks automatic candidate provenance, or a link has an invalid row.
        This function never substitutes nearby frequencies for missing IDs.
    """
    plans, rows = {}, {}
    interfaces = {item["interface"] for item in matched.inputs}
    if len(interfaces) != 1:
        msg = "Cross-sweep plans must select one interface"
        raise ValueError(msg)
    for item in matched.inputs:
        key = item["key"]
        plans[key], rows[key] = _input_plan(item, indexes, matched.input(key))

    edges = []

    def edge(dataset, row, reference, row_ref, source):
        if (
            dataset not in rows
            or reference not in rows
            or not 0 <= row < len(rows[dataset])
            or not 0 <= row_ref < len(rows[reference])
        ):
            msg = "Candidate correspondence contains an invalid input row"
            raise ValueError(msg)
        rid, rid_ref = rows[dataset][row], rows[reference][row_ref]
        edges.append(
            {
                "dataset": dataset,
                "resonator_id": rid,
                "dataset_ref": reference,
                "resonator_id_ref": rid_ref,
                "input_row": row,
                "input_row_ref": row_ref,
                "source": source,
                "status": "candidate" if rid and rid_ref else "unmapped",
            }
        )

    reference = matched.reference["key"]
    for item in matched.inputs:
        key = item["key"]
        if key == reference:
            continue
        pairs = matched.pair(key).matched.matched
        for iq, ir in zip(pairs["idx_query"], pairs["idx_ref"], strict=True):
            edge(key, int(iq), reference, int(ir), "tone_match")
    names = (
        "dataset",
        "resonator_id",
        "dataset_ref",
        "resonator_id_ref",
        "input_row",
        "input_row_ref",
        "source",
        "status",
    )
    links = xr.Dataset(
        {
            name: (
                "row",
                np.asarray(
                    [r[name] for r in edges],
                    dtype=(int if name in {"input_row", "input_row_ref"} else str),
                ),
            )
            for name in names
        }
    )
    return group_sweep_plans(plans, links)


def _input_plan(item, indexes, table):
    index = indexes[DataProdRef.model_validate(item["product"])]
    if str(index.run.uid) != item["run_uid"]:
        msg = "Segment input run differs from the matching input"
        raise ValueError(msg)
    with open_kids_product(index, interface=item["interface"]) as tree:
        block = tree.toltec_kids.blocks[item["block"]]
        if block.meta.uid_raw_obs != item["uid_raw_obs"]:
            msg = "Segment input acquisition differs from the matching input"
            raise ValueError(msg)
        finding = KidsFind.view(block.tree)
        plan = _block_plan(block, finding)
        input_rows = _finding_rows(table, finding, plan)
        plan.attrs.update(input=item, matching_rows=input_rows)
        return plan, input_rows


def _block_plan(block, finding):
    if "model_fit" in block.tree.children:
        node = block.tree["model_fit"]
        if node.attrs.get("plan_source") != "kids_find":
            msg = "Supplied ModelFit plans require explicit component correspondences"
            raise ValueError(msg)
        return node["plan"].to_dataset(inherit=False).load().copy(deep=True)
    sweep = block.sweep
    mask = np.broadcast_to(
        np.asarray(sweep.chan_axis_table["tone_mask"], bool)[:, None],
        sweep.I.shape,
    ).copy()
    if SweepCheck.result_path() in block.tree.children:
        check = SweepCheck.view(block.tree)
        mask &= ~check.mask_spike & ~check.mask_chan_bad[:, None]
    return ModelFit.make_plan(sweep, finding, mask=mask)


def _finding_rows(table, finding, plan):
    detected = {int(r["idx"]): r for r in finding.detected}
    if "idx_det" not in table.colnames:
        msg = "Sweep grouping requires saved finding IDs in the matching input"
        raise ValueError(msg)
    available = set(plan.resonator_id.values)
    rows = []
    for row in table:
        rid = row["idx_det"]
        if np.ma.is_masked(rid) or int(rid) < 0:
            rows.append("")
            continue
        if int(rid) not in detected or not np.isclose(
            row["f_det"].to_value("Hz"),
            detected[int(rid)]["f"].to_value("Hz"),
            rtol=0,
            atol=1e-6,
        ):
            msg = "Matching row and exact saved finding disagree"
            raise ValueError(msg)
        rows.append(str(rid) if str(rid) in available else "")
    return rows
