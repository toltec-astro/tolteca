"""Bind stationary model branches to recorded TolTEC acquisition order."""

from __future__ import annotations

import numpy as np

from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kidsproc.models import NONLINEARITY_UNIQUE_MAX


def _recorded_history(block):
    """Read chronological probe intervals without loading detector samples."""
    if not block.data_kind & ToltecDataKind.raw_sweep:
        msg = "Sweep history requires recorded raw-acquisition probe intervals"
        raise ValueError(msg)
    probe = block.probe_axis
    mapper, schema = probe.mapper, probe.mapper.schema
    lo = mapper.get_arr(probe.data_source, schema.f_lo).u.to("Hz").values
    start = mapper.get_arr(probe.data_source, schema.sample_start).values
    end = mapper.get_arr(probe.data_source, schema.sample_end).values
    if (
        lo.ndim != 1
        or len(lo) < 2  # noqa: PLR2004 -- At least two probe steps establish direction.
        or start.shape != lo.shape
        or end.shape != lo.shape
        or start.dtype.kind not in "iu"
        or end.dtype.kind not in "iu"
        or np.any(start < 0)
        or np.any(end <= start)
        or np.any(start[1:] < end[:-1])
        or not np.all(np.isfinite(lo) & (lo > 0))
    ):
        msg = "Sweep history requires chronological, disjoint probe sample intervals"
        raise ValueError(msg)
    delta = np.diff(lo)
    if np.all(delta > 0):
        direction = "increasing"
    elif np.all(delta < 0):
        direction = "decreasing"
    else:
        msg = "Recorded LO steps do not establish one monotonic sweep direction"
        raise ValueError(msg)
    # Reduced frequency coordinates can be sorted independently of acquisition.
    # Do not mistake their order for evidence of the physical sweep history.
    if "sweep" in block.tree.children:
        reduced_lo = block.sweep.f_lo
        if reduced_lo is None or not np.array_equal(reduced_lo.u.to("Hz").values, lo):
            msg = "Reduced LO order differs from recorded probe intervals"
            raise ValueError(msg)
    meta = block.meta
    return {
        "status": "recorded",
        "source": "probe_sample_intervals",
        "uid_raw_obs": meta.uid_raw_obs,
        "interface": meta.interface,
        "block": int(meta.block),
        "direction": direction,
        "n_steps": len(lo),
        "n_samples": int(np.sum(end - start)),
        "sample_start": int(start[0]),
        "sample_end": int(end[-1]),
        "f_lo_first_hz": float(lo[0]),
        "f_lo_last_hz": float(lo[-1]),
    }


def bind_analysis(block, analysis):
    """Resolve a directional analysis and retain its acquisition evidence."""
    requires_history = analysis.nonlinear and (
        analysis.nonlinear_branch != "unique"
        or analysis.a_nl_max > NONLINEARITY_UNIQUE_MAX
    )
    try:
        history = _recorded_history(block)
    except (KeyError, ValueError) as error:
        if requires_history:
            msg = f"Directional ModelFit requires acquisition history: {error}"
            raise ValueError(msg) from error
        return analysis, {"status": "unavailable", "reason": str(error)}
    if requires_history:
        branch = analysis.nonlinear_branch
        if branch != "unique" and branch != history["direction"]:
            msg = "Configured nonlinear branch contradicts recorded sweep direction"
            raise ValueError(msg)
        analysis = analysis.model_copy(
            update={"nonlinear_branch": history["direction"]}
        )
    return analysis, history
