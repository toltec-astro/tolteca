"""Cross-acquisition frequency matching with production v2 numerical behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.table import (
    QTable,  # noqa: TC002 -- public dataclass signature introspection.
)

from tolteca_kids.match1d import (
    Match1DResult,  # noqa: TC001 -- public dataclass signature introspection.
)

if TYPE_CHECKING:
    from astropy.units import Quantity

    from tolteca_kids.match1d import Match1D


@dataclass
class ToneMatchResult:
    """Query and reference tables with their frequency-match diagnostic."""

    query: QTable
    reference: QTable
    matched: Match1DResult


def match_tones(  # noqa: PLR0913 -- two tables, two frequency vectors and matching parameters.
    matcher: Match1D,
    tbl_kids: QTable,
    f_kids: Quantity,
    tbl_kids_ref: QTable,
    f_kids_ref: Quantity,
    *,
    match_shift_max: Quantity = 10 << u.MHz,
    match_shift_step: Quantity = 2 << u.kHz,
    query_key: str | None = None,
) -> ToneMatchResult:
    """Match query and reference tone tables with the production v2 calculation.

    Parameters
    ----------
    matcher : Match1D
        Configured frequency matcher.
    tbl_kids, tbl_kids_ref : astropy.table.QTable
        Query and reference tones with idx_chan, f_chan and amp_tone columns.
        Optional Qr columns enable phase-offset diagnostics. Query metadata must
        include uid_raw_obs; inputs are not modified.
    f_kids, f_kids_ref : astropy.units.Quantity
        Frequencies to match. VNA inputs use detections; other sweeps use tones.
    match_shift_max : astropy.units.Quantity, optional
        Maximum frequency shift, default 10 MHz.
    match_shift_step : astropy.units.Quantity, optional
        Frequency-shift search step, default 2 kHz (v2 bare-function default).
    query_key : str or None, optional
        Suffix identifying this query dataset in projected column names. Defaults
        to its uid_raw_obs metadata; does not change the matching calculation.

    Returns
    -------
    ToneMatchResult
        Query-oriented and reference-oriented tables, including uniqueness flags,
        and the match diagnostic. Column values, units and order follow v2.
    """

    def _match_postproc(r: Match1DResult) -> Match1DResult:
        matched = r.matched
        iq = matched["idx_query"]
        matched["idx_chan"] = tbl_kids["idx_chan"][iq]
        matched["f_chan"] = tbl_kids["f_chan"][iq]
        matched["amp_tone"] = tbl_kids["amp_tone"][iq]
        if "Qr" in tbl_kids.colnames:
            Qr = matched["Qr"] = tbl_kids["Qr"][iq]
        else:
            Qr = None
        ir = matched["idx_ref"]
        matched["idx_chan_ref"] = tbl_kids_ref["idx_chan"][ir]
        matched["f_chan_ref"] = tbl_kids_ref["f_chan"][ir]
        matched["amp_tone_ref"] = tbl_kids_ref["amp_tone"][ir]
        xx = matched["x_match"] = matched["dist"] / matched["ref"]
        if "Qr" in tbl_kids_ref.colnames:
            Qr_ref = matched["Qr_ref"] = tbl_kids_ref["Qr"][ir]
        else:
            Qr_ref = None
        if not (Qr is None or Qr_ref is None):
            rr = 0.5 / Qr
            matched["d_phi_match"] = np.rad2deg(np.arctan2(xx, rr)) << u.deg
        return r

    matched = matcher(
        query=f_kids.to(u.MHz),
        ref=f_kids_ref.to(u.MHz),
        postproc_hook=_match_postproc,
        shift_kw={"shift_max": match_shift_max, "dx": match_shift_step},
    )

    def _make_unique(tbl, idx_key, idx_other_key, dist_key):
        t = tbl.copy()
        t.sort(dist_key)
        _, unique_indices = np.unique(t[idx_other_key], return_index=True)
        tbl["mask_match_unique"] = False
        tbl["mask_match_unique"][t[idx_key][unique_indices]] = True
        return tbl

    tbl_matched = tbl_kids.copy()
    tq = matched.data["query_matched"]
    tbl_matched["f"] = tq["query"]
    tbl_matched["f_ref"] = tq["ref"]
    for colname in [
        "idx_query",
        "idx_ref",
        "dist",
        "dist_shifted",
        "adist_shifted",
        "idx_chan_ref",
        "f_chan_ref",
        "amp_tone_ref",
        "x_match",
        "Qr_ref",
        "d_phi_match",
    ]:
        if colname in tq.colnames:
            tbl_matched[colname] = tq[colname]
    tbl_matched = _make_unique(tbl_matched, "idx_query", "idx_ref", "adist_shifted")
    tbl_matched_ref = tbl_kids_ref.copy()
    matched_suffix = (
        query_key if query_key is not None else tbl_kids.meta["uid_raw_obs"]
    )
    tr = matched.data["ref_matched"]
    tbl_matched_ref["idx_ref"] = tr["idx_ref"]
    tbl_matched_ref["f_ref"] = tr["ref"]
    tbl_matched_ref[f"idx_{matched_suffix}"] = tr["idx_query"]
    tbl_matched_ref[f"f_{matched_suffix}"] = tr["query"]
    for colname in [
        "dist",
        "dist_shifted",
        "adist_shifted",
        "idx_chan",
        "f_chan",
        "amp_tone",
        "x_match",
        "Qrd_phi_match",  # v2 concatenated key; see tone-matching design notes.
    ]:
        if colname in tr.colnames:
            tbl_matched_ref[f"{colname}_{matched_suffix}"] = tr[colname]
    tbl_matched_ref = _make_unique(
        tbl_matched_ref,
        "idx_ref",
        f"idx_{matched_suffix}",
        f"adist_shifted_{matched_suffix}",
    )
    tbl_matched_ref[f"mask_match_unique_{matched_suffix}"] = tbl_matched_ref[
        "mask_match_unique"
    ]
    tbl_matched_ref.meta.update({f"tone_match_{matched_suffix}": tbl_kids.meta})
    return ToneMatchResult(tbl_matched, tbl_matched_ref, matched)
