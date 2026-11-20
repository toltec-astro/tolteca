"""Conservative candidate identity joins for saved model evolution."""

import numpy as np
import plotly.graph_objects as go
from tollan.plot.plotly import PlotlyViz, make_subplots


def model_evolution_rows(  # noqa: C901, PLR0912, PLR0913 -- explicit inputs and rejection reasons for distinct identity ambiguities.
    fit,
    finding_pairs,
    tone_input,
    tone_pairs,
    *,
    reference=False,
    candidate_input=False,
):
    """Join fitted candidates to reference rows using existing correspondences.

    Parameters
    ----------
    fit : xarray.Dataset
        Saved ModelFit association rows from an automatic KidsFind plan.
        Candidate labels must be that finding result's detection IDs.
        Supplied plans require explicit correspondence evidence before this join.
    finding_pairs, tone_input, tone_pairs : astropy.table.Table
        Complete finding-candidate/channel relations, this matching input's
        original table, and complete query/reference tone edges. The latter may
        be None for the reference input itself.
    reference : bool, optional
        This is the matching reference. Defaults to False.
    candidate_input : bool, optional
        The matching input represents VNA detections rather than probe channels.
        Defaults to False. Channel tables may also contain projected detection
        columns; those projections must not substitute for full finding edges.

    Returns
    -------
    list of dict
        One row per fitted candidate, including unresolved reasons. Reference
        rows identify tones within the pinned matching result, not global IDs.
        A blend is not resolved by choosing its strongest or nearest component.
    """
    edges = {}
    for pair in finding_pairs:
        candidate, chan = str(int(pair["idx_det"])), int(pair["idx_chan"])
        edges.setdefault(candidate, set()).add(chan)
    by_channel = {}
    for candidate, channels in edges.items():
        for chan in channels:
            by_channel.setdefault(chan, set()).add(candidate)
    tone_edges, reverse = {}, {}
    if reference:
        tone_edges = {i: {i} for i in range(len(tone_input))}
        reverse = tone_edges
    else:
        for pair in tone_pairs:
            q, r = int(pair["idx_query"]), int(pair["idx_ref"])
            tone_edges.setdefault(q, set()).add(r)
            reverse.setdefault(r, set()).add(q)
    # Read each saved column once; joins are indexed by their explicit IDs.
    # Repeated xarray selection here otherwise rereads Zarr per candidate.
    candidates = {}
    group_ids = fit.group_id.values
    parameters = {key: fit[key].values for key in ("fr", "Qr", "a_nl")}
    for i, candidate in enumerate(fit.resonator_id.values.astype(str)):
        candidates.setdefault(candidate, []).append(i)
    input_rows = {}
    for i, value in enumerate(tone_input["idx_det" if candidate_input else "idx_chan"]):
        input_rows.setdefault(int(value), []).append(i)
    result = []
    for candidate, positions in candidates.items():
        record = {"candidate": candidate, "reference": None, "status": "unresolved"}
        groups = set(group_ids[positions].tolist())
        record["group"] = ", ".join(sorted(str(g) for g in groups))
        record.update(
            {key: float(values[positions[0]]) for key, values in parameters.items()}
        )
        channels = edges.get(candidate, set())
        values = {int(candidate)} if candidate_input else channels
        possible_inputs = {i for value in values for i in input_rows.get(value, [])}
        record["reference_alternatives"] = sorted(
            {r for i in possible_inputs for r in tone_edges.get(i, set())}
        )
        if len(groups) != 1:
            reason = "candidate appears in multiple fits"
        elif not candidate_input and (
            len(channels) != 1 or any(len(by_channel[c]) != 1 for c in channels)
        ):
            reason = "ambiguous candidate/channel relation"
        else:
            value = int(candidate) if candidate_input else next(iter(channels))
            indexes = input_rows.get(value, [])
            refs = (
                tone_edges.get(int(indexes[0]), set()) if len(indexes) == 1 else set()
            )
            if len(refs) == 1 and len(reverse[next(iter(refs))]) == 1:
                record.update(reference=next(iter(refs)), status="linked")
                reason = ""
            else:
                reason = "missing or ambiguous tone correspondence"
        record["reason"] = reason
        result.append(record)
    # Two candidates may never silently claim one reference identity.
    counts = {}
    for record in result:
        if record["reference"] is not None:
            counts[record["reference"]] = counts.get(record["reference"], 0) + 1
    for record in result:
        if record["reference"] is not None and counts[record["reference"]] > 1:
            record.update(
                reference=None,
                status="unresolved",
                reason="multiple candidates claim this reference",
            )
    return result


class ModelEvolutionViz(PlotlyViz):
    """Parameter history on unambiguous reference-tone identities."""

    def __call__(self, rows, labels, *, identity="reference"):
        """Plot frequency changes, Qr and nonlinearity across pinned input blocks.

        Parameters
        ----------
        rows : list of dict
            Candidate records with input ordinal, reference row and parameters.
        labels : list of str
            Ordered acquisition/block labels.
        identity : {"reference", "detector", "identity"}, optional
            Reference-tone rows by default, or UIDs from one explicitly selected
            APT. Only accepted/tentative observations enter the APT history;
            unresolved alternatives remain gaps. Hover labels retain status.

        Returns
        -------
        plotly.graph_objects.Figure
            Gaps retain unresolved and unavailable measurements.
        """
        linked = [
            r
            for r in rows
            if r["status"] in {"linked", "accepted", "tentative"}
            and r.get(identity) is not None
        ]
        ids = sorted({r[identity] for r in linked})
        title = {
            "detector": "Detector UID",
            "identity": "Detector / candidate",
            "reference": "Reference tone",
        }[identity]
        fig = make_subplots(3, 1, shared_xaxes=True, vertical_spacing=0.09)
        for panel, key in enumerate(("fr", "Qr", "a_nl"), 1):
            matrix = np.full((len(ids), len(labels)), np.nan)
            positions = {value: i for i, value in enumerate(ids)}
            for record in linked:
                matrix[positions[record[identity]], record["input"]] = record[key]
            states = np.full(matrix.shape, "unavailable", dtype=object)
            for record in linked:
                states[positions[record[identity]], record["input"]] = record["status"]
            if key == "fr":
                for row in matrix:
                    valid = np.flatnonzero(np.isfinite(row))
                    if len(valid):
                        row[:] -= row[valid[0]]
                matrix /= 1e3
            name = {"fr": "Δfr (kHz)", "Qr": "Qr", "a_nl": "a_nl"}[key]
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=labels,
                    y=[str(uid) for uid in ids],
                    customdata=states,
                    name=name,
                    colorscale="Viridis",
                    colorbar={"title": name, "len": 0.24, "y": 1 - (panel - 0.5) / 3},
                    hovertemplate="%{x}<br>"
                    + title
                    + " %{y}<br>%{customdata}<br>"
                    + name
                    + "=%{z:.5g}<extra></extra>",
                ),
                row=panel,
                col=1,
            )
            fig.update_yaxes(title_text=title, type="category", row=panel, col=1)
        fig.update_layout(height=850, margin={"l": 70, "r": 100, "t": 30, "b": 100})
        return fig
