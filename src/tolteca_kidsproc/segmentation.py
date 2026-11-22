"""Frequency support and local joint-analysis groups for measured sweeps."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import xarray as xr


def ordered_match_support(allowed):
    """Retain every ordered assignment covering the smaller candidate set.

    Parameters
    ----------
    allowed : array_like of bool, shape (n, m)
        Admissible edges between two strictly frequency-ordered feature lists.
        The caller owns registration, frequency gates and group boundaries.

    Returns
    -------
    tuple of numpy.ndarray
        Feasible edges and forced edges, both shape ``(n, m)``. A forced edge
        occurs in every complete, order-preserving, one-to-one assignment.
        The larger list may have unmatched features. If the smaller list cannot
        be covered, both arrays are false; no partial best assignment is chosen.

    Notes
    -----
    This is conditional correspondence evidence, not physical identity. It
    never ranks edges by distance or ignores a competing complete assignment.
    Unknown identities must remain in the candidate lists. Frequencies that
    cannot be strictly ordered must be rejected by the caller.
    """
    allowed = np.asarray(allowed, dtype=bool)
    dimensions = 2
    if allowed.ndim != dimensions:
        msg = "Expected a two-dimensional candidate matrix"
        raise ValueError(msg)
    n, m = allowed.shape
    if n < m:
        feasible, forced = ordered_match_support(allowed.T)
        return feasible.T, forced.T
    feasible = np.zeros_like(allowed)
    if not m:
        return feasible, feasible.copy()
    forward = np.zeros((n + 1, m + 1), dtype=bool)
    forward[0, 0] = True
    for i in range(n):
        forward[i + 1] = forward[i]
        forward[i + 1, 1:] |= forward[i, :-1] & allowed[i]
    if not forward[n, m]:
        return feasible, feasible.copy()
    backward = np.zeros_like(forward)
    backward[n, m] = True
    for i in range(n - 1, -1, -1):
        backward[i] = backward[i + 1]
        backward[i, :-1] |= backward[i + 1, 1:] & allowed[i]
    feasible = forward[:-1, :-1] & allowed & backward[1:, 1:]
    forced = feasible & (feasible.sum(axis=0)[None, :] == 1)
    return feasible, forced


class _Components:
    def __init__(self, keys):
        self.parents = {key: key for key in keys}
        self.sizes = dict.fromkeys(self.parents, 1)
        self.labels = dict(self.parents)

    def root(self, key):
        while key != self.parents[key]:
            self.parents[key] = self.parents[self.parents[key]]
            key = self.parents[key]
        return key

    def join(self, first, second):
        a, b = self.root(first), self.root(second)
        if a == b:
            return
        if self.sizes[a] < self.sizes[b]:
            a, b = b, a
        self.parents[b] = a
        self.sizes[a] += self.sizes[b]
        self.labels[a] = min(self.labels[a], self.labels[b])

    def label(self, key):
        return self.labels[self.root(key)]


def group_intervals(bounds, groups):
    """Close initial groups over intersecting frequency-support intervals.

    Parameters
    ----------
    bounds : array_like, shape (n, 2)
        Finite inclusive lower and upper frequencies, in the same units.
        Intervals may be asymmetric, nested or in arbitrary order.
    groups : sequence of str
        Initial group labels aligned with the intervals. Existing membership
        is retained; intersecting groups are joined transitively.

    Returns
    -------
    numpy.ndarray of str
        Group per input interval, using the lexicographically smallest original
        label in each component. Input order is preserved. Labels identify local
        computation groups, not physical detectors.

    Raises
    ------
    ValueError
        Bounds are nonfinite, reversed, empty or not aligned with nonempty labels.
    """
    bounds = np.asarray(bounds, dtype=float)
    groups = np.asarray(groups, dtype=str)
    if (
        bounds.shape != (groups.size, 2)
        or groups.ndim != 1
        or np.any(groups == "")
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        msg = "Support needs aligned nonempty group labels and finite increasing bounds"
        raise ValueError(msg)
    components = _Components(groups)
    active, stop = None, -np.inf
    for i in np.argsort(bounds[:, 0], kind="stable"):
        start, end = bounds[i]
        if active is not None and start <= stop:
            components.join(groups[active], groups[i])
            stop = max(stop, end)
        else:
            active, stop = i, end
    return np.asarray([components.label(g) for g in groups], dtype=str)


def validate_segment_ownership(plan):
    """Reject acquired sample intervals owned by different fitting groups.

    Parameters
    ----------
    plan : xarray.Dataset
        One sweep's association rows with ``group_id``, ``chan``,
        ``sample_start`` and exclusive ``sample_stop``. Multiple overlapping
        rows within one group are permitted and select their sample union.

    Raises
    ------
    ValueError
        Different groups select the same channel/sample interval. Join their
        responses in one group before independent fitting.
    """
    by_channel = defaultdict(list)
    for chan, start, stop, group in zip(
        plan.chan.values,
        plan.sample_start.values,
        plan.sample_stop.values,
        plan.group_id.values,
        strict=True,
    ):
        by_channel[chan].append((int(start), int(stop), str(group)))
    for chan, rows in by_channel.items():
        end, owner = -1, ""
        for start, stop, group in sorted(rows):
            if start < end and group != owner:
                msg = (
                    f"Channel {chan} samples overlap between "
                    f"groups {owner} and {group}; "
                    "join their responses in one segment-plan group"
                )
                raise ValueError(msg)
            if start >= end:
                owner = group
            end = max(end, stop)


@dataclass
class CrossSweepPlan:
    """Complete per-sweep groups joined by explicit candidate correspondences.

    ``plans`` maps exact input keys to per-block segment tables. Their ``input``
    attributes retain the workflow's product/interface/block bindings. ``links``
    retains all supplied candidate edges, including ambiguous and unmapped ones.
    ``groups`` maps each local group to its joint-analysis group. These groups
    delimit computation; they do not accept matches or allocate detector UIDs.
    """

    plans: dict[str, xr.Dataset]
    links: xr.Dataset
    groups: xr.Dataset

    def select(self, joint_group_id):
        """Select complete local plans for one joint-analysis group.

        Parameters
        ----------
        joint_group_id : str
            A value from ``groups.joint_group_id``.

        Returns
        -------
        dict[str, xarray.Dataset]
            Per-input segments including every component and covering channel
            of the selected local groups. Original input attributes are retained.
            Unrelated groups are excluded; no scientific arrays are read.

        Raises
        ------
        KeyError
            The joint group does not exist.
        """
        rows = self.groups.where(
            self.groups.joint_group_id == joint_group_id, drop=True
        )
        if not rows.sizes["row"]:
            raise KeyError(joint_group_id)
        result = {}
        for dataset in np.unique(rows.dataset.values):
            local = rows.group_id.values[rows.dataset.values == dataset]
            plan = self.plans[str(dataset)]
            result[str(dataset)] = plan.isel(
                row=np.flatnonzero(np.isin(plan.group_id.values, local))
            )
        return result

    def to_tree(self):
        """Return portable segment, correspondence and membership tables."""
        return xr.DataTree.from_dict(
            {
                "plans": None,
                "groups": self.groups,
                "links": self.links,
                **{f"plans/{key}": value for key, value in self.plans.items()},
            }
        )

    @classmethod
    def from_tree(cls, tree):
        """Load saved tables while their source store remains open."""
        return cls(
            {
                key: node.to_dataset(inherit=False).load()
                for key, node in tree["plans"].children.items()
            },
            tree["links"].to_dataset(inherit=False).load(),
            tree["groups"].to_dataset(inherit=False).load(),
        )


def _sweep_owners(plans):
    nodes, owners = [], {}
    for key, plan in sorted(plans.items()):
        if not key or "/" in key or key in {".", ".."}:
            msg = "Sweep input keys must be nonempty single path components"
            raise ValueError(msg)
        validate_segment_ownership(plan)
        for group, component in zip(
            plan.group_id.values, plan.resonator_id.values, strict=True
        ):
            node = (key, str(group))
            identity = (key, str(component))
            if identity in owners and owners[identity] != node:
                msg = "A component must have one group in its source sweep"
                raise ValueError(msg)
            owners[identity] = node
            nodes.append(node)
    return sorted(set(nodes)), owners


def group_sweep_plans(plans, links):
    """Connect whole sweep groups through supplied component correspondences.

    Parameters
    ----------
    plans : mapping of str to xarray.Dataset
        Segment plans keyed by unique exact input labels. Each component has
        one local group. Every plan retains its original sample indices.
    links : xarray.Dataset
        Row columns ``dataset``, ``resonator_id``, ``dataset_ref`` and
        ``resonator_id_ref``. Multiple edges are candidate alternatives, not
        accepted identities. An empty component ID denotes an explicitly
        unmapped endpoint and does not connect groups. Additional evidence
        columns are retained. Missing nonempty IDs are errors.

    Returns
    -------
    CrossSweepPlan
        Connected local problems for this finite input selection, including
        isolated groups. A blend connects all involved groups without merging
        component IDs. Input tables are copied and remain unchanged.

    Raises
    ------
    ValueError
        Input keys, component ownership, segment ownership or link endpoints
        are invalid. Inputs must belong to one interface; workflows enforce
        their exact binding and measurement provenance before this call.
    """
    nodes, owners = _sweep_owners(plans)
    components = _Components(nodes)
    fields = ("dataset", "resonator_id", "dataset_ref", "resonator_id_ref")
    if any(name not in links or links[name].dims != ("row",) for name in fields):
        msg = f"Correspondences need row columns {fields}"
        raise ValueError(msg)
    for key, rid, ref, rid_ref in zip(*(links[n].values for n in fields), strict=True):
        if key not in plans or ref not in plans:
            msg = "Correspondence refers to an unknown sweep input"
            raise ValueError(msg)
        for endpoint in ((key, rid), (ref, rid_ref)):
            if endpoint[1] and endpoint not in owners:
                msg = f"Correspondence refers to absent component {endpoint}"
                raise ValueError(msg)
        if rid and rid_ref:
            components.join(owners[key, rid], owners[ref, rid_ref])
    roots = {
        root: str(i)
        for i, root in enumerate(sorted({components.label(node) for node in nodes}))
    }
    groups = xr.Dataset(
        {
            "dataset": ("row", np.asarray([n[0] for n in nodes], dtype=str)),
            "group_id": ("row", np.asarray([n[1] for n in nodes], dtype=str)),
            "joint_group_id": (
                "row",
                np.asarray(
                    [roots[components.label(node)] for node in nodes], dtype=str
                ),
            ),
        }
    )
    return CrossSweepPlan(
        {k: v.copy(deep=True) for k, v in plans.items()},
        links.copy(deep=True),
        groups,
    )
