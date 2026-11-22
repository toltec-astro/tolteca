"""A sweep-edge channel is not automatically a simultaneous readout probe."""

from types import SimpleNamespace

import numpy as np
import xarray as xr

from tolteca_kids.model_joint import joint_models


def test_joint_export_keeps_components_and_only_selected_probe_rows():
    """A singlet and doublet share sweep coverage but no readout channel."""
    fr = np.array(
        [
            [666e6, np.nan],
            [666e6, np.nan],
            [667e6, np.nan],
            [667e6, 667.2e6],
            [667e6, 667.2e6],
        ]
    )
    valid = np.isfinite(fr)
    rows = xr.Dataset(
        {
            "component_valid": (("response", "component"), valid),
            "resonator_id": (
                ("response", "component"),
                [["a", ""], ["a", ""], ["b", ""], ["b", "c"], ["b", "c"]],
            ),
            "fr": (("response", "component"), fr),
            "r": (("response", "component"), np.full(fr.shape, 5e-5)),
            "a_nl": (("response", "component"), np.full(fr.shape, 0.2)),
            "c_re": (("response", "component"), valid * 1e-5),
            "c_im": (("response", "component"), np.zeros(fr.shape)),
            "gain_family": ("response", ["affine"] * 5),
            "f_g": ("response", [666e6, 666e6, 667e6, 667e6, 667.2e6]),
            "H0_re": ("response", np.ones(5)),
            "H0_im": ("response", np.zeros(5)),
            "H1_re": ("response", np.zeros(5)),
            "H1_im": ("response", np.zeros(5)),
        },
        coords={
            "group_id": ("response", ["left", "left", "pair", "pair", "pair"]),
            "chan": ("response", [622, 623, 622, 623, 624]),
        },
    )
    props = xr.Dataset(
        {
            "group_id": ("chan", ["left", "pair", "pair"]),
            "fp": ("chan", [666e6, 667e6, 667.2e6]),
        },
        coords={"chan": [622, 623, 624]},
    )
    quality = xr.Dataset(
        {
            "group_id": ("group", ["left", "pair"]),
            "group_flag": ("group", [0, 0]),
            "channel_flag": ("chan", [0, 0, 0]),
        },
        coords={"chan": [622, 623, 624]},
    )
    view = SimpleNamespace(components=lambda: rows, chan_prop=props)
    left, pair = joint_models(view, quality, include_single=True)["groups"]
    assert left["channels"] == [622]
    assert pair["channels"] == [623, 624]
    assert pair["resonator_ids"] == ["b", "c"]
    assert np.asarray(pair["coupling_re"]).shape == (2, 2)
    assert pair["probes"] == [667e6, 667.2e6]
    assert pair["a_nl"] == [0.2, 0.2]
    # Restricting probes cannot silently drop a component seen only at a sweep edge.
    props["group_id"] = ("chan", ["left", "pair", ""])
    rows["component_valid"][3, 1] = False
    [pair] = joint_models(view, quality)["groups"]
    assert pair["channels"] == [623]
    assert pair["resonator_ids"] == ["b", "c"]
    assert pair["coupling_re"][0][1] == 0
