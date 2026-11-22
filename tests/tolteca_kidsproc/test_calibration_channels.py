"""Fixed-probe routing is distinct from sweep sample membership."""

import numpy as np
import xarray as xr

from tolteca_kidsproc.calibration import calibration_channels


def fit_rows():
    """Two groups share edge samples, but have distinct actual probes."""
    return xr.Dataset(
        {
            "group_id": (
                "row",
                ["left", "left", "pair", "pair", "pair", "pair", "pair"],
            ),
            "chan": ("row", [622, 623, 622, 623, 623, 624, 624]),
            "resonator_id": ("row", ["a", "a", "b", "b", "c", "b", "c"]),
            "f_min": ("row", [100.0, 110.0, 170.0, 170.0, 170.0, 210.0, 210.0]),
            "f_max": ("row", [150.0, 120.0, 180.0, 240.0, 240.0, 280.0, 280.0]),
            "fit_status": ("row", ["optimizer_converged"] * 7),
        }
    )


def test_probe_support_resolves_sweep_overlap_without_changing_fits():
    fit = fit_rows()
    before = fit.copy(deep=True)
    props = calibration_channels(
        fit, [622, 623, 624, 700], [125.0, 200.0, 250.0, 300.0]
    )
    assert props.group_id.values.tolist() == ["left", "pair", "pair", ""]
    assert props.recovery_reason.values.tolist() == [
        "fixed_probe_recovery_not_assessed"
    ] * 3 + ["no_fit"]
    xr.testing.assert_identical(fit, before)


def test_overlap_at_actual_probe_stays_ambiguous_even_if_one_fit_failed():
    fit = fit_rows()
    fit["f_max"][1] = 220.0
    fit["fit_status"][1] = "optimizer_failed"
    props = calibration_channels(fit, [622, 623, 624], [125.0, 200.0, 250.0])
    assert props.sel(chan=623).group_id.item() == ""
    assert (
        props.sel(chan=623).recovery_reason.item()
        == "multiple_group_responses_require_selection"
    )
    # Losing one simultaneous probe makes the doublet unresolved, not a singlet.
    assert (
        props.sel(chan=624).recovery_reason.item()
        == "insufficient_simultaneous_measurements"
    )


def test_unbound_and_out_of_range_probes_do_not_guess_a_group():
    props = calibration_channels(fit_rows(), [622, 623, 624], [np.nan, 300.0, 250.0])
    assert props.sel(chan=622).group_id.item() == ""
    assert props.sel(chan=623).recovery_reason.item() == "probe_outside_fit_window"
    assert props.sel(chan=624).recovery_mode.item() == "unresolved"
