"""Backend-independent flags in KIDs model reports."""

from enum import IntFlag


class ModelFitFlag(IntFlag):
    """Common kids.fit/1 model-report bits, shared with kidscpp model/flags.h.

    Warnings preserve numerical results. ``fit_unusable`` forbids solving;
    the accompanying bits retain the reason. Zero is not scientific acceptance.
    """

    fit_good = 0
    fit_first_pass_better = 1 << 0
    seed_large_offset = 1 << 1
    seed_not_converged = 1 << 2
    seed_out_of_range = 1 << 3
    seed_qr_out_of_range = 1 << 4
    fit_large_offset = 1 << 5
    fit_not_converged = 1 << 6
    fit_out_of_range = 1 << 7
    fit_qr_out_of_range = 1 << 8
    fit_low_gain = 1 << 9
    fit_insufficient_data = 1 << 11
    fit_invalid_data = 1 << 12
    fit_unusable = 1 << 13
    fit_at_bound = 1 << 14
    fit_rank_deficient = 1 << 15
    fit_condition_high = 1 << 16
    fit_residual_high = 1 << 17
    fit_initialization_fallback = 1 << 18
    fit_no_model = 1 << 19
    fit_selection_required = 1 << 20
    fit_probe_outside_window = 1 << 21
    fit_joint_required = 1 << 22
    fit_recovery_unresolved = 1 << 24
    fit_power_high = 1 << 25
    fit_power_low = 1 << 26
    fit_power_reference_unmatched = 1 << 27
    fit_nonlinear_limit = 1 << 28
    fit_resonance_outside_window = 1 << 29
