"""PCA controls must not leak flagged values into valid target channels."""

import copy
import importlib
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[2] / "examples/tune_model_comparison")
    )
    return tuple(
        importlib.import_module(name)
        for name in ("pca_trace", "pca_trace_analysis", "pca_flag_projection")
    )


def test_unequal_chunks_use_cumulative_decimated_raw_ranges(modules):
    _, analysis, _ = modules
    raw = np.array([[2, 5, 0, 7], [6, 11, 4, 13], [12, 15, 10, 17]])
    np.testing.assert_array_equal(
        analysis.chunk_bounds(raw, 2, 7), [[0, 2], [2, 5], [5, 7]]
    )
    with pytest.raises(ValueError, match="do not account"):
        analysis.chunk_bounds(raw, 2, 8)


def test_selection_requires_every_sample_valid_without_a_fit_quality_cut(modules):
    trace, _, _ = modules
    apt = Table(
        {
            "array": [0, 0, 0, 0, 1],
            "flag": [0, 0, 0, 1, 0],
            "nw": [0, 0, 0, 0, 7],
            "kids_tone": [4, 5, 6, 7, 0],
        }
    )
    good = {
        "nw": np.array([0, 0, 0, 0, 7]),
        "channel": np.array([4, 5, 6, 7, 0]),
        "finite": np.ones(5, dtype=bool),
        "valid": np.ones(5, dtype=bool),
    }
    other = copy.deepcopy(good)
    other["valid"][1] = False
    other["finite"][2] = False
    selected, ledger = trace.matched_selection(apt, {"legacy": good, "v3": other})
    np.testing.assert_array_equal(selected, [True, False, False, False, False])
    assert len(ledger) == 5
    assert not ledger["v3_all_samples_valid"][1]
    other["channel"][0] = 10
    with pytest.raises(AssertionError):
        trace.matched_selection(apt, {"v3": other})


def test_valid_mean_ignores_stored_flagged_values(modules):
    _, _, projection = modules
    values = np.array([[1.0, 99.0], [3.0, 42.0], [5.0, 10.0]])
    flags = np.array([[False, True], [False, True], [False, False]])
    centered = projection.center_valid(values, flags)
    np.testing.assert_equal(centered[:, 0], [-2, 0, 2])
    assert centered[2, 1] == 0


def test_flagged_samples_cannot_change_valid_least_squares_output(modules):
    _, _, projection = modules
    basis = np.ones((3, 1)) / np.sqrt(3)
    flags = np.array([[True, False, False], [False, False, False]])
    first = np.array([[1e6, 2.0, 2.0], [3.0, 3.0, 3.0]])
    second = first.copy()
    second[0, 0] = -1e9
    a = projection.masked_projection(first, flags, basis)
    b = projection.masked_projection(second, flags, basis)
    assert abs(a[0][0, 1] - b[0][0, 1]) > 1e8  # Reproduces the native leakage.
    np.testing.assert_allclose(a[2][~flags], b[2][~flags], atol=1e-10)
    np.testing.assert_allclose(a[2][~flags], 0, atol=1e-10)


def test_full_valid_projection_equals_native_and_rank_loss_is_explicit(modules):
    _, _, projection = modules
    basis = np.eye(3)[:, :2]
    values = np.arange(12.0, dtype=float).reshape(4, 3)
    flags = np.zeros_like(values, dtype=bool)
    ordinary, zeroed, ls, leak = projection.masked_projection(values, flags, basis)
    np.testing.assert_array_equal(ordinary, ls)
    np.testing.assert_array_equal(ordinary, zeroed)
    np.testing.assert_array_equal(leak, 0)
    flags[:, 0] = True
    with pytest.raises(ValueError, match="Insufficient valid probes"):
        projection.masked_projection(values, flags, basis)


def test_basis_exchange_decomposition_preserves_cross_term(modules):
    _, analysis, _ = modules
    rng = np.random.default_rng(57)
    left = rng.normal(size=(40, 8))
    right = left + 0.2 * rng.normal(size=left.shape)
    basis_a = np.linalg.qr(rng.normal(size=(8, 2)))[0]
    basis_b = np.linalg.qr(rng.normal(size=(8, 2)))[0]
    input_term, basis_term = analysis.difference_terms(left, right, basis_a, basis_b)
    difference = analysis.remove_modes(right, basis_b) - analysis.remove_modes(
        left, basis_a
    )
    np.testing.assert_allclose(input_term + basis_term, difference, atol=1e-14)
    weights = np.arange(1.0, 9.0)
    terms = analysis.weighted_power(input_term, weights) + analysis.weighted_power(
        basis_term, weights
    )
    cross = 2 * np.sum(input_term * basis_term * weights) / (len(left) * weights.sum())
    assert terms + cross == pytest.approx(analysis.weighted_power(difference, weights))
