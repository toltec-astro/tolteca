"""Response evidence must not be replaced by fit labels or numerical success."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def report():
    """Load the example without running its command-line driver."""
    path = Path(__file__).parents[2] / "examples/tune_model_comparison"
    return runpy.run_path(str(path / "network_populations.py"))


def test_identity_follows_original_peak_not_old_model_integer(report):
    """A refit changes component IDs while tentative identity stays tentative."""
    source = [
        {
            "detector": 2,
            "status": "tentative",
            "peaks": "186332",
            "group": "old",
            "candidate": "318",
        }
    ]
    links = pd.DataFrame(
        [
            {
                "peak_id": 186332,
                "group_id": "new",
                "resonator_id": "7",
                "status": "linked",
            },
            {
                "peak_id": 9,
                "group_id": "old",
                "resonator_id": "318",
                "status": "linked",
            },
        ]
    )
    result = report["candidate_links"](source, links)
    assert set(result) == {("new", "7")}
    assert result["new", "7"][0]["status"] == "tentative"
    links.loc[0, "status"] = "unresolved"
    assert report["candidate_links"](source, links) == {}


def test_conflicting_peak_attribution_is_not_resolved_by_frequency(report):
    """A reference component split across two current terms remains unresolved."""
    source = [{"detector": 2, "status": "tentative", "peaks": "10,11"}]
    links = pd.DataFrame(
        [
            {"peak_id": 10, "group_id": "1", "resonator_id": "7", "status": "linked"},
            {"peak_id": 11, "group_id": "1", "resonator_id": "8", "status": "linked"},
        ]
    )
    assert report["candidate_links"](source, links) == {}


def test_high_qr_and_complete_native_record_do_not_qualify_optical_response(report):
    """A quiet bounded dark candidate must not become a qualified active detector."""
    components = pd.DataFrame(
        [
            {
                "group_id": "0",
                "resonator_id": "1",
                "Qr": 27100,
                "a_nl": 1.1,
                "at_bound": False,
                "native_fraction": 1.0,
            },
            {
                "group_id": "2",
                "resonator_id": "3",
                "Qr": 100000,
                "a_nl": 5.49,
                "at_bound": True,
                "native_fraction": 1.0,
            },
        ]
    )
    links = pd.DataFrame(
        [
            {"peak_id": 9, "group_id": "0", "resonator_id": "1", "status": "linked"},
        ]
    )
    candidates = {("2", "3"): [{"apt_uid": 2, "status": "tentative", "peaks": [12]}]}
    rows = report["qualify_components"](components, links, [9], candidates)
    assert rows.response_class.tolist() == ["unknown", "dark_candidate"]
    assert rows.finder_dark_candidate.tolist() == [True, False]
    assert rows.association_status.tolist() == ["model_local_only", "tentative"]
    assert not rows.science_qualified.any()
    assert rows.at_bound.tolist() == [False, True]
    assert rows.native_fraction.tolist() == [1.0, 1.0]


def test_reference_join_rejects_changed_acquired_values(report):
    """Identical peak indices cannot authorize a different sweep's identity join."""
    fields = {
        "sweep": ("I", "Q", "f_chan", "f_lo"),
        "kids_find/d21/peaks": ("idx", "x", "Qr", "height", "bitmask"),
        "kids_find/det_grouped": ("group", "peak_id", "subdet", "x"),
    }
    current = {f"{g}/{k}": np.arange(3.0) for g, names in fields.items() for k in names}
    reference = {key: value.copy() for key, value in current.items()}
    assert len(report["verify_measurements"](current, reference)) == 13
    reference["sweep/I"][0] += 1e-6
    with pytest.raises(ValueError, match="Different acquisition/finding evidence"):
        report["verify_measurements"](current, reference)


def test_isotropic_iq_noise_and_optical_signal_transform_together():
    """Nonlinear output PSD inequality need not lose two-coordinate information."""
    path = (
        Path(__file__).parents[2] / "examples/tune_model_comparison/network_report.py"
    )
    sensitivity = runpy.run_path(str(path))["sensitivity"]
    r, xp, a = 3e-5, 1e-5, 0.9
    b = 2 * a * r**3
    gain, coupling = 1.1 - 0.3j, 3e-6 + 1e-6j
    z = gain * (1 - coupling / (r + 1j * xp))

    def recover(iq):
        w = coupling / (1 - iq / gain)
        return np.array([w.real, w.imag]), np.array(
            [w.real, w.imag - b / (w.real**2 + w.imag**2)]
        )

    step = 1e-7
    prime, optical = [], []
    for direction in (1, 1j):
        plus, minus = recover(z + step * direction), recover(z - step * direction)
        prime.append((plus[0] - minus[0]) / (2 * step))
        optical.append((plus[1] - minus[1]) / (2 * step))
    g_prime, g_opt = np.array(prime).T, np.array(optical).T
    covariance_prime = g_prime @ g_prime.T
    covariance_opt = g_opt @ g_opt.T
    np.testing.assert_allclose(
        covariance_prime / covariance_prime[0, 0], np.eye(2), atol=1e-8
    )
    values = sensitivity(r, xp, b)
    jacobian = np.array([[1, 0], [values["mu"], values["D"]]])
    np.testing.assert_allclose(
        covariance_opt, jacobian @ covariance_prime @ jacobian.T, rtol=1e-8
    )
    np.testing.assert_allclose(
        covariance_opt[1, 1] / covariance_opt[0, 0],
        values["D"] ** 2 + values["mu"] ** 2,
        rtol=1e-8,
    )
    # A synthetic optical response tests the transformation, not a detector class.
    signal_opt = np.array([0.03, 1.0])
    signal_prime = np.linalg.solve(jacobian, signal_opt)
    information_opt = signal_opt @ np.linalg.solve(covariance_opt, signal_opt)
    information_prime = signal_prime @ np.linalg.solve(covariance_prime, signal_prime)
    np.testing.assert_allclose(information_opt, information_prime, rtol=1e-8)
