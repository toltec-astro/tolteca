"""Complete CLI sweep publication, saved replay and production-v2 artifacts."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from typer.testing import CliRunner

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_kids import KidsFind, SweepCheck, TimestreamCheck
from tolteca_kids.cli import app
from tolteca_kids.product import open_sweep_product, read_product_provenance
from tolteca_kids.tlaloc_output import TlalocOutput

from .test_kids_find_persistence import FIGURES, _contract
from .test_sweep_check_plot_parity import _assert_equal, _decode, _figure_contract
from .test_timestream_check import _allow_roundoff

CHECK_FIGURES = ("chan_summary", "s21_f", "s21_f_grid", "i_q_grid", "roach_tone_power")
TONE_FILES = (
    "targ_freqs.dat",
    "targ_power.dat",
    "random_phases.dat",
    "default_targ_masks.dat",
)


@pytest.fixture(
    scope="module",
    params=[(145605, "vnasweep"), (145485, "targsweep"), (145609, "tune")],
)
def published(request, tmp_path_factory):
    """Use the real CLI with every sweep producer enabled in the same invocation."""
    obsnum, suffix = request.param
    data_root = request.getfixturevalue("data_lmt_path")
    source = next((data_root / "toltec/tcs/toltec0").glob(f"*{obsnum}*_{suffix}.nc"))
    root = tmp_path_factory.mktemp(f"workflow-{obsnum}")
    config = {
        "kids": {
            **{
                key: {"enabled": True}
                for key in (
                    "sweep_reduce",
                    "timestream_check",
                    "sweep_check",
                    "kids_find",
                )
            },
            "timestream_check_plot": {
                "enabled": True,
                "artifacts": {"save_formats": ["html", "json"]},
            },
            "sweep_check_plot": {
                "enabled": True,
                "artifacts": {"save_formats": ["html", "json"]},
            },
            "kids_find_plot": {
                "enabled": True,
                "artifacts": {"figure": {"save_formats": ["html", "json"]}},
            },
            "tone_prep": {
                "enabled": True,
                "tone_phases": {"method": "random", "seed": 42},
                "phase_optimization": {"trials": 64, "seed": 7},
            },
            "output": {
                "enabled": True,
                "path": str(root / "products"),
                "namespace": "sweep-qa",
            },
        }
    }
    # JSON is YAML; retain the exact invocation config next to its products.
    config_path = root / "workflow.yaml"
    config_path.write_text(json.dumps(config, indent=2))
    result = CliRunner().invoke(
        app, ["--config_paths", str(config_path), "--", str(source)]
    )
    (root / "cli.log").write_text(result.output)
    assert result.exit_code == 0, result.exception
    stores = list((root / "products").rglob("array.zarr"))
    assert len(stores) == 1
    return source, root, stores[0].parent


def _artifact(product, block, filename):
    matches = list(product.glob(f"artifacts/toltec0/blocks/{block}/**/{filename}"))
    assert len(matches) == 1, (block, filename, matches)
    return matches[0]


def test_complete_cli_publication_and_saved_tone_installation(published, monkeypatch):
    source, root, product = published
    count = 2 if "tune.nc" in source.name else 1
    config = read_product_provenance(product).inputs["toltec0"].config
    assert config is not None
    assert config.timestream_check.psd.nperseg == 512
    assert config.sweep_check.readout_power is not None
    assert config.tone_prep.phase_optimization is not None
    assert config.tone_prep.phase_optimization.trials == 64
    index = DataProdIndex.read(product)
    indexed = {
        item.source.path.as_posix() for item in index.product.data_items if item.source
    }
    files = {
        p.relative_to(product).as_posix()
        for p in (product / "artifacts").rglob("*")
        if p.is_file()
    }
    assert files <= indexed
    assert len(files) == count * (13 * 2 + 3 + len(TONE_FILES))

    def forbid_computation(*args, **kwargs):
        pytest.fail("Reopening a publication must not repeat scientific computation")

    for cls in (KidsFind, SweepCheck, TimestreamCheck):
        monkeypatch.setattr(cls, "run", forbid_computation)
    with open_sweep_product(product) as saved:
        assert len(saved.toltec_kids.blocks) == count
        for block_id, block in enumerate(saved.toltec_kids.blocks):
            scope = block.tree
            assert {
                "sweep",
                "sweep_check",
                "timestream_check",
                "kids_find",
                "tlaloc",
            } <= scope.children.keys()
            check = TimestreamCheck.view(scope)
            assert check.psd.f_psd.size == 257
            assert check.psd.psd_analysis == check.config.psd
            for name in (*CHECK_FIGURES, *FIGURES, "noise_psd"):
                html = _artifact(product, block_id, f"{name}.html").read_text()
                assert "Plotly.newPlot" in html
                assert 'src="https://cdn.plot.ly/plotly-' not in html
            for name, table in KidsFind.view(scope).export_tables().items():
                actual = Table.read(_artifact(product, block_id, f"{name}.ecsv"))
                for column in actual.colnames:
                    np.testing.assert_array_equal(
                        actual[column], table[column], err_msg=column
                    )
            replay = root / f"install-block-{block_id}"
            network = replay / "toltec0"
            network.mkdir(parents=True)
            TlalocOutput(enabled=True, path=replay)(scope)
            np.testing.assert_array_equal(
                np.loadtxt(network / "targ_power.dat"), scope["tlaloc"].power_target_db
            )
            for name in TONE_FILES:
                assert (network / name).read_bytes() == _artifact(
                    product, block_id, name
                ).read_bytes()


def _reference(script, source, destination, block, *extra):
    interpreter = os.environ.get("TOLTECA_V2_PYTHON")
    if not interpreter:
        pytest.skip("Set TOLTECA_V2_PYTHON for production-v2 comparison")
    subprocess.run(  # noqa: S603 -- explicitly selected local production interpreter
        [
            interpreter,
            str(Path(__file__).with_name(script)),
            str(source),
            str(destination),
            str(block),
            *extra,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )


def _assert_plots(product, block, refs):
    for name in CHECK_FIGURES:
        actual = json.loads(_artifact(product, block, f"{name}.json").read_text())
        expected = json.loads((refs / f"{name}.json").read_text())
        if name == "roach_tone_power":
            actual, expected = _contract(actual), _contract(expected)
            for a, e in zip(actual["data"], expected["data"], strict=True):
                if e.get("xaxis") == "x2":
                    assert a["type"] == "scattergl" and e["type"] == "scatter"
                    a["type"] = "scatter"
        else:
            actual, expected = _figure_contract(actual), _figure_contract(expected)
            if name == "chan_summary":
                for a, e in zip(actual["data"], expected["data"], strict=True):
                    if "x" not in e:
                        np.testing.assert_array_equal(
                            a.pop("x"), np.arange(e["z"].shape[-1])
                        )
        _assert_equal(actual, expected, name)
    for name in FIGURES:
        actual = _contract(
            json.loads(_artifact(product, block, f"{name}.json").read_text())
        )
        expected = _contract(
            json.loads((refs / "finding" / f"{name}.json").read_text())
        )
        _assert_equal(actual, expected, name)
    _assert_noise_plot(product, block, refs)


def _assert_noise_plot(product, block, refs):
    actual = _decode(
        json.loads(_artifact(product, block, "noise_psd.json").read_text())
    )
    expected = _decode(json.loads((refs / "v2.json").read_text()))
    for a, e in zip(actual["data"], expected["data"], strict=True):
        if a["type"] == "scattergl":
            assert e["type"] == "scatter"
            a["type"] = "scatter"
    for obj in (actual, expected):
        template = obj["layout"].pop("template")
        obj["layout"]["colorway"] = template["layout"]["colorway"]
    _allow_roundoff(actual, expected)
    _assert_equal(actual, expected, "noise_psd")


def test_cumulative_saved_results_and_all_figures_equal_v2(published):
    source, root, product = published
    with open_sweep_product(product) as saved:
        for block_id, block in enumerate(saved.toltec_kids.blocks):
            refs = root / f"v2-block-{block_id}"
            refs.mkdir()
            for script, destination, extra in (
                ("v2_sweep_reference.py", refs / "reduced.npz", ()),
                ("v2_sweep_check_reference.py", refs / "check.npz", ("true",)),
                ("v2_sweep_check_plot_reference.py", refs, ()),
                ("v2_kids_find_plot_reference.py", refs / "finding", ("chan",)),
                ("v2_tlaloc_reference.py", refs / "tones", ()),
                ("v2_noise_psd_reference.py", refs, ()),
            ):
                _reference(script, source, destination, block_id, *extra)
            _assert_scientific(block, refs)
            for name in ("kids_find", "chan_prop"):
                actual = Table.read(_artifact(product, block_id, f"{name}.ecsv"))
                expected = Table.read(refs / "tones" / f"{name}.ecsv")
                assert actual.colnames == expected.colnames
                for column in expected.colnames:
                    np.testing.assert_array_equal(
                        actual[column], expected[column], err_msg=column
                    )
                    assert actual[column].unit == expected[column].unit
            _assert_tone_files(product, block_id, refs, block)
            _assert_plots(product, block_id, refs)


def _assert_scientific(block, refs):
    scope = block.tree
    sweep = block.sweep
    with np.load(refs / "reduced.npz") as expected:
        for name in (
            "I",
            "Q",
            "unc_I",
            "unc_Q",
            "f_chan",
            "sweep",
            "frequency",
        ):
            np.testing.assert_array_equal(
                getattr(sweep, name), expected[name], err_msg=name
            )
    check = SweepCheck.view(scope)
    with np.load(refs / "check.npz") as expected:
        for name in expected.files:
            if name.endswith("_unit") or name in {"S21_checked", "aS21_unc_db"}:
                continue
            actual = getattr(check, name)
            if name.endswith("_slices"):
                actual = [(part.start, part.stop) for part in actual]
            np.testing.assert_array_equal(
                getattr(actual, "value", actual), expected[name], err_msg=name
            )
        np.testing.assert_array_equal(
            check.sweep.I + 1j * check.sweep.Q, expected["S21_checked"]
        )
    with np.load(refs / "check.power.npz") as expected:
        for name in expected.files:
            np.testing.assert_array_equal(
                getattr(check.readout_power, name), expected[name], err_msg=name
            )


def _assert_tone_files(product, block_id, refs, block):
    for name in TONE_FILES:
        if name == "random_phases.dat":
            assert np.isfinite(np.loadtxt(_artifact(product, block_id, name))).all()
            continue
        if name == "targ_power.dat":
            actual = np.loadtxt(_artifact(product, block_id, name))
            expected = np.loadtxt(refs / "tones/toltec0/default_targ_amps.dat")
            amplitudes = np.zeros_like(actual)
            active = np.isfinite(actual)
            amplitudes[active] = 10 ** ((actual[active] - np.max(actual[active])) / 20)
            np.testing.assert_allclose(amplitudes, expected / np.max(expected))
            continue
        if name == "targ_freqs.dat":
            actual = Table.read(_artifact(product, block_id, name), format="ascii.ecsv")
            expected = Table.read(refs / "tones/toltec0" / name, format="ascii.ecsv")
            assert actual.colnames == expected.colnames
            for column in expected.colnames:
                np.testing.assert_array_equal(
                    actual[column], expected[column], err_msg=column
                )
        else:
            np.testing.assert_array_equal(
                np.loadtxt(_artifact(product, block_id, name)),
                np.loadtxt(refs / "tones/toltec0" / name),
            )
