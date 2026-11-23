"""Portable handoff must use recorded identities and explicit recipient files."""

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from tolteca_datamodels.toltec.metadata import ToltecRawObsMetadata

SOURCE = (
    Path(__file__).parents[2]
    / "examples/tune_model_comparison/handoff/run_observation.py"
)
SPEC = importlib.util.spec_from_file_location("run_observation", SOURCE)
assert SPEC is not None and SPEC.loader is not None
RECIPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECIPE)


def test_filepaths_are_relative_to_config_not_cwd(tmp_path):
    source = tmp_path / "source.nc"
    source.touch()
    config = [{"filepath": "source.nc", "stubs": [{"filepath": "source.nc"}]}]
    RECIPE.resolve_filepaths(config, tmp_path)
    assert config[0]["filepath"] == str(source)
    assert config[0]["stubs"][0]["filepath"] == str(source)
    with pytest.raises(FileNotFoundError):
        RECIPE.resolve_filepaths({"filepath": "absent.nc"}, tmp_path)


def header(interface, uid, obs_type):
    """Metadata can reuse a calibration from a different observation."""
    obsnum, subobsnum, scannum = map(int, uid.split("-"))
    return ToltecRawObsMetadata.model_construct(
        interface=interface,
        master=None,
        obsnum=obsnum,
        subobsnum=subobsnum,
        scannum=scannum,
        obs_type=obs_type,
    )


def test_calibration_identity_comes_from_header_not_same_obsnum(tmp_path, monkeypatch):
    headers = {
        "tune.nc": header("toltec0", "10-0-1", "tune"),
        "target.nc": header("toltec0", "20-0-2", "timestream"),
    }
    monkeypatch.setattr(
        RECIPE,
        "open_toltec_kids",
        lambda path: nullcontext(
            SimpleNamespace(toltec_kids=SimpleNamespace(meta=headers[Path(path).name]))
        ),
    )
    config = {
        "inputs": [
            {
                "data_items": [
                    {"meta": {"interface": "toltec0"}, "filepath": "target.nc"}
                ]
            }
        ]
    }
    rows, cal_uid, target_uid = RECIPE.acquisition_manifest(
        config, [tmp_path / "tune.nc"], obsnum=20
    )
    assert (cal_uid, target_uid) == ("10-0-1", "20-0-2")
    assert rows[0]["interface"] == "toltec0"
    with pytest.raises(ValueError, match="header"):
        RECIPE.acquisition_manifest(config, [tmp_path / "tune.nc"], obsnum=21)
    with pytest.raises(ValueError, match="More than one"):
        RECIPE.acquisition_manifest(config, [tmp_path / "tune.nc"] * 2)
    config["inputs"][0]["data_items"][0]["meta"]["interface"] = "toltec1"
    with pytest.raises(ValueError, match="header"):
        RECIPE.acquisition_manifest(config, [tmp_path / "tune.nc"])


def test_legacy_label_cannot_be_applied_to_selected_kids_stubs():
    telescope = {
        "meta": {"interface": "lmt"},
        "stubs": [{"filepath": "coords_stub.nc"}],
    }
    kids = {"meta": {"interface": "toltec0"}}
    config = {
        "inputs": [{"data_items": [telescope, kids]}],
        "timestream": {"type": "xs"},
    }
    RECIPE.validate_reference(config, also_legacy=True)
    kids["stubs"] = [{"filepath": "model_stub.nc"}]
    with pytest.raises(ValueError, match="original legacy-model"):
        RECIPE.validate_reference(config, also_legacy=True)
    RECIPE.validate_reference(config, also_legacy=False)


@pytest.mark.parametrize("quantity", ["is", "qs"])
def test_component_rows_require_solved_map_quantity(quantity):
    with pytest.raises(ValueError, match="solved timestream"):
        RECIPE.validate_reference({"timestream": {"type": quantity}})
