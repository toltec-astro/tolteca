"""Typed figures, retained artifact settings, and block-owned publication."""

import json
import pickle

import numpy as np
import plotly.io as pio
import pytest
import xarray as xr
from pydantic import Field
from tollan.accessor.xarray import ensure_datatree
from tollan.config import RuntimeContext
from tollan.pipeline import (
    ArtifactContext,
    ArtifactFile,
    ArtifactSpec,
    Step,
    StepConfig,
    StepContext,
)
from tollan.plot.plotly import PlotlyFigShowInDash

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_kids import SweepCheck, SweepCheckPlot, SweepReduce
from tolteca_kids.product import open_kids_product, write_kids_product
from tolteca_kids.runner import run_data
from tolteca_kids.sweep_check_viz import (
    ChannelSummaryViz,
    IQGridViz,
    SweepCheckArtifacts,
)


@pytest.fixture
def checked(target_sweep_path):
    with xr.open_dataset(target_sweep_path) as raw:
        tree = ensure_datatree(raw.toltec_kids.select_channels(range(0, 135, 5)).load())
    SweepReduce()(tree)
    block = tree["blocks/0"]
    SweepCheck()(block)
    return tree, block, SweepCheck.view(block)


def test_configured_viz_and_step_retain_effective_specs(checked, tmp_path, monkeypatch):
    _, block, check = checked
    spec = SweepCheckArtifacts(
        save_formats=["json"],
        i_q_grid={"page_size": 10, "height": 600, "alias": "preview"},
        s21_f_grid={"page_size": 10, "columns": 2},
        chan_summary={"save": False},
    )
    shown = []
    monkeypatch.setattr(
        PlotlyFigShowInDash,
        "__call__",
        lambda self, figures, **kwargs: shown.append(figures),
    )
    _, context = SweepCheckPlot(artifacts=spec, save_path=tmp_path, show=True)(
        block, return_context=True
    )
    assert context.completed and len(context.artifacts) == 4
    entry = context.artifacts[spec.i_q_grid]
    assert entry.spec.save_formats == ["json"]
    assert entry.spec.page_size == 10
    assert entry.spec.show_plotly is spec.show_plotly
    assert entry.data.layout.height == 600
    assert IQGridViz not in context.artifacts  # The configured alias owns this entry.
    assert shown[0]["i_q_grid"] is entry.data
    assert [frame.name for frame in entry.data.frames] == ["0", "10", "20"]
    np.testing.assert_array_equal(
        entry.data.frames[-1].data[0].x, check.sweep.I.values[20]
    )
    assert len(context.files) == 3
    assert {item.format for item in context.files} == {"json"}
    assert json.loads(
        pio.read_json(tmp_path / "i_q_grid.json").to_json()
    ) == json.loads(entry.data.to_json())
    assert not (tmp_path / "chan_summary.json").exists()
    assert (
        "artifacts" not in context.model_dump()
    )  # Live objects are not persisted in attrs.
    assert context.model_dump()["config"]["artifacts"]["i_q_grid"]["page_size"] == 10
    restored = SweepCheckPlot.load_context(SweepCheckPlot.dump_context(context))
    assert not restored.artifacts and restored.config == context.config
    checkpoint = pickle.loads(pickle.dumps(context))  # noqa: S301 -- local checkpoint
    assert json.loads(checkpoint.artifacts[spec.i_q_grid].data.to_json()) == json.loads(
        entry.data.to_json()
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Export must not construct a new figure")

    monkeypatch.setattr(IQGridViz, "__call__", unexpected)
    receipts = entry.spec.write(entry.data, path=tmp_path / "deferred")
    assert [item.path.name for item in receipts] == ["deferred.json"]


class TextSpec(ArtifactSpec[str]):
    """A native text artifact."""

    def write(self, data, *, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        return [ArtifactFile(path, "text")]


class TextArtifacts(ArtifactSpec[ArtifactContext]):
    """A non-figure producer exercising generic product collection."""

    note: TextSpec = Field(default_factory=TextSpec)

    def __call__(self, text):
        artifacts = ArtifactContext()
        artifacts.add(self.note, text)
        return artifacts

    def write(self, data, *, path):
        entry = data[self.note]
        return entry.spec.write(entry.data, path=path / "note.txt")


class NoteConfig(StepConfig):
    artifacts: TextArtifacts = Field(default_factory=TextArtifacts)


class NoteContext(StepContext["Note", NoteConfig]):
    pass


class Note(Step[NoteConfig, NoteContext]):
    @classmethod
    def run(cls, data, context):
        context.artifacts = context.config.artifacts("Checked block")


def test_writer_collects_generic_and_aliased_block_artifacts(
    checked, tmp_path, monkeypatch
):
    tree, block, _ = checked
    spec = SweepCheckArtifacts(save_formats=["json"])
    SweepCheckPlot(artifacts=spec)(block)
    SweepCheckPlot(artifacts=spec, alias="preview")(block)
    Note()(block)
    output = write_kids_product(tree, tmp_path)
    index = DataProdIndex.read(output)
    files = [
        item for item in index.data_items if item.meta["format"] in {"json", "text"}
    ]
    assert len(files) == 9
    assert len({item.source_uri for item in files}) == 9
    assert all("artifacts/toltec0/blocks/0/" in item.source_uri for item in files)
    assert (
        next(
            (output / item.source_uri).read_text()
            for item in files
            if item.meta["format"] == "text"
        )
        == "Checked block"
    )
    with open_kids_product(output) as saved:
        attrs = saved["blocks/0/tolteca_kids"].attrs
        restored = SweepCheckPlot.load_context(
            attrs[SweepCheckPlot.get_context_key(alias="preview")]
        )
        assert restored.config.artifacts == spec
        assert not restored.artifacts

    current = output.parent / "index.yaml"
    before = current.read_bytes()

    def fail(*args, **kwargs):
        message = "export failed"
        raise OSError(message)

    monkeypatch.setattr(TextArtifacts, "write", fail)
    with pytest.raises(OSError, match="export failed"):
        write_kids_product(tree, tmp_path)
    assert current.read_bytes() == before
    assert not list((output.parent / ".staging").iterdir())


def test_acquisition_driver_owns_iteration_and_preserves_block_records(
    data_lmt_path, tmp_path
):
    source = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    with xr.open_dataset(source) as native:
        tree = ensure_datatree(
            native.toltec_kids.select_channels([0, 5, 10, 15]).load()
        )
    runtime = RuntimeContext(
        {
            "kids": {
                "sweep_check": {"enabled": True},
                "sweep_check_plot": {
                    "enabled": True,
                    "artifacts": {"save_formats": ["json"]},
                },
            }
        }
    )
    assert run_data(tree, source, runtime) is tree
    assert not SweepCheckPlot.has_context(tree)
    for block in tree.toltec_kids.blocks:
        assert SweepReduce.get_context(block.tree).completed
        assert (
            SweepCheck.get_context(block.tree).data.data_source
            is block.tree.sweep_check
        )
        context = SweepCheckPlot.get_context(block.tree)
        assert context.completed and len(context.artifacts) == 4
    output = write_kids_product(tree, tmp_path)
    index = DataProdIndex.read(output)
    assert sum(item.meta["format"] == "json" for item in index.data_items) == 8
    with open_kids_product(output) as saved:
        for block in saved.toltec_kids.blocks:
            check = block.sweep_check
            entry = SweepCheckPlot.get_context(
                tree[f"blocks/{block.meta.block}"]
            ).artifacts[ChannelSummaryViz]
            assert ChannelSummaryViz()(check).to_json() == entry.data.to_json()
