"""Acquisition feedback selects blocks without changing scientific results."""

import numpy as np
import pytest
import xarray as xr
from tollan.accessor.xarray import ensure_datatree

from tolteca_kids import SweepCheck, SweepReduce
from tolteca_kids.quicklook import KidsQuicklook


@pytest.fixture
def tune(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    with xr.open_dataset(path) as native:
        yield native.toltec_kids.select_channels([0, 5, 10, 15])


def test_last_sweep_is_explicit_when_check_missing(tune):
    working = ensure_datatree(tune)
    SweepReduce()(working)
    blocks = list(working.toltec_kids.blocks)
    assert len(blocks) == 2
    SweepCheck()(blocks[0].tree)
    trees = {working.toltec_kids.meta.interface: working}
    last = KidsQuicklook()(trees)
    assert last.document.interfaces[0].blocks == [0]
    assert last.document.interfaces[0].block is None
    assert last.document.interfaces[0].n_flagged is None
    selected = KidsQuicklook(block=0)(trees)
    view = SweepCheck.view(blocks[0].tree)
    assert selected.document.interfaces[0].n_flagged == np.count_nonzero(
        view.mask_chan_bad
    )
    assert selected.document.interfaces[0].n_channels == len(view.mask_chan_bad)
    assert selected.document.acquisitions == [working.toltec_kids.meta.uid_raw_obs]
    assert len(selected.figure.data) == 4


def test_feedback_export_failure_does_not_advance_tag(tune, tmp_path, monkeypatch):
    from tollan.pipeline import ArtifactContext

    from tolteca_kids.product import write_kids_product

    working = ensure_datatree(tune)
    SweepReduce()(working)
    spec = KidsQuicklook(summary={"save": False})
    artifacts = ArtifactContext()
    artifacts.add(spec, spec({working.toltec_kids.meta.interface: working}))
    first = write_kids_product(working, tmp_path, artifacts=artifacts)
    index_before = (first.parent / "index.yaml").read_bytes()

    def fail(*args, **kwargs):
        msg = "Renderer unavailable"
        raise OSError(msg)

    monkeypatch.setattr(KidsQuicklook, "write", fail)
    with pytest.raises(OSError, match="Renderer unavailable"):
        write_kids_product(working, tmp_path, artifacts=artifacts)
    assert (first.parent / "index.yaml").read_bytes() == index_before
    assert (first / "index.yaml").is_file()
