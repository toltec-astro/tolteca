"""Live group figures use the same scientific curves as standalone figures."""

import numpy as np
import pytest

from tolteca_kids.model_fit_quality import ModelFitQualityConfig, model_fit_quality
from tolteca_kids.model_fit_viz import ModelFitGroupViz, _group_figure

from .test_model_fit_diagnostics import (
    fitted,  # noqa: F401 -- shared nontrivial grouped fit fixture
)


def test_live_group_preserves_all_pair_curves(fitted):  # noqa: F811 -- pytest fixture injection
    view = fitted.kids.model_fit
    quality = model_fit_quality(view, ModelFitQualityConfig())
    original = _group_figure(
        view.fit, view.samples, quality.isel(group=0), view.joint.isel(group=0)
    )
    live = ModelFitGroupViz()(view, quality, group="pair")
    assert len(live.layout.updatemenus[0].buttons) == 4
    assert len(live.data) == len(original.data)
    for old, new in zip(original.data, live.data, strict=True):
        assert new.type == "scattergl"
        np.testing.assert_array_equal(old.x, new.x)
        np.testing.assert_array_equal(old.y, new.y)
        assert old.visible == new.visible
        assert old.name == new.name


def test_caller_selects_resonator_without_removing_other_pairs(fitted):  # noqa: F811 -- pytest fixture injection
    view = fitted.kids.model_fit
    quality = model_fit_quality(view, ModelFitQualityConfig())
    selected = str(view.fit.resonator_id.values[-1])
    figure = ModelFitGroupViz()(view, quality, group="pair", resonator=selected)
    assert f"resonator {selected}" in figure.layout.title.text
    assert len(figure.layout.updatemenus[0].buttons) == 4
    assert len(figure.data) == len(ModelFitGroupViz()(view, quality, group="pair").data)
    with pytest.raises(ValueError, match="absent from model group"):
        ModelFitGroupViz()(view, quality, group="pair", resonator="missing")
