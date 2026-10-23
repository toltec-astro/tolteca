"""Finding diagnostics reject unavailable inputs before figure construction."""

import pytest
import xarray as xr
from pydantic import ValidationError
from tollan.pipeline import StepFailureKind

from tolteca_kids.config import KidsConfig
from tolteca_kids.kids_find_plot import KidsFindPlot


def test_finding_plot_requires_finding_in_recipe():
    with pytest.raises(ValidationError, match="kids_find_plot requires kids_find"):
        KidsConfig(kids_find_plot={"enabled": True})


def test_finding_plot_records_missing_results():
    _, context = KidsFindPlot()(xr.DataTree(), return_context=True)
    assert not context.completed
    assert context.failures[0].kind is StepFailureKind.missing_input
    assert not context.artifacts
