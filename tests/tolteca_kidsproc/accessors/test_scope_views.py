"""Input and analysis views from explicitly selected tree scopes."""

from dataclasses import replace

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from tollan.accessor import NameMapping, SubpathMapping

from tolteca_kidsproc.accessors.kids import KidsAccessor, KidsMapper
from tolteca_kidsproc.accessors.views import S21View, SweepView, TimestreamView
from tolteca_kidsproc.analysis import D21Analysis, PSDAnalysis


@pytest.fixture
def sweep():
    return xr.Dataset(
        {
            "I": ("sample", da.arange(64, chunks=16)),
            "Q": ("sample", da.zeros(64, chunks=16)),
        },
        coords={"frequency": ("sample", np.arange(64) + 1e9, {"units": "Hz"})},
    )


@pytest.fixture
def timestream():
    return xr.Dataset(
        {"I": ("time", np.arange(64.0)), "Q": ("time", np.zeros(64))},
        coords={"time": ("time", np.arange(64) / 100, {"units": "s"})},
        attrs={"f_smp": 100.0},
    )


def test_dataset_and_payload_node_retain_input(sweep):
    for source in (sweep, xr.DataTree(dataset=sweep)):
        accessor = source.kids
        assert accessor.data_source is source
        assert accessor.sweep.mapper is accessor.mapper
        assert accessor.sweep.I.data is sweep.I.data
        assert accessor.sweep.data_source.identical(sweep)
        assert accessor.s21.mapper is accessor.sweep.mapper


def test_accessor_resolves_and_reuses_mapper(sweep, monkeypatch):
    calls = []
    from_data_source = KidsMapper.from_data_source

    def resolve(source):
        calls.append(source)
        return from_data_source(source)

    monkeypatch.setattr(KidsMapper, "from_data_source", staticmethod(resolve))
    accessor = sweep.kids
    assert accessor.data_source is sweep
    assert len(calls) == 1
    assert calls[0] is sweep
    assert accessor.s21.I.data is sweep.I.data
    assert accessor.sweep.I.data is sweep.I.data
    assert len(calls) == 1
    assert calls[0] is sweep


def test_analysis_facet_uses_its_own_mapper_without_requiring_signal_fields(sweep):
    result = D21Analysis(method="gradient")(sweep.kids.sweep)
    accessor = result.kids
    assert accessor.mapper.schema.I not in accessor.mapper
    assert accessor.mapper.schema.Q not in accessor.mapper
    view = accessor.d21
    assert view.mapper is not accessor.mapper
    assert view.d21.shape == sweep.I.shape


def test_analyzed_scope_exposes_input_and_results(sweep, timestream, tmp_path):
    derivative = D21Analysis(method="gradient")
    result = derivative(sweep.kids.sweep)
    assert result.kids.sweep.I.data is sweep.I.data
    before = result.kids.d21.data_source
    assert derivative(result.kids.sweep, output=result) is result
    assert result.kids.d21.data_source is before

    target = tmp_path / "sweep.zarr"
    result.to_zarr(target, zarr_format=2)
    with xr.open_datatree(target, engine="zarr") as reopened:
        assert reopened.kids.sweep.data_source.equals(sweep)
        xr.testing.assert_equal(reopened.kids.d21.d21, result.kids.d21.d21)

    spectra = PSDAnalysis(nperseg=16)(timestream.kids.timestream)
    assert spectra.kids.timestream.I.data is timestream.I.data
    assert spectra.kids.psd.signal_names == ("I", "Q")


def test_block_selection_is_explicit_and_local(sweep, timestream):
    other = sweep.assign(I=sweep.I + 10)
    tree = xr.DataTree.from_dict(
        {
            "blocks/0/sweep": sweep,
            "blocks/0/timestream": timestream,
            "blocks/19/sweep": other,
        }
    )
    for scope in (tree, tree["blocks"]):
        with pytest.raises(ValueError, match="Missing required field"):
            _ = scope.kids.sweep
    for name, expected in (("0", sweep), ("19", other)):
        block = tree[f"blocks/{name}"]
        for scope in (block, block.copy(deep=False)):
            assert scope.kids.sweep.data_source.identical(expected)
            assert scope.kids.sweep.I.data is expected.I.data
    block = tree.children["blocks"].children["0"]
    assert block.kids.s21.I.data is sweep.I.data
    assert block.kids.timestream.I.data is timestream.I.data
    D21Analysis(method="gradient")(block.kids.sweep, output=block)
    assert block.kids.d21.d21.dims == ("sample",)
    assert set(tree["blocks/19"].children) == {"sweep"}


def test_multi_channel_views_select_their_own_input(sweep, timestream):
    sweep = sweep.expand_dims(chan=[0, 1])
    timestream = timestream.expand_dims(chan=[0, 1, 2])
    scope = xr.DataTree.from_dict({"sweep": sweep, "timestream": timestream})
    assert scope.kids.multi_sweep.n_chans == 2
    assert scope.kids.multi_timestream.n_chans == 3
    assert scope.kids.multi_sweep.data_source.identical(sweep)
    assert scope.kids.multi_timestream.data_source.identical(timestream)
    assert scope.kids.multi_sweep.mapper is scope.kids.sweep.mapper
    assert scope.kids.multi_timestream.mapper is scope.kids.timestream.mapper
    assert scope.kids.multi_sweep.data_source is scope.kids.sweep.data_source
    assert scope.kids.multi_timestream.data_source is scope.kids.timestream.data_source


def test_direct_views_use_supplied_payload_without_subtree_selection(sweep, timestream):
    scope = xr.DataTree.from_dict({"sweep": sweep, "timestream": timestream})
    before = scope.copy(deep=False)
    for view_cls, name, expected in (
        (S21View, "sweep", sweep),
        (SweepView, "sweep", sweep),
        (TimestreamView, "timestream", timestream),
    ):
        with pytest.raises(ValueError, match="Missing required field"):
            view_cls(scope)
        view = view_cls(scope[name])
        assert view._data_source is scope[name]
        assert view.data_source.identical(expected)
        assert view.mapper.schema.I in view.mapper
        assert view.I.data is expected.I.data
    xr.testing.assert_identical(scope, before)


def test_direct_payload_view_retains_supplied_mapper(sweep):
    original = KidsMapper.from_data_source(sweep)
    mapped_fields = dict(original.mapped_fields)
    field = original.schema.I
    mapped_fields[field] = replace(mapped_fields[field], name="in_phase")
    mapper = KidsMapper(mapped_fields)
    scope = xr.DataTree.from_dict({"sweep": sweep.rename(I="in_phase")})
    view = SweepView(scope["sweep"], mapper=mapper)
    assert view._data_source is scope["sweep"]
    assert view.mapper is mapper
    assert view.I.data is sweep.I.data


def test_subview_uses_resolved_subpath_in_supplied_mapper(sweep):
    other = sweep.assign(I=sweep.I + 10)
    scope = xr.DataTree.from_dict({"sweep": sweep, "scan": other})
    original = KidsMapper.from_data_source(scope)
    mapped_fields = dict(original.mapped_fields)
    field = original.schema.sweep_subpath
    mapped_fields[field] = replace(mapped_fields[field], name="scan")
    mapper = KidsMapper(mapped_fields)
    view = KidsAccessor(scope, mapper=mapper).sweep
    assert view._data_source is scope["scan"]
    assert view.data_source.identical(other)
    assert view.mapper is not mapper
    assert view.mapper.schema.I in view.mapper


def test_schema_selects_input_path_and_variable_candidates(sweep, monkeypatch):
    schema = replace(
        KidsMapper.schema,
        sweep_subpath=SubpathMapping.path(("missing", "products/scan")),
        I=NameMapping.var(("missing_I", "in_phase")),
    )
    monkeypatch.setattr(KidsMapper, "schema", schema)
    source = sweep.rename(I="in_phase")
    scope = xr.DataTree.from_dict({"products/scan": source})
    assert scope.kids.sweep.data_source.identical(source)
    assert scope.kids.sweep.I.data is sweep.I.data
    assert scope.kids.sweep.mapper.get_name(schema.I) == "in_phase"


def test_selected_input_is_validated_without_trying_another_path(sweep):
    scope = xr.DataTree.from_dict({"sweep": xr.Dataset(), "_input": sweep})
    with pytest.raises(ValueError, match="Missing required field"):
        _ = scope.kids.sweep


def test_explicit_payload_mapper_is_retained(sweep):
    source = sweep.rename(I="in_phase")
    for original_source, data in (
        (sweep, source),
        (xr.DataTree(dataset=sweep), xr.DataTree(dataset=source)),
    ):
        original = KidsMapper.from_data_source(original_source)
        schema = original.schema
        mapped_fields = dict(original.mapped_fields)
        mapped_fields[schema.I] = replace(mapped_fields[schema.I], name="in_phase")
        mapper = KidsMapper(mapped_fields)
        view = KidsAccessor(data, mapper=mapper).sweep
        assert view.mapper is mapper
        assert view.I.data is sweep.I.data
