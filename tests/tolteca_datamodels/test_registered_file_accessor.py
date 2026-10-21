"""Registered file-table population across independent instruments."""

from contextlib import ExitStack
from enum import Flag
from itertools import permutations
from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
import pytest
import xarray as xr
from astropy import units as u
from astropy.table import QTable
from astropy.time import Time
from pydantic import ValidationError
from tollan.accessor.pandas.model_table import (
    ModelColumn,
    RecordRuleMapper,
)

from tolteca_datamodels.core import data_model as data_model_registry
from tolteca_datamodels.core.data_model import DataModelSpec
from tolteca_datamodels.core.file import source_trace_issues
from tolteca_datamodels.core.file._capabilities import _FileCapabilities
from tolteca_datamodels.core.obs_spec import ObsSpecContext
from tolteca_datamodels.core.source import (
    SourceOpener,
    SourceReaderRules,
)
from tolteca_datamodels.core.types import DataModelInfo
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.tel import LmtTelIOMapper
from tolteca_datamodels.lmt.types import LmtDataModel
from tolteca_datamodels.lmt_base.filepath import FilepathInfoMixin
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_dummy_instru.file import DummyInstruFileAccessor
from tolteca_dummy_instru.metadata import DummyInstruMetadata
from tolteca_dummy_instru.testing import make_dataset
from tolteca_dummy_instru.types import DummyInstruDataKind


class _TestDataModel(DataModelInfo):
    name = "test"


class _FirstDataModel(DataModelInfo):
    name = "first"


class _SecondDataModel(DataModelInfo):
    name = "second"


class _ConflictingFlagsDataModel(DataModelInfo):
    name = "conflicting_flags"


class _ConflictingLabelsDataModel(DataModelInfo):
    name = "conflicting_labels"


class _TableDataModel(DataModelInfo):
    name = "table"


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(data_model_registry._data_models, "_entries", {})
    for cls in (LmtFileAccessor, ToltecFileAccessor, DummyInstruFileAccessor):
        DataModelSpec(file_accessor_cls=cls).register()


@pytest.mark.parametrize("invalid", [object(), str, None])
def test_data_model_validates_accessor_before_registration(invalid):
    previous = dict(data_model_registry._data_models)
    with pytest.raises(ValidationError, match="subclass of FileAccessor"):
        DataModelSpec(file_accessor_cls=invalid)
    assert data_model_registry._data_models == previous


def test_base_accessor_is_not_concrete_support():
    with pytest.raises(ValueError, match="concrete"):
        DataModelSpec(file_accessor_cls=FileAccessor)


def test_registration_name_cannot_diverge_from_accessor_definition():
    before = dict(data_model_registry._data_models)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DataModelSpec.model_validate(
            {"name": "other", "file_accessor_cls": ToltecFileAccessor}
        )
    assert data_model_registry._data_models == before


def test_handlers_do_not_widen_concrete_accessor():
    with pytest.raises(ValueError, match="Concrete file accessors"):
        ToltecFileAccessor.from_sources([], accessor_types=(DummyInstruFileAccessor,))


@pytest.fixture
def paths(data_lmt_path, dummy_tel_path, tmp_path):
    dummy = tmp_path / "chan01-tcs-1-0-2-20251031T120000.nc"
    metadata = DummyInstruMetadata(
        data_models=["dummy_instru"],
        instru="dummy_instru",
        interface="chan01",
        interface_kind="counter",
        channel=1,
        master="tcs",
        obsnum=1,
        subobsnum=0,
        scannum=2,
        filepath_orig=str(dummy),
        t0=Time(1761912000.0, format="unix"),
        t1=Time(1761912003.0, format="unix"),
        t_int=3 * u.s,
    )
    make_dataset(metadata, [1, 2]).to_netcdf(dummy)
    return [
        data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        next((data_lmt_path / "toltec/tcs/toltec0").glob("*145605*_vnasweep.nc")),
        dummy,
        dummy_tel_path,
        "archive:/unidentified.nc",
    ]


def test_scoped_enrichment_keeps_generic_filename_information(paths, monkeypatch):
    parsed = []
    parse = FilepathInfo.from_source

    def trace(source, *, raise_on_error=True):
        parsed.append(source)
        return parse(source, raise_on_error=raise_on_error)

    monkeypatch.setattr(FilepathInfo, "from_source", trace)
    handlers = (ToltecFileAccessor,)
    table = FileAccessor.from_sources(paths, accessor_types=handlers)
    assert parsed == table.source_loc.tolist()
    assert table.interface.iloc[:4].tolist() == ["tel", "toltec0", "chan01", "tel"]
    assert table.data_models.iloc[:4].tolist() == [
        ["lmt", "toltec"],
        ["toltec"],
        ["dummy_instru"],
        ["lmt", "dummy_instru"],
    ]
    assert table.uid_raw_obs.iloc[2:].isna().all()
    assert "data_kind_dummy_instru" not in table
    assert table.data_kind_toltec.iloc[2:].eq(ToltecDataKind.unknown.value).all()
    assert FileAccessor(table, accessor_types=handlers).query(
        "bits_any_set(data_kind_toltec, ToltecDataKind.raw_sweep)"
    ).index.tolist() == [1]
    assert FileAccessor(table, accessor_types=handlers).query_obs_spec("*/1").empty


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
@pytest.mark.parametrize(
    ("handlers", "supported"),
    [
        ((LmtFileAccessor,), {0, 3}),
        ((ToltecFileAccessor,), {0, 1}),
        ((DummyInstruFileAccessor,), {2, 3}),
        ((ToltecFileAccessor, DummyInstruFileAccessor), {0, 1, 2, 3}),
        ((), set()),
    ],
)
def test_generic_factory_reads_only_selected_filename_scopes(
    paths, handlers, supported, mode, monkeypatch
):
    attempted = []

    def read_metadata(self, rows, *, trace):
        attempted.extend(position for position, _ in rows)

    monkeypatch.setattr(FileAccessor, "_read_metadata", read_metadata)
    table = FileAccessor.from_sources(paths, accessor_types=handlers, mode=mode)
    expected = {
        "guess": [],
        "guess_or_read": [4],
        "read": sorted(supported | {4}),
    }
    assert attempted == expected[mode]
    assert table.source.tolist() == paths
    assert table.file_prefix.iloc[:4].notna().all()


@pytest.mark.parametrize("operation", ["build", "read", "open"])
def test_selected_readers_stay_scoped_through_table_copies(
    paths, monkeypatch, operation
):
    def unexpected(data):
        pytest.fail("Unselected data-model identifier ran")

    monkeypatch.setattr(
        DummyInstruFileAccessor,
        "_reader_rules",
        SourceReaderRules(
            rules=DummyInstruFileAccessor._reader_rules.rules,
            prepare_record=unexpected,
        ),
    )
    handlers = (ToltecFileAccessor,)
    table = FileAccessor.from_sources(paths, accessor_types=handlers)
    accessor = FileAccessor(table, accessor_types=handlers)

    def check(enriched):
        assert "data_kind_dummy_instru" not in enriched
        assert enriched.t0_mjd.notna().tolist() == [True, True, False, False, False]
        assert enriched.file_data_models.iloc[2] == ["dummy_instru"]
        assert enriched.data_kind_toltec.iloc[2:].eq(ToltecDataKind.unknown.value).all()

    if operation == "build":
        check(FileAccessor.from_sources(paths, mode="read", accessor_types=handlers))
    elif operation == "read":
        check(accessor.update_from_source_metadata())
    else:
        with accessor.open() as opened:
            FileAccessor(opened, accessor_types=handlers).update_from_source_metadata(
                inplace=True
            )
            check(opened)
            assert opened.io_obj.notna().tolist() == [True, True, False, False, False]
        assert "io_obj" not in opened
    assert "t0_mjd" not in table


def test_no_handlers_still_projects_registered_filenames(paths):
    handlers = ()
    table = FileAccessor.from_sources(paths, accessor_types=handlers)
    assert table.interface.iloc[:4].tolist() == ["tel", "toltec0", "chan01", "tel"]
    assert "data_kind_toltec" not in table
    assert "data_kind_dummy_instru" not in table
    assert "uid_raw_obs" not in table


@pytest.mark.parametrize("accessor_cls", [FileAccessor, ToltecFileAccessor])
@pytest.mark.parametrize("mode", ["guess", "read"])
def test_membership_queries_include_companions_without_other_instruments(
    paths, accessor_cls, mode
):
    table = accessor_cls.from_sources(paths, mode=mode)
    table.index = ["same", "same", "dummy", "dummy", "unknown"]
    accessor = accessor_cls(table)
    for name, positions in (
        ("toltec", [0, 1]),
        ("lmt", [0, 3]),
        ("dummy_instru", [2, 3]),
    ):
        pd.testing.assert_frame_equal(
            accessor.query(f'contains(data_models, "{name}")'), table.iloc[positions]
        )
    pd.testing.assert_frame_equal(
        accessor.query('contains(data_models, "toltec") and obsnum > 145600'),
        table.iloc[[0, 1]],
    )
    pd.testing.assert_frame_equal(
        accessor.query(
            'contains(data_models, "toltec") & '
            f"bits_any_set(data_kind_toltec, {ToltecDataKind.raw_sweep.value})"
        ),
        table.iloc[[1]],
    )


@pytest.mark.parametrize("memberships", [["lmt", "toltec"], ["toltec", "lmt"]])
def test_memberships_control_scope_independently_of_filename_class(
    memberships, paths, monkeypatch
):
    class OtherFilepathInfo(FilepathInfo, FilepathInfoMixin):
        pass

    info = FilepathInfo.from_source(paths[0])
    assert info is not None
    other = OtherFilepathInfo.model_validate(
        info.model_dump(round_trip=True) | {"data_models": memberships}
    )
    monkeypatch.setattr(FilepathInfo, "from_source", lambda *args, **kwargs: other)
    attempts = []

    def read(self, rows, *, trace):
        attempts.extend(position for position, row in rows)

    monkeypatch.setattr(FileAccessor, "_read_metadata", read)
    table = ToltecFileAccessor.from_sources(paths[:1], mode="read")
    assert attempts == [0]
    assert table.uid_raw_obs.tolist() == ["tcs-145609-0-2"]
    assert table.data_kind_toltec.tolist() == [ToltecDataKind.lmt_tel.value]
    assert table.data_models.tolist() == [memberships]
    attempts.clear()
    DummyInstruFileAccessor.from_sources(paths[:1], mode="read")
    assert attempts == []


def test_format_openers_use_filename_columns_without_enrichment_handlers(
    paths, monkeypatch
):
    from tolteca_datamodels.core.source import opener as source_module

    def open_sweep(path):
        pytest.fail("Opener selection must not open the source")

    monkeypatch.setattr(
        FileAccessor,
        "_source_opener",
        SourceOpener(
            rules=(
                ({"suffix": "vnasweep", "ext": "nc"}, open_sweep),
                *source_module.source_opener.rules,
            )
        ),
    )
    handlers = ()
    table = FileAccessor.from_sources(paths, accessor_types=handlers)
    assert FileAccessor(table, accessor_types=handlers)._get_opener_choices()[1] == {
        FileAccessor: open_sweep
    }


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_generic_modes_parse_once_and_read_metadata_once(paths, mode, monkeypatch):
    parsed, read = [], []
    parse = FilepathInfo.from_source
    metadata = LmtTelIOMapper.get_metadata

    def trace_parse(source, **kwargs):
        parsed.append(source)
        return parse(source, **kwargs)

    def trace_metadata(self, data_source):
        read.append(data_source)
        return metadata(self, data_source)

    monkeypatch.setattr(FilepathInfo, "from_source", trace_parse)
    monkeypatch.setattr(LmtTelIOMapper, "get_metadata", trace_metadata)
    table = FileAccessor.from_sources(iter(paths), mode=mode)
    # Header metadata can parse filepath_orig separately; filename-only work
    # parses each source_loc once, never once per registered accessor.
    if mode != "read":
        assert parsed == table.source_loc.tolist()
    assert len(read) == (2 if mode == "read" else 0)
    assert table.source.tolist() == paths
    assert "data_kind" not in table
    assert table.data_kind_toltec.tolist() == [
        ToltecDataKind.lmt_tel.value,
        ToltecDataKind.vnasweep.value,
        *[ToltecDataKind.unknown.value] * 3,
    ]
    assert table.data_kind_dummy_instru.tolist() == [
        *[DummyInstruDataKind.unknown.value] * 2,
        DummyInstruDataKind.counts.value,
        DummyInstruDataKind.telescope.value,
        DummyInstruDataKind.unknown.value,
    ]
    accessor = FileAccessor(table)
    assert accessor.query(
        "bits_any_set(data_kind_toltec, ToltecDataKind.raw_sweep)"
    ).index.tolist() == [1]
    assert accessor.query(
        "bits_any_set(data_kind_dummy_instru, DummyInstruDataKind.counts)"
    ).index.tolist() == [2]
    assert accessor.query_obs_spec("*/1").interface.tolist() == ["chan01"]
    assert table.source_trace.map(source_trace_issues).iloc[4][0].stage == "filepath"


def test_reader_failure_preserves_guesses_and_does_not_try_foreign_reader(paths):
    with xr.open_dataset(paths[2]) as data:
        sources = [{"source": data, "source_loc": paths[1]}]
        guessed = FileAccessor.from_sources(sources)
        assert guessed.data_kind_toltec.iloc[0] == ToltecDataKind.vnasweep.value
        assert guessed.roach.notna().all()
        failed = FileAccessor(guessed).update_from_source_metadata()
        pd.testing.assert_frame_equal(
            failed.drop(columns="source_trace"), guessed.drop(columns="source_trace")
        )
        assert (
            failed.source_trace.map(source_trace_issues).iloc[0][-1].stage == "metadata"
        )
        assert not failed.metadata_read.iloc[0]
        assert not data["counts"].variable._in_memory
        assert data._close is not None


def test_header_membership_correction_controls_subsequent_scoped_open(paths):
    # Same telescope layout, but its header identifies the other supported realm.
    with xr.open_dataset(paths[3]) as data:
        guessed = FileAccessor.from_sources([{"source": data, "source_loc": paths[0]}])
        assert guessed.data_models.tolist() == [["lmt", "toltec"]]
        corrected = FileAccessor(guessed).update_from_source_metadata(on_error="raise")
        assert corrected.data_models.tolist() == [["lmt", "dummy_instru"]]
        assert corrected.data_kind_toltec.tolist() == [ToltecDataKind.unknown.value]
        assert corrected.data_kind_dummy_instru.tolist() == [
            DummyInstruDataKind.telescope.value
        ]
        provenance = [name for name in guessed if str(name).startswith("file_")]
        pd.testing.assert_frame_equal(corrected[provenance], guessed[provenance])
        with ToltecFileAccessor(corrected).open(on_error="raise") as foreign:
            assert foreign.io_obj.iloc[0] is None
            assert foreign.accessor_obj.iloc[0] is None
        with DummyInstruFileAccessor(corrected).open(on_error="raise") as own:
            assert own.io_obj.iloc[0] is data
            assert isinstance(own.accessor_obj.iloc[0], type(data.lmt_tel))
            assert own.accessor_obj.iloc[0].data_source is data
        assert data._close is not None


def test_unregistered_receiver_keeps_general_telescope_metadata(paths):
    with xr.open_dataset(paths[0]) as original:
        data = original.assign({"Header.Dcs.Receiver": "other"})
        table = FileAccessor.from_sources(
            [{"source": data, "source_loc": paths[0]}], mode="read"
        )
        assert table.data_models.tolist() == [["lmt", "other"]]
        assert "receiver" not in table
        assert table.instru.tolist() == ["lmt"]
        assert table.data_kind_toltec.eq(ToltecDataKind.unknown.value).all()
        assert table.data_kind_dummy_instru.eq(DummyInstruDataKind.unknown.value).all()


@pytest.mark.parametrize(
    "accessor_cls",
    [FileAccessor, LmtFileAccessor, ToltecFileAccessor, DummyInstruFileAccessor],
)
def test_opened_scope_does_not_depend_on_filename(accessor_cls, paths, data_lmt_path):
    hwpr = next((data_lmt_path / "toltec/tcs/hwpr").glob("*.nc"))
    files = [paths[0], paths[1], hwpr, paths[2], paths[3]]
    expected = {
        FileAccessor: [True, True, True, True, True],
        LmtFileAccessor: [True, False, False, False, True],
        ToltecFileAccessor: [True, True, True, False, False],
        DummyInstruFileAccessor: [False, False, False, True, True],
    }[accessor_cls]
    with ExitStack() as stack:
        objects = [stack.enter_context(xr.open_dataset(path)) for path in files]
        table = accessor_cls.from_sources(
            [{"source": obj, "source_loc": "renamed.nc"} for obj in objects]
        )
        with accessor_cls(table).open() as opened:
            assert [obj is not None for obj in opened.accessor_obj] == expected
            assert opened.io_obj.notna().all()
            accessor_cls(opened).update_from_source_metadata(inplace=True)
            assert opened.obsnum.notna().tolist() == expected
            for obj, supported in zip(objects, expected, strict=True):
                if supported and "Data.TelescopeBackend.TelTime" in obj:
                    assert obj.lmt_tel.meta.instru == "lmt"
        assert all(obj._close is not None for obj in objects)


@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("operation", ["read", "open"])
def test_selected_telescope_reader_failure_never_falls_back(
    paths, monkeypatch, generic, operation
):
    class TelescopeFiles(FileAccessor[LmtDataModel]):
        _reader_rules = LmtFileAccessor._reader_rules
        _metadata_columns = LmtFileAccessor._metadata_columns

    cls = FileAccessor if generic else TelescopeFiles
    with xr.open_dataset(paths[0]) as original:
        corrupt = original.drop_vars("Header.Source.Ra")
        table = cls.from_sources(
            [{"source": obj, "source_loc": "renamed.nc"} for obj in (corrupt, original)]
        )

        def check(result):
            assert result.obsnum.isna().tolist() == [True, False]
            failure = result.source_trace.map(source_trace_issues).iloc[0][-1]
            assert failure.stage == "metadata"
            assert failure.exception_type == "KeyError"
            assert "Header.Source.Ra" in failure.message
            assert [
                item.stage
                for item in result.source_trace.map(source_trace_issues).iloc[1]
            ] == ["filepath"]

        if operation == "read":
            check(cls(table).update_from_source_metadata())
        else:
            with cls(table).open() as opened:
                cls(opened).update_from_source_metadata(inplace=True)
                check(opened)
                assert opened.io_obj.iloc[0] is corrupt
                assert opened.io_obj.iloc[1] is original


def test_unrelated_reader_overlap_raises_before_reading(paths):
    class Reader:
        def __init__(self, data):
            pytest.fail("Ambiguous source was bound")

    class OtherReader(Reader):
        pass

    class FirstFiles(FileAccessor[LmtDataModel]):
        _reader_rules = SourceReaderRules(
            rules=(({}, Reader),), prepare_record=lambda obj: obj
        )

    class SecondFiles(FileAccessor[LmtDataModel]):
        _reader_rules = SourceReaderRules(
            rules=(({}, OtherReader),), prepare_record=lambda obj: obj
        )

    with pytest.raises(RuntimeError, match="Ambiguous source readers"):
        FileAccessor.from_sources(
            paths[:1], mode="read", accessor_types=(FirstFiles, SecondFiles)
        )


def test_unidentified_name_and_failed_header_can_be_retried(paths):
    with xr.open_dataset(paths[2]) as data:
        incomplete = data.copy(deep=False)
        del incomplete.attrs["master"]
        table = FileAccessor.from_sources(
            [{"source": incomplete, "source_loc": "renamed.nc"}],
            mode="guess_or_read",
            on_error="record",
        )
        assert [
            item.stage for item in table.source_trace.map(source_trace_issues).iloc[0]
        ] == [
            "filepath",
            "metadata",
        ]
        assert table.data_kind_toltec.eq(ToltecDataKind.unknown.value).all()
        assert table.data_kind_dummy_instru.eq(DummyInstruDataKind.unknown.value).all()
        with FileAccessor(table).open(io_objs=[data], on_error="raise") as recovered:
            FileAccessor(recovered).update_from_source_metadata(
                inplace=True, on_error="raise"
            )
        assert (
            recovered.data_kind_dummy_instru.iloc[0] == DummyInstruDataKind.counts.value
        )
        assert [
            item.stage
            for item in recovered.source_trace.map(source_trace_issues).iloc[0]
        ] == ["filepath"]
        assert recovered.obsnum.tolist() == [1]
        assert not data["counts"].variable._in_memory


def test_scoped_refresh_preserves_other_instrument_columns(paths):
    generic = FileAccessor.from_sources(paths[:2])
    refreshed = generic.toltec_file.update_from_source_metadata()
    pd.testing.assert_series_equal(
        refreshed.data_kind_dummy_instru, generic.data_kind_dummy_instru
    )
    assert refreshed.data_kind_toltec.iloc[0] == ToltecDataKind.lmt_tel.value
    assert refreshed.toltec_file.query(
        "bits_any_set(data_kind_toltec, raw_sweep)"
    ).index.tolist() == [1]


def test_open_retains_positions_and_borrows_resources(paths):
    with ExitStack() as stack:
        objects = [stack.enter_context(xr.open_dataset(path)) for path in paths[:4]]
        table = FileAccessor.from_sources(
            [{"source": data, "source_loc": "unidentified.nc"} for data in objects]
        )
        table.index = [4, 4, 2, 1]
        with FileAccessor(table).open(on_error="raise") as opened:
            FileAccessor(opened).update_from_source_metadata(
                inplace=True, on_error="raise"
            )
            pd.testing.assert_index_equal(opened.index, table.index)
            assert all(
                actual is expected
                for actual, expected in zip(opened.io_obj, objects, strict=True)
            )
            assert opened.data_models.iloc[0] == ["lmt", "toltec"]
            assert opened.data_models.iloc[3] == ["lmt", "dummy_instru"]
            assert (
                opened.data_kind_dummy_instru.iloc[2]
                == DummyInstruDataKind.counts.value
            )
        assert "io_obj" not in opened
        assert all(obj._close is not None for obj in objects)


@pytest.mark.parametrize(
    "order",
    list(permutations((LmtFileAccessor, ToltecFileAccessor, DummyInstruFileAccessor))),
)
@pytest.mark.parametrize("mode", ["guess", "read"])
def test_overlapping_support_is_independent_of_registration_order(
    order, paths, monkeypatch, mode
):
    monkeypatch.setattr(data_model_registry._data_models, "_entries", {})
    for cls in order:
        DataModelSpec(file_accessor_cls=cls).register()
    table = FileAccessor.from_sources(paths, mode=mode)
    assert table.data_kind_toltec.iloc[0] == ToltecDataKind.lmt_tel.value
    assert table.data_kind_dummy_instru.iloc[3] == DummyInstruDataKind.telescope.value


def test_overlapping_filename_handlers_contribute_to_the_same_row(paths):
    class ExtraTelescopeColumns(LmtFileAccessor):
        @staticmethod
        def _guess_derived_columns(table):
            return table[["obsnum"]].rename(columns={"obsnum": "extra_obsnum"})

    table = FileAccessor.from_sources(
        paths,
        accessor_types=(
            ExtraTelescopeColumns,
            ToltecFileAccessor,
            DummyInstruFileAccessor,
        ),
    )
    assert (
        table.extra_obsnum.take([0, 3]).tolist() == table.obsnum.take([0, 3]).tolist()
    )
    assert table.extra_obsnum.take([1, 2, 4]).isna().all()
    assert table.uid_raw_obs.iloc[0] == "tcs-145609-0-2"
    assert table.data_kind_toltec.iloc[0] == ToltecDataKind.lmt_tel.value
    assert table.data_kind_dummy_instru.iloc[3] == DummyInstruDataKind.telescope.value


@pytest.mark.parametrize("positions", [[], [0, 2], [0, 1, 2]])
def test_computed_columns_are_staged_by_position_without_mutating_input(positions):
    class ComputedFiles(FileAccessor[_TestDataModel]):
        @staticmethod
        def _guess_derived_columns(table):
            return pd.DataFrame(
                {"value": table.value + 10, "doubled": table.value * 2},
                index=table.index,
            )

    table = pd.DataFrame(
        {"value": pd.array([1, 2, 3], dtype="Int64")}, index=["same", "other", "same"]
    )
    before = table.copy()
    handlers = _FileCapabilities[FileAccessor](
        accessor_types=(ComputedFiles,), default_accessor=FileAccessor
    )
    computed = handlers._guess_derived_columns(table, {frozenset({"test"}): positions})
    pd.testing.assert_frame_equal(table, before)
    pd.testing.assert_index_equal(computed.index, table.index)
    expected = before.copy()
    expected["doubled"] = pd.array([None] * len(table), dtype="Int64")
    for position in positions:
        expected.iloc[position, 0] += 10
        expected.iloc[position, 1] = before.iloc[position, 0] * 2
    pd.testing.assert_frame_equal(computed, expected)
    table[computed.columns] = computed
    pd.testing.assert_frame_equal(table, expected)


@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_computed_updates_to_existing_columns_do_not_depend_on_scope(
    paths, generic, mixed
):
    class ComputedFiles(LmtFileAccessor):
        @staticmethod
        def _guess_derived_columns(table):
            # Mechanical hook check, not a telescope metadata correction.
            return pd.DataFrame({"obsnum": table.obsnum + 1}, index=table.index)

    sources = paths[:2] if mixed else paths[:1]
    if generic:
        table = FileAccessor.from_sources(sources, accessor_types=(ComputedFiles,))
    else:
        table = ComputedFiles.from_sources(sources)
    assert table.obsnum.iloc[0] == table.file_obsnum.iloc[0] + 1
    if mixed:
        assert table.obsnum.iloc[1] == table.file_obsnum.iloc[1]


@pytest.mark.parametrize("failure", ["raise", "wrong_index", "short_output"])
def test_failed_computation_does_not_install_earlier_outputs(failure):
    class First(FileAccessor[_TestDataModel]):
        @staticmethod
        def _guess_derived_columns(table):
            return pd.DataFrame({"value": table.value + 1}, index=table.index)

    class Second(First):
        @staticmethod
        def _guess_derived_columns(table):
            if failure == "raise":
                msg = "Computation failed"
                raise ValueError(msg)
            if failure == "wrong_index":
                return table.reset_index(drop=True)
            return table.iloc[:0]

    table = pd.DataFrame({"value": [1, 2]}, index=[4, 8])
    before = table.copy()
    handlers = _FileCapabilities[FileAccessor](
        accessor_types=(First, Second), default_accessor=FileAccessor
    )
    match = "Computation failed" if failure == "raise" else "preserve.*index"
    with pytest.raises(ValueError, match=match):
        handlers._guess_derived_columns(table, {frozenset({"test"}): [0, 1]})
    pd.testing.assert_frame_equal(table, before)


def test_generic_registration_does_not_widen_concrete_enrichment(paths):
    table = LmtFileAccessor.from_sources(paths)
    assert table.obsnum.notna().tolist() == [True, True, True, True, False]
    assert table.uid_raw_obs.notna().tolist() == [True, False, False, True, False]
    assert "data_kind_toltec" not in table
    table = ToltecFileAccessor.from_sources(paths)
    assert table.data_kind_toltec.tolist() == [
        ToltecDataKind.lmt_tel.value,
        ToltecDataKind.vnasweep.value,
        ToltecDataKind.unknown.value,
        ToltecDataKind.unknown.value,
        ToltecDataKind.unknown.value,
    ]
    assert str(table.data_kind_toltec.dtype) == "Int64"


def test_empty_table_retains_registered_columns():
    table = FileAccessor.from_sources([])
    assert table.empty
    assert table.data_kind_toltec.dtype == "Int64"
    assert table.data_kind_dummy_instru.dtype == "Int64"
    assert {"source", "obsnum", "file_prefix", "roach", "channel"} <= set(table)


def test_classifier_scope_precedes_scalar_rule_matches(paths, monkeypatch):
    # An overly broad rule cannot claim a row outside its memberships.
    monkeypatch.setattr(
        DummyInstruFileAccessor,
        "_data_kind_mapper",
        RecordRuleMapper(
            rules=[({"interface": "tel"}, DummyInstruDataKind.telescope.value)],
            default=DummyInstruDataKind.unknown.value,
        ),
    )
    table = FileAccessor.from_sources(paths[:1])
    assert table.data_kind_toltec.tolist() == [ToltecDataKind.lmt_tel.value]
    assert table.data_kind_dummy_instru.tolist() == [DummyInstruDataKind.unknown.value]
    assert "data_kind" not in table
    assert FileAccessor(table).query(
        "bits_any_set(data_kind_toltec, ToltecDataKind.lmt_tel) & "
        "bits_any_set(data_kind_dummy_instru, DummyInstruDataKind.unknown)"
    ).index.tolist() == [0]


def test_distinct_classifiers_cannot_overwrite_the_same_column(monkeypatch):
    monkeypatch.setattr(
        DummyInstruFileAccessor, "_data_kind_column", "data_kind_toltec"
    )
    with pytest.raises(ValueError, match="Conflicting classification column"):
        FileAccessor.from_sources([])


def test_query_combines_independent_masks_with_equal_numeric_bits(paths):
    assert ToltecDataKind.vnasweep.value == DummyInstruDataKind.counts.value
    table = FileAccessor.from_sources(paths)
    selected = FileAccessor(table).query(
        "bits_any_set(data_kind_toltec, ToltecDataKind.raw_sweep) | "
        "bits_any_set(data_kind_dummy_instru, DummyInstruDataKind.counts)"
    )
    assert selected.index.tolist() == [1, 2]


@pytest.mark.parametrize("reverse", [False, True])
def test_selected_reader_owns_its_projection_to_a_shared_column(reverse):
    class Reader:
        def __init__(self, data):
            self.data = data

        @property
        def meta(self):
            return self.data.meta

    class FirstFiles(FileAccessor[_FirstDataModel]):
        _reader_rules = SourceReaderRules(
            rules=(({"label": "first"}, Reader),),
            prepare_record=lambda obj: obj,
        )
        _metadata_columns: ClassVar = {
            "count": ModelColumn("foo", dtype="Int64", default=-7),
            "first_only": ModelColumn("foo", dtype="Int64", default=-7),
        }

    class SecondFiles(FileAccessor[_SecondDataModel]):
        _reader_rules = SourceReaderRules(
            rules=(({"label": "second"}, Reader),),
            prepare_record=lambda obj: obj,
        )
        _metadata_columns: ClassVar = {
            "count": ModelColumn("bar", converter=int, dtype="Int64")
        }

    handlers = (SecondFiles, FirstFiles) if reverse else (FirstFiles, SecondFiles)
    data = [
        SimpleNamespace(label="second", meta=SimpleNamespace(bar="20")),
        SimpleNamespace(label="first", meta=SimpleNamespace(foo=10)),
        SimpleNamespace(label="second", meta=SimpleNamespace(bar=None)),
        SimpleNamespace(label="first", meta=SimpleNamespace()),
    ]
    table = FileAccessor.from_sources(
        [{"source": obj, "source_loc": "unidentified.nc"} for obj in data],
        accessor_types=handlers,
    )
    table.index = ["same", "same", "last", "last"]
    table["count"] = pd.array([99] * 4, dtype="Int64")
    result = FileAccessor(table, accessor_types=handlers).update_from_source_metadata(
        on_error="raise"
    )
    assert result["count"].take([0, 1, 3]).tolist() == [20, 10, -7]
    assert pd.isna(result["count"].iloc[2])
    assert result["first_only"].take([1, 3]).tolist() == [10, -7]
    assert result["first_only"].take([0, 2]).isna().all()
    assert result["count"].dtype == table["count"].dtype
    pd.testing.assert_index_equal(result.index, table.index)
    assert table["count"].tolist() == [99] * 4


def test_distinct_flag_classes_cannot_share_a_query_name():
    class ConflictingFlags(FileAccessor[_ConflictingFlagsDataModel]):
        _data_kind_type = Flag("ToltecDataKind", {"counts": 1})  # ty: ignore[mismatched-type-name] -- Deliberate query-name collision.

    DataModelSpec(file_accessor_cls=ConflictingFlags).register()
    with pytest.raises(ValueError, match="Conflicting flag class name"):
        FileAccessor.from_sources([])


@pytest.mark.parametrize(
    "accessor_cls", [FileAccessor, ToltecFileAccessor, DummyInstruFileAccessor]
)
def test_unordered_interfaces_preserve_wildcard_position_semantics(paths, accessor_cls):
    table = accessor_cls.from_sources(paths[:4])
    assert "tel" in table.interface.tolist()
    pd.testing.assert_frame_equal(accessor_cls(table).query_obs_spec("*/[-1]"), table)


def test_accessor_context_composition_rejects_conflicting_label_numbers():
    class ConflictingLabels(FileAccessor[_ConflictingLabelsDataModel]):
        _obs_spec_context = ObsSpecContext(
            field_label_numbers={"interface": {"toltec0": 1}}
        )

    with pytest.raises(ValueError, match="Conflicting numeric interpretation"):
        FileAccessor.from_sources(
            [], accessor_types=(ToltecFileAccessor, ConflictingLabels)
        )


def test_mixed_formats_open_independently_of_instrument_handlers(paths, tmp_path):
    identified = []

    class TableReader:
        def __init__(self, data):
            self.data = data

        @property
        def meta(self):
            return SimpleNamespace(row_count=len(self.data))

    def identify_table(data):
        identified.append(data)
        return {} if isinstance(data, QTable) else None

    class TableFiles(FileAccessor[_TableDataModel]):
        _reader_rules = SourceReaderRules(
            rules=(({"table": True}, TableReader),),
            prepare_record=lambda obj: (
                {"table": True} if identify_table(obj) is not None else None
            ),
        )
        _metadata_columns: ClassVar = {
            "row_count": ModelColumn("row_count", dtype="Int64", default=None)
        }

    DataModelSpec(file_accessor_cls=TableFiles).register()

    data = QTable({"signal": [1, 2]})
    cached = tmp_path / "cached-without-extension"
    data.write(cached, format="ascii.ecsv")
    table = FileAccessor.from_sources(
        [
            paths[0],
            {"source": cached, "source_loc": "archive:/measurement.ecsv"},
            {"source": data, "source_loc": "archive:/borrowed.ecsv"},
        ]
    )
    with FileAccessor(table).open(on_error="raise") as opened:
        assert isinstance(opened.io_obj.iloc[0], xr.Dataset)
        assert isinstance(opened.io_obj.iloc[1], QTable)
        assert opened.io_obj.iloc[2] is data
        FileAccessor(opened).update_from_source_metadata(inplace=True, on_error="raise")
        assert opened.row_count.iloc[1:].tolist() == [2, 2]
        assert opened.data_kind_toltec.iloc[0] == ToltecDataKind.lmt_tel.value
        assert len(identified) == 2
    assert len(data) == 2
    assert "io_obj" not in opened


@pytest.mark.parametrize(
    "accessor_cls", [FileAccessor, LmtFileAccessor, ToltecFileAccessor]
)
def test_filename_projection_does_not_require_a_registered_accessor(
    monkeypatch, accessor_cls
):
    import re

    from tolteca_datamodels.core import filepath as _filepath

    monkeypatch.setattr(
        _filepath, "_filepath_parsers", _filepath._filepath_parsers.copy()
    )

    class UnhandledFilepath(FilepathInfo):
        serial: int

    _filepath.register_filepath_parser(
        pattern=re.compile(r"^unhandled-(?P<serial>\d+)\.nc$"),
        parsers=(),
        filepath_info_cls=UnhandledFilepath,
    )

    table = accessor_cls.from_sources(["unhandled-42.nc", "unidentified.nc"])
    assert table.file_serial.iloc[0] == 42
    assert pd.isna(table.file_serial.iloc[1])
    if accessor_cls is not LmtFileAccessor:
        assert table.data_kind_toltec.eq(ToltecDataKind.unknown.value).all()
    if accessor_cls is FileAccessor:
        assert table.data_kind_dummy_instru.eq(DummyInstruDataKind.unknown.value).all()
    assert table.source_trace.map(source_trace_issues).iloc[0] == []
