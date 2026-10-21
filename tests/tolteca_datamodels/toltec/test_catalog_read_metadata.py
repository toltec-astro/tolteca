"""Header enrichment of file tables supplied independently of discovery."""

from contextlib import contextmanager

import pandas as pd
import pytest
import xarray as xr
from tollan.utils.fileloc import fileloc
from xarray.backends.netCDF4_ import NetCDF4DataStore

from tests.source_helpers import patch_metadata_read
from tolteca_datamodels.core.file import source_trace_issues
from tolteca_datamodels.core.source import SourceReaderRules
from tolteca_datamodels.input_discovery import InputDiscovery
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.tel import LmtTelAccessor
from tolteca_datamodels.toltec.file import (
    ToltecFileAccessor,
    get_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.kids.core import ToltecKidsAccessor
from tolteca_datamodels.toltec.types import ToltecDataKind


@pytest.mark.parametrize(
    "filename", ["cached.nc", "tel_dummy_instru_2025-10-31_145609_00_0002.nc"]
)
def test_telescope_kind_uses_receiver_even_without_toltec_filename(
    data_lmt_path, filename
):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with xr.open_dataset(path) as original:
        data = original.assign({"Header.File.Name": filename})
        table = ToltecFileAccessor.from_sources(
            [{"source": data, "source_loc": "archive:/cached.nc"}], mode="read"
        )
        assert table.data_kind_toltec.tolist() == [ToltecDataKind.lmt_tel.value]
        assert table.data_models.tolist() == [["lmt", "toltec"]]
        assert "receiver" not in table
        assert table.instru.tolist() == ["lmt"]
        assert table.file_prefix.isna().all()
        assert table.obsnum.tolist() == [145609]
        assert table.filepath_orig.tolist() == [filename]
        assert not data["Data.TelescopeBackend.TelAzAct"].variable._in_memory


def test_telescope_metadata_corrects_current_affiliation_after_rule_binding(
    data_lmt_path,
):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with xr.open_dataset(path) as original:
        # Deliberate header/location mismatch tests the authority of the receiver.
        data = original.assign({"Header.Dcs.Receiver": "Dummy_Instru"})
        sources = [{"source": data, "source_loc": path}]
        lmt = LmtFileAccessor.from_sources(sources, mode="read")
        assert lmt.data_models.tolist() == [["lmt", "dummy_instru"]]
        table = ToltecFileAccessor.from_sources(sources, mode="read")
        assert table.data_kind_toltec.tolist() == [ToltecDataKind.unknown.value]
        assert table.data_models.tolist() == [["lmt", "dummy_instru"]]
        assert table.file_data_models.tolist() == [["lmt", "toltec"]]
        assert "receiver" not in table
        assert table.source_trace.map(source_trace_issues).iloc[0] == []
        with table.toltec_file.open() as opened:
            assert opened.io_obj.iloc[0] is None
        assert original._close is not None


def test_metadata_uses_the_same_reader_scope_as_open(data_lmt_path):
    class TelescopeFiles(ToltecFileAccessor):
        _reader_rules = SourceReaderRules(
            rules=ToltecFileAccessor._reader_rules.rules[:1],
            prepare_record=ToltecFileAccessor._reader_rules.prepare_record,
        )

    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    table = TelescopeFiles.from_sources([path], mode="read")
    assert table.data_kind_toltec.tolist() == [ToltecDataKind.tune.value]
    assert "atten_drive" not in table
    assert [
        item.stage for item in table.source_trace.map(source_trace_issues).iloc[0]
    ] == ["identify"]
    with TelescopeFiles(table).open() as opened:
        assert opened.io_obj.iloc[0] is not None
        assert opened.accessor_obj.iloc[0] is None
        assert any(
            item.stage == "identify"
            for item in opened.source_trace.map(source_trace_issues).iloc[0]
        )

    enriched = ToltecFileAccessor.from_sources([path], mode="read")
    inspected = TelescopeFiles(enriched).update_from_source_metadata(force=True)
    pd.testing.assert_frame_equal(
        inspected.drop(columns="source_trace"), enriched.drop(columns="source_trace")
    )
    assert (
        inspected.source_trace.map(source_trace_issues).iloc[0][-1].stage == "identify"
    )


def test_read_inspects_payload_when_current_identity_is_unknown(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    location = path.with_name(path.name.replace("toltec0", "hwpr"))
    table = ToltecFileAccessor.from_sources([{"source": path, "source_loc": location}])
    # Clear uncertain current identity without changing filename provenance.
    table[["interface", "interface_kind"]] = None
    read = ToltecFileAccessor(table).update_from_source_metadata(on_error="raise")
    assert read.interface.tolist() == ["toltec0"]
    assert read.roach.tolist() == [0]
    assert read.data_kind_toltec.tolist() == [ToltecDataKind.tune.value]
    assert table.file_interface.tolist() == ["hwpr"]


@pytest.mark.parametrize("mode", ["read", "guess_or_read"])
def test_tree_metadata_reads_the_explicit_acquisition_node(data_lmt_path, mode):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    with xr.open_dataset(path) as raw:
        tree = xr.DataTree.from_dict({"raw": raw, "unrelated": xr.Dataset()})
        table = ToltecFileAccessor.from_sources(
            [{"source": tree["raw"], "source_loc": "archive:/cached.nc"}], mode=mode
        )
        assert table.source.iloc[0] is tree["raw"]
        assert table.data_kind_toltec.tolist() == [ToltecDataKind.tune.value]
        assert table.obsnum.tolist() == [145609]
        assert table.atten_drive.tolist() == [7.25]
        assert list(tree.children) == ["raw", "unrelated"]
        assert not raw["Data.Toltec.Is"].variable._in_memory
        assert raw._close is not None


def test_reader_policy_accepts_a_duck_typed_accessor_class(data_lmt_path):
    class Reader:
        def __init__(self, data):
            self.reader = ToltecKidsAccessor(data)

        @property
        def meta(self):
            return self.reader.meta

        @property
        def data_kind(self):
            return self.reader.data_kind

    class Files(ToltecFileAccessor):
        _reader_rules = SourceReaderRules(
            rules=(({}, Reader),),
        )

    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    table = Files.from_sources([path], mode="read")
    assert table.data_kind_toltec.tolist() == [ToltecDataKind.tune.value]
    assert table.readout_code.tolist() == [0]
    assert table.atten_drive.tolist() == [7.25]


def test_catalog_enrichment_preserves_rows_and_does_not_discover(
    data_lmt_path, monkeypatch
):
    tel = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    (kids,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    catalog = guess_info_from_sources([kids, tel, kids])
    catalog.index = pd.Index([8, 3, 8], name="input_row")
    catalog["batch"] = ["first", "second", "third"]
    catalog.attrs["label"] = "selected inputs"
    before = catalog.copy(deep=True)
    open_dataset = xr.open_dataset
    opened = []

    def unexpected(*args, **kwargs):
        pytest.fail("Catalog enrichment invoked file discovery")

    def open_data(path, **kwargs):
        opened.append(path)
        return open_dataset(path, **kwargs)

    monkeypatch.setattr(InputDiscovery, "discover", unexpected)
    monkeypatch.setattr(xr, "open_dataset", open_data)
    result = catalog.toltec_file.update_from_source_metadata()
    assert result is not catalog
    pd.testing.assert_frame_equal(catalog, before)
    pd.testing.assert_index_equal(result.index, catalog.index)
    pd.testing.assert_series_equal(result["batch"], catalog["batch"])
    pd.testing.assert_series_equal(result["file_prefix"], catalog["file_prefix"])
    assert result.attrs == catalog.attrs
    assert opened == [kids, tel, kids]
    assert result["interface"].tolist() == ["toltec0", "tel", "toltec0"]


def test_catalog_uses_identification_location_but_opens_local_cache(
    data_lmt_path, tmp_path, monkeypatch
):
    source = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    cache = tmp_path / "download.cache"
    cache.symlink_to(source)
    remote = fileloc(f"archive:/data_lmt/tel/{source.name}")
    info = get_info_from_source(cache, source_loc=remote)
    assert info is not None
    catalog = guess_info_from_sources([{"source": cache, "source_loc": remote}])
    open_dataset = xr.open_dataset
    opened = []

    def open_data(path, **kwargs):
        opened.append(path)
        assert path == cache
        return open_dataset(path, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", open_data)
    result = catalog.toltec_file.update_from_source_metadata()
    assert opened == [cache]
    [identified] = list(result.itertuples(index=False))
    assert identified.source == info.source
    assert identified.source_loc is remote
    assert result["file_prefix"].iloc[0] == info.file_prefix


def test_remote_rows_keep_existing_header_snapshot(data_lmt_path, monkeypatch):
    (source,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    catalog = guess_info_from_sources(
        [source]
    ).toltec_file.update_from_source_metadata()
    catalog["source"] = f"archive:{source}"

    def unexpected(*args, **kwargs):
        pytest.fail("Remote source reached the local backend")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    result = catalog.toltec_file.update_from_source_metadata(force=True)
    pd.testing.assert_frame_equal(
        result.drop(columns="source_trace"), catalog.drop(columns="source_trace")
    )
    assert result.source_trace.iloc[0][-1].stage == "open"
    assert result.source_trace.iloc[0][-1].outcome == "error"
    assert len(result.toltec_file.query("atten_drive == 7.25")) == 1


def test_empty_catalog_enrichment():
    catalog = guess_info_from_sources([])
    result = catalog.toltec_file.update_from_source_metadata()
    pd.testing.assert_frame_equal(result, catalog)


@pytest.mark.parametrize("inplace", [True, False])
def test_header_failure_leaves_catalog_unchanged_and_closes_sources(
    data_lmt_path, monkeypatch, inplace
):
    paths = sorted((data_lmt_path / "tel").glob("*.nc"))[:2]
    assert len(paths) == 2
    catalog = guess_info_from_sources(paths)
    before = catalog.copy(deep=True)
    read_source = LmtTelAccessor.meta.func
    close_store = NetCDF4DataStore.close
    read = []
    closed = []

    def fail_second(data, **kwargs):
        read.append(paths[len(read)])
        if len(read) == 2:
            msg = "Test second-header failure"
            raise ValueError(msg)
        return read_source(data, **kwargs)

    def close(store, **kwargs):
        close_store(store, **kwargs)
        closed.append(store)

    patch_metadata_read(monkeypatch, ToltecFileAccessor._reader_rules, fail_second)
    monkeypatch.setattr(NetCDF4DataStore, "close", close)
    with pytest.raises(ValueError, match="second-header failure"):
        catalog.toltec_file.update_from_source_metadata(
            inplace=inplace, on_error="raise"
        )
    assert read == paths
    assert len(closed) == 2
    pd.testing.assert_frame_equal(catalog, before)


def test_metadata_replaces_prefilled_values_without_mutating_input(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    catalog = guess_info_from_sources([path]).toltec_file.update_from_source_metadata()
    catalog["atten_drive"] = 1.25
    refreshed = catalog.toltec_file.update_from_source_metadata(force=True)
    assert catalog["atten_drive"].tolist() == [1.25]
    assert refreshed["atten_drive"].tolist() == [7.25]
    pd.testing.assert_frame_equal(
        catalog.drop(columns=["atten_drive", "source_trace"]),
        refreshed.drop(columns=["atten_drive", "source_trace"]),
    )


def test_header_identifiers_and_filename_provenance_are_distinct(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    table = guess_info_from_sources([path])
    guessed_obsnum = int(table["obsnum"].iloc[0])
    with xr.open_dataset(path) as original:
        # Deliberate header/filename disagreement exercises authoritative reading.
        data = original.assign({"Header.Toltec.ObsNum": guessed_obsnum + 1})
        with table.toltec_file.open(io_objs=[data]) as read:
            read.toltec_file.update_from_source_metadata(inplace=True)
            assert read["obsnum"].iloc[0] == guessed_obsnum + 1
            assert read["file_obsnum"].iloc[0] == guessed_obsnum
            assert read.filepath_info.iloc[0] is table.filepath_info.iloc[0]
            assert read.filepath_info.iloc[0].obsnum == guessed_obsnum
            assert "metadata" not in read
            assert next(iter(read.itertuples(index=False))).obsnum == guessed_obsnum + 1
    assert table["obsnum"].iloc[0] == guessed_obsnum


def test_partial_refresh_keeps_skipped_headers_and_duplicate_indices(
    data_lmt_path, tmp_path, monkeypatch
):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    cache = tmp_path / "cached.nc"
    cache.symlink_to(path)
    table = guess_info_from_sources([f"archive:{path}", cache, f"archive:{path}"])
    # Borrowed objects allow initial reading without opening the remote resources.
    with (
        xr.open_dataset(path) as data,
        table.toltec_file.open(
            io_objs=[data, data, data], update_from_metadata=True
        ) as populated,
    ):
        table = populated
    table.index = pd.Index([8, 8, 3], name="input_row")
    table["batch"] = ["first", "second", "third"]
    table.attrs["label"] = "selected inputs"
    before = table.copy(deep=True)
    open_dataset = xr.open_dataset
    opened = []

    @contextmanager
    def tracked(source, **kwargs):
        opened.append(source)
        with open_dataset(source, **kwargs) as data:
            yield data.assign({"Header.Toltec.DriveAtten": 12.5})

    monkeypatch.setattr(xr, "open_dataset", tracked)
    refreshed = table.toltec_file.update_from_source_metadata(force=True)
    assert opened == [cache]
    assert refreshed["atten_drive"].tolist() == [7.25, 12.5, 7.25]
    pd.testing.assert_frame_equal(table, before)
    pd.testing.assert_index_equal(refreshed.index, table.index)
    pd.testing.assert_series_equal(refreshed["batch"], table["batch"])
    pd.testing.assert_series_equal(refreshed["file_prefix"], table["file_prefix"])
    assert refreshed.attrs == table.attrs


def test_no_metadata_model_does_not_erase_existing_metadata(
    data_lmt_path, tmp_path, monkeypatch
):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    cache = tmp_path / "cached.nc"
    cache.symlink_to(path)
    table = guess_info_from_sources(
        [path, cache]
    ).toltec_file.update_from_source_metadata()
    table["batch"] = ["kept", "refreshed"]

    # Supported products can be classified without raw acquisition metadata.
    patch_metadata_read(
        monkeypatch, ToltecFileAccessor._reader_rules, lambda reader: None
    )
    monkeypatch.setattr(
        ToltecFileAccessor,
        "_classify_metadata",
        staticmethod(lambda reader, meta: ToltecDataKind.unknown),
    )
    refreshed = table.toltec_file.update_from_source_metadata(force=True)
    assert refreshed["atten_drive"].iloc[0] == 7.25
    assert refreshed["atten_drive"].iloc[1] == 7.25
    assert refreshed["filepath_orig"].iloc[1] == table["filepath_orig"].iloc[1]
    assert table["atten_drive"].tolist() == [7.25, 7.25]
    pd.testing.assert_series_equal(refreshed["batch"], table["batch"])
    assert refreshed["data_kind_toltec"].iloc[1] == ToltecDataKind.unknown.value
    pd.testing.assert_frame_equal(
        refreshed.drop(
            columns=["data_kind_toltec", "source_trace", "_source_identification"]
        ),
        table.drop(
            columns=["data_kind_toltec", "source_trace", "_source_identification"]
        ),
    )
    assert [
        [item.stage for item in cell]
        for cell in refreshed.source_trace.map(source_trace_issues)
    ] == [
        [],
        ["filepath"],
    ]
    assert [
        [item.stage for item in cell]
        for cell in table.source_trace.map(source_trace_issues)
    ] == [
        [],
        ["filepath"],
    ]
