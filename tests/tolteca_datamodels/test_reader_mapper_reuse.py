"""Shared native-reader mappings with independent source values and caches."""

import pytest
import xarray as xr

from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.tel import LmtTelAccessor
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.hwpr import ToltecHwprAccessor
from tolteca_datamodels.toltec.kids.core import ToltecKidsAccessor


@pytest.mark.parametrize(
    ("accessor_cls", "patterns"),
    [
        (LmtTelAccessor, ("tel/*.nc",)),
        (
            ToltecKidsAccessor,
            ("toltec/tcs/toltec*/*_tune.nc", "toltec/tcs/toltec*/*_targsweep.nc"),
        ),
        (
            ToltecKidsAccessor,
            ("toltec/tcs/toltec*/*_0002_*.nc",),
        ),
        (
            ToltecKidsAccessor,
            ("toltec/tcs/toltec*/*_vnasweep.nc",),
        ),
        (ToltecHwprAccessor, ("toltec/tcs/hwpr/*.nc",)),
    ],
    ids=["telescope", "kids-sweep", "kids-timestream", "kids-vna", "hwpr"],
)
@pytest.mark.parametrize("as_tree", [False, True])
def test_mapper_reuse_across_reference_files(
    data_lmt_path, monkeypatch, accessor_cls, patterns, as_tree
):
    paths = [path for pattern in patterns for path in data_lmt_path.glob(pattern)]
    assert paths
    expected = []
    mappings = []
    for path in paths:
        with xr.open_dataset(path) as dataset:
            reader = accessor_cls(dataset)
            expected.append(reader.meta.model_dump(mode="json"))
            mappings.append(dict(reader.mapper.mapped_fields))
    mapper = reader.mapper
    assert all(mapping == mappings[0] for mapping in mappings)

    def unexpected(*args, **kwargs):
        pytest.fail("An explicitly supplied mapper must not be resolved again")

    monkeypatch.setattr(type(mapper), "from_data_source", unexpected)
    # All original sources have closed. Reopen to check both repeated access and
    # sharing across files with different header values and dimension lengths.
    previous = []
    for path, metadata in zip(paths, expected, strict=True):
        with xr.open_dataset(path) as dataset:
            source = dataset
            if as_tree:
                source = xr.DataTree(dataset)
            reader = accessor_cls(source, mapper=mapper)
            assert reader.mapper is mapper
            cached = reader.meta
            assert cached.model_dump(mode="json") == metadata
            assert reader.meta is cached
            assert dict(mapper.mapped_fields) == mappings[0]
            previous.append(reader.meta)
    assert [meta.model_dump(mode="json") for meta in previous] == expected


@pytest.mark.parametrize(
    "reader_case",
    [
        ("lmt_tel", LmtFileAccessor, "tel/*.nc"),
        ("toltec_kids", ToltecFileAccessor, "toltec/tcs/toltec0/*_tune.nc"),
        ("toltec_hwpr", ToltecFileAccessor, "toltec/tcs/hwpr/*.nc"),
    ],
)
@pytest.mark.parametrize("generic", [False, True])
def test_table_enrichment_reuses_live_accessor_metadata(
    data_lmt_path, monkeypatch, reader_case, generic
):
    accessor_name, file_accessor_cls, pattern = reader_case
    path = next(data_lmt_path.glob(pattern))
    if generic:
        file_accessor_cls = FileAccessor
    with xr.open_dataset(path) as source:
        reader = getattr(source, accessor_name)
        metadata = reader.meta
        kind = getattr(reader, "data_kind", None)

        table = file_accessor_cls.from_sources(
            [
                {
                    "source": source,
                    "source_loc": path.with_name(
                        path.name.replace(f"{metadata.obsnum:06d}", "000000")
                    ),
                }
            ],
            mode="guess",
        )
        assert table.obsnum.tolist() == [0]
        with file_accessor_cls(table).open() as opened:
            assert opened.io_obj.iloc[0] is source
            live_reader = opened.accessor_obj.iloc[0]
            assert live_reader is not reader
            assert live_reader.meta.model_dump() == metadata.model_dump()
            assert getattr(live_reader, "data_kind", None) == kind

            def unexpected(*args, **kwargs):
                pytest.fail("Table enrichment bypassed the live accessor's cache")

            monkeypatch.setattr(type(live_reader.mapper), "get_metadata", unexpected)
            if accessor_name == "toltec_kids":
                monkeypatch.setattr(
                    type(live_reader.mapper), "_get_data_kind", unexpected
                )
            enriched = file_accessor_cls(opened).update_from_source_metadata()
            assert enriched.obsnum.tolist() == [metadata.obsnum]
            assert enriched.uid_raw_obs_file.tolist() == [metadata.uid_raw_obs_file]
            assert table.obsnum.tolist() == [0]
            refreshed = file_accessor_cls(enriched).update_from_source_metadata(
                force=True
            )
            assert refreshed.obsnum.tolist() == [metadata.obsnum]
            file_accessor_cls(opened).update_from_source_metadata(
                inplace=True, force=True
            )
            assert opened.obsnum.tolist() == [metadata.obsnum]
        assert reader.meta is metadata
        if kind is not None:
            assert enriched.data_kind_toltec.tolist() == [kind.value]
