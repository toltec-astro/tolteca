"""Raw files become reusable observation inventories without manual relations."""

from pathlib import Path

import pytest
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.data_prod import (
    DataProdBatch,
    DataProdIndex,
    DataProdStore,
    Location,
    build_raw_obs_batch,
)
from tolteca_datamodels.raw_obs_context import RawObsContext, RawObsDataItem
from tolteca_datamodels.source import SourceSourceLoc
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.kids.core import ToltecKidsIOMapper

ROOT = (Path(__file__).parents[3] / "toltec_astro_dev/run/data_lmt").resolve()
LOCATIONS = {"data_lmt": Location(root=ROOT.as_uri(), location_type="local")}
TUNE = "tcs-145609-0-1"
SCIENCE = "tcs-145609-0-2"


@pytest.fixture
def files():
    return sorted(
        p
        for p in ROOT.rglob("*.nc")
        if "_145609_" in p.name
        and (p.name.startswith("toltec0_") or p.name.startswith("tel_toltec_"))
    )


def _indexes(batch):
    assert isinstance(batch, DataProdBatch)
    assert all(isinstance(index, DataProdIndex) for index in batch.items)
    return {index.product.uid: index for index in batch.items}


def _snapshot(batch):
    return batch.model_dump(mode="json")


def _context(batch):
    indexes = _indexes(batch)
    return RawObsContext.from_product(
        indexes[SCIENCE].product,
        data_items=(RawObsDataItem(interface="toltec0", block=0),),
        cal_products={
            uid: i.product
            for uid, i in indexes.items()
            if i.product.meta.data_prod_type == "dp_raw_obs"
        },
    )


def test_unordered_mixed_files_publish_and_reopen_context(files, tmp_path, monkeypatch):
    original = ToltecKidsIOMapper.get_arr

    def without_iq(self, data, field, *args, **kwargs):
        assert field not in (self.schema.I, self.schema.Q)
        return original(self, data, field, *args, **kwargs)

    monkeypatch.setattr(ToltecKidsIOMapper, "get_arr", without_iq)
    batch = build_raw_obs_batch(reversed(files), locations=LOCATIONS)
    indexes = _indexes(batch)
    assert set(indexes) == {"tcs-145609", TUNE, SCIENCE}
    assert _snapshot(batch) == _snapshot(
        build_raw_obs_batch(files, locations=LOCATIONS)
    )
    assert {i.meta["interface"] for i in indexes[SCIENCE].product.data_items} == {
        "toltec0",
        "tel",
    }
    assert {a.uid for a in indexes["tcs-145609"].product.assocs} == {TUNE, SCIENCE}
    assert [
        b.kind for b in indexes[TUNE].product.cal_refs[0].meta.block_history.blocks
    ] == ["sweep", "sweep"]
    context = _context(batch)
    sweep = context.data_items[0].cal_items[0]
    assert sweep.raw_obs.uid == TUNE
    assert sweep.data_items[0].block == 1
    store = DataProdStore(tmp_path)
    reopened = {}
    for uid, index in indexes.items():
        path = store.write_index(index, primary_uid=uid)
        reopened[uid] = DataProdIndex.read(path)
    reopened_batch = DataProdBatch(
        meta={"selection": "reference"}, items=list(reopened.values())
    )
    assert _context(reopened_batch) == context
    # Generated index artifacts are retained during enrichment, without reopening them.
    updated = build_raw_obs_batch(files, locations=LOCATIONS, existing=reopened_batch)
    assert updated.meta == reopened_batch.meta
    assert updated.meta is not reopened_batch.meta
    assert _context(updated) == context
    for uid in indexes:
        assert len(_indexes(updated)[uid].product.data_items) == len(
            reopened[uid].product.data_items
        )


def test_calibration_arriving_later_resolves_without_mutating_existing(files):
    science = [
        p for p in files if p.name.startswith("toltec0_") and "tune" not in p.name
    ]
    tune = [p for p in files if p.name.startswith("toltec0_") and "tune" in p.name]
    before = build_raw_obs_batch(science, locations=LOCATIONS)
    assert _indexes(before)[SCIENCE].product.cal_refs[0].target_uid is None
    saved = _snapshot(before)
    after = build_raw_obs_batch(tune, locations=LOCATIONS, existing=before)
    assert _snapshot(before) == saved
    assert _indexes(after)[SCIENCE].product.cal_refs[0].target_uid == TUNE
    assert _context(after).data_items[0].cal_items[0].data_items[0].block == 1
    assert _snapshot(
        build_raw_obs_batch(science + tune, locations=LOCATIONS, existing=after)
    ) == _snapshot(after)


def test_bad_file_and_conflicting_history_leave_previous_inventory_intact(
    files, tmp_path
):
    existing = build_raw_obs_batch(files, locations=LOCATIONS)
    saved = _snapshot(existing)
    with pytest.raises(OSError):
        build_raw_obs_batch(
            [ROOT / "missing.nc"], locations=LOCATIONS, existing=existing
        )
    assert _snapshot(existing) == saved
    ref = _indexes(existing)[TUNE].product.cal_refs[0]
    _indexes(existing)[TUNE].product.cal_refs[0] = ref.model_copy(
        update={
            "meta": ref.meta.model_copy(update={"cal_obsnum": 999999}),
        }
    )
    saved = _snapshot(existing)
    with pytest.raises(ValueError, match="Conflicting calibration reference"):
        build_raw_obs_batch(files, locations=LOCATIONS, existing=existing)
    assert _snapshot(existing) == saved


def test_headers_determine_identity_and_cached_source_keeps_openable_uri(
    files, tmp_path
):
    original = next(
        p for p in files if p.name.startswith("toltec0_") and "tune" in p.name
    )
    renamed = tmp_path / "unfamiliar 50%23#?.nc"
    renamed.symlink_to(original.resolve())
    sources = [SourceSourceLoc(source=renamed, source_loc="archive:/original.nc")]
    batch = build_raw_obs_batch(
        sources, locations={"cache": Location(root=tmp_path.as_uri())}
    )
    indexes = _indexes(batch)
    assert set(indexes) == {"tcs-145609", TUNE}
    item = indexes[TUNE].product.data_items[0]
    assert indexes[TUNE].get_source_uri(item.source) == fileloc(renamed).url
    assert (item.meta["master"], item.meta["obsnum"], item.meta["scannum"]) == (
        "tcs",
        145609,
        1,
    )


def test_canonical_source_under_symlinked_archive(files, tmp_path):
    source = next(p for p in files if p.name.startswith("toltec0_")).resolve()
    alias = tmp_path / "named archive"
    alias.symlink_to(source.parent, target_is_directory=True)
    locations = {"archive": Location(root=alias.as_uri())}
    batch = build_raw_obs_batch([source], locations=locations)
    index = next(
        i for i in batch.items if i.product.meta.data_prod_type == "dp_raw_obs"
    )
    ref = index.product.data_items[0].source
    assert ref.location == "archive"
    assert ref.path.name == source.name
    assert index.get_source_uri(ref) == (alias / source.name).as_uri()
    with pytest.raises(ValueError, match="matched 2"):
        build_raw_obs_batch(
            [source],
            locations={
                **locations,
                "physical": Location(root=source.parent.as_uri()),
            },
        )


def test_multiple_networks_and_hwpr_group_by_acquisition():
    paths = sorted(
        p
        for p in ROOT.rglob("*.nc")
        if "_156755_" in p.name
        and p.name.startswith(("toltec7_", "toltec11_", "hwpr_"))
    )
    assert len(paths) == 5
    indexes = _indexes(build_raw_obs_batch(paths, locations=LOCATIONS))
    raw = indexes["tcs-156755-0-2"].product
    assert {item.meta["interface"] for item in raw.data_items} == {
        "toltec7",
        "toltec11",
        "hwpr",
    }
    assert {ref.interface for ref in raw.cal_refs} == {"toltec7", "toltec11"}
    assert all(ref.target_uid == "tcs-156755-0-1" for ref in raw.cal_refs)


def test_borrowed_file_is_not_closed_and_duplicate_source_is_read_once(
    files, monkeypatch
):
    import tolteca_datamodels.data_prod.raw_obs as implementation

    path = next(p for p in files if p.name.startswith("toltec0_") and "tune" in p.name)
    original = implementation.read_raw_obs_index
    reads = []

    def read(source, **kwargs):
        reads.append(source)
        return original(source, **kwargs)

    monkeypatch.setattr(implementation, "read_raw_obs_index", read)
    with open_toltec_kids(path) as data:
        source = SourceSourceLoc(source=data, source_loc=path)
        build_raw_obs_batch([source, source], locations=LOCATIONS)
        signal = data["Data.Toltec.Is"]
        assert signal.isel({signal.dims[0]: 0}).load().size > 0
    assert len(reads) == 1


def test_declared_roots_are_portable_and_invalid_placement_reads_nothing(
    files, tmp_path, monkeypatch
):
    import tolteca_datamodels.data_prod.raw_obs as implementation

    batch = build_raw_obs_batch(files, locations=LOCATIONS)
    for index in batch.items:
        if index.product.data_items:
            assert set(index.locations) == {"data_lmt"}
        for item in index.product.data_items:
            assert item.source.location == "data_lmt"
            assert not item.source.path.is_absolute()
            assert fileloc(index.get_source_uri(item.source)).path in files
    assert DataProdBatch.model_validate_json(batch.model_dump_json()) == batch

    def forbid_read(*args, **kwargs):
        pytest.fail("Invalid placement must be detected before file inspection")

    monkeypatch.setattr(implementation, "read_raw_obs_index", forbid_read)
    source = files[0]
    for locations in (
        {},
        {"elsewhere": Location(root=tmp_path.as_uri())},
        {**LOCATIONS, "nested": Location(root=source.parent.as_uri())},
    ):
        with pytest.raises(ValueError, match="exactly one declared location"):
            build_raw_obs_batch([source], locations=locations)
    with pytest.raises(ValueError, match="absolute root"):
        build_raw_obs_batch([source], locations={"relative": Location(root=".")})


def test_existing_batch_rejects_unsupported_entries_duplicates_and_root_changes(
    files, tmp_path
):
    existing = build_raw_obs_batch(files, locations=LOCATIONS)
    duplicate = DataProdBatch(items=[existing.items[0], existing.items[0]])
    with pytest.raises(ValueError, match="Duplicate product"):
        build_raw_obs_batch([], locations=LOCATIONS, existing=duplicate)

    store = DataProdStore(tmp_path)
    store.ensure_namespace()
    unsupported = DataProdBatch(items=[store.read_namespace_index()])
    with pytest.raises(ValueError, match="raw/observation product indexes"):
        build_raw_obs_batch([], locations=LOCATIONS, existing=unsupported)

    saved = _snapshot(existing)
    with pytest.raises(ValueError, match="Conflicting storage location"):
        build_raw_obs_batch(
            [],
            locations={"data_lmt": Location(root=tmp_path.as_uri())},
            existing=existing,
        )
    assert _snapshot(existing) == saved
    assert build_raw_obs_batch([], locations=LOCATIONS).items == []
