"""Create, reopen and package a small TolTECA product tree without a database.

Run with ``python examples/data_product_store.py NEW_OUTPUT_DIRECTORY``.
The acquisition identifiers follow the reference-data convention. The text
artifacts demonstrate publication; they are not scientific reduction results.
"""

# ruff: noqa: INP001, T201 -- standalone example with a printed output path

from contextlib import ExitStack
from pathlib import Path

from tollan.pipeline import Artifact, ArtifactFile, ArtifactSpec

from tolteca_datamodels.data_prod import (
    CounterRunNameGenerator,
    DataProd,
    DataProdAssoc,
    DataProdBatch,
    DataProdIndex,
    DataProdStore,
    WorkflowRun,
    resolve_index,
    write_artifacts,
)
from tolteca_datamodels.data_prod.metadata import RawObsMeta, RawObsProcMeta


def describe_acquisition(scan: int) -> DataProdIndex:
    """Describe one logical acquisition without opening any input files."""
    return DataProdIndex(
        product=DataProd(
            uid=f"tcs-145609-0-{scan}",
            meta=RawObsMeta(
                name=f"Acquisition {scan}",
                data_prod_type="dp_raw_obs",
                master="tcs",
                obsnum=145609,
                subobsnum=0,
                scannum=scan,
            ),
        ),
    )


def describe_result(raw: DataProdIndex, run: WorkflowRun) -> DataProdIndex:
    """Attach the acquisition primary and producing run to a result draft."""
    meta = raw.product.meta
    assert isinstance(meta, RawObsMeta)
    return DataProdIndex(
        product=DataProd(
            uid=f"{raw.product.uid}_proc",
            meta=RawObsProcMeta(
                name="Example processing result",
                data_prod_type="dp_raw_obs_proc",
                master=meta.master,
                obsnum=meta.obsnum,
                subobsnum=meta.subobsnum,
                scannum=meta.scannum,
            ),
            assocs=[
                DataProdAssoc(
                    uid=raw.product.uid,
                    is_primary=True,
                    data_prod_assoc_type="dpa_raw_obs_proc_raw_obs",
                )
            ],
        ),
        run=run,
    )


class TextSummary(ArtifactSpec[str]):
    """Write a one-line text report as ``summary.txt``."""

    def write(self, data: str, *, path: Path) -> list[ArtifactFile]:
        """Write the report beneath ``path`` and return its file."""
        target = path / "summary.txt"
        target.write_text(data + "\n", encoding="utf-8")
        return [ArtifactFile(path=target, format="text", name="summary")]


def publish_results(
    store: DataProdStore,
    acquisitions: list[DataProdIndex],
    *,
    key: str,
) -> tuple[WorkflowRun, list[DataProdIndex]]:
    """Publish several products in one execution and save its final outcome."""
    results = []
    with (
        ExitStack() as saved_runs,
        WorkflowRun.create(
            context={
                "inputs": [raw.product.uid for raw in acquisitions],
                "detail": key,
            },
        ) as run,
    ):
        saved_runs.callback(store.write_run, run)
        for raw in acquisitions:
            draft = describe_result(raw, run)
            with store.materialize(
                draft,
                run_name=CounterRunNameGenerator(key=key),
                tag="main",
            ) as output_index:
                report = Artifact(data=f"{key}: {raw.product.uid}", spec=TextSummary())
                write_artifacts(
                    output_index, [report], producer=TextSummary.get_context_key()
                )
            results.append(output_index)
    return run, results


def build_example(root: Path) -> DataProdBatch:
    """Create a new output tree, check its saved records and return a batch."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    # setup-start
    store = DataProdStore(root / "dataprod_toltec", namespace="scratch")
    acquisitions = [describe_acquisition(scan) for scan in (1, 2)]
    for raw in acquisitions:
        store.write_index(raw)
    # setup-end

    # publish-start
    first_run, first_results = publish_results(store, acquisitions, key="example")
    second_run, second_results = publish_results(store, acquisitions[:1], key="recheck")
    # publish-end

    # read-start
    group_path = store.get_product_path("tcs-145609-0-1", "dp_raw_obs_proc")
    selected_path = resolve_index(group_path, tag="main")
    selected = DataProdIndex.read(selected_path)
    summary = next(
        item
        for item in selected.product.data_items
        if item.meta.get("format") == "text"
    )
    source = selected.resolve_location(summary.source.location).joinpath(
        summary.source.path
    )
    assert source.path.read_text(encoding="utf-8") == "recheck: tcs-145609-0-1\n"

    group = DataProdIndex.read(group_path)
    assert len(group.products) == 2  # noqa: PLR2004 -- two published versions
    earlier = DataProdIndex.read(group_path, run_uids=[first_run.uid])
    assert len(earlier.products) == 1
    assert earlier.products[0].product.reference == first_results[0].product.reference
    # read-end

    for index in first_results:
        assert index.store_run is not None
        assert index.store_run.name == "0000-example"
    repeated = second_results[0]
    assert repeated.store_run is not None
    assert repeated.store_run.name == "0001-recheck"
    assert selected.product.reference == second_results[0].product.reference
    assert all(
        store.read_run(run.uid).status == "completed" for run in (first_run, second_run)
    )

    # batch-start
    second_group_path = store.get_product_path("tcs-145609-0-2", "dp_raw_obs_proc")
    batch = DataProdBatch(
        items=[
            *acquisitions,
            group,
            DataProdIndex.read(second_group_path),
            store.read_namespace_index(run_uids=[first_run.uid, second_run.uid]),
        ]
    )
    # batch-end
    assert DataProdIndex.read(resolve_index(second_group_path)).product.reference == (
        first_results[1].product.reference
    )
    assert len(batch.items) == len(acquisitions) + len(acquisitions) + 1
    return batch


if __name__ == "__main__":
    import sys

    build_example(Path(sys.argv[1]))
    print(Path(sys.argv[1]).resolve())
