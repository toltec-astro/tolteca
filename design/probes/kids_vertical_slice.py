"""Local integration probe, not a proposed DB registration or catalog API.

Run with the workspace interpreter, supplying a reference root and a NEW output
directory. Uses no existing database. All generated state remains under output.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from tolteca_db.ingest.file_scanner import guess_info_from_file
from tolteca_db.ingest.ingest import DataIngestor
from tolteca_db.models.metadata import ProcessContext, ReducedObsMeta, RoachInterfaceMeta
from tolteca_db.models.orm import (
    Base, DataProd, DataProdAssoc, DataProdAssocType, DataProdSource, DataProdType, Location,
)
from tolteca_kids.product import open_sweep_product
from tolteca_kids.sweep_reduce import SweepReduceView
from tolteca_web.obs.catalog import ParquetCatalogBackend
from tolteca_web.sweep.reduced import reduced_sweep_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root, output = args.reference_root.absolute(), args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    engine = create_engine(f"sqlite:///{output / 'catalog.sqlite'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        raw_location = Location(label="fixture", location_type="filesystem", root_uri=root.as_uri())
        result_location = Location(label="products", location_type="filesystem", root_uri=output.as_uri())
        raw_type = DataProdType(label="dp_raw_obs", level=0)
        reduced_type = DataProdType(label="dp_reduced_obs", level=1)
        association_type = DataProdAssocType(label="dpa_reduced_obs_raw_obs")
        session.add_all([raw_location, result_location, raw_type, reduced_type, association_type])
        session.flush()
        for suffix in ("targsweep", "vnasweep", "tune"):
            source = sorted((root / "toltec/tcs/toltec0").glob(f"*_{suffix}.nc"))[0]
            info = guess_info_from_file(source)
            assert info is not None
            raw, raw_source = DataIngestor(session, raw_location.pk, master="tcs", nw_id=info.roach).ingest_file(info)
            assert raw is not None and raw_source is not None
            product = output / suffix
            # Same subprocess entry point an orchestrator or a user can rerun.
            subprocess.run([sys.executable, "-m", "tolteca", "kids", "reduce", str(source),
                            "--output", str(output), "--kids.output.name", suffix], check=True)
            manifest = json.loads((product / "manifest.json").read_text())
            for key in ("obsnum", "subobsnum", "scannum", "roach", "file_suffix"):
                assert manifest["input"][key] == getattr(info, key), key
            with open_sweep_product(product) as tree:
                context = SweepReduceView(tree).context
            obs = {key: getattr(info, key) for key in ("obsnum", "subobsnum", "scannum")}
            reduced = DataProd(
                data_prod_type_fk=reduced_type.pk, availability_state="available",
                meta=ReducedObsMeta(**obs, master="tcs", name=product.name,
                                    data_prod_type="dp_reduced_obs", reduction_method="SweepReduce"),
            )
            session.add(reduced)
            session.flush()
            session.add(DataProdSource(
                source_uri=str(product.relative_to(output)), location_fk=result_location.pk,
                data_prod_fk=reduced.pk, availability_state="available",
                meta=RoachInterfaceMeta(**obs, master="tcs", roach=info.roach,
                                        nw_id=info.roach, interface=info.interface),
            ))
            session.add(DataProdAssoc(
                data_prod_assoc_type_fk=association_type.pk,
                src_data_prod_fk=reduced.pk, dst_data_prod_fk=raw.pk,
                context=ProcessContext(module="tolteca_kids.sweep_reduce", version=manifest["software"]["tolteca"],
                                       config=context.config.model_dump(mode="json")),
            ))
        session.commit()

    # Derive the catalog from committed ORM records, not the in-memory inputs.
    rows = []
    with Session(engine) as session:
        statement = (select(DataProdSource).join(DataProd).join(DataProdType)
                     .where(DataProdType.label == "dp_reduced_obs"))
        for source in session.scalars(statement):
            meta = source.meta
            rows.append({
                "product_pk": source.data_prod_fk,
                "master": meta.master, "obsnum": meta.obsnum, "subobsnum": meta.subobsnum,
                "scannum": meta.scannum, "nw": meta.nw_id,
                "data_kind": "reduced_sweep", "product_path": source.source_uri,
                "zarr_path": f"{source.source_uri}/arrays.zarr",
            })
        edges = session.scalars(select(DataProdAssoc)).all()
        assert len(edges) == len(rows) == 3
        for edge in edges:
            assert session.get(DataProd, edge.src_data_prod_fk).meta.tag == "reduced_obs"
            assert session.get(DataProd, edge.dst_data_prod_fk).meta.tag == "raw_obs"
    pl.DataFrame(rows).write_parquet(output / "catalog.parquet")
    catalog = ParquetCatalogBackend(output / "catalog.parquet", output)
    matches = catalog.query(data_kinds=["reduced_sweep"], nw_list=[0], master="tcs")
    assert matches.height == 3
    for row in matches.iter_rows(named=True):
        assert catalog.get_zarr_path(row["master"], row["obsnum"], row["subobsnum"],
                                     row["scannum"], row["nw"]) == output / row["zarr_path"]
        with open_sweep_product(output / row["product_path"]) as tree:
            view = SweepReduceView(tree)
            figure = reduced_sweep_figure(view, [0, 1])
            for i, channel in enumerate((0, 1)):
                np.testing.assert_array_equal(figure.data[2*i].y, view.selected_block.I.values[channel])
                np.testing.assert_array_equal(figure.data[2*i+1].y, view.selected_block.Q.values[channel])
                np.testing.assert_array_equal(figure.data[2*i].x, view.selected_block.frequency.values[channel])
            figure.write_html(output / f"{row['product_path']}.html", include_plotlyjs=True)
    engine.dispose()
    print(f"PASS: CLI → Zarr → DB lineage → Parquet discovery → Web I/Q figures: {output}")


if __name__ == "__main__":
    main()
