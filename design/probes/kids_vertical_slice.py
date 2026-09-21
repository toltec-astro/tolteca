"""Exercise the local reducer-only vertical slice against v2 reference values.

All inputs are read-only. OUTPUT must be new; products, logs, comparison arrays,
SQLite and Parquet state are created only there. Run with the workspace Python.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy import units as u
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from tolteca_db.ingest.file_scanner import guess_info_from_file
from tolteca_db.ingest.ingest import DataIngestor
from tolteca_db.ingest.sweep_product import register_sweep_product
from tolteca_db.models.orm import (
    Base,
    DataProdAssoc,
    DataProdAssocType,
    DataProdSource,
    DataProdType,
    Location,
)
from tolteca_db.sweep_catalog import write_sweep_catalog
from tolteca_web.obs.catalog import ParquetCatalogBackend
from tolteca_web.sweep.products import SweepProductService, create_app

from tolteca_kids.product import open_sweep_product, read_sweep_manifest
from tolteca_kids.sweep_reduce import SweepReduce


def _run(argv, cwd, log):
    result = subprocess.run(
        argv, cwd=cwd, check=False, capture_output=True, text=True, timeout=180
    )
    log.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr)
    result.check_returncode()
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--v2_python", type=Path, required=True)
    parser.add_argument(
        "--deploy_source",
        type=Path,
        required=True,
        help="Isolated Deploy worktree's src directory",
    )
    args = parser.parse_args()
    root, output = args.reference_root.absolute(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Import precisely the isolated Deploy implementation under review.
    sys.path.insert(0, str(args.deploy_source.resolve()))
    from tolteca_deploy.launch import plan_kids

    v2_script = Path(__file__).parents[2] / "tests/tolteca_kids/v2_sweep_reference.py"
    engine = create_engine(f"sqlite:///{output / 'catalog.sqlite'}")
    Base.metadata.create_all(engine)
    cases = []
    with Session(engine) as session:
        raw_location = Location(label="fixture", root_uri=root.as_uri())
        products = Location(label="products", root_uri=output.as_uri())
        session.add_all(
            [
                raw_location,
                products,
                DataProdType(label="dp_raw_obs", level=0),
                DataProdType(label="dp_reduced_obs", level=1),
                DataProdAssocType(label="dpa_reduced_obs_raw_obs"),
            ]
        )
        session.commit()
        location_pk = products.pk
        for suffix, block in (
            ("vnasweep", -1),
            ("targsweep", -1),
            ("tune", 0),
            ("tune", -1),
        ):
            source = sorted((root / "toltec/tcs/toltec0").glob(f"*_{suffix}.nc"))[0]
            name = f"{suffix}-{'last' if block == -1 else block}"
            app_args = [
                "--kids.output.enabled",
                "true",
                "--kids.output.name",
                name,
                "--kids.sweep_reduce.block",
                str(block),
                "--",
                str(source),
            ]
            plan = plan_kids(
                app_args,
                environ={
                    "TOLTECPIPE_DATA_LMT_ROOT": str(root),
                    "TOLTECPIPE_DATAPROD_TOLTEC_ROOT": str(output),
                },
            )
            (output / f"{name}.command.json").write_text(plan.model_dump_json(indent=2))
            if not cases:
                # Verify no-op equivalence before the first reduction.
                noop = plan_kids(["--dry_run", "--", str(source)], environ={})
                direct = _run(
                    [
                        sys.executable,
                        "-m",
                        "tolteca_kids.cli",
                        "--dry_run",
                        "--",
                        str(source),
                    ],
                    noop.cwd,
                    output / "noop-direct.log",
                )
                launched = _run(noop.argv, noop.cwd, output / "noop-deployed.log")
                assert direct == launched
            _run(plan.argv, plan.cwd, output / f"{name}.log")
            path = output / name
            manifest = read_sweep_manifest(path)
            assert manifest.input is not None and manifest.config is not None
            assert manifest.config.sweep_reduce.block == block
            expected_path = output / f"{name}.v2.npz"
            _run(
                [
                    str(args.v2_python),
                    str(v2_script),
                    str(source),
                    str(expected_path),
                    str(block),
                ],
                plan.cwd,
                output / f"{name}.v2.log",
            )
            with np.load(expected_path) as expected, open_sweep_product(path) as tree:
                invocation = SweepReduce.view(tree)
                view = invocation.selected_block
                arrays = ("I", "Q", "unc_I", "unc_Q", "f_chan", "sweep", "frequency")
                for field in arrays:
                    value = getattr(view, field)
                    assert value is not None, field
                    np.testing.assert_array_equal(
                        value.values, expected[field], err_msg=f"{name}: {field}"
                    )
                for field, expected_unit in zip(
                    ("I", "Q", "unc_I", "unc_Q"), expected["units"], strict=True
                ):
                    unit = getattr(view, field).u.unit
                    assert (
                        u.dimensionless_unscaled if unit is None else unit
                    ) == u.Unit(expected_unit)
                assert view.frequency.u.unit == u.Hz
                info = manifest.input
                uri = str(source.relative_to(root))
                raw_source = session.get(DataProdSource, uri)
                if raw_source is None:
                    parsed = guess_info_from_file(source)
                    assert parsed is not None
                    _, raw_source = DataIngestor(
                        session, raw_location.pk, master=info.master, nw_id=info.roach
                    ).ingest_file(parsed)
                assert raw_source is not None
                product, _ = register_sweep_product(
                    session, path, location=products, raw_source=raw_source
                )
                cases.append(
                    {
                        "name": name,
                        "source": str(source),
                        "product_pk": product.pk,
                        "block_requested": block,
                        "block_index": invocation.context.block_index,
                        "arrays_exact": list(arrays),
                        "units_exact": True,
                        "I_shape": list(invocation.I.shape),
                        "I_chunks": list(invocation.I.encoding["chunks"]),
                    }
                )
        session.commit()

    # Export from committed ORM state through the DB API, not manually built rows.
    with Session(engine) as session:
        products = session.get(Location, location_pk)
        assert products is not None
        write_sweep_catalog(session, output / "catalog.parquet", location=products)
        assert len(session.scalars(select(DataProdAssoc)).all()) == len(cases)
    engine.dispose()
    catalog = ParquetCatalogBackend(output / "catalog.parquet", output)
    service = SweepProductService(catalog, output)
    assert service.query(master="tcs", nw_list=[0]).height == 4
    for case in cases:
        figure = service.figure(case["product_pk"], [0, 1])
        with np.load(output / f"{case['name']}.v2.npz") as expected:
            for i, channel in enumerate((0, 1)):
                np.testing.assert_array_equal(
                    figure.data[2 * i].y, expected["I"][channel]
                )
                np.testing.assert_array_equal(
                    figure.data[2 * i + 1].y, expected["Q"][channel]
                )
                np.testing.assert_array_equal(
                    figure.data[2 * i].x, expected["frequency"][channel]
                )
        figure.write_html(output / f"{case['name']}.html", include_plotlyjs=True)
        case["web_traces_exact"] = True
    app = create_app(output / "catalog.parquet", output)
    client = app.server.test_client()
    assert client.get("/_dash-layout").status_code == 200
    result = client.post(
        "/_dash-update-component",
        json={
            "output": "product.options",
            "outputs": {"id": "product", "property": "options"},
            "inputs": [{"id": "reload", "property": "n_clicks", "value": 0}],
            "state": [],
            "changedPropIds": ["reload.n_clicks"],
        },
    )
    assert result.status_code == 200, result.data
    assert len(result.json["response"]["product"]["options"]) == 4
    (output / "report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "scope": "reducer-only, not later KIDs science stages",
                "cases": cases,
                "dash_catalog_callback": "passed",
                "noop_command_equivalence": "passed",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"PASS: {len(cases)} exact v2 cases → CLI/Zarr/DB/Parquet/Web: {output}")


if __name__ == "__main__":
    main()
