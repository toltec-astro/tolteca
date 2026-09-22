"""Inspect raw-header requiredness and reduced metadata without changing inputs.

Run with the workspace interpreter. JSON is written to stdout. Missing-field
cases deliberately remove one mapped header from an in-memory Dataset view;
the acquisition files and existing product stores remain read-only.
"""

# ruff: noqa: INP001, T201 -- standalone diagnostic

import argparse
import json
from pathlib import Path

from tolteca_datamodels.toltec import open_toltec
from tolteca_datamodels.toltec.kids import ReducedSweepView
from tolteca_kids.product import open_sweep_product


def inspect_raw(path):
    """Record existing raw metadata behavior for deliberately incomplete views."""
    fields = (
        "obs_type",
        "atten_drive",
        "atten_sense",
        "cal_obsnum",
        "cal_subobsnum",
        "cal_scannum",
        "n_chans_max",
        "filepath_orig",
        "repeat",
        "adc_snap",
        "f_lo_center",
        "n_sweepreps",
    )
    with open_toltec(path) as data:
        mapper = data.toltec_kids.mapper
        meta = data.toltec_kids.meta
        result = {
            "type": type(meta).__name__,
            "instru_component": meta.instru_component,
            "repeat": meta.repeat,
            "n_blocks": getattr(meta, "n_blocks", None),
            "cases": {},
        }
        for field in fields:
            mapping = getattr(mapper.schema, field)
            if mapping not in mapper:
                continue
            name = mapper.get_name(mapping)
            changed = data.drop_vars(name)
            try:
                changed_meta = changed.toltec_kids.meta
            except Exception as exc:  # noqa: BLE001 -- record actual behavior
                observed = {"error": f"{type(exc).__name__}: {exc}"}
            else:
                observed = {
                    "type": type(changed_meta).__name__,
                    "value": getattr(changed_meta, field, "<field not declared>"),
                    "n_blocks": getattr(changed_meta, "n_blocks", None),
                }
            result["cases"][field] = {"removed": name, **observed}
        assert not data["Data.Toltec.Is"].variable._in_memory
        assert not data["Data.Toltec.Qs"].variable._in_memory
        assert not data["Header.Toltec.AdcSnapData"].variable._in_memory
        return result


def inspect_product(path):
    """Read the existing raw accessor's behavior on a saved reduced child."""
    with open_sweep_product(path) as tree:
        data = ReducedSweepView(tree).data_source
        mapper = data.toltec_kids.mapper
        field = mapper.schema.obs_type
        obs_type = mapper.get_arr(data, field)
        result = {"obs_type_dims": obs_type.dims, "obs_type_shape": obs_type.shape}
        try:
            meta = data.toltec_kids.meta
        except Exception as exc:  # noqa: BLE001 -- record current unsupported cases
            result["error"] = f"{type(exc).__name__}: {exc}"
        else:
            result.update(
                type=type(meta).__name__, n_times=meta.n_times, t_exp=str(meta.t_exp)
            )
        return result


def main():
    """Report raw-file and saved-product metadata contracts separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("product_root", type=Path)
    args = parser.parse_args()
    result = {
        "raw": {
            path.name: inspect_raw(path)
            for path in sorted((args.data_root / "toltec/tcs/toltec0").glob("*.nc"))
        },
        "products": {
            name: inspect_product(args.product_root / name)
            for name in ("vnasweep-last", "targsweep-last", "tune-0", "tune-last")
        },
    }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
