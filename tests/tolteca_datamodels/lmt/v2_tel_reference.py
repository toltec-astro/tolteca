"""Extract telescope values with a separate v2 interpreter."""

import json
import sys

import numpy as np


def main():
    """Read a reference file and write comparison values into a temporary NPZ."""
    # Import only in the v2 subprocess, not during v3 doctest collection.
    from tolteca_datamodels.lmt.tel import (  # noqa: PLC0415
        LmtTelFileIO,  # ty: ignore[unresolved-import]
    )

    source, destination = sys.argv[1:]
    with LmtTelFileIO(source).open() as reader:
        meta = reader.meta
        fields = {
            name: getattr(meta, name)
            for name in (
                "master",
                "master_name",
                "interface",
                "instru",
                "instru_component",
                "obsnum",
                "subobsnum",
                "scannum",
                "mapping_type",
                "obs_goal",
                "file_suffix",
                "file_ext",
            )
        }
        fields.update(
            source=str(meta.source.path),
            file_timestamp=meta.file_timestamp.isoformat(),
            t0=meta.t0.unix,
            t0_format=meta.t0.format,
            target=[meta.target.ra.rad, meta.target.dec.rad],
            target_off=[meta.target_off.ra.rad, meta.target_off.dec.rad],
            target_frame=meta.target.frame.name,
            target_off_frame=meta.target_off.frame.name,
        )
        data = reader.read()
        values = {
            name: getattr(data, name)
            for name in (
                "time",
                "az",
                "alt",
                "ra",
                "dec",
                "holdflag",
            )
        }
        fields["units"] = {
            name: str(value.unit)
            for name, value in values.items()
            if hasattr(value, "unit")
        }
        np.savez(destination, metadata=json.dumps(fields), **values)


if __name__ == "__main__":
    main()
